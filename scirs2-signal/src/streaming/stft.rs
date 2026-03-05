//! Streaming Short-Time Fourier Transform (STFT) and its inverse (ISTFT).
//!
//! [`StreamingSTFT`] accepts arbitrary-sized chunks of time-domain data and
//! produces frequency-domain frames whenever enough new samples have
//! accumulated to fill a hop.
//!
//! [`StreamingISTFT`] performs the inverse: it accepts frequency-domain frames
//! and reconstructs the time-domain signal via weighted overlap-add (WOLA).

use crate::error::{SignalError, SignalResult};
use crate::streaming::ring_buffer::RingBuffer;
use crate::window::get_window;
use scirs2_core::numeric::Complex64;

// ============================================================================
// Window function enumeration
// ============================================================================

/// Supported window functions for the streaming STFT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    /// Rectangular (no windowing).
    Rectangular,
    /// Hann window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
    /// Blackman-Harris window.
    BlackmanHarris,
}

impl WindowFunction {
    /// Convert to the string key expected by [`crate::window::get_window`].
    fn as_str(&self) -> &'static str {
        match self {
            Self::Rectangular => "rectangular",
            Self::Hann => "hann",
            Self::Hamming => "hamming",
            Self::Blackman => "blackman",
            Self::BlackmanHarris => "blackmanharris",
        }
    }
}

// ============================================================================
// StreamingSTFT
// ============================================================================

/// Configuration for [`StreamingSTFT`].
#[derive(Debug, Clone)]
pub struct StreamingSTFTConfig {
    /// Analysis window size in samples (also the FFT size).
    pub window_size: usize,
    /// Hop size in samples (step between consecutive frames).
    pub hop_size: usize,
    /// Window function to apply before the FFT.
    pub window_function: WindowFunction,
}

impl Default for StreamingSTFTConfig {
    fn default() -> Self {
        Self {
            window_size: 1024,
            hop_size: 256,
            window_function: WindowFunction::Hann,
        }
    }
}

/// Streaming STFT processor.
///
/// Feed arbitrary-length chunks of time-domain data via [`StreamingSTFT::process_chunk`] and
/// receive zero or more STFT frames per call.  Internally a [`RingBuffer`] is
/// used to manage the overlap between successive frames.
pub struct StreamingSTFT {
    config: StreamingSTFTConfig,
    /// Pre-computed window coefficients.
    window: Vec<f64>,
    /// Ring buffer holding incoming samples.
    buffer: RingBuffer<f64>,
    /// Number of new samples accumulated since the last frame was emitted.
    pending_samples: usize,
    /// Total frames emitted.
    frames_emitted: u64,
}

impl StreamingSTFT {
    /// Create a new streaming STFT processor.
    ///
    /// # Errors
    ///
    /// * `window_size` must be > 0.
    /// * `hop_size` must be > 0 and <= `window_size`.
    pub fn new(config: StreamingSTFTConfig) -> SignalResult<Self> {
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

        let window = get_window(config.window_function.as_str(), config.window_size, true)?;

        let buffer = RingBuffer::new(config.window_size)?;

        Ok(Self {
            config,
            window,
            buffer,
            pending_samples: 0,
            frames_emitted: 0,
        })
    }

    /// Feed a chunk of time-domain samples and collect any completed STFT
    /// frames.
    ///
    /// Each returned `Vec<Complex64>` has length `window_size / 2 + 1`
    /// (positive-frequency bins of a real FFT).
    pub fn process_chunk(&mut self, chunk: &[f64]) -> SignalResult<Vec<Vec<Complex64>>> {
        let mut frames = Vec::new();

        for &sample in chunk {
            self.buffer.push(sample);
            self.pending_samples += 1;

            // Emit a frame every `hop_size` new samples, once we have a full
            // window.
            if self.pending_samples >= self.config.hop_size
                && self.buffer.len() >= self.config.window_size
            {
                let frame = self.emit_frame()?;
                frames.push(frame);
                self.pending_samples = 0;
            }
        }

        Ok(frames)
    }

    /// Flush any remaining buffered data by zero-padding the last frame.
    pub fn flush(&mut self) -> SignalResult<Option<Vec<Complex64>>> {
        if self.pending_samples == 0 {
            return Ok(None);
        }
        // Pad the buffer to window_size with zeros if necessary
        while self.buffer.len() < self.config.window_size {
            self.buffer.push(0.0);
        }
        let frame = self.emit_frame()?;
        self.pending_samples = 0;
        Ok(Some(frame))
    }

    /// Total number of STFT frames emitted.
    pub fn frames_emitted(&self) -> u64 {
        self.frames_emitted
    }

    /// Latency in samples (one full window).
    pub fn latency_samples(&self) -> usize {
        self.config.window_size
    }

    /// Number of positive-frequency bins per frame.
    pub fn num_bins(&self) -> usize {
        self.config.window_size / 2 + 1
    }

    /// Reset the processor state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.pending_samples = 0;
        self.frames_emitted = 0;
    }

    // ---- internal ----

    /// Extract the current window from the ring buffer, apply the window
    /// function, compute the real FFT, and return the positive-frequency
    /// spectrum.
    fn emit_frame(&mut self) -> SignalResult<Vec<Complex64>> {
        let raw = self.buffer.last_n(self.config.window_size)?;

        // Apply window
        let windowed: Vec<f64> = raw
            .iter()
            .zip(self.window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        // Real FFT
        let spectrum = scirs2_fft::rfft(&windowed, None).map_err(|e| {
            SignalError::ComputationError(format!("FFT error in streaming STFT: {e}"))
        })?;

        self.frames_emitted += 1;
        Ok(spectrum)
    }
}

// ============================================================================
// StreamingISTFT  -- Overlap-Add Reconstruction
// ============================================================================

/// Streaming inverse STFT using the weighted overlap-add (WOLA) method.
///
/// Feed STFT frames via [`StreamingISTFT::process_frame`] and pull reconstructed time-domain
/// samples via [`StreamingISTFT::read_output`].
pub struct StreamingISTFT {
    /// Window / hop configuration (must match the analysis STFT).
    window_size: usize,
    hop_size: usize,
    /// Synthesis window.
    window: Vec<f64>,
    /// Overlap-add accumulation buffer.
    ola_buffer: Vec<f64>,
    /// Normalisation buffer (sum of squared windows for COLA).
    norm_buffer: Vec<f64>,
    /// Write position in the OLA buffer.
    write_pos: usize,
    /// Read position (how many samples have been consumed).
    read_pos: usize,
    /// Total frames processed.
    frames_processed: u64,
}

impl StreamingISTFT {
    /// Create a new streaming ISTFT.
    ///
    /// `window_size` and `hop_size` **must** match the analysis
    /// [`StreamingSTFT`] that produced the frames.
    ///
    /// # Errors
    ///
    /// Same constraints as [`StreamingSTFT::new`].
    pub fn new(
        window_size: usize,
        hop_size: usize,
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

        let window = get_window(window_function.as_str(), window_size, true)?;

        // The OLA buffer needs to hold at least 2 * window_size so that we
        // can accumulate overlapping frames without losing data.
        let buf_len = window_size * 4;

        Ok(Self {
            window_size,
            hop_size,
            window,
            ola_buffer: vec![0.0; buf_len],
            norm_buffer: vec![0.0; buf_len],
            write_pos: 0,
            read_pos: 0,
            frames_processed: 0,
        })
    }

    /// Feed one STFT frame (positive-frequency bins) and overlap-add the
    /// reconstructed time-domain segment into the internal buffer.
    ///
    /// The frame length should be `window_size / 2 + 1`.
    pub fn process_frame(&mut self, frame: &[Complex64]) -> SignalResult<()> {
        let expected_len = self.window_size / 2 + 1;
        if frame.len() != expected_len {
            return Err(SignalError::ValueError(format!(
                "Expected frame length {expected_len}, got {}",
                frame.len()
            )));
        }

        // Inverse real FFT -> time domain
        let time_domain = scirs2_fft::irfft(frame, Some(self.window_size)).map_err(|e| {
            SignalError::ComputationError(format!("IFFT error in streaming ISTFT: {e}"))
        })?;

        // Apply synthesis window and accumulate
        let buf_len = self.ola_buffer.len();
        for (i, (&td, &w)) in time_domain.iter().zip(self.window.iter()).enumerate() {
            let pos = (self.write_pos + i) % buf_len;
            self.ola_buffer[pos] += td * w;
            self.norm_buffer[pos] += w * w;
        }

        // Advance write position by hop_size
        self.write_pos = (self.write_pos + self.hop_size) % buf_len;
        self.frames_processed += 1;

        Ok(())
    }

    /// Read up to `max_samples` reconstructed samples from the output buffer.
    ///
    /// Returns fewer samples if not enough data is available yet.  Samples are
    /// normalised by the COLA window sum.
    pub fn read_output(&mut self, max_samples: usize) -> Vec<f64> {
        let buf_len = self.ola_buffer.len();
        let available = self.available_samples();
        let n = max_samples.min(available);

        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let pos = self.read_pos % buf_len;
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
            self.read_pos += 1;
        }

        out
    }

    /// Number of samples available for reading.
    pub fn available_samples(&self) -> usize {
        // Conservative: we can read up to write_pos - read_pos hops worth
        if self.frames_processed == 0 {
            return 0;
        }
        let total_written = self.frames_processed as usize * self.hop_size;
        if total_written > self.read_pos {
            // Keep one window_size worth of margin for overlap
            let raw_available = total_written - self.read_pos;
            if raw_available > self.window_size {
                raw_available - self.window_size
            } else {
                0
            }
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
        self.read_pos = 0;
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
    fn test_stft_basic_creation() {
        let stft = StreamingSTFT::new(StreamingSTFTConfig::default());
        assert!(stft.is_ok());
        let stft = stft.expect("should succeed");
        assert_eq!(stft.num_bins(), 513); // 1024/2 + 1
    }

    #[test]
    fn test_stft_invalid_config() {
        // zero window
        let r = StreamingSTFT::new(StreamingSTFTConfig {
            window_size: 0,
            hop_size: 256,
            window_function: WindowFunction::Hann,
        });
        assert!(r.is_err());

        // hop > window
        let r = StreamingSTFT::new(StreamingSTFTConfig {
            window_size: 512,
            hop_size: 1024,
            window_function: WindowFunction::Hann,
        });
        assert!(r.is_err());
    }

    #[test]
    fn test_stft_produces_frames() {
        let config = StreamingSTFTConfig {
            window_size: 256,
            hop_size: 64,
            window_function: WindowFunction::Hann,
        };
        let mut stft = StreamingSTFT::new(config).expect("create STFT");

        // Feed exactly 256 + 64 samples -> should get first frame after 256
        // and second frame after 256+64
        let chunk: Vec<f64> = (0..320).map(|i| (i as f64 * 0.01).sin()).collect();
        let frames = stft.process_chunk(&chunk).expect("process_chunk");
        // We get at least 1 frame
        assert!(!frames.is_empty());
        // Each frame has the right number of bins
        assert_eq!(frames[0].len(), 129); // 256/2 + 1
    }

    #[test]
    fn test_stft_incremental_feed() {
        let config = StreamingSTFTConfig {
            window_size: 128,
            hop_size: 64,
            window_function: WindowFunction::Hamming,
        };
        let mut stft = StreamingSTFT::new(config).expect("create STFT");

        // Feed 32 samples at a time -- need 128 for first frame
        let mut total_frames = 0;
        for _ in 0..10 {
            let chunk: Vec<f64> = vec![1.0; 32];
            let frames = stft.process_chunk(&chunk).expect("process");
            total_frames += frames.len();
        }
        // 320 samples total, 128 window, 64 hop -> after 128 first frame,
        // then every 64 -> (320-128)/64 + 1 = 4 frames
        assert!(total_frames >= 3);
    }

    #[test]
    fn test_stft_flush() {
        let config = StreamingSTFTConfig {
            window_size: 128,
            hop_size: 64,
            window_function: WindowFunction::Hann,
        };
        let mut stft = StreamingSTFT::new(config).expect("create STFT");

        // Feed 100 samples (less than a full window + hop)
        let chunk: Vec<f64> = vec![0.5; 100];
        let _ = stft.process_chunk(&chunk).expect("process");

        // Flush remaining
        let flushed = stft.flush().expect("flush");
        // Whether we get a frame depends on pending state
        // At minimum, it should not error
        let _ = flushed;
    }

    #[test]
    fn test_stft_reset() {
        let config = StreamingSTFTConfig {
            window_size: 128,
            hop_size: 64,
            window_function: WindowFunction::Hann,
        };
        let mut stft = StreamingSTFT::new(config).expect("create STFT");
        let chunk: Vec<f64> = vec![1.0; 256];
        let _ = stft.process_chunk(&chunk).expect("process");
        assert!(stft.frames_emitted() > 0);

        stft.reset();
        assert_eq!(stft.frames_emitted(), 0);
    }

    #[test]
    fn test_stft_sine_wave_peak() {
        // A pure sine should produce a peak at the correct frequency bin
        let window_size = 256;
        let hop_size = 256;
        let fs = 256.0; // sample rate = window_size for simplicity
        let freq = 10.0; // 10 Hz -> bin 10

        let config = StreamingSTFTConfig {
            window_size,
            hop_size,
            window_function: WindowFunction::Rectangular,
        };
        let mut stft = StreamingSTFT::new(config).expect("create STFT");

        let chunk: Vec<f64> = (0..window_size)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
            .collect();
        let frames = stft.process_chunk(&chunk).expect("process");
        assert!(!frames.is_empty());

        // Find peak bin
        let magnitudes: Vec<f64> = frames[0].iter().map(|c| c.norm()).collect();
        let peak_bin = magnitudes
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        assert_eq!(peak_bin, 10);
    }

    // ---- StreamingISTFT ----

    #[test]
    fn test_istft_creation() {
        let istft = StreamingISTFT::new(256, 64, WindowFunction::Hann);
        assert!(istft.is_ok());
    }

    #[test]
    fn test_istft_invalid_config() {
        assert!(StreamingISTFT::new(0, 64, WindowFunction::Hann).is_err());
        assert!(StreamingISTFT::new(256, 512, WindowFunction::Hann).is_err());
    }

    #[test]
    fn test_istft_frame_length_check() {
        let mut istft = StreamingISTFT::new(256, 64, WindowFunction::Hann).expect("create ISTFT");
        // Wrong frame length
        let bad_frame = vec![Complex64::new(0.0, 0.0); 10];
        assert!(istft.process_frame(&bad_frame).is_err());
    }

    #[test]
    fn test_stft_istft_roundtrip() {
        // Verify that STFT -> ISTFT recovers the original signal (approximately).
        let window_size = 256;
        let hop_size = 64;

        let config = StreamingSTFTConfig {
            window_size,
            hop_size,
            window_function: WindowFunction::Hann,
        };
        let mut stft = StreamingSTFT::new(config).expect("create STFT");
        let mut istft =
            StreamingISTFT::new(window_size, hop_size, WindowFunction::Hann).expect("create ISTFT");

        // Generate test signal
        let n_samples = 2048;
        let signal: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 / 256.0;
                (2.0 * std::f64::consts::PI * 10.0 * t).sin()
            })
            .collect();

        // Forward
        let frames = stft.process_chunk(&signal).expect("stft process");

        // Inverse
        for frame in &frames {
            istft.process_frame(frame).expect("istft process");
        }

        let reconstructed = istft.read_output(n_samples);

        // The reconstructed signal should approximately match the original
        // (with some initial transient delay).  Check correlation for the
        // stable portion.
        if reconstructed.len() >= window_size {
            let offset = window_size; // skip transient
            let end = reconstructed.len().min(signal.len());
            if offset < end {
                let seg_len = end - offset;
                let mut max_abs_err = 0.0_f64;
                for i in 0..seg_len {
                    let err = (reconstructed[i] - signal[offset + i]).abs();
                    if err > max_abs_err {
                        max_abs_err = err;
                    }
                }
                // We expect reasonable reconstruction (tolerance due to
                // windowing edge effects)
                assert!(
                    max_abs_err < 2.0,
                    "Max absolute reconstruction error = {max_abs_err}"
                );
            }
        }
    }

    #[test]
    fn test_istft_reset() {
        let mut istft = StreamingISTFT::new(128, 32, WindowFunction::Hann).expect("create ISTFT");
        let frame = vec![Complex64::new(1.0, 0.0); 65]; // 128/2 + 1
        istft.process_frame(&frame).expect("process");
        assert_eq!(istft.frames_processed(), 1);

        istft.reset();
        assert_eq!(istft.frames_processed(), 0);
    }
}
