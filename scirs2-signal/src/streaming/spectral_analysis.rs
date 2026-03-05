//! Real-time spectral analysis processors.
//!
//! ## Provided components
//!
//! | Struct | Description |
//! |--------|-------------|
//! | [`SpectralFlux`] | Onset detection via spectral difference |
//! | [`PitchDetector`] | Autocorrelation-based pitch detection (simplified YIN) |
//! | [`StreamingPowerSpectrum`] | Running PSD with exponential averaging |

use crate::error::{SignalError, SignalResult};
use crate::streaming::ring_buffer::RingBuffer;
use crate::streaming::stft::{StreamingSTFT, StreamingSTFTConfig, WindowFunction};
use scirs2_core::numeric::Complex64;

// ============================================================================
// SpectralFlux
// ============================================================================

/// Onset detection via spectral flux (the L2 norm of the positive difference
/// between consecutive magnitude spectra).
///
/// A spike in the flux value indicates a sudden change in spectral content
/// which typically corresponds to a musical onset or transient.
pub struct SpectralFlux {
    /// Internal STFT used to produce magnitude spectra.
    stft: StreamingSTFT,
    /// Previous magnitude spectrum for differencing.
    prev_magnitude: Option<Vec<f64>>,
    /// Adaptive threshold state (exponential moving average of flux).
    threshold_ema: f64,
    /// Smoothing coefficient for the adaptive threshold.
    threshold_alpha: f64,
    /// Total onsets detected.
    onsets_detected: u64,
}

/// Result from one call to [`SpectralFlux::process_chunk`].
#[derive(Debug, Clone)]
pub struct FluxResult {
    /// Raw spectral flux values (one per STFT frame produced).
    pub flux_values: Vec<f64>,
    /// Boolean onset decisions (using adaptive threshold).
    pub onsets: Vec<bool>,
}

impl SpectralFlux {
    /// Create a new spectral flux onset detector.
    ///
    /// # Arguments
    ///
    /// * `window_size` - STFT window size.
    /// * `hop_size` - STFT hop size.
    /// * `threshold_alpha` - Smoothing factor for the adaptive threshold
    ///   (0.0 .. 1.0, smaller = slower adaptation).  A typical value is 0.1.
    pub fn new(window_size: usize, hop_size: usize, threshold_alpha: f64) -> SignalResult<Self> {
        if !(0.0..=1.0).contains(&threshold_alpha) {
            return Err(SignalError::ValueError(
                "threshold_alpha must be in [0, 1]".to_string(),
            ));
        }
        let config = StreamingSTFTConfig {
            window_size,
            hop_size,
            window_function: WindowFunction::Hann,
        };
        let stft = StreamingSTFT::new(config)?;
        Ok(Self {
            stft,
            prev_magnitude: None,
            threshold_ema: 0.0,
            threshold_alpha,
            onsets_detected: 0,
        })
    }

    /// Feed a chunk of audio and return flux values and onset decisions.
    pub fn process_chunk(&mut self, chunk: &[f64]) -> SignalResult<FluxResult> {
        let frames = self.stft.process_chunk(chunk)?;
        let mut flux_values = Vec::with_capacity(frames.len());
        let mut onsets = Vec::with_capacity(frames.len());

        for frame in &frames {
            let magnitude: Vec<f64> = frame.iter().map(|c| c.norm()).collect();

            let flux = if let Some(ref prev) = self.prev_magnitude {
                // Half-wave rectified spectral difference
                magnitude
                    .iter()
                    .zip(prev.iter())
                    .map(|(&curr, &prv)| {
                        let diff = curr - prv;
                        if diff > 0.0 {
                            diff * diff
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    .sqrt()
            } else {
                0.0
            };

            // Adaptive threshold
            self.threshold_ema =
                self.threshold_alpha * flux + (1.0 - self.threshold_alpha) * self.threshold_ema;
            let is_onset = flux > self.threshold_ema * 1.5;
            if is_onset {
                self.onsets_detected += 1;
            }

            flux_values.push(flux);
            onsets.push(is_onset);
            self.prev_magnitude = Some(magnitude);
        }

        Ok(FluxResult {
            flux_values,
            onsets,
        })
    }

    /// Total onsets detected so far.
    pub fn onsets_detected(&self) -> u64 {
        self.onsets_detected
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.stft.reset();
        self.prev_magnitude = None;
        self.threshold_ema = 0.0;
        self.onsets_detected = 0;
    }
}

// ============================================================================
// PitchDetector  -- simplified YIN algorithm
// ============================================================================

/// Autocorrelation-based pitch detector using a simplified YIN algorithm.
///
/// The YIN algorithm computes a cumulative mean normalised difference function
/// over a buffer of samples and finds the first dip below a threshold to
/// estimate the fundamental period.
///
/// Reference: de Cheveign\'e, A. & Kawahara, H. (2002).  "YIN, a fundamental
/// frequency estimator for speech and music."  JASA 111(4).
pub struct PitchDetector {
    /// Sample rate in Hz.
    sample_rate: f64,
    /// Analysis frame size (at least 2x the maximum expected period).
    frame_size: usize,
    /// Hop between consecutive analyses.
    hop_size: usize,
    /// YIN absolute threshold (typically 0.10 .. 0.20).
    yin_threshold: f64,
    /// Ring buffer holding incoming samples.
    buffer: RingBuffer<f64>,
    /// Pending samples since last analysis.
    pending: usize,
    /// Minimum frequency to detect (limits the search range).
    min_freq: f64,
    /// Maximum frequency to detect.
    max_freq: f64,
}

/// Result from pitch detection.
#[derive(Debug, Clone)]
pub struct PitchResult {
    /// Estimated fundamental frequency in Hz.  `None` if no pitch detected.
    pub frequency: Option<f64>,
    /// YIN confidence (lower is better; 0 = perfect periodicity).
    pub confidence: f64,
    /// Whether a voiced segment was detected.
    pub is_voiced: bool,
}

impl PitchDetector {
    /// Create a new YIN pitch detector.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate of the input signal.
    /// * `frame_size` - Analysis frame length.  Should be >= 2 * (sample_rate / min_freq).
    /// * `hop_size` - Number of new samples between consecutive analyses.
    /// * `min_freq` - Minimum detectable frequency in Hz (e.g. 50).
    /// * `max_freq` - Maximum detectable frequency in Hz (e.g. 2000).
    /// * `yin_threshold` - Absolute threshold for the CMND function (0.10 - 0.20).
    pub fn new(
        sample_rate: f64,
        frame_size: usize,
        hop_size: usize,
        min_freq: f64,
        max_freq: f64,
        yin_threshold: f64,
    ) -> SignalResult<Self> {
        if sample_rate <= 0.0 {
            return Err(SignalError::ValueError(
                "sample_rate must be positive".to_string(),
            ));
        }
        if frame_size == 0 {
            return Err(SignalError::ValueError(
                "frame_size must be > 0".to_string(),
            ));
        }
        if hop_size == 0 || hop_size > frame_size {
            return Err(SignalError::ValueError(
                "hop_size must be > 0 and <= frame_size".to_string(),
            ));
        }
        if min_freq <= 0.0 || max_freq <= min_freq {
            return Err(SignalError::ValueError(
                "Require 0 < min_freq < max_freq".to_string(),
            ));
        }
        if yin_threshold <= 0.0 || yin_threshold >= 1.0 {
            return Err(SignalError::ValueError(
                "yin_threshold must be in (0, 1)".to_string(),
            ));
        }

        let buffer = RingBuffer::new(frame_size)?;

        Ok(Self {
            sample_rate,
            frame_size,
            hop_size,
            yin_threshold,
            buffer,
            pending: 0,
            min_freq,
            max_freq,
        })
    }

    /// Feed a chunk of samples and return pitch estimates.
    pub fn process_chunk(&mut self, chunk: &[f64]) -> SignalResult<Vec<PitchResult>> {
        let mut results = Vec::new();

        for &sample in chunk {
            self.buffer.push(sample);
            self.pending += 1;

            if self.pending >= self.hop_size && self.buffer.len() >= self.frame_size {
                let result = self.analyse()?;
                results.push(result);
                self.pending = 0;
            }
        }

        Ok(results)
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.pending = 0;
    }

    // ---- internal YIN ----

    fn analyse(&self) -> SignalResult<PitchResult> {
        let frame = self.buffer.last_n(self.frame_size)?;
        let half = self.frame_size / 2;

        // Lag range corresponding to [min_freq, max_freq]
        let tau_min = (self.sample_rate / self.max_freq).floor() as usize;
        let tau_max = ((self.sample_rate / self.min_freq).ceil() as usize).min(half);

        if tau_min >= tau_max || tau_max > half {
            return Ok(PitchResult {
                frequency: None,
                confidence: 1.0,
                is_voiced: false,
            });
        }

        // Step 2: Difference function d(tau)
        let mut diff = vec![0.0_f64; tau_max + 1];
        for tau in 1..=tau_max {
            let mut sum = 0.0;
            for j in 0..half {
                let delta = frame[j] - frame[j + tau];
                sum += delta * delta;
            }
            diff[tau] = sum;
        }

        // Step 3: Cumulative mean normalised difference function (CMNDF)
        let mut cmndf = vec![0.0_f64; tau_max + 1];
        cmndf[0] = 1.0;
        let mut running_sum = 0.0;
        for tau in 1..=tau_max {
            running_sum += diff[tau];
            cmndf[tau] = if running_sum > 1e-30 {
                diff[tau] * tau as f64 / running_sum
            } else {
                1.0
            };
        }

        // Step 4: Absolute threshold -- find first tau >= tau_min where
        // cmndf[tau] < threshold and is a local minimum.
        let mut best_tau: Option<usize> = None;
        let mut best_val = f64::MAX;

        for tau in tau_min..=tau_max {
            if cmndf[tau] < self.yin_threshold {
                // Check that it is a local minimum
                let is_local_min = if tau + 1 <= tau_max {
                    cmndf[tau] <= cmndf[tau + 1]
                } else {
                    true
                };
                if is_local_min && cmndf[tau] < best_val {
                    best_val = cmndf[tau];
                    best_tau = Some(tau);
                    break; // take the first dip below threshold
                }
            }
        }

        // If no dip below threshold, fall back to global minimum in range
        if best_tau.is_none() {
            for tau in tau_min..=tau_max {
                if cmndf[tau] < best_val {
                    best_val = cmndf[tau];
                    best_tau = Some(tau);
                }
            }
        }

        match best_tau {
            Some(tau) => {
                // Step 5: Parabolic interpolation for sub-sample accuracy
                let refined_tau = self.parabolic_interp(&cmndf, tau, tau_max);
                let freq = self.sample_rate / refined_tau;
                let is_voiced = best_val < self.yin_threshold;
                Ok(PitchResult {
                    frequency: Some(freq),
                    confidence: best_val,
                    is_voiced,
                })
            }
            None => Ok(PitchResult {
                frequency: None,
                confidence: 1.0,
                is_voiced: false,
            }),
        }
    }

    /// Parabolic interpolation around `tau` for sub-sample precision.
    fn parabolic_interp(&self, cmndf: &[f64], tau: usize, tau_max: usize) -> f64 {
        if tau < 1 || tau >= tau_max {
            return tau as f64;
        }
        let s0 = cmndf[tau - 1];
        let s1 = cmndf[tau];
        let s2 = cmndf[tau + 1];
        let denom = 2.0 * s1 - s2 - s0;
        if denom.abs() < 1e-30 {
            return tau as f64;
        }
        tau as f64 + (s0 - s2) / (2.0 * denom)
    }
}

// ============================================================================
// StreamingPowerSpectrum
// ============================================================================

/// Running power spectral density with exponential averaging.
///
/// Each time a new STFT frame is produced, the squared magnitude spectrum is
/// blended with the running estimate using an exponential moving average (EMA).
pub struct StreamingPowerSpectrum {
    /// Internal STFT.
    stft: StreamingSTFT,
    /// Running PSD estimate.
    psd: Option<Vec<f64>>,
    /// Smoothing factor (0 < alpha <= 1).  Larger = faster adaptation.
    alpha: f64,
    /// Total frames accumulated.
    frames_accumulated: u64,
}

impl StreamingPowerSpectrum {
    /// Create a new streaming power spectrum estimator.
    ///
    /// # Arguments
    ///
    /// * `window_size` - STFT analysis window size.
    /// * `hop_size` - STFT hop size.
    /// * `alpha` - EMA smoothing factor in (0, 1].
    pub fn new(window_size: usize, hop_size: usize, alpha: f64) -> SignalResult<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(SignalError::ValueError(
                "alpha must be in (0, 1]".to_string(),
            ));
        }
        let config = StreamingSTFTConfig {
            window_size,
            hop_size,
            window_function: WindowFunction::Hann,
        };
        let stft = StreamingSTFT::new(config)?;
        Ok(Self {
            stft,
            psd: None,
            alpha,
            frames_accumulated: 0,
        })
    }

    /// Feed a chunk of audio and update the running PSD.
    ///
    /// Returns the latest PSD snapshot (if at least one frame was processed).
    pub fn process_chunk(&mut self, chunk: &[f64]) -> SignalResult<Option<Vec<f64>>> {
        let frames = self.stft.process_chunk(chunk)?;

        for frame in &frames {
            let power: Vec<f64> = frame.iter().map(|c| c.norm_sqr()).collect();

            self.psd = Some(match self.psd.take() {
                Some(mut prev) => {
                    for (p, &new) in prev.iter_mut().zip(power.iter()) {
                        *p = self.alpha * new + (1.0 - self.alpha) * *p;
                    }
                    prev
                }
                None => power,
            });

            self.frames_accumulated += 1;
        }

        Ok(self.psd.clone())
    }

    /// Get the current PSD estimate (may be `None` if no frames processed yet).
    pub fn current_psd(&self) -> Option<&[f64]> {
        self.psd.as_deref()
    }

    /// Total frames accumulated.
    pub fn frames_accumulated(&self) -> u64 {
        self.frames_accumulated
    }

    /// Number of frequency bins.
    pub fn num_bins(&self) -> usize {
        self.stft.num_bins()
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.stft.reset();
        self.psd = None;
        self.frames_accumulated = 0;
    }
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ---- SpectralFlux ----

    #[test]
    fn test_spectral_flux_creation() {
        let sf = SpectralFlux::new(256, 64, 0.1);
        assert!(sf.is_ok());
    }

    #[test]
    fn test_spectral_flux_invalid_alpha() {
        assert!(SpectralFlux::new(256, 64, -0.1).is_err());
        assert!(SpectralFlux::new(256, 64, 1.5).is_err());
    }

    #[test]
    fn test_spectral_flux_constant_signal() {
        let mut sf = SpectralFlux::new(256, 64, 0.1).expect("create SpectralFlux");
        // Constant signal -> no spectral change -> flux near 0
        let chunk = vec![1.0; 1024];
        let result = sf.process_chunk(&chunk).expect("process");
        // After the first frame (which has no previous), all flux values
        // should be near zero for a constant signal.
        for (i, &f) in result.flux_values.iter().enumerate() {
            if i > 0 {
                assert!(
                    f < 1e-6,
                    "Expected near-zero flux for constant signal, got {f}"
                );
            }
        }
    }

    #[test]
    fn test_spectral_flux_onset_detection() {
        let mut sf = SpectralFlux::new(256, 128, 0.05).expect("create SpectralFlux");

        // Silence followed by a burst
        let mut signal = vec![0.0; 512];
        for i in 0..512 {
            signal.push((i as f64 * 0.3).sin());
        }
        let result = sf.process_chunk(&signal).expect("process");
        // There should be at least one non-zero flux value
        let has_nonzero = result.flux_values.iter().any(|&f| f > 1e-6);
        assert!(has_nonzero, "Expected some spectral flux at onset");
    }

    #[test]
    fn test_spectral_flux_reset() {
        let mut sf = SpectralFlux::new(256, 128, 0.1).expect("create SpectralFlux");
        let chunk = vec![1.0; 512];
        let _ = sf.process_chunk(&chunk).expect("process");
        sf.reset();
        assert_eq!(sf.onsets_detected(), 0);
    }

    #[test]
    fn test_spectral_flux_incremental() {
        let mut sf = SpectralFlux::new(128, 64, 0.1).expect("create SpectralFlux");
        // Feed small chunks
        for _ in 0..10 {
            let chunk = vec![0.5; 32];
            let _ = sf.process_chunk(&chunk).expect("process");
        }
        // Should not crash and should accumulate frames
    }

    // ---- PitchDetector ----

    #[test]
    fn test_pitch_detector_creation() {
        let pd = PitchDetector::new(8000.0, 1024, 512, 50.0, 2000.0, 0.15);
        assert!(pd.is_ok());
    }

    #[test]
    fn test_pitch_detector_invalid_params() {
        // Negative sample rate
        assert!(PitchDetector::new(-1.0, 1024, 512, 50.0, 2000.0, 0.15).is_err());
        // min_freq >= max_freq
        assert!(PitchDetector::new(8000.0, 1024, 512, 2000.0, 50.0, 0.15).is_err());
        // threshold out of range
        assert!(PitchDetector::new(8000.0, 1024, 512, 50.0, 2000.0, 0.0).is_err());
        assert!(PitchDetector::new(8000.0, 1024, 512, 50.0, 2000.0, 1.0).is_err());
    }

    #[test]
    fn test_pitch_detector_pure_sine() {
        let sample_rate = 8000.0;
        let freq = 440.0;
        let frame_size = 2048;
        let hop_size = 1024;

        let mut pd = PitchDetector::new(sample_rate, frame_size, hop_size, 50.0, 4000.0, 0.15)
            .expect("create PitchDetector");

        // Generate 440 Hz sine
        let n = 4096;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / sample_rate).sin())
            .collect();

        let results = pd.process_chunk(&signal).expect("process");
        assert!(!results.is_empty(), "Should produce at least one result");

        // Check that at least one result detects the correct frequency
        let mut found_correct = false;
        for r in &results {
            if let Some(f) = r.frequency {
                if (f - freq).abs() < 20.0 {
                    found_correct = true;
                    break;
                }
            }
        }
        assert!(
            found_correct,
            "Should detect ~440 Hz; got: {:?}",
            results.iter().map(|r| r.frequency).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_pitch_detector_silence() {
        let mut pd = PitchDetector::new(8000.0, 1024, 512, 50.0, 2000.0, 0.15)
            .expect("create PitchDetector");

        let signal = vec![0.0; 2048];
        let results = pd.process_chunk(&signal).expect("process");
        // For silence, no pitched content should be detected
        for r in &results {
            // Either no frequency or very low confidence
            assert!(
                !r.is_voiced || r.frequency.is_none(),
                "Silence should not be voiced"
            );
        }
    }

    #[test]
    fn test_pitch_detector_reset() {
        let mut pd = PitchDetector::new(8000.0, 1024, 512, 50.0, 2000.0, 0.15)
            .expect("create PitchDetector");
        let signal = vec![1.0; 2048];
        let _ = pd.process_chunk(&signal);
        pd.reset();
        // After reset the buffer should be empty
        assert!(pd.buffer.is_empty());
    }

    // ---- StreamingPowerSpectrum ----

    #[test]
    fn test_power_spectrum_creation() {
        let ps = StreamingPowerSpectrum::new(256, 64, 0.3);
        assert!(ps.is_ok());
    }

    #[test]
    fn test_power_spectrum_invalid_alpha() {
        assert!(StreamingPowerSpectrum::new(256, 64, 0.0).is_err());
        assert!(StreamingPowerSpectrum::new(256, 64, 1.5).is_err());
    }

    #[test]
    fn test_power_spectrum_accumulation() {
        let mut ps = StreamingPowerSpectrum::new(256, 128, 0.5).expect("create PSD");
        let chunk = vec![1.0; 512];
        let result = ps.process_chunk(&chunk).expect("process");
        assert!(result.is_some());
        let psd = result.expect("should have PSD");
        assert_eq!(psd.len(), 129); // 256/2 + 1
                                    // All values should be non-negative
        for &v in &psd {
            assert!(v >= 0.0, "PSD values must be non-negative");
        }
    }

    #[test]
    fn test_power_spectrum_sine_peak() {
        let window_size = 256;
        let hop_size = 128;
        let fs = 256.0;
        let freq = 20.0;

        let mut ps = StreamingPowerSpectrum::new(window_size, hop_size, 1.0).expect("create PSD");

        let n = 1024;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin())
            .collect();

        let result = ps.process_chunk(&signal).expect("process");
        if let Some(psd) = result {
            // Find the peak bin
            let peak_bin = psd
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            // The expected bin for freq Hz at fs sample rate with window_size FFT
            let expected_bin = (freq * window_size as f64 / fs).round() as usize;
            assert!(
                (peak_bin as i64 - expected_bin as i64).unsigned_abs() <= 2,
                "Peak bin {peak_bin} should be near expected bin {expected_bin}"
            );
        }
    }

    #[test]
    fn test_power_spectrum_reset() {
        let mut ps = StreamingPowerSpectrum::new(128, 64, 0.5).expect("create PSD");
        let chunk = vec![1.0; 256];
        let _ = ps.process_chunk(&chunk).expect("process");
        assert!(ps.current_psd().is_some());

        ps.reset();
        assert!(ps.current_psd().is_none());
        assert_eq!(ps.frames_accumulated(), 0);
    }

    #[test]
    fn test_power_spectrum_smoothing() {
        let mut ps = StreamingPowerSpectrum::new(128, 128, 0.5).expect("create PSD");

        // First chunk: signal
        let chunk1: Vec<f64> = (0..128)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / 128.0).sin())
            .collect();
        let _ = ps.process_chunk(&chunk1).expect("process");
        let psd1 = ps.current_psd().expect("should have psd").to_vec();

        // Second chunk: silence
        let chunk2 = vec![0.0; 128];
        let _ = ps.process_chunk(&chunk2).expect("process");
        let psd2 = ps.current_psd().expect("should have psd").to_vec();

        // With alpha=0.5, the PSD after the second (silent) frame should be
        // roughly half the first PSD
        let ratio = psd2.iter().sum::<f64>() / psd1.iter().sum::<f64>();
        assert!(
            ratio < 0.75 && ratio > 0.0,
            "Smoothing should blend old and new; ratio = {ratio}"
        );
    }
}
