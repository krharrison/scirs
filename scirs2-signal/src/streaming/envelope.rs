//! Streaming envelope detection / followers.
//!
//! ## Provided components
//!
//! | Struct | Description |
//! |--------|-------------|
//! | [`HilbertEnvelope`] | Approximate analytic signal envelope via a single-pole IIR |
//! | [`RmsEnvelope`] | RMS envelope with a configurable sliding window |
//! | [`PeakEnvelopeFollower`] | Peak follower with independent attack / release |

use crate::error::{SignalError, SignalResult};

// ============================================================================
// HilbertEnvelope
// ============================================================================

/// Streaming envelope approximation inspired by the analytic signal.
///
/// Instead of a full Hilbert transform (which requires the entire signal), we
/// approximate the envelope by passing the *squared* signal through a
/// single-pole lowpass IIR and then taking the square root.  The cutoff
/// frequency of the lowpass controls how closely the envelope tracks amplitude
/// modulations.
///
/// ```text
///  env[n] = sqrt( alpha * x[n]^2 + (1 - alpha) * env[n-1]^2 )
/// ```
///
/// where `alpha = 1 - exp(-2 * pi * cutoff / sample_rate)`.
pub struct HilbertEnvelope {
    /// IIR coefficient.
    alpha: f64,
    /// Squared envelope state.
    env_sq: f64,
    /// Total samples processed.
    samples_processed: u64,
}

impl HilbertEnvelope {
    /// Create a new Hilbert-like envelope follower.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Input sample rate in Hz.
    /// * `cutoff_hz` - Lowpass cutoff for envelope smoothing.  A typical value
    ///   is 10 - 30 Hz for music, 1 - 5 Hz for speech.
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` or `cutoff_hz` is non-positive, or if
    /// `cutoff_hz >= sample_rate / 2`.
    pub fn new(sample_rate: f64, cutoff_hz: f64) -> SignalResult<Self> {
        if sample_rate <= 0.0 {
            return Err(SignalError::ValueError(
                "sample_rate must be positive".to_string(),
            ));
        }
        if cutoff_hz <= 0.0 {
            return Err(SignalError::ValueError(
                "cutoff_hz must be positive".to_string(),
            ));
        }
        if cutoff_hz >= sample_rate / 2.0 {
            return Err(SignalError::ValueError(
                "cutoff_hz must be < sample_rate / 2".to_string(),
            ));
        }

        let alpha = 1.0 - (-2.0 * std::f64::consts::PI * cutoff_hz / sample_rate).exp();

        Ok(Self {
            alpha,
            env_sq: 0.0,
            samples_processed: 0,
        })
    }

    /// Process a single sample and return the envelope value.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let x_sq = input * input;
        self.env_sq = self.alpha * x_sq + (1.0 - self.alpha) * self.env_sq;
        self.samples_processed += 1;
        self.env_sq.sqrt()
    }

    /// Process a chunk.
    pub fn process_chunk(&mut self, chunk: &[f64]) -> Vec<f64> {
        chunk.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Current envelope value.
    pub fn current_value(&self) -> f64 {
        self.env_sq.sqrt()
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.env_sq = 0.0;
        self.samples_processed = 0;
    }

    /// Total samples processed.
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }
}

// ============================================================================
// RmsEnvelope
// ============================================================================

/// Streaming RMS (root-mean-square) envelope with a configurable window.
///
/// Maintains a running sum of squares over a sliding window of `window_size`
/// samples for O(1) per-sample updates.
pub struct RmsEnvelope {
    window_size: usize,
    /// Circular buffer of sample squares.
    buffer: Vec<f64>,
    /// Write position.
    pos: usize,
    /// Number of valid entries.
    count: usize,
    /// Running sum of squares.
    sum_sq: f64,
    samples_processed: u64,
}

impl RmsEnvelope {
    /// Create a new RMS envelope follower.
    ///
    /// # Errors
    ///
    /// Returns an error if `window_size` is 0.
    pub fn new(window_size: usize) -> SignalResult<Self> {
        if window_size == 0 {
            return Err(SignalError::ValueError(
                "window_size must be > 0".to_string(),
            ));
        }
        Ok(Self {
            window_size,
            buffer: vec![0.0; window_size],
            pos: 0,
            count: 0,
            sum_sq: 0.0,
            samples_processed: 0,
        })
    }

    /// Process a single sample and return the current RMS value.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let sq = input * input;

        // Remove the oldest squared value
        if self.count == self.window_size {
            self.sum_sq -= self.buffer[self.pos];
        }

        self.buffer[self.pos] = sq;
        self.sum_sq += sq;
        self.pos = (self.pos + 1) % self.window_size;

        if self.count < self.window_size {
            self.count += 1;
        }
        self.samples_processed += 1;

        // Guard against tiny negative floating point drift
        let mean_sq = if self.sum_sq > 0.0 {
            self.sum_sq / self.count as f64
        } else {
            0.0
        };
        mean_sq.sqrt()
    }

    /// Process a chunk.
    pub fn process_chunk(&mut self, chunk: &[f64]) -> Vec<f64> {
        chunk.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Current RMS value.
    pub fn current_value(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mean_sq = if self.sum_sq > 0.0 {
            self.sum_sq / self.count as f64
        } else {
            0.0
        };
        mean_sq.sqrt()
    }

    /// Reset state.
    pub fn reset(&mut self) {
        self.buffer.iter_mut().for_each(|v| *v = 0.0);
        self.pos = 0;
        self.count = 0;
        self.sum_sq = 0.0;
        self.samples_processed = 0;
    }

    /// Window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Total samples processed.
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }
}

// ============================================================================
// PeakEnvelopeFollower
// ============================================================================

/// Peak envelope follower with independent attack and release time constants.
///
/// The envelope rises quickly (attack) when the signal amplitude exceeds the
/// current envelope, and decays slowly (release) when the signal is below.
///
/// ```text
/// if |x[n]| > env[n-1]:
///     env[n] = attack_coeff * env[n-1] + (1 - attack_coeff) * |x[n]|
/// else:
///     env[n] = release_coeff * env[n-1]
/// ```
pub struct PeakEnvelopeFollower {
    attack_coeff: f64,
    release_coeff: f64,
    envelope: f64,
    samples_processed: u64,
}

impl PeakEnvelopeFollower {
    /// Create a new peak envelope follower.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz.
    /// * `attack_time_s` - Attack time constant in seconds.  Smaller = faster.
    /// * `release_time_s` - Release time constant in seconds.  Larger = slower decay.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is non-positive.
    pub fn new(sample_rate: f64, attack_time_s: f64, release_time_s: f64) -> SignalResult<Self> {
        if sample_rate <= 0.0 {
            return Err(SignalError::ValueError(
                "sample_rate must be positive".to_string(),
            ));
        }
        if attack_time_s <= 0.0 {
            return Err(SignalError::ValueError(
                "attack_time_s must be positive".to_string(),
            ));
        }
        if release_time_s <= 0.0 {
            return Err(SignalError::ValueError(
                "release_time_s must be positive".to_string(),
            ));
        }

        let attack_coeff = (-1.0 / (attack_time_s * sample_rate)).exp();
        let release_coeff = (-1.0 / (release_time_s * sample_rate)).exp();

        Ok(Self {
            attack_coeff,
            release_coeff,
            envelope: 0.0,
            samples_processed: 0,
        })
    }

    /// Process a single sample and return the current envelope value.
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let abs_input = input.abs();
        if abs_input > self.envelope {
            // Attack
            self.envelope =
                self.attack_coeff * self.envelope + (1.0 - self.attack_coeff) * abs_input;
        } else {
            // Release
            self.envelope *= self.release_coeff;
        }
        self.samples_processed += 1;
        self.envelope
    }

    /// Process a chunk.
    pub fn process_chunk(&mut self, chunk: &[f64]) -> Vec<f64> {
        chunk.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Current envelope value.
    pub fn current_value(&self) -> f64 {
        self.envelope
    }

    /// Reset the follower state.
    pub fn reset(&mut self) {
        self.envelope = 0.0;
        self.samples_processed = 0;
    }

    /// Total samples processed.
    pub fn samples_processed(&self) -> u64 {
        self.samples_processed
    }
}

// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    // ---- HilbertEnvelope ----

    #[test]
    fn test_hilbert_creation() {
        let h = HilbertEnvelope::new(44100.0, 20.0);
        assert!(h.is_ok());
    }

    #[test]
    fn test_hilbert_invalid_params() {
        assert!(HilbertEnvelope::new(0.0, 20.0).is_err());
        assert!(HilbertEnvelope::new(44100.0, 0.0).is_err());
        assert!(HilbertEnvelope::new(44100.0, 30000.0).is_err()); // >= Nyquist
    }

    #[test]
    fn test_hilbert_constant_signal() {
        let mut h = HilbertEnvelope::new(1000.0, 50.0).expect("create HilbertEnvelope");

        // Feed a constant amplitude signal for long enough to converge
        let amp = 0.8;
        let mut last = 0.0;
        for _ in 0..5000 {
            last = h.process_sample(amp);
        }
        // Envelope should converge toward amp
        assert!(
            (last - amp).abs() < 0.1,
            "Envelope {last} should converge to {amp}"
        );
    }

    #[test]
    fn test_hilbert_sine_envelope() {
        let fs = 8000.0;
        let freq = 440.0;
        let mut h = HilbertEnvelope::new(fs, 20.0).expect("create HilbertEnvelope");

        let n = 8000; // 1 second
        let env: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let sample = (2.0 * std::f64::consts::PI * freq * t).sin();
                h.process_sample(sample)
            })
            .collect();

        // After convergence, envelope of a unit sine should be near 1/sqrt(2)
        // (RMS of sine) or between 0.5 and 1.0
        let tail_avg: f64 = env[n - 1000..].iter().sum::<f64>() / 1000.0;
        assert!(
            tail_avg > 0.3 && tail_avg < 1.0,
            "Tail average envelope = {tail_avg}, expected in (0.3, 1.0)"
        );
    }

    #[test]
    fn test_hilbert_process_chunk() {
        let mut h = HilbertEnvelope::new(1000.0, 30.0).expect("create");
        let chunk = vec![0.5; 100];
        let out = h.process_chunk(&chunk);
        assert_eq!(out.len(), 100);
        // All values should be non-negative
        for &v in &out {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn test_hilbert_reset() {
        let mut h = HilbertEnvelope::new(1000.0, 30.0).expect("create");
        h.process_sample(1.0);
        assert!(h.current_value() > 0.0);
        h.reset();
        assert!((h.current_value() - 0.0).abs() < 1e-15);
        assert_eq!(h.samples_processed(), 0);
    }

    // ---- RmsEnvelope ----

    #[test]
    fn test_rms_creation() {
        let r = RmsEnvelope::new(256);
        assert!(r.is_ok());
        assert!(RmsEnvelope::new(0).is_err());
    }

    #[test]
    fn test_rms_constant_signal() {
        let mut r = RmsEnvelope::new(100).expect("create RmsEnvelope");
        // Feed 200 samples of amplitude 3.0
        let mut last = 0.0;
        for _ in 0..200 {
            last = r.process_sample(3.0);
        }
        // RMS of a constant = that constant
        assert!((last - 3.0).abs() < 1e-10, "Expected RMS 3.0, got {last}");
    }

    #[test]
    fn test_rms_sine() {
        let fs = 1000.0;
        let freq = 100.0;
        let window_size = 100; // one full cycle at 100 Hz / 1000 Hz sample rate

        let mut r = RmsEnvelope::new(window_size).expect("create RmsEnvelope");

        // Feed two full cycles
        let mut last = 0.0;
        for i in 0..200 {
            let t = i as f64 / fs;
            let sample = (2.0 * std::f64::consts::PI * freq * t).sin();
            last = r.process_sample(sample);
        }
        // RMS of sine = 1/sqrt(2) ~ 0.7071
        let expected = 1.0 / 2.0_f64.sqrt();
        assert!(
            (last - expected).abs() < 0.05,
            "Expected RMS ~{expected}, got {last}"
        );
    }

    #[test]
    fn test_rms_process_chunk() {
        let mut r = RmsEnvelope::new(4).expect("create RmsEnvelope");
        let out = r.process_chunk(&[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(out.len(), 4);
        assert!((out[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rms_reset() {
        let mut r = RmsEnvelope::new(10).expect("create RmsEnvelope");
        r.process_chunk(&[5.0; 20]);
        assert!(r.current_value() > 0.0);
        r.reset();
        assert!((r.current_value() - 0.0).abs() < 1e-15);
        assert_eq!(r.samples_processed(), 0);
    }

    #[test]
    fn test_rms_window_slides() {
        let mut r = RmsEnvelope::new(4).expect("create RmsEnvelope");
        // Feed 4 samples of 2.0 -> RMS = 2.0
        r.process_chunk(&[2.0; 4]);
        assert!((r.current_value() - 2.0).abs() < 1e-10);
        // Now feed 4 samples of 0.0 -> RMS should go to 0
        r.process_chunk(&[0.0; 4]);
        assert!(
            r.current_value() < 0.01,
            "RMS should be near 0, got {}",
            r.current_value()
        );
    }

    // ---- PeakEnvelopeFollower ----

    #[test]
    fn test_peak_creation() {
        let p = PeakEnvelopeFollower::new(44100.0, 0.001, 0.1);
        assert!(p.is_ok());
    }

    #[test]
    fn test_peak_invalid_params() {
        assert!(PeakEnvelopeFollower::new(0.0, 0.001, 0.1).is_err());
        assert!(PeakEnvelopeFollower::new(44100.0, 0.0, 0.1).is_err());
        assert!(PeakEnvelopeFollower::new(44100.0, 0.001, 0.0).is_err());
    }

    #[test]
    fn test_peak_attack() {
        let fs = 1000.0;
        let mut p = PeakEnvelopeFollower::new(fs, 0.001, 0.1).expect("create PeakFollower");

        // Sudden burst should cause the envelope to rise quickly
        let env_before = p.current_value();
        for _ in 0..10 {
            p.process_sample(1.0);
        }
        let env_after = p.current_value();
        assert!(
            env_after > env_before + 0.5,
            "Attack should raise envelope quickly: before={env_before}, after={env_after}"
        );
    }

    #[test]
    fn test_peak_release() {
        let fs = 1000.0;
        let mut p = PeakEnvelopeFollower::new(fs, 0.001, 0.05).expect("create PeakFollower");

        // Charge the envelope
        for _ in 0..100 {
            p.process_sample(1.0);
        }
        let peak = p.current_value();

        // Now silence
        for _ in 0..200 {
            p.process_sample(0.0);
        }
        let after_release = p.current_value();

        assert!(
            after_release < peak * 0.5,
            "Release should decay: peak={peak}, after={after_release}"
        );
    }

    #[test]
    fn test_peak_process_chunk() {
        let mut p = PeakEnvelopeFollower::new(1000.0, 0.001, 0.1).expect("create PeakFollower");
        let out = p.process_chunk(&[0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(out.len(), 5);
        // The peak should happen at or after index 2
        let max_idx = out
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert!(max_idx >= 2, "Peak should be at or after the impulse");
    }

    #[test]
    fn test_peak_reset() {
        let mut p = PeakEnvelopeFollower::new(1000.0, 0.001, 0.1).expect("create PeakFollower");
        p.process_chunk(&[1.0; 100]);
        assert!(p.current_value() > 0.0);
        p.reset();
        assert!((p.current_value() - 0.0).abs() < 1e-15);
        assert_eq!(p.samples_processed(), 0);
    }

    #[test]
    fn test_peak_negative_input() {
        let mut p = PeakEnvelopeFollower::new(1000.0, 0.001, 0.1).expect("create PeakFollower");
        // Negative amplitudes should be tracked by absolute value
        let env = p.process_sample(-0.9);
        assert!(env > 0.0, "Should track absolute value of negative input");
    }
}
