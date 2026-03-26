//! Coherence-based double-talk detector and high-level EchoCanceller.
//!
//! Provides:
//! - `DoubleTalkConfig` / `DoubleTalkDetector` — coherence-based DTD.
//! - `EchoCanceller` — combines an NLMS adaptive filter with the DTD to
//!   freeze weights during double-talk events.

use std::collections::VecDeque;

use super::nlms::{NlmsConfig, NlmsFilter};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the coherence-based double-talk detector.
#[derive(Debug, Clone)]
pub struct DoubleTalkConfig {
    /// Block size for computing statistics (in samples).
    pub window_size: usize,
    /// Coherence threshold: double-talk declared when coherence < threshold.
    /// (High coherence ≈ pure echo; low coherence ≈ near-end contamination.)
    pub coherence_threshold: f64,
    /// Voice activity threshold: ratio of error power to reference power.
    /// If error_power < vad_threshold * ref_power, single-talk assumed.
    pub vad_threshold: f64,
}

impl Default for DoubleTalkConfig {
    fn default() -> Self {
        Self {
            window_size: 256,
            coherence_threshold: 0.9,
            vad_threshold: 0.01,
        }
    }
}

// ── Coherence-based DTD ───────────────────────────────────────────────────────

/// Coherence-based double-talk detector.
///
/// Computes the normalised coherence between reference and error signals:
///
/// ```text
/// coherence = |cross_corr(ref, error)|² / (power(ref) * power(error))
/// ```
///
/// When near-end speech is present the error contains uncorrelated near-end
/// energy, so coherence drops below the threshold.
#[derive(Debug, Clone)]
pub struct DoubleTalkDetector {
    config: DoubleTalkConfig,
    ref_buf: VecDeque<f64>,
    err_buf: VecDeque<f64>,
}

impl DoubleTalkDetector {
    /// Create a new DoubleTalkDetector.
    pub fn new(config: DoubleTalkConfig) -> Self {
        let n = config.window_size;
        Self {
            config,
            ref_buf: VecDeque::from(vec![0.0; n]),
            err_buf: VecDeque::from(vec![0.0; n]),
        }
    }

    /// Update buffers with one sample and query double-talk status.
    ///
    /// Returns `true` if double-talk is detected.
    pub fn update_sample(&mut self, reference: f64, error: f64) -> bool {
        self.ref_buf.pop_back();
        self.ref_buf.push_front(reference);
        self.err_buf.pop_back();
        self.err_buf.push_front(error);
        // Lazy: coherence computed over the current window
        self.compute_double_talk()
    }

    /// Detect double-talk over a block of (reference, error) pairs.
    ///
    /// Returns `true` if the block as a whole is classified as double-talk.
    pub fn detect(&mut self, reference: &[f64], error: &[f64]) -> bool {
        let len = reference.len().min(error.len());
        if len == 0 {
            return false;
        }
        // Accumulate both buffers, then check coherence at the end
        for i in 0..len {
            self.ref_buf.pop_back();
            self.ref_buf.push_front(reference[i]);
            self.err_buf.pop_back();
            self.err_buf.push_front(error[i]);
        }
        self.compute_double_talk()
    }

    /// Compute current double-talk status from stored window.
    fn compute_double_talk(&self) -> bool {
        let n = self.config.window_size;

        let ref_slice: Vec<f64> = self.ref_buf.iter().copied().collect();
        let err_slice: Vec<f64> = self.err_buf.iter().copied().collect();

        let ref_power: f64 = ref_slice.iter().map(|x| x * x).sum::<f64>() / n as f64;
        let err_power: f64 = err_slice.iter().map(|x| x * x).sum::<f64>() / n as f64;

        // If both signals are silent, no double-talk
        if ref_power < 1e-15 && err_power < 1e-15 {
            return false;
        }

        // VAD: if error power very small relative to reference, pure far-end
        if err_power < self.config.vad_threshold * ref_power.max(1e-30) {
            return false;
        }

        // Cross-correlation at lag 0 (normalised)
        let cross: f64 = ref_slice
            .iter()
            .zip(err_slice.iter())
            .map(|(r, e)| r * e)
            .sum::<f64>()
            / n as f64;

        let denom = ref_power * err_power;
        if denom < 1e-30 {
            return false;
        }

        let coherence = (cross * cross) / denom;

        // Double-talk when error is uncorrelated with reference
        coherence < self.config.coherence_threshold
    }

    /// Simple energy-based Voice Activity Detector.
    ///
    /// Returns `true` if the signal has non-trivial energy.
    pub fn voice_activity(&self, signal: &[f64]) -> bool {
        if signal.is_empty() {
            return false;
        }
        let power: f64 = signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64;
        power > 1e-8
    }
}

// ── EchoCanceller ─────────────────────────────────────────────────────────────

/// High-level echo canceller combining NLMS adaptation with coherence-based
/// double-talk detection.
///
/// During double-talk events, filter weights are frozen to prevent divergence.
#[derive(Debug, Clone)]
pub struct EchoCanceller {
    nlms: NlmsFilter,
    detector: DoubleTalkDetector,
}

impl EchoCanceller {
    /// Create a new EchoCanceller.
    ///
    /// # Arguments
    ///
    /// * `nlms_config` – Configuration for the underlying NLMS adaptive filter.
    /// * `dt_config`   – Configuration for the coherence-based double-talk detector.
    pub fn new(nlms_config: NlmsConfig, dt_config: DoubleTalkConfig) -> Self {
        Self {
            nlms: NlmsFilter::new(nlms_config),
            detector: DoubleTalkDetector::new(dt_config),
        }
    }

    /// Process a block of audio samples.
    ///
    /// Double-talk is evaluated block-by-block.  When detected, the NLMS weights
    /// are frozen for the duration of the block (the echo estimate is still
    /// computed and subtracted using the current weights).
    ///
    /// # Arguments
    ///
    /// * `reference`  – Far-end (loudspeaker) signal.
    /// * `microphone` – Microphone signal (echo + near-end speech).
    ///
    /// # Returns
    ///
    /// Echo-cancelled output signal.
    pub fn process(&mut self, reference: &[f64], microphone: &[f64]) -> Vec<f64> {
        let len = reference.len().min(microphone.len());
        if len == 0 {
            return Vec::new();
        }

        // Compute an "error preview" using current weights (no update)
        let preview_errors: Vec<f64> = (0..len)
            .map(|i| {
                let echo: f64 = self
                    .nlms
                    .weights
                    .iter()
                    .zip(self.nlms.buffer.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                microphone[i] - echo
            })
            .collect();

        // Detect double-talk from reference vs. preview error
        let double_talk = self.detector.detect(reference, &preview_errors);

        if double_talk {
            // Freeze: compute output without updating weights.
            // We still want the echo subtracted even if weights are frozen.
            (0..len)
                .map(|i| {
                    // Update buffer only (not weights)
                    self.nlms.buffer.pop_back();
                    self.nlms.buffer.push_front(reference[i]);
                    let echo: f64 = self
                        .nlms
                        .weights
                        .iter()
                        .zip(self.nlms.buffer.iter())
                        .map(|(w, x)| w * x)
                        .sum();
                    microphone[i] - echo
                })
                .collect()
        } else {
            // Adapt: normal NLMS update
            self.nlms.process_block(reference, microphone)
        }
    }

    /// Reset the echo canceller to its initial state.
    pub fn reset(&mut self) {
        self.nlms.reset();
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_talk_detector_fires_for_uncorrelated_signals() {
        let cfg = DoubleTalkConfig {
            window_size: 128,
            coherence_threshold: 0.9,
            vad_threshold: 0.001,
        };
        let mut det = DoubleTalkDetector::new(cfg);

        // Reference: sine wave; error: completely different sine (uncorrelated)
        let reference: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
        let error: Vec<f64> = (0..128).map(|i| (i as f64 * 0.73 + 1.0).sin()).collect();

        let dt = det.detect(&reference, &error);
        assert!(
            dt,
            "Should detect double-talk when ref and error are uncorrelated"
        );
    }

    #[test]
    fn double_talk_detector_no_fire_for_correlated_signals() {
        let cfg = DoubleTalkConfig::default();
        let mut det = DoubleTalkDetector::new(cfg);

        // Reference: sine wave; error: 0.5× same sine (highly correlated = pure echo)
        let reference: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        // error = 0.0  →  error power very small → vad_threshold kicks in → single-talk
        let error: Vec<f64> = reference.iter().map(|r| r * 0.001).collect();

        let dt = det.detect(&reference, &error);
        assert!(!dt, "Should not detect double-talk for small error power");
    }

    #[test]
    fn voice_activity_silent_signal() {
        let det = DoubleTalkDetector::new(DoubleTalkConfig::default());
        assert!(!det.voice_activity(&[0.0; 100]));
    }

    #[test]
    fn voice_activity_active_signal() {
        let det = DoubleTalkDetector::new(DoubleTalkConfig::default());
        let sig: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        assert!(det.voice_activity(&sig));
    }

    #[test]
    fn echo_canceller_produces_correct_length() {
        let mut ec = EchoCanceller::new(
            NlmsConfig {
                filter_length: 32,
                ..Default::default()
            },
            DoubleTalkConfig::default(),
        );
        let r: Vec<f64> = (0..256).map(|i| (i as f64 * 0.05).sin()).collect();
        let m: Vec<f64> = r.iter().map(|x| x * 0.8).collect();
        let out = ec.process(&r, &m);
        assert_eq!(out.len(), 256);
    }

    #[test]
    fn echo_canceller_cancels_pure_echo() {
        let mut ec = EchoCanceller::new(
            NlmsConfig {
                filter_length: 16,
                step_size: 0.7,
                ..Default::default()
            },
            DoubleTalkConfig {
                window_size: 64,
                coherence_threshold: 0.5,
                vad_threshold: 0.001,
            },
        );
        // 3 blocks of 512 samples; mic == 0.9 * ref (pure echo)
        let mut total_err = 0.0_f64;
        let mut count = 0;
        for block in 0..6 {
            let r: Vec<f64> = (0..512)
                .map(|i| ((block * 512 + i) as f64 * 0.07).sin())
                .collect();
            let m: Vec<f64> = r.iter().map(|x| x * 0.9).collect();
            let out = ec.process(&r, &m);
            if block >= 3 {
                total_err += out.iter().map(|e| e * e).sum::<f64>();
                count += out.len();
            }
        }
        let rms = (total_err / count as f64).sqrt();
        assert!(
            rms < 0.3,
            "EchoCanceller should reduce pure echo: rms={rms:.4}"
        );
    }
}
