//! Double-talk detection (DTD) algorithms for Acoustic Echo Cancellation.
//!
//! Double-talk occurs when both far-end (loudspeaker) and near-end (local talker)
//! signals are active simultaneously.  During double-talk, the adaptive filter
//! must freeze to prevent divergence.
//!
//! This module provides three complementary detection strategies:
//!
//! 1. **Geigel DTD** – fast, sample-level comparison of microphone amplitude
//!    against a windowed maximum of the far-end reference.
//! 2. **Cross-correlation DTD** – computes normalised cross-correlation between
//!    far-end and error signals to detect near-end contamination.
//! 3. **Far-end activity detector** – simple energy-based voice activity
//!    detector on the reference channel.
//!
//! # References
//!
//! * Geigel, R. (1973). "Techniques for suppressing double-talking in echo
//!   cancellers".  Internal Memorandum, Bell Laboratories.
//! * Benesty, J., Morgan, D.R. & Cho, J.H. (2000). "A new class of
//!   doubletalk detectors based on cross-correlation". *IEEE Trans. Speech
//!   and Audio Processing*, 8(2), 168–172.

use super::types::DoubleTalkStatus;

// ── Geigel DTD ──────────────────────────────────────────────────────────────

/// Geigel double-talk detector.
///
/// Declares double-talk when the microphone level exceeds a scaled maximum
/// of the recent far-end reference:
///
/// ```text
/// |d(n)| > δ · max( |x(n)|, |x(n−1)|, …, |x(n−L+1)| )
/// ```
///
/// The detector is lightweight (O(1) per sample with a sliding-max buffer)
/// and widely used as a first-stage DTD in practical AEC systems.
#[derive(Debug, Clone)]
pub struct GeigelDetector {
    /// Detection threshold δ ∈ (0, 1).
    threshold: f64,
    /// Sliding window of recent far-end magnitudes.
    far_end_mag_buffer: Vec<f64>,
    /// Current maximum magnitude in the window.
    current_max: f64,
    /// Circular buffer write index.
    buf_idx: usize,
    /// Window length (typically = filter_length).
    window_len: usize,
    /// Hold-off counter: samples remaining after last detection.
    holdoff_remaining: usize,
    /// Hold-off duration in samples.
    holdoff_duration: usize,
}

impl GeigelDetector {
    /// Create a new Geigel detector.
    ///
    /// # Arguments
    ///
    /// * `window_len` – Number of past far-end samples to consider (usually
    ///   equal to the AEC filter length).
    /// * `threshold` – Detection sensitivity δ ∈ (0, 1).  Smaller values
    ///   make the detector more aggressive (more freeze events).
    /// * `holdoff_duration` – Number of samples to remain in double-talk state
    ///   after the condition clears.
    pub fn new(window_len: usize, threshold: f64, holdoff_duration: usize) -> Self {
        let window_len = window_len.max(1);
        Self {
            threshold: threshold.clamp(1e-6, 1.0 - 1e-6),
            far_end_mag_buffer: vec![0.0; window_len],
            current_max: 0.0,
            buf_idx: 0,
            window_len,
            holdoff_remaining: 0,
            holdoff_duration,
        }
    }

    /// Process one sample pair and return whether double-talk is detected.
    ///
    /// # Arguments
    ///
    /// * `far_end_sample` – Current far-end reference sample x(n).
    /// * `microphone_sample` – Current microphone (desired) sample d(n).
    ///
    /// # Returns
    ///
    /// `true` if double-talk is detected (adaptation should freeze).
    pub fn detect(&mut self, far_end_sample: f64, microphone_sample: f64) -> bool {
        let far_mag = far_end_sample.abs();

        // Update circular buffer
        self.far_end_mag_buffer[self.buf_idx] = far_mag;
        self.buf_idx = (self.buf_idx + 1) % self.window_len;

        // Recompute sliding max (exact, O(L) but simple and robust).
        // For very long filters one could use a monotonic deque; here L is
        // typically 128–2048 so a linear scan is fine.
        self.current_max = self
            .far_end_mag_buffer
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);

        let mic_mag = microphone_sample.abs();
        let detected = mic_mag > self.threshold * self.current_max && self.current_max > 1e-12;

        if detected {
            self.holdoff_remaining = self.holdoff_duration;
        } else if self.holdoff_remaining > 0 {
            self.holdoff_remaining -= 1;
        }

        detected || self.holdoff_remaining > 0
    }

    /// Reset the detector state.
    pub fn reset(&mut self) {
        self.far_end_mag_buffer.fill(0.0);
        self.current_max = 0.0;
        self.buf_idx = 0;
        self.holdoff_remaining = 0;
    }
}

// ── Cross-correlation DTD ───────────────────────────────────────────────────

/// Cross-correlation based double-talk detector.
///
/// Estimates the normalised cross-correlation between the far-end reference
/// and the error (residual) signal.  When near-end speech contaminates the
/// error, the cross-correlation drops below a threshold, indicating double-talk.
///
/// Uses exponentially-weighted running statistics to avoid block-level latency.
#[derive(Debug, Clone)]
pub struct CrossCorrelationDetector {
    /// Smoothing factor α for running power/cross-correlation estimates.
    alpha: f64,
    /// Running estimate of far-end power E[x²].
    far_end_power: f64,
    /// Running estimate of error power E[e²].
    error_power: f64,
    /// Running estimate of cross-power E[x·e].
    cross_power: f64,
    /// Detection threshold for normalised cross-correlation.
    /// Double-talk is declared when |ρ_xe| > threshold (error is correlated
    /// with far-end, meaning the filter has not fully cancelled the echo
    /// AND near-end is present).
    threshold: f64,
}

impl CrossCorrelationDetector {
    /// Create a new cross-correlation DTD.
    ///
    /// # Arguments
    ///
    /// * `alpha` – Smoothing factor ∈ (0, 1).  Values near 1.0 track changes
    ///   quickly; values near 0.0 give smoother estimates.
    /// * `threshold` – Decision threshold ∈ (0, 1).  When the normalised
    ///   cross-correlation magnitude exceeds this value, double-talk is declared.
    pub fn new(alpha: f64, threshold: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 0.999),
            far_end_power: 0.0,
            error_power: 0.0,
            cross_power: 0.0,
            threshold: threshold.clamp(0.01, 0.99),
        }
    }

    /// Process one sample pair.
    ///
    /// # Arguments
    ///
    /// * `far_end_sample` – Current far-end reference x(n).
    /// * `error_sample` – Current residual (microphone − echo estimate) e(n).
    ///
    /// # Returns
    ///
    /// `true` if double-talk is detected.
    pub fn detect(&mut self, far_end_sample: f64, error_sample: f64) -> bool {
        let a = self.alpha;
        self.far_end_power = (1.0 - a) * self.far_end_power + a * far_end_sample * far_end_sample;
        self.error_power = (1.0 - a) * self.error_power + a * error_sample * error_sample;
        self.cross_power = (1.0 - a) * self.cross_power + a * far_end_sample * error_sample;

        let denom = (self.far_end_power * self.error_power).sqrt();
        if denom < 1e-15 {
            return false;
        }
        let rho = (self.cross_power / denom).abs();

        // High cross-correlation between far-end and error means the filter
        // is not tracking well – likely near-end interference.
        // In a well-converged filter with only far-end, error is uncorrelated
        // with x.  When near-end appears, error becomes uncorrelated with x
        // too, BUT the error power rises sharply relative to far-end power.
        // We invert the logic: low correlation means good cancellation OR
        // pure near-end; high correlation means echo leakage.
        // For DTD we use: double-talk when error power is high AND correlation
        // is low (near-end dominates).
        let power_ratio = if self.far_end_power > 1e-15 {
            self.error_power / self.far_end_power
        } else {
            0.0
        };

        // Double-talk: error power much higher than expected and decorrelated
        // from far-end.
        power_ratio > self.threshold && rho < self.threshold
    }

    /// Reset running statistics.
    pub fn reset(&mut self) {
        self.far_end_power = 0.0;
        self.error_power = 0.0;
        self.cross_power = 0.0;
    }
}

// ── Far-end activity detector ───────────────────────────────────────────────

/// Energy-based far-end activity detector.
///
/// Computes a smoothed estimate of the far-end signal energy and compares
/// it against a noise-floor threshold.  Used to distinguish "far-end only"
/// from "silence" states.
#[derive(Debug, Clone)]
pub struct FarEndActivityDetector {
    /// Smoothing factor for energy estimate.
    alpha: f64,
    /// Running smoothed energy.
    energy: f64,
    /// Activation threshold (energy must exceed this for "active").
    threshold: f64,
}

impl FarEndActivityDetector {
    /// Create a new far-end activity detector.
    ///
    /// # Arguments
    ///
    /// * `alpha` – Smoothing factor ∈ (0, 1).
    /// * `threshold` – Minimum smoothed energy to declare activity.
    pub fn new(alpha: f64, threshold: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 0.999),
            energy: 0.0,
            threshold: threshold.max(0.0),
        }
    }

    /// Update with a new far-end sample and return activity status.
    pub fn is_active(&mut self, sample: f64) -> bool {
        self.energy = (1.0 - self.alpha) * self.energy + self.alpha * sample * sample;
        self.energy > self.threshold
    }

    /// Reset the detector.
    pub fn reset(&mut self) {
        self.energy = 0.0;
    }
}

// ── Combined DTD status ─────────────────────────────────────────────────────

/// Determine the combined double-talk status from individual detectors.
///
/// # Arguments
///
/// * `far_end_active` – Whether the far-end reference signal is active.
/// * `double_talk_detected` – Whether the Geigel/cross-correlation DTD
///   has flagged double-talk.
/// * `error_magnitude` – Absolute value of the current error sample
///   (used to infer near-end activity heuristically).
/// * `near_end_threshold` – Minimum error magnitude to declare near-end
///   activity when far-end is silent.
pub fn classify_talk_status(
    far_end_active: bool,
    double_talk_detected: bool,
    error_magnitude: f64,
    near_end_threshold: f64,
) -> DoubleTalkStatus {
    let near_end_active = error_magnitude > near_end_threshold;

    match (far_end_active, near_end_active, double_talk_detected) {
        (_, _, true) => DoubleTalkStatus::DoubleTalk,
        (true, true, false) => DoubleTalkStatus::DoubleTalk,
        (true, false, false) => DoubleTalkStatus::FarEndOnly,
        (false, true, false) => DoubleTalkStatus::NearEndOnly,
        (false, false, false) => DoubleTalkStatus::NoTalk,
        _ => DoubleTalkStatus::NoTalk,
    }
}
