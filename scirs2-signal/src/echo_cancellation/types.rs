//! Types and configuration for Acoustic Echo Cancellation (AEC).
//!
//! This module defines the core data structures used throughout the AEC pipeline:
//! configuration parameters, filter state, double-talk status indicators,
//! and per-sample processing results.

use scirs2_core::ndarray::Array1;

/// Configuration for the Acoustic Echo Canceller.
///
/// Controls filter dimensions, adaptation speed, double-talk detection
/// sensitivity, and optional sub-band decomposition.
///
/// # Example
///
/// ```
/// use scirs2_signal::echo_cancellation::AECConfig;
///
/// let cfg = AECConfig {
///     filter_length: 512,
///     step_size: 0.3,
///     ..AECConfig::default()
/// };
/// assert_eq!(cfg.filter_length, 512);
/// ```
#[derive(Debug, Clone)]
pub struct AECConfig {
    /// Number of taps in the adaptive filter (echo-path model length).
    ///
    /// Longer filters can capture longer echo paths but converge more slowly.
    /// Typical values: 128–2048 (at 8 kHz, 128 taps ≈ 16 ms).
    pub filter_length: usize,

    /// NLMS normalised step size μ ∈ (0, 2).
    ///
    /// Controls the trade-off between convergence speed and steady-state
    /// misadjustment.  Values around 0.3–0.8 are common for AEC.
    pub step_size: f64,

    /// Regularisation constant ε for NLMS normalisation.
    ///
    /// Prevents division by zero when input power is very small.
    pub regularization: f64,

    /// Geigel double-talk detection threshold δ ∈ (0, 1).
    ///
    /// The Geigel detector declares double-talk when
    /// |d(n)| > δ · max(|x(n)|, …, |x(n−L+1)|).
    /// Smaller values make the detector more sensitive (more freezes).
    /// For echo paths with gain < 1 (typical), values near 1.0 are
    /// appropriate; 0.7 is a safe default.
    pub dtd_threshold: f64,

    /// Number of sub-bands for the multi-delay filter bank.
    ///
    /// When > 1 the impulse response is partitioned into `num_subbands`
    /// sub-filters of length `filter_length / num_subbands`, enabling
    /// more efficient processing of long echo paths.
    /// Set to 1 to disable partitioning.
    pub num_subbands: usize,

    /// Exponential smoothing factor for ERLE power estimates α ∈ (0, 1].
    ///
    /// Controls how quickly the ERLE measurement tracks changes.
    /// Smaller values give a smoother (more stable) ERLE reading.
    pub erle_smoothing: f64,

    /// Hold-off duration (in samples) after double-talk detection.
    ///
    /// The filter remains frozen for this many samples after the
    /// last double-talk detection event, reducing near-end divergence.
    pub dtd_holdoff_samples: usize,
}

impl Default for AECConfig {
    fn default() -> Self {
        Self {
            filter_length: 256,
            step_size: 0.5,
            regularization: 1e-6,
            dtd_threshold: 0.7,
            num_subbands: 1,
            erle_smoothing: 0.98,
            dtd_holdoff_samples: 16,
        }
    }
}

/// Persistent state of the adaptive echo canceller.
///
/// Holds filter coefficients, circular input buffer, and running
/// statistics used for ERLE computation and double-talk detection.
#[derive(Debug, Clone)]
pub struct AECState {
    /// Adaptive filter coefficients (echo-path estimate).
    pub coefficients: Array1<f64>,
    /// ERLE history (in dB) – one entry per processed sample.
    pub erle_history: Vec<f64>,
    /// Circular far-end input buffer (length = filter_length).
    pub(crate) far_end_buffer: Vec<f64>,
    /// Write index into the circular buffer.
    pub(crate) buf_idx: usize,
    /// Smoothed desired-signal power (for ERLE numerator).
    pub(crate) desired_power: f64,
    /// Smoothed error-signal power (for ERLE denominator).
    pub(crate) error_power: f64,
    /// Samples remaining in the DTD hold-off window.
    pub(crate) dtd_holdoff_counter: usize,
}

impl AECState {
    /// Create a fresh AEC state for the given filter length.
    pub fn new(filter_length: usize) -> Self {
        Self {
            coefficients: Array1::zeros(filter_length),
            erle_history: Vec::new(),
            far_end_buffer: vec![0.0; filter_length],
            buf_idx: 0,
            desired_power: 0.0,
            error_power: 0.0,
            dtd_holdoff_counter: 0,
        }
    }

    /// Reset the state to initial conditions without reallocating.
    pub fn reset(&mut self) {
        self.coefficients.fill(0.0);
        self.erle_history.clear();
        self.far_end_buffer.fill(0.0);
        self.buf_idx = 0;
        self.desired_power = 0.0;
        self.error_power = 0.0;
        self.dtd_holdoff_counter = 0;
    }
}

/// Double-talk detection status.
///
/// Indicates the current talk activity as detected by the DTD subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DoubleTalkStatus {
    /// Silence – neither far-end nor near-end activity detected.
    NoTalk,
    /// Only far-end signal is active (safe to adapt).
    FarEndOnly,
    /// Only near-end signal is active (no echo present).
    NearEndOnly,
    /// Both far-end and near-end are active (freeze adaptation).
    DoubleTalk,
}

/// Result of processing a single AEC sample.
///
/// Contains the echo-cancelled output, current ERLE measurement,
/// and double-talk status.
#[derive(Debug, Clone)]
pub struct AECResult {
    /// Echo-cancelled output sample (near-end estimate).
    pub output: f64,
    /// Echo Return Loss Enhancement in dB.
    ///
    /// ERLE = 10 log10(`E[d^2] / E[e^2]`), higher is better.
    /// Returns `f64::NEG_INFINITY` when error power is zero.
    pub erle_db: f64,
    /// Current double-talk detection status.
    pub dtd_status: DoubleTalkStatus,
}
