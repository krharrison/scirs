//! NLMS-based Acoustic Echo Canceller with double-talk detection.
//!
//! Implements a full AEC pipeline: NLMS adaptive filtering, Geigel + far-end
//! activity double-talk detection, ERLE monitoring, and optional multi-delay
//! filter bank partitioning for long echo paths.
//!
//! # Algorithm overview
//!
//! 1. Push far-end sample into circular buffer.
//! 2. Compute echo estimate ŷ = wᵀ x.
//! 3. Compute error e = d − ŷ (echo-cancelled output).
//! 4. Run double-talk detector(s).
//! 5. If no double-talk: adapt w ← w + (μ / (ε + ‖x‖²)) e x.
//! 6. Update smoothed ERLE.
//!
//! # Multi-delay filter bank
//!
//! For long impulse responses the filter can be partitioned into `K` sub-filters
//! of length `L/K`.  Each sub-filter covers a different delay segment of the
//! echo path.  This is conceptually equivalent to a single filter of length `L`
//! but allows independent step-size normalisation per partition, improving
//! convergence for echo paths with widely varying energy across delays.
//!
//! # References
//!
//! * Haykin, S. (2002). *Adaptive Filter Theory*, 4th ed. Prentice Hall.
//! * Soo, J. & Pang, K. (1990). "Multidelay block frequency domain adaptive
//!   filter". *IEEE Trans. ASSP*, 38(2), 373–376.

use scirs2_core::ndarray::Array1;

use super::double_talk::{classify_talk_status, FarEndActivityDetector, GeigelDetector};
use super::types::{AECConfig, AECResult, AECState, DoubleTalkStatus};
use crate::error::{SignalError, SignalResult};

/// Acoustic Echo Canceller.
///
/// Combines NLMS adaptive filtering with Geigel double-talk detection
/// and ERLE monitoring.  Supports optional multi-delay partitioning.
///
/// # Example
///
/// ```
/// use scirs2_signal::echo_cancellation::{AECConfig, AcousticEchoCanceller};
///
/// let cfg = AECConfig { filter_length: 64, step_size: 0.5, ..AECConfig::default() };
/// let mut aec = AcousticEchoCanceller::new(cfg).expect("create AEC");
///
/// // Simulate: far-end goes through an echo path, microphone picks up echo
/// let echo_path = [0.0, 0.0, 0.8, 0.0];  // 2-sample delay, gain 0.8
/// for n in 0..500 {
///     let far = (n as f64 * 0.1).sin();
///     let echo: f64 = echo_path.iter().enumerate()
///         .map(|(k, &h)| if n >= k { h * ((n - k) as f64 * 0.1).sin() } else { 0.0 })
///         .sum();
///     let mic = echo; // no near-end for this test
///     let result = aec.process_sample(far, mic);
///     // result.output should converge toward 0 as the filter learns
/// }
/// ```
#[derive(Debug, Clone)]
pub struct AcousticEchoCanceller {
    /// AEC configuration.
    config: AECConfig,
    /// Adaptive filter state.
    state: AECState,
    /// Geigel double-talk detector.
    geigel: GeigelDetector,
    /// Far-end activity detector.
    far_end_vad: FarEndActivityDetector,
    /// Number of sub-filters (partitions).
    num_partitions: usize,
    /// Length of each partition.
    partition_len: usize,
    /// Sample counter (for warm-up period).
    sample_count: usize,
}

impl AcousticEchoCanceller {
    /// Create a new AEC instance from the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `SignalError::ValueError` if configuration parameters are
    /// out of valid range.
    pub fn new(config: AECConfig) -> SignalResult<Self> {
        if config.filter_length == 0 {
            return Err(SignalError::ValueError(
                "filter_length must be >= 1".to_string(),
            ));
        }
        if config.step_size <= 0.0 || config.step_size >= 2.0 {
            return Err(SignalError::ValueError(
                "step_size must be in (0, 2) for NLMS convergence".to_string(),
            ));
        }
        if config.num_subbands == 0 {
            return Err(SignalError::ValueError(
                "num_subbands must be >= 1".to_string(),
            ));
        }

        let num_partitions = config.num_subbands.min(config.filter_length);
        // Round up partition length so total >= filter_length
        let partition_len = (config.filter_length + num_partitions - 1) / num_partitions;
        let actual_filter_len = partition_len * num_partitions;

        let state = AECState {
            coefficients: Array1::zeros(actual_filter_len),
            erle_history: Vec::new(),
            far_end_buffer: vec![0.0; actual_filter_len],
            buf_idx: 0,
            desired_power: 0.0,
            error_power: 0.0,
            dtd_holdoff_counter: 0,
        };

        let geigel = GeigelDetector::new(
            actual_filter_len,
            config.dtd_threshold,
            config.dtd_holdoff_samples,
        );

        // Far-end VAD: use a low energy threshold for activity detection.
        let far_end_vad = FarEndActivityDetector::new(0.1, 1e-8);

        Ok(Self {
            config,
            state,
            geigel,
            far_end_vad,
            num_partitions,
            partition_len,
            sample_count: 0,
        })
    }

    /// Process a single sample pair.
    ///
    /// # Arguments
    ///
    /// * `far_end` – Far-end reference sample (loudspeaker signal).
    /// * `near_end` – Microphone sample (echo + near-end + noise).
    ///
    /// # Returns
    ///
    /// An [`AECResult`] containing the echo-cancelled output, current ERLE,
    /// and double-talk status.
    pub fn process_sample(&mut self, far_end: f64, near_end: f64) -> AECResult {
        let filter_len = self.state.coefficients.len();
        let alpha = self.config.erle_smoothing;
        let eps = self.config.regularization;

        // 1. Push far-end into circular buffer
        self.state.far_end_buffer[self.state.buf_idx] = far_end;

        // 2. Build input vector x(n) from circular buffer (most recent first)
        let x = self.build_input_vector();

        // 3. Compute echo estimate: ŷ = wᵀ x
        let echo_estimate = self.dot_product(&x);

        // 4. Compute error (echo-cancelled output)
        let error = near_end - echo_estimate;

        // 5. Double-talk detection
        let far_active = self.far_end_vad.is_active(far_end);

        // Geigel DTD on the raw microphone signal: declares double-talk when
        // |d(n)| > δ · max(|x(n)|, …, |x(n−L+1)|).
        // With δ close to 1.0, this only fires when near-end speech pushes
        // the microphone level above the far-end reference (echo path gain
        // is typically < 1.0 in practice).
        let geigel_dtd = self.geigel.detect(far_end, near_end);
        self.sample_count = self.sample_count.saturating_add(1);

        let dtd_status = classify_talk_status(
            far_active,
            geigel_dtd,
            error.abs(),
            1e-6, // near-end threshold for silence detection
        );

        // 6. Adapt filter when far-end is active and DTD does not freeze.
        let should_adapt = far_active && !geigel_dtd;

        if should_adapt {
            if self.num_partitions <= 1 {
                // Standard NLMS update
                self.nlms_update(&x, error, eps);
            } else {
                // Multi-delay partitioned NLMS update
                self.partitioned_nlms_update(&x, error, eps);
            }
        }

        // 7. Advance circular buffer index
        self.state.buf_idx = (self.state.buf_idx + 1) % filter_len;

        // 8. Update smoothed ERLE
        self.state.desired_power =
            alpha * self.state.desired_power + (1.0 - alpha) * near_end * near_end;
        self.state.error_power = alpha * self.state.error_power + (1.0 - alpha) * error * error;

        let erle_db = compute_erle_db(self.state.desired_power, self.state.error_power);
        self.state.erle_history.push(erle_db);

        AECResult {
            output: error,
            erle_db,
            dtd_status,
        }
    }

    /// Process a block of samples.
    ///
    /// Equivalent to calling [`process_sample`](Self::process_sample) for each
    /// element, but returns a collected vector of results.
    ///
    /// # Errors
    ///
    /// Returns `SignalError::DimensionMismatch` if the two slices differ in length.
    pub fn process_block(
        &mut self,
        far_end_block: &[f64],
        near_end_block: &[f64],
    ) -> SignalResult<Vec<AECResult>> {
        if far_end_block.len() != near_end_block.len() {
            return Err(SignalError::DimensionMismatch(format!(
                "far_end_block length {} != near_end_block length {}",
                far_end_block.len(),
                near_end_block.len()
            )));
        }
        let results = far_end_block
            .iter()
            .zip(near_end_block.iter())
            .map(|(&f, &n)| self.process_sample(f, n))
            .collect();
        Ok(results)
    }

    /// Return a reference to the current filter coefficients.
    pub fn coefficients(&self) -> &Array1<f64> {
        &self.state.coefficients
    }

    /// Return the ERLE history (in dB).
    pub fn erle_history(&self) -> &[f64] {
        &self.state.erle_history
    }

    /// Return a reference to the internal state.
    pub fn state(&self) -> &AECState {
        &self.state
    }

    /// Reset the canceller to its initial state.
    pub fn reset(&mut self) {
        self.state.reset();
        self.geigel.reset();
        self.far_end_vad.reset();
        self.sample_count = 0;
    }

    // ── private helpers ─────────────────────────────────────────────────────

    /// Build the input vector x(n) from the circular buffer.
    ///
    /// x = [x(n), x(n−1), …, x(n−L+1)] where x(n) is at buf_idx.
    fn build_input_vector(&self) -> Vec<f64> {
        let len = self.state.far_end_buffer.len();
        let mut x = Vec::with_capacity(len);
        for k in 0..len {
            let idx = (self.state.buf_idx + len - k) % len;
            x.push(self.state.far_end_buffer[idx]);
        }
        x
    }

    /// Dot product of coefficients with an input vector.
    fn dot_product(&self, x: &[f64]) -> f64 {
        self.state
            .coefficients
            .iter()
            .zip(x.iter())
            .map(|(&w, &xi)| w * xi)
            .sum()
    }

    /// Standard NLMS weight update.
    fn nlms_update(&mut self, x: &[f64], error: f64, eps: f64) {
        let power: f64 = x.iter().map(|xi| xi * xi).sum();
        let step = self.config.step_size / (eps + power);
        for (w, &xi) in self.state.coefficients.iter_mut().zip(x.iter()) {
            *w += step * error * xi;
        }
    }

    /// Multi-delay (partitioned) NLMS update.
    ///
    /// Each partition normalises its step size independently based on the
    /// energy of the corresponding segment of the input vector.  To prevent
    /// divergence in partitions with very low energy, the global input power
    /// is used as a lower bound for normalisation.
    fn partitioned_nlms_update(&mut self, x: &[f64], error: f64, eps: f64) {
        // Compute global power for stability lower bound
        let global_power: f64 = x.iter().map(|xi| xi * xi).sum();
        let min_partition_power = global_power / (self.num_partitions as f64).max(1.0);

        for p in 0..self.num_partitions {
            let start = p * self.partition_len;
            let end = ((p + 1) * self.partition_len).min(x.len());
            if start >= x.len() {
                break;
            }

            let segment = &x[start..end];
            let partition_power: f64 = segment.iter().map(|xi| xi * xi).sum();
            // Use at least min_partition_power to avoid blow-up in silent partitions
            let effective_power = partition_power.max(min_partition_power);
            let step = self.config.step_size / (eps + effective_power);

            for k in start..end {
                if k < self.state.coefficients.len() {
                    self.state.coefficients[k] += step * error * x[k];
                }
            }
        }
    }
}

/// Compute ERLE in dB from smoothed desired and error powers.
///
/// ERLE = 10 log₁₀(P_desired / P_error)
///
/// Returns `f64::NEG_INFINITY` when error power is effectively zero,
/// and `0.0` when desired power is effectively zero.
fn compute_erle_db(desired_power: f64, error_power: f64) -> f64 {
    if desired_power < 1e-30 {
        return 0.0;
    }
    if error_power < 1e-30 {
        return f64::NEG_INFINITY.max(-200.0); // cap at −200 dB
    }
    10.0 * (desired_power / error_power).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a simple echo scenario: far-end convolved with an impulse
    /// response gives the microphone signal (no near-end, no noise).
    fn generate_echo_scenario(num_samples: usize, echo_path: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut far_end = Vec::with_capacity(num_samples);
        let mut mic = Vec::with_capacity(num_samples);

        for n in 0..num_samples {
            let x = (n as f64 * 0.05).sin() + 0.5 * (n as f64 * 0.13).cos();
            far_end.push(x);

            let mut echo = 0.0;
            for (k, &h) in echo_path.iter().enumerate() {
                if n >= k {
                    let xk =
                        (((n - k) as f64) * 0.05).sin() + 0.5 * (((n - k) as f64) * 0.13).cos();
                    echo += h * xk;
                }
            }
            mic.push(echo);
        }
        (far_end, mic)
    }

    #[test]
    fn test_aec_creation_default() {
        let aec = AcousticEchoCanceller::new(AECConfig::default());
        assert!(aec.is_ok());
    }

    #[test]
    fn test_aec_invalid_config() {
        let cfg = AECConfig {
            filter_length: 0,
            ..AECConfig::default()
        };
        assert!(AcousticEchoCanceller::new(cfg).is_err());

        let cfg = AECConfig {
            step_size: 0.0,
            ..AECConfig::default()
        };
        assert!(AcousticEchoCanceller::new(cfg).is_err());

        let cfg = AECConfig {
            step_size: 2.5,
            ..AECConfig::default()
        };
        assert!(AcousticEchoCanceller::new(cfg).is_err());

        let cfg = AECConfig {
            num_subbands: 0,
            ..AECConfig::default()
        };
        assert!(AcousticEchoCanceller::new(cfg).is_err());
    }

    #[test]
    fn test_echo_suppression_known_path() {
        let echo_path = [0.0, 0.0, 0.8, -0.3, 0.1];
        let (far, mic) = generate_echo_scenario(2000, &echo_path);

        let cfg = AECConfig {
            filter_length: 32,
            step_size: 0.5,
            ..AECConfig::default()
        };
        let mut aec = AcousticEchoCanceller::new(cfg).expect("create AEC");

        let results = aec.process_block(&far, &mic).expect("process block");

        // After convergence, output (error) should be near zero
        let last_100: Vec<f64> = results[1900..].iter().map(|r| r.output).collect();
        let rms: f64 = (last_100.iter().map(|e| e * e).sum::<f64>() / 100.0).sqrt();
        assert!(
            rms < 0.05,
            "RMS of last 100 samples should be near zero, got {rms}"
        );
    }

    #[test]
    fn test_erle_improves_over_time() {
        let echo_path = [0.0, 0.7, 0.3];
        let (far, mic) = generate_echo_scenario(3000, &echo_path);

        let cfg = AECConfig {
            filter_length: 32,
            step_size: 0.5,
            ..AECConfig::default()
        };
        let mut aec = AcousticEchoCanceller::new(cfg).expect("create AEC");
        let results = aec.process_block(&far, &mic).expect("process");

        // Average ERLE in first 500 samples vs last 500 samples
        let early_erle: f64 = results[100..600]
            .iter()
            .map(|r| r.erle_db)
            .filter(|e| e.is_finite())
            .sum::<f64>()
            / 500.0;
        let late_erle: f64 = results[2500..]
            .iter()
            .map(|r| r.erle_db)
            .filter(|e| e.is_finite())
            .sum::<f64>()
            / 500.0;

        assert!(
            late_erle > early_erle,
            "ERLE should improve: early={early_erle:.1} dB, late={late_erle:.1} dB"
        );
    }

    #[test]
    fn test_filter_converges_to_echo_path() {
        // Use a low-gain echo path so that the Geigel DTD cannot spuriously
        // fire (total echo gain 0.3 is far below any reasonable threshold).
        // A pseudo-random far-end signal provides good persistence of
        // excitation for fast convergence.
        let echo_path = [0.0, 0.2, -0.1];
        let num_samples = 15000;

        // Generate a pseudo-random far-end signal using a simple LCG
        let mut rng_state: u64 = 12345;
        let mut far = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (rng_state >> 33) as f64 / (u32::MAX as f64) - 0.5;
            far.push(val);
        }

        // Generate mic = convolution of far with echo_path (pure echo, no near-end)
        let mut mic = vec![0.0; num_samples];
        for n in 0..num_samples {
            for (k, &h) in echo_path.iter().enumerate() {
                if n >= k {
                    mic[n] += h * far[n - k];
                }
            }
        }

        let cfg = AECConfig {
            filter_length: 16,
            step_size: 0.5,
            // Threshold near 1.0 and no hold-off: pure echo (gain < 1)
            // should never trigger Geigel.
            dtd_threshold: 0.999,
            dtd_holdoff_samples: 0,
            ..AECConfig::default()
        };
        let mut aec = AcousticEchoCanceller::new(cfg).expect("create AEC");
        let _results = aec.process_block(&far, &mic).expect("process");

        let w = aec.coefficients();

        // Verify that the learned coefficients approximate the true echo path
        for (k, &h) in echo_path.iter().enumerate() {
            assert!(
                (w[k] - h).abs() < 0.15,
                "w[{k}]={:.4} should be near h[{k}]={h:.4}",
                w[k]
            );
        }
        // Remaining taps should be near zero
        for k in echo_path.len()..8 {
            assert!(w[k].abs() < 0.15, "w[{k}]={:.4} should be near zero", w[k]);
        }
    }

    #[test]
    fn test_dtd_identifies_double_talk() {
        let echo_path = [0.0, 0.5];
        let (far, mic_echo) = generate_echo_scenario(1000, &echo_path);

        // Add strong near-end signal in samples 500..700
        let mut mic: Vec<f64> = mic_echo;
        for n in 500..700 {
            mic[n] += 5.0 * (n as f64 * 0.3).sin(); // loud near-end
        }

        let cfg = AECConfig {
            filter_length: 32,
            step_size: 0.5,
            dtd_threshold: 0.4,
            ..AECConfig::default()
        };
        let mut aec = AcousticEchoCanceller::new(cfg).expect("create AEC");
        let results = aec.process_block(&far, &mic).expect("process");

        // During the double-talk region, at least some samples should be flagged
        let dt_count = results[500..700]
            .iter()
            .filter(|r| matches!(r.dtd_status, DoubleTalkStatus::DoubleTalk))
            .count();
        assert!(
            dt_count > 10,
            "Should detect double-talk in at least 10 of 200 samples, got {dt_count}"
        );
    }

    #[test]
    fn test_no_adaptation_during_double_talk() {
        // Near-end signal much louder than far-end triggers Geigel DTD
        // and prevents adaptation.
        let echo_path = [0.0, 0.3];
        let num_samples = 2000;

        // Use a pseudo-random far-end signal so the Geigel buffer fills
        // with representative values quickly.
        let mut rng_state: u64 = 54321;
        let mut far = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (rng_state >> 33) as f64 / (u32::MAX as f64) - 0.5;
            far.push(val);
        }

        // Generate mic = echo + huge near-end that dominates.
        // mic >> far_end ensures Geigel fires (|mic| > threshold * max(|far|)).
        let mut mic = vec![0.0; num_samples];
        for n in 0..num_samples {
            // echo component
            for (k, &h) in echo_path.iter().enumerate() {
                if n >= k {
                    mic[n] += h * far[n - k];
                }
            }
            // Dominant near-end: 200x louder than far-end (~0.5 amplitude)
            mic[n] += 100.0 * (n as f64 * 0.2).sin();
        }

        let cfg = AECConfig {
            filter_length: 16,
            step_size: 0.5,
            dtd_threshold: 0.3, // sensitive: triggers when mic > 0.3 * max(far_end)
            dtd_holdoff_samples: 50,
            ..AECConfig::default()
        };
        let mut aec = AcousticEchoCanceller::new(cfg).expect("create AEC");
        let results = aec.process_block(&far, &mic).expect("process");

        // After the initial warm-up period, most samples should detect
        // double-talk.  Check the second half only (after the Geigel
        // sliding-max buffer and far-end VAD have stabilised).
        let second_half = &results[num_samples / 2..];
        let dt_count = second_half
            .iter()
            .filter(|r| matches!(r.dtd_status, DoubleTalkStatus::DoubleTalk))
            .count();
        let half_len = second_half.len();
        assert!(
            dt_count > half_len / 2,
            "Should detect double-talk in most second-half samples, got {dt_count}/{half_len}"
        );

        // Filter should remain near zero because DTD prevents adaptation.
        // Allow slightly larger norm since a few warm-up samples may adapt.
        let w = aec.coefficients();
        let w_norm: f64 = w.iter().map(|wi| wi * wi).sum::<f64>();
        assert!(
            w_norm < 10.0,
            "Weight norm should be small due to DTD freeze, got {w_norm:.4}"
        );
    }

    #[test]
    fn test_multi_delay_handles_long_ir() {
        // Long echo path (20 taps) with moderate total gain.
        let mut echo_path = vec![0.0; 20];
        echo_path[2] = 0.5;
        echo_path[10] = -0.3;
        echo_path[18] = 0.15;

        // Use more samples and a pseudo-random far-end for reliable
        // convergence with a 20-tap echo path.
        let num_samples = 15000;
        let mut rng_state: u64 = 99999;
        let mut far = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (rng_state >> 33) as f64 / (u32::MAX as f64) - 0.5;
            far.push(val);
        }

        // Generate mic = convolution of far with echo_path
        let mut mic = vec![0.0; num_samples];
        for n in 0..num_samples {
            for (k, &h) in echo_path.iter().enumerate() {
                if n >= k {
                    mic[n] += h * far[n - k];
                }
            }
        }

        let cfg = AECConfig {
            filter_length: 32,
            step_size: 0.5,
            num_subbands: 4, // partition into 4 sub-filters of 8 taps
            // High threshold + no hold-off to avoid spurious DTD freezing.
            dtd_threshold: 0.999,
            dtd_holdoff_samples: 0,
            ..AECConfig::default()
        };
        let mut aec = AcousticEchoCanceller::new(cfg).expect("create AEC");
        let results = aec.process_block(&far, &mic).expect("process");

        // Check convergence: last 200 samples should have low residual
        let tail_start = num_samples - 200;
        let last_n: Vec<f64> = results[tail_start..].iter().map(|r| r.output).collect();
        let rms: f64 = (last_n.iter().map(|e| e * e).sum::<f64>() / last_n.len() as f64).sqrt();
        assert!(
            rms < 0.15,
            "Multi-delay AEC should converge, last-200 RMS = {rms}"
        );
    }

    #[test]
    fn test_process_block_matches_sample_by_sample() {
        let echo_path = [0.0, 0.5, -0.3];
        let (far, mic) = generate_echo_scenario(200, &echo_path);

        let cfg = AECConfig {
            filter_length: 16,
            step_size: 0.4,
            ..AECConfig::default()
        };

        // Block processing
        let mut aec_block = AcousticEchoCanceller::new(cfg.clone()).expect("create");
        let block_results = aec_block.process_block(&far, &mic).expect("block");

        // Sample-by-sample processing
        let mut aec_sample = AcousticEchoCanceller::new(cfg).expect("create");
        let sample_results: Vec<AECResult> = far
            .iter()
            .zip(mic.iter())
            .map(|(&f, &m)| aec_sample.process_sample(f, m))
            .collect();

        // Outputs must be bit-identical
        for (i, (br, sr)) in block_results.iter().zip(sample_results.iter()).enumerate() {
            assert!(
                (br.output - sr.output).abs() < 1e-14,
                "Sample {i}: block={} sample={}",
                br.output,
                sr.output
            );
        }
    }

    #[test]
    fn test_reset() {
        let cfg = AECConfig::default();
        let mut aec = AcousticEchoCanceller::new(cfg).expect("create");

        // Process some data
        for n in 0..100 {
            aec.process_sample((n as f64 * 0.1).sin(), (n as f64 * 0.1).sin() * 0.5);
        }
        assert!(!aec.erle_history().is_empty());

        aec.reset();
        assert!(aec.erle_history().is_empty());
        assert!(aec.coefficients().iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_dimension_mismatch() {
        let cfg = AECConfig::default();
        let mut aec = AcousticEchoCanceller::new(cfg).expect("create");
        let result = aec.process_block(&[1.0, 2.0], &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_silence_detection() {
        let cfg = AECConfig::default();
        let mut aec = AcousticEchoCanceller::new(cfg).expect("create");

        // Process silence
        let result = aec.process_sample(0.0, 0.0);
        assert!(
            matches!(result.dtd_status, DoubleTalkStatus::NoTalk),
            "Silence should give NoTalk status"
        );
    }

    #[test]
    fn test_far_end_only() {
        let cfg = AECConfig {
            filter_length: 8,
            step_size: 0.5,
            ..AECConfig::default()
        };
        let mut aec = AcousticEchoCanceller::new(cfg).expect("create");

        // Process far-end signal with small echo (well-modelled)
        // First warm up with enough samples so the VAD activates
        for n in 0..200 {
            let far = (n as f64 * 0.1).sin();
            let mic = 0.5 * far; // simple echo
            let result = aec.process_sample(far, mic);
            // After warm-up, should see FarEndOnly
            if n > 50 {
                assert!(
                    !matches!(result.dtd_status, DoubleTalkStatus::NearEndOnly),
                    "Should not be NearEndOnly when far-end is active"
                );
            }
        }
    }

    #[test]
    fn test_erle_db_computation() {
        // Direct test of the helper
        let erle = compute_erle_db(1.0, 0.01);
        // 10 * log10(1.0 / 0.01) = 10 * 2 = 20 dB
        assert!((erle - 20.0).abs() < 0.01, "Expected ~20 dB, got {erle}");

        let erle_zero_err = compute_erle_db(1.0, 0.0);
        // Should be capped, not infinity
        assert!(erle_zero_err.is_finite());

        let erle_zero_des = compute_erle_db(0.0, 1.0);
        assert!((erle_zero_des - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_multi_partition_vs_single() {
        // Both should converge, multi-partition may converge differently
        // but both should achieve reasonable cancellation
        let echo_path = [0.0, 0.6, -0.2, 0.1];
        let (far, mic) = generate_echo_scenario(3000, &echo_path);

        let cfg_single = AECConfig {
            filter_length: 16,
            step_size: 0.5,
            num_subbands: 1,
            ..AECConfig::default()
        };
        let cfg_multi = AECConfig {
            filter_length: 16,
            step_size: 0.5,
            num_subbands: 4,
            ..AECConfig::default()
        };

        let mut aec_s = AcousticEchoCanceller::new(cfg_single).expect("create");
        let mut aec_m = AcousticEchoCanceller::new(cfg_multi).expect("create");

        let res_s = aec_s.process_block(&far, &mic).expect("single");
        let res_m = aec_m.process_block(&far, &mic).expect("multi");

        let rms_s: f64 = (res_s[2800..]
            .iter()
            .map(|r| r.output * r.output)
            .sum::<f64>()
            / 200.0)
            .sqrt();
        let rms_m: f64 = (res_m[2800..]
            .iter()
            .map(|r| r.output * r.output)
            .sum::<f64>()
            / 200.0)
            .sqrt();

        assert!(rms_s < 0.1, "Single partition RMS = {rms_s}");
        assert!(rms_m < 0.1, "Multi partition RMS = {rms_m}");
    }

    #[test]
    fn test_geigel_detector_standalone() {
        let mut det = GeigelDetector::new(8, 0.5, 4);

        // Far-end active, mic follows (echo only) — should NOT trigger
        for n in 0..50 {
            let x = (n as f64 * 0.1).sin();
            let d = 0.3 * x; // echo is weaker than far-end
            let dtd = det.detect(x, d);
            // Should generally not trigger since echo < threshold * max(far-end)
            if n > 10 {
                assert!(
                    !dtd,
                    "Geigel should not trigger for echo-only at sample {n}"
                );
            }
        }

        det.reset();

        // Far-end active, mic has strong near-end — should trigger
        for n in 0..50 {
            let x = (n as f64 * 0.1).sin();
            let d = 5.0 * (n as f64 * 0.3).cos(); // much louder near-end
            let dtd = det.detect(x, d);
            if n > 10 {
                assert!(dtd, "Geigel should trigger for loud near-end at sample {n}");
            }
        }
    }

    #[test]
    fn test_cross_correlation_detector() {
        use super::super::double_talk::CrossCorrelationDetector;

        let mut det = CrossCorrelationDetector::new(0.1, 0.5);

        // Correlated signals — no double-talk
        for n in 0..100 {
            let x = (n as f64 * 0.1).sin();
            let e = 0.01 * (n as f64 * 0.07).cos(); // small uncorrelated error
            let _dtd = det.detect(x, e);
        }

        det.reset();

        // Large error relative to far-end — double-talk
        let mut dt_count = 0;
        for n in 0..200 {
            let x = (n as f64 * 0.1).sin() * 0.1;
            let e = 5.0 * (n as f64 * 0.3).cos(); // large independent error
            if det.detect(x, e) {
                dt_count += 1;
            }
        }
        assert!(
            dt_count > 50,
            "Cross-correlation DTD should fire for large independent error, got {dt_count}"
        );
    }

    #[test]
    fn test_far_end_activity_detector() {
        let mut vad = FarEndActivityDetector::new(0.1, 1e-4);

        // Silence
        for _ in 0..50 {
            assert!(!vad.is_active(0.0));
        }

        // Active signal
        for n in 0..100 {
            let x = (n as f64 * 0.1).sin();
            vad.is_active(x);
        }
        assert!(vad.is_active(1.0));

        vad.reset();
        assert!(!vad.is_active(0.0));
    }

    #[test]
    fn test_empty_block() {
        let cfg = AECConfig::default();
        let mut aec = AcousticEchoCanceller::new(cfg).expect("create");
        let results = aec.process_block(&[], &[]).expect("empty");
        assert!(results.is_empty());
    }

    #[test]
    fn test_classify_talk_status() {
        assert_eq!(
            classify_talk_status(false, false, 0.0, 1e-6),
            DoubleTalkStatus::NoTalk
        );
        assert_eq!(
            classify_talk_status(true, false, 0.0, 1e-6),
            DoubleTalkStatus::FarEndOnly
        );
        assert_eq!(
            classify_talk_status(false, false, 1.0, 1e-6),
            DoubleTalkStatus::NearEndOnly
        );
        assert_eq!(
            classify_talk_status(true, true, 1.0, 1e-6),
            DoubleTalkStatus::DoubleTalk
        );
        assert_eq!(
            classify_talk_status(true, false, 1.0, 1e-6),
            DoubleTalkStatus::DoubleTalk
        );
    }
}
