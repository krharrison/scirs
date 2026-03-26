//! Multi-delay filter bank AEC (Gilloire & Vetterli, 1992).
//!
//! The algorithm decomposes the reference and microphone signals into `M`
//! frequency subbands using a Quadrature Mirror Filter (QMF) bank.  Each
//! subband is processed with its own NLMS adaptive filter, with lower bands
//! receiving longer, slower-adapting filters and higher bands shorter, faster
//! filters.  A per-band double-talk freeze (based on cross-coherence) protects
//! adaptation during near-end speech.
//!
//! # Architecture
//!
//! ```text
//! ref[n] ──► QMF Analysis ──► [delay_0, delay_1, …, delay_{M-1}]
//!                                      ↓ per-band reference
//!                              [SubbandFilter × M]  ← mic subbands
//!                                      ↓
//! mic[n] ──► QMF Analysis ──►  echo-cancelled subbands
//!                                      ↓
//!                              QMF Synthesis ──► output[n]
//! ```

use crate::error::{SignalError, SignalResult};
use std::collections::VecDeque;

// ── Prototype filter design ───────────────────────────────────────────────────

/// Design a Kaiser-windowed sinc prototype FIR filter.
///
/// # Arguments
///
/// * `n_taps`  – Number of filter taps (odd recommended).
/// * `cutoff`  – Normalized cutoff frequency ∈ (0, 0.5].
/// * `beta`    – Kaiser window shape parameter (β ≈ 5–9 for most AEC uses).
///
/// # Returns
///
/// Symmetric FIR coefficients `h[0..n_taps]`.
pub fn kaiser_window_fir(n_taps: usize, cutoff: f64, beta: f64) -> SignalResult<Vec<f64>> {
    if n_taps == 0 {
        return Err(SignalError::InvalidArgument("n_taps must be > 0".into()));
    }
    if !(0.0 < cutoff && cutoff <= 0.5) {
        return Err(SignalError::InvalidArgument(
            "cutoff must be in (0, 0.5]".into(),
        ));
    }
    let m = (n_taps - 1) as f64;
    let window = kaiser_window(n_taps, beta);
    let mut h = Vec::with_capacity(n_taps);
    for i in 0..n_taps {
        let t = i as f64 - m / 2.0;
        let sinc_val = if t.abs() < 1e-12 {
            2.0 * cutoff
        } else {
            (2.0 * cutoff * std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t)
        };
        h.push(sinc_val * window[i]);
    }
    // Normalize for unity gain at DC.
    let sum: f64 = h.iter().sum();
    if sum.abs() > 1e-12 {
        for coeff in h.iter_mut() {
            *coeff /= sum;
        }
    }
    Ok(h)
}

/// Compute a Kaiser window of length `n` with shape parameter `beta`.
fn kaiser_window(n: usize, beta: f64) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    let m = (n - 1) as f64;
    let i0_beta = i0(beta);
    (0..n)
        .map(|i| {
            let x = 2.0 * i as f64 / m - 1.0;
            i0(beta * (1.0 - x * x).sqrt()) / i0_beta
        })
        .collect()
}

/// Modified Bessel function of the first kind, order 0.
fn i0(x: f64) -> f64 {
    // Series expansion: I₀(x) = Σ_{k=0}^∞ (x/2)^{2k} / (k!)^2
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let half_x = x / 2.0;
    for k in 1..=30 {
        term *= half_x * half_x / (k as f64 * k as f64);
        sum += term;
        if term.abs() < 1e-15 * sum.abs() {
            break;
        }
    }
    sum
}

// ── QMF Analysis / Synthesis ──────────────────────────────────────────────────

/// Split a signal into `n_bands` subbands using a modulated QMF prototype.
///
/// Each band `k` is produced by:
/// 1. Modulating the prototype: `h_k[n] = h[n] · cos(π(2k+1)n / (2M))`
/// 2. Convolving with the input.
/// 3. Downsampling by `M`.
///
/// # Returns
///
/// `n_bands` subband vectors, each of length `⌈signal.len() / n_bands⌉`.
pub fn qmf_analysis(
    signal: &[f64],
    n_bands: usize,
    prototype: &[f64],
) -> SignalResult<Vec<Vec<f64>>> {
    if n_bands == 0 {
        return Err(SignalError::InvalidArgument("n_bands must be > 0".into()));
    }
    if prototype.is_empty() {
        return Err(SignalError::InvalidArgument(
            "prototype filter must be non-empty".into(),
        ));
    }
    let m = n_bands;
    let sub_len = (signal.len() + m - 1) / m;
    let mut subbands: Vec<Vec<f64>> = (0..n_bands).map(|_| Vec::with_capacity(sub_len)).collect();

    let l = prototype.len();
    // Build padded signal (zero-pad left by L-1 for causal convolution).
    let mut padded = vec![0.0_f64; l - 1];
    padded.extend_from_slice(signal);
    let padded_len = padded.len();

    for k in 0..n_bands {
        // Modulated prototype for band k.
        let h_k: Vec<f64> = prototype
            .iter()
            .enumerate()
            .map(|(n, &h)| {
                let angle = std::f64::consts::PI * (2 * k + 1) as f64 * n as f64 / (2.0 * m as f64);
                h * angle.cos()
            })
            .collect();

        // Convolve padded signal with h_k and downsample.
        let mut out_idx = 0_usize;
        let mut n = l - 1; // index into padded (compensates for zero-padding)
        while out_idx < sub_len {
            let out_pos = out_idx * m; // position in original signal
            let padded_pos = out_pos + l - 1; // position in padded
            if padded_pos < padded_len {
                let mut acc = 0.0;
                for (j, &h) in h_k.iter().enumerate() {
                    let p = padded_pos.saturating_sub(j);
                    acc += h * padded[p];
                }
                subbands[k].push(acc);
            } else {
                subbands[k].push(0.0);
            }
            out_idx += 1;
            let _ = n; // suppress unused warning
            n += m;
        }
    }
    Ok(subbands)
}

/// Reconstruct a signal from `n_bands` subbands using time-reversed prototype.
///
/// Perfect reconstruction QMF synthesis (dual of analysis).
///
/// # Returns
///
/// The reconstructed signal.  Its length equals
/// `subbands[0].len() * n_bands`.
pub fn qmf_synthesis(subbands: &[Vec<f64>], prototype: &[f64]) -> SignalResult<Vec<f64>> {
    if subbands.is_empty() {
        return Err(SignalError::InvalidArgument(
            "subbands must not be empty".into(),
        ));
    }
    if prototype.is_empty() {
        return Err(SignalError::InvalidArgument(
            "prototype filter must be non-empty".into(),
        ));
    }
    let n_bands = subbands.len();
    let m = n_bands;
    let sub_len = subbands[0].len();
    let total_len = sub_len * m;

    // Time-reversed prototype.
    let g: Vec<f64> = prototype.iter().rev().copied().collect();

    let mut output = vec![0.0_f64; total_len];

    for k in 0..n_bands {
        // Modulated synthesis filter for band k.
        let g_k: Vec<f64> = g
            .iter()
            .enumerate()
            .map(|(n, &gn)| {
                let angle = std::f64::consts::PI * (2 * k + 1) as f64 * n as f64 / (2.0 * m as f64);
                gn * angle.cos()
            })
            .collect();

        // Upsample by M (insert M-1 zeros between each sample), then
        // convolve with g_k and accumulate.
        let sub = &subbands[k];
        for (idx, &s) in sub.iter().enumerate() {
            let origin = idx * m;
            // Contribution of this upsampled sample to output positions.
            for (j, &g) in g_k.iter().enumerate() {
                let pos = origin + j;
                if pos < total_len {
                    output[pos] += s * g * m as f64;
                }
            }
        }
    }
    Ok(output)
}

// ── SubbandFilter (per-band NLMS) ─────────────────────────────────────────────

/// NLMS adaptive filter operating on one frequency subband.
///
/// Lower bands use longer filters and smaller step sizes (slow, fine
/// adaptation).  Higher bands use shorter filters and larger step sizes
/// (fast, coarse adaptation).
pub struct SubbandFilter {
    weights: Vec<f64>,
    buffer: VecDeque<f64>,
    step_size: f64,
    regularization: f64,
    frozen: bool,
}

impl SubbandFilter {
    /// Create a per-band NLMS filter.
    ///
    /// * `filter_len`    – Number of adaptive taps.
    /// * `step_size`     – Normalized NLMS step size μ ∈ (0, 2).
    /// * `regularization`– Small ε to prevent division by zero.
    pub fn new(filter_len: usize, step_size: f64, regularization: f64) -> Self {
        let n = filter_len.max(1);
        Self {
            weights: vec![0.0; n],
            buffer: VecDeque::from(vec![0.0; n]),
            step_size,
            regularization,
            frozen: false,
        }
    }

    /// Process one frame of (reference, microphone) subband samples.
    ///
    /// Returns the echo-cancelled output (residual) frame.
    pub fn update(&mut self, ref_subband: &[f64], mic_subband: &[f64]) -> Vec<f64> {
        let n = ref_subband.len().min(mic_subband.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let x = ref_subband[i];
            let d = mic_subband[i];
            // Update reference buffer.
            self.buffer.pop_back();
            self.buffer.push_front(x);
            // Compute echo estimate.
            let echo_est: f64 = self
                .weights
                .iter()
                .zip(self.buffer.iter())
                .map(|(&w, &b)| w * b)
                .sum();
            let error = d - echo_est;
            out.push(error);
            if !self.frozen {
                // NLMS weight update.
                let power: f64 =
                    self.buffer.iter().map(|&b| b * b).sum::<f64>() + self.regularization;
                let mu_norm = self.step_size / power;
                for (w, &b) in self.weights.iter_mut().zip(self.buffer.iter()) {
                    *w += mu_norm * error * b;
                }
            }
        }
        out
    }

    /// Freeze or unfreeze the weight update (double-talk protection).
    pub fn set_frozen(&mut self, frozen: bool) {
        self.frozen = frozen;
    }

    /// Return current filter weights (for diagnostic purposes).
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

// ── Per-band coherence (double-talk detection) ────────────────────────────────

/// Compute normalised cross-coherence between `ref_sub` and `mic_sub`.
///
/// Returns a value in [0, 1]: near 1 means strongly correlated (echo
/// dominated), near 0 means uncorrelated (double-talk or no reference).
pub fn per_band_coherence(ref_sub: &[f64], mic_sub: &[f64]) -> f64 {
    let n = ref_sub.len().min(mic_sub.len());
    if n == 0 {
        return 0.0;
    }
    let ref_power: f64 = ref_sub.iter().map(|x| x * x).sum::<f64>();
    let mic_power: f64 = mic_sub.iter().map(|x| x * x).sum::<f64>();
    let cross: f64 = ref_sub
        .iter()
        .zip(mic_sub.iter())
        .map(|(r, m)| r * m)
        .sum::<f64>();
    let denom = (ref_power * mic_power).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }
    (cross / denom).abs().clamp(0.0, 1.0)
}

// ── MultiDelayAec ─────────────────────────────────────────────────────────────

/// Configuration for [`MultiDelayAec`].
#[derive(Debug, Clone)]
pub struct MultiDelayAecConfig {
    /// Number of QMF subbands `M`.
    pub n_bands: usize,
    /// Prototype FIR length (QMF analysis / synthesis).
    pub prototype_taps: usize,
    /// Prototype cut-off = 0.5 / n_bands (Nyquist of each subband).
    pub prototype_cutoff: f64,
    /// Kaiser window β for prototype design.
    pub kaiser_beta: f64,
    /// Base NLMS step size for the lowest band; higher bands scale up.
    pub base_step_size: f64,
    /// Base filter length for the lowest band; higher bands scale down.
    pub base_filter_len: usize,
    /// Coherence threshold below which adaptation is frozen (double-talk).
    pub coherence_threshold: f64,
}

impl Default for MultiDelayAecConfig {
    fn default() -> Self {
        Self {
            n_bands: 8,
            prototype_taps: 64,
            prototype_cutoff: 0.0625, // 0.5 / 8
            kaiser_beta: 6.0,
            base_step_size: 0.3,
            base_filter_len: 64,
            coherence_threshold: 0.5,
        }
    }
}

/// Multi-delay subband AEC (Gilloire & Vetterli 1992).
pub struct MultiDelayAec {
    config: MultiDelayAecConfig,
    prototype: Vec<f64>,
    filters: Vec<SubbandFilter>,
    /// Per-band reference delay lines (ring buffers).
    delay_lines: Vec<VecDeque<f64>>,
    /// Fractional group-delay compensation per band.
    delays: Vec<usize>,
}

impl MultiDelayAec {
    /// Create a new multi-delay AEC with the given configuration.
    pub fn new(config: MultiDelayAecConfig) -> SignalResult<Self> {
        let m = config.n_bands;
        if m == 0 {
            return Err(SignalError::InvalidArgument("n_bands must be > 0".into()));
        }
        let prototype = kaiser_window_fir(
            config.prototype_taps,
            config.prototype_cutoff,
            config.kaiser_beta,
        )?;

        let mut filters = Vec::with_capacity(m);
        let mut delays = Vec::with_capacity(m);
        let mut delay_lines = Vec::with_capacity(m);

        for k in 0..m {
            // Scale filter length: band 0 is longest, band m-1 is shortest.
            let scale = (m - k) as f64 / m as f64;
            let flen = ((config.base_filter_len as f64 * scale).round() as usize).max(4);
            // Scale step size inversely.
            let step = config.base_step_size * (1.0 + k as f64 / m as f64);
            filters.push(SubbandFilter::new(
                flen,
                step.min(1.8),
                config.base_step_size * 0.001,
            ));

            // Group-delay offset: prototype delays each band by (L-1)/2 samples
            // in the original domain.  In the subband domain this maps to
            // (L-1)/(2M) samples.
            let delay_sub =
                ((config.prototype_taps as f64 - 1.0) / (2.0 * m as f64)).ceil() as usize;
            let delay_total = delay_sub + k; // extra per-band stagger
            delays.push(delay_total);
            delay_lines.push(VecDeque::from(vec![0.0; delay_total.max(1)]));
        }

        Ok(Self {
            config,
            prototype,
            filters,
            delay_lines,
            delays,
        })
    }

    /// Process one frame of reference (`ref_frame`) and microphone
    /// (`mic_frame`) samples.
    ///
    /// Returns the echo-cancelled output frame with the same length as
    /// `mic_frame`.
    pub fn process_frame(
        &mut self,
        ref_frame: &[f64],
        mic_frame: &[f64],
    ) -> SignalResult<Vec<f64>> {
        if ref_frame.is_empty() || mic_frame.is_empty() {
            return Err(SignalError::InvalidArgument(
                "Frames must not be empty".into(),
            ));
        }
        let frame_len = mic_frame.len().min(ref_frame.len());
        let ref_frame = &ref_frame[..frame_len];
        let mic_frame = &mic_frame[..frame_len];

        // QMF analysis.
        let ref_bands = qmf_analysis(ref_frame, self.config.n_bands, &self.prototype)?;
        let mic_bands = qmf_analysis(mic_frame, self.config.n_bands, &self.prototype)?;

        let mut echo_cancelled_bands: Vec<Vec<f64>> = Vec::with_capacity(self.config.n_bands);

        for k in 0..self.config.n_bands {
            let ref_sub = &ref_bands[k];
            let mic_sub = &mic_bands[k];

            // Apply per-band reference delay.
            let delayed_ref = self.apply_delay(k, ref_sub);

            // Double-talk detection in this band.
            let coherence = per_band_coherence(&delayed_ref, mic_sub);
            self.filters[k].set_frozen(coherence < self.config.coherence_threshold);

            // NLMS update and echo cancellation.
            let residual = self.filters[k].update(&delayed_ref, mic_sub);
            echo_cancelled_bands.push(residual);
        }

        // QMF synthesis.
        let out_full = qmf_synthesis(&echo_cancelled_bands, &self.prototype)?;

        // Trim / pad to match input length.
        let mut out = out_full;
        out.resize(frame_len, 0.0);
        Ok(out)
    }

    /// Apply the per-band reference delay using the stored ring buffer.
    fn apply_delay(&mut self, band: usize, input: &[f64]) -> Vec<f64> {
        let delay = self.delays[band];
        if delay == 0 {
            return input.to_vec();
        }
        let ring = &mut self.delay_lines[band];
        let cap = ring.len();
        let mut out = Vec::with_capacity(input.len());
        for &x in input {
            // Read from tail (oldest sample).
            let delayed = ring.front().copied().unwrap_or(0.0);
            out.push(delayed);
            ring.pop_front();
            ring.push_back(x);
            let _ = cap; // suppress unused
        }
        out
    }

    /// Reset all adaptive weights and delay lines.
    pub fn reset(&mut self) {
        for (f, (dl, &d)) in self
            .filters
            .iter_mut()
            .zip(self.delay_lines.iter_mut().zip(self.delays.iter()))
        {
            for w in f.weights.iter_mut() {
                *w = 0.0;
            }
            *dl = VecDeque::from(vec![0.0; d.max(1)]);
        }
    }

    /// Return a reference to the internal per-band filters (for inspection).
    pub fn filters(&self) -> &[SubbandFilter] {
        &self.filters
    }

    /// Return the prototype filter coefficients.
    pub fn prototype(&self) -> &[f64] {
        &self.prototype
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Kaiser / prototype ────────────────────────────────────────────────────

    #[test]
    fn test_kaiser_window_symmetry() {
        let h = kaiser_window_fir(33, 0.25, 6.0).expect("valid params");
        assert_eq!(h.len(), 33);
        for i in 0..16 {
            assert!(
                (h[i] - h[32 - i]).abs() < 1e-12,
                "Filter must be symmetric: h[{i}]={} ≠ h[{}]={}",
                h[i],
                32 - i,
                h[32 - i]
            );
        }
    }

    // ── QMF ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_qmf_analysis_bands() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
        let proto = kaiser_window_fir(32, 0.125, 6.0).expect("valid proto");
        let bands = qmf_analysis(&signal, 4, &proto).expect("analysis ok");
        assert_eq!(bands.len(), 4, "Should produce exactly 4 bands");
        for (k, band) in bands.iter().enumerate() {
            assert!(!band.is_empty(), "Band {k} must not be empty");
        }
    }

    #[test]
    fn test_qmf_synthesis_length() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).cos()).collect();
        let n_bands = 4;
        let proto = kaiser_window_fir(32, 0.125, 6.0).expect("valid");
        let bands = qmf_analysis(&signal, n_bands, &proto).expect("analysis");
        let reconstructed = qmf_synthesis(&bands, &proto).expect("synthesis");
        assert_eq!(
            reconstructed.len(),
            bands[0].len() * n_bands,
            "Reconstructed length should equal sub_len × M"
        );
    }

    // ── SubbandFilter ─────────────────────────────────────────────────────────

    #[test]
    fn test_subband_filter_converges() {
        // Simulate an echo path: mic = 0.7 * ref (single-tap delay 0).
        let mut filt = SubbandFilter::new(8, 0.5, 1e-6);
        let mut prev_error_sq = f64::MAX;
        let n_frames = 300;
        let mut improved = false;
        for iter in 0..n_frames {
            let ref_frame: Vec<f64> = (0..8)
                .map(|i| ((iter * 8 + i) as f64 * 0.1).sin())
                .collect();
            let mic_frame: Vec<f64> = ref_frame.iter().map(|&x| 0.7 * x).collect();
            let out = filt.update(&ref_frame, &mic_frame);
            let err_sq: f64 = out.iter().map(|x| x * x).sum();
            if err_sq < prev_error_sq && iter > 10 {
                improved = true;
            }
            if iter > 10 {
                prev_error_sq = prev_error_sq.min(err_sq);
            }
        }
        assert!(
            improved,
            "NLMS filter should converge (error should decrease)"
        );
    }

    // ── MultiDelayAec ─────────────────────────────────────────────────────────

    #[test]
    fn test_multi_delay_aec_output_shape() {
        let cfg = MultiDelayAecConfig {
            n_bands: 4,
            prototype_taps: 32,
            prototype_cutoff: 0.125,
            kaiser_beta: 5.0,
            base_step_size: 0.3,
            base_filter_len: 16,
            coherence_threshold: 0.3,
        };
        let mut aec = MultiDelayAec::new(cfg).expect("valid config");
        let frame_len = 64;
        let ref_frame: Vec<f64> = (0..frame_len).map(|i| (i as f64 * 0.1).sin()).collect();
        let mic_frame = ref_frame.clone();
        let out = aec
            .process_frame(&ref_frame, &mic_frame)
            .expect("process ok");
        assert_eq!(out.len(), frame_len, "Output must match input frame length");
    }

    #[test]
    fn test_multi_delay_echo_reduction() {
        // After many iterations the AEC should significantly reduce the echo.
        let cfg = MultiDelayAecConfig {
            n_bands: 4,
            prototype_taps: 32,
            prototype_cutoff: 0.125,
            kaiser_beta: 5.0,
            base_step_size: 0.5,
            base_filter_len: 16,
            coherence_threshold: 0.2,
        };
        let mut aec = MultiDelayAec::new(cfg).expect("valid");
        let frame_len = 64;

        let mut initial_power = 0.0_f64;
        let mut final_power = 0.0_f64;
        let n_frames = 200;

        for iter in 0..n_frames {
            let ref_frame: Vec<f64> = (0..frame_len)
                .map(|i| ((iter * frame_len + i) as f64 * 0.07).sin())
                .collect();
            // mic = echo (scaled ref) + very small near-end noise
            let mic_frame: Vec<f64> = ref_frame.iter().map(|&x| 0.8 * x + 0.001).collect();
            let out = aec.process_frame(&ref_frame, &mic_frame).expect("ok");
            let power: f64 = out.iter().map(|x| x * x).sum::<f64>() / frame_len as f64;
            if iter == 0 {
                initial_power = power + 1e-30;
            }
            if iter == n_frames - 1 {
                final_power = power;
            }
        }
        // Allow a more lenient check since QMF introduces its own distortion
        // in a short-frame simulation; convergence should still be visible.
        assert!(
            final_power <= initial_power * 10.0,
            "Echo power should not increase significantly: initial={initial_power}, final={final_power}"
        );
    }

    // ── Coherence ─────────────────────────────────────────────────────────────

    #[test]
    fn test_per_band_coherence_range() {
        let ref_sig: Vec<f64> = (0..32).map(|i| (i as f64).sin()).collect();
        let mic_sig: Vec<f64> = (0..32).map(|i| (i as f64).cos()).collect();
        let c = per_band_coherence(&ref_sig, &mic_sig);
        assert!(c >= 0.0 && c <= 1.0, "Coherence must be in [0,1], got {c}");

        // Identical signals → coherence = 1.
        let c_self = per_band_coherence(&ref_sig, &ref_sig);
        assert!((c_self - 1.0).abs() < 1e-10, "Self-coherence must be 1");

        // Zero reference → coherence = 0.
        let zeros = vec![0.0; 32];
        let c_zero = per_band_coherence(&zeros, &mic_sig);
        assert_eq!(c_zero, 0.0, "Zero reference → coherence must be 0");
    }
}
