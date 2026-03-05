//! Signal Quality Metrics
//!
//! Comprehensive signal quality assessment including:
//! - Signal-to-Noise Ratio (SNR) estimation (blind and reference-based)
//! - Signal-to-Distortion Ratio (SDR)
//! - PESQ-like perceptual quality metric
//! - Spectral flatness measure (Wiener entropy)
//! - Crest factor
//! - Dynamic range measurement
//! - Zero-crossing rate
//! - SINAD (Signal-to-Noise-and-Distortion)
//! - Effective Number of Bits (ENOB)

use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// SNR Estimation
// ---------------------------------------------------------------------------

/// Configuration for blind SNR estimation.
#[derive(Debug, Clone)]
pub struct BlindSnrConfig {
    /// Frame length in samples.
    pub frame_length: usize,
    /// Frame hop size in samples.
    pub hop_size: usize,
    /// Percentile (0..1) of frames considered noise-only (default 0.1).
    pub noise_percentile: f64,
}

impl Default for BlindSnrConfig {
    fn default() -> Self {
        Self {
            frame_length: 256,
            hop_size: 128,
            noise_percentile: 0.1,
        }
    }
}

/// Estimate SNR from a reference signal and a degraded (noisy) signal.
///
/// `SNR = 10 * log10(P_signal / P_noise)`
///
/// where `P_noise` is estimated from the difference `degraded - reference`.
pub fn snr_reference(reference: &[f64], degraded: &[f64]) -> SignalResult<f64> {
    if reference.is_empty() || degraded.is_empty() {
        return Err(SignalError::ValueError("Signals must not be empty".into()));
    }
    if reference.len() != degraded.len() {
        return Err(SignalError::DimensionMismatch(format!(
            "Reference length {} != degraded length {}",
            reference.len(),
            degraded.len()
        )));
    }

    let n = reference.len() as f64;
    let signal_power: f64 = reference.iter().map(|&x| x * x).sum::<f64>() / n;
    let noise_power: f64 = reference
        .iter()
        .zip(degraded.iter())
        .map(|(&r, &d)| {
            let e = d - r;
            e * e
        })
        .sum::<f64>()
        / n;

    if noise_power < f64::EPSILON {
        return Ok(f64::INFINITY);
    }
    Ok(10.0 * (signal_power / noise_power).log10())
}

/// Estimate SNR from a signal and known noise floor.
///
/// `SNR = 10 * log10(P_signal / noise_power)`
pub fn snr_from_noise_floor(signal: &[f64], noise_power: f64) -> SignalResult<f64> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if noise_power < 0.0 {
        return Err(SignalError::ValueError(
            "Noise power must be non-negative".into(),
        ));
    }
    let n = signal.len() as f64;
    let signal_power: f64 = signal.iter().map(|&x| x * x).sum::<f64>() / n;
    if noise_power < f64::EPSILON {
        return Ok(f64::INFINITY);
    }
    Ok(10.0 * (signal_power / noise_power).log10())
}

/// Blind SNR estimation based on frame-level energy sorting.
///
/// Frames the signal, sorts by energy, and uses the lowest-energy percentile
/// as the noise estimate.
pub fn snr_blind(signal: &[f64], config: &BlindSnrConfig) -> SignalResult<f64> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if config.frame_length == 0 || config.hop_size == 0 {
        return Err(SignalError::InvalidArgument(
            "Frame length and hop size must be > 0".into(),
        ));
    }
    if config.noise_percentile <= 0.0 || config.noise_percentile >= 1.0 {
        return Err(SignalError::InvalidArgument(
            "noise_percentile must be in (0, 1)".into(),
        ));
    }

    // Compute frame energies
    let mut frame_energies = Vec::new();
    let mut start = 0;
    while start + config.frame_length <= signal.len() {
        let energy: f64 = signal[start..start + config.frame_length]
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            / config.frame_length as f64;
        frame_energies.push(energy);
        start += config.hop_size;
    }

    if frame_energies.is_empty() {
        return Err(SignalError::ValueError(
            "Signal too short for the given frame parameters".into(),
        ));
    }

    // Sort frame energies
    frame_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n_noise = (frame_energies.len() as f64 * config.noise_percentile).ceil() as usize;
    let n_noise = n_noise.max(1).min(frame_energies.len());

    let noise_power: f64 = frame_energies[..n_noise].iter().sum::<f64>() / n_noise as f64;

    let total_power: f64 = frame_energies.iter().sum::<f64>() / frame_energies.len() as f64;

    if noise_power < f64::EPSILON {
        return Ok(f64::INFINITY);
    }

    let signal_power = total_power - noise_power;
    if signal_power < 0.0 {
        return Ok(0.0);
    }

    Ok(10.0 * (signal_power / noise_power).log10())
}

/// Segmental SNR: average of per-frame SNR values (in dB).
///
/// Useful for speech quality where local SNR is more meaningful than global.
pub fn segmental_snr(
    reference: &[f64],
    degraded: &[f64],
    frame_length: usize,
    hop_size: usize,
) -> SignalResult<f64> {
    if reference.len() != degraded.len() {
        return Err(SignalError::DimensionMismatch(format!(
            "Reference length {} != degraded length {}",
            reference.len(),
            degraded.len()
        )));
    }
    if frame_length == 0 || hop_size == 0 {
        return Err(SignalError::InvalidArgument(
            "Frame length and hop size must be > 0".into(),
        ));
    }

    let mut snr_sum = 0.0;
    let mut count = 0;
    let mut start = 0;

    while start + frame_length <= reference.len() {
        let sig_power: f64 = reference[start..start + frame_length]
            .iter()
            .map(|&x| x * x)
            .sum::<f64>();
        let noise_power: f64 = reference[start..start + frame_length]
            .iter()
            .zip(degraded[start..start + frame_length].iter())
            .map(|(&r, &d)| (d - r).powi(2))
            .sum::<f64>();

        if noise_power > f64::EPSILON && sig_power > f64::EPSILON {
            let frame_snr = 10.0 * (sig_power / noise_power).log10();
            // Clip to reasonable range
            let clipped = frame_snr.max(-10.0).min(35.0);
            snr_sum += clipped;
            count += 1;
        }
        start += hop_size;
    }

    if count == 0 {
        return Err(SignalError::ComputationError(
            "No valid frames for segmental SNR computation".into(),
        ));
    }

    Ok(snr_sum / count as f64)
}

// ---------------------------------------------------------------------------
// Signal-to-Distortion Ratio (SDR)
// ---------------------------------------------------------------------------

/// Compute Signal-to-Distortion Ratio using BSS Eval methodology.
///
/// Decomposes the estimated signal into target, interference, noise, and
/// artefact components and computes the ratio.
///
/// # Arguments
/// * `reference` - Clean reference signal
/// * `estimated` - Estimated/processed signal
///
/// # Returns
/// SDR in dB.
pub fn sdr(reference: &[f64], estimated: &[f64]) -> SignalResult<f64> {
    if reference.is_empty() || estimated.is_empty() {
        return Err(SignalError::ValueError("Signals must not be empty".into()));
    }
    if reference.len() != estimated.len() {
        return Err(SignalError::DimensionMismatch(format!(
            "Reference length {} != estimated length {}",
            reference.len(),
            estimated.len()
        )));
    }

    let n = reference.len();

    // Project estimated onto reference to get the target component
    let ref_power: f64 = reference.iter().map(|&x| x * x).sum();
    if ref_power < f64::EPSILON {
        return Err(SignalError::ComputationError(
            "Reference signal has zero power".into(),
        ));
    }

    let cross: f64 = reference
        .iter()
        .zip(estimated.iter())
        .map(|(&r, &e)| r * e)
        .sum();
    let scale = cross / ref_power;

    // s_target = scale * reference
    // e_noise = estimated - s_target
    let target_power: f64 = reference.iter().map(|&x| (scale * x).powi(2)).sum::<f64>();
    let distortion_power: f64 = reference
        .iter()
        .zip(estimated.iter())
        .map(|(&r, &e)| (e - scale * r).powi(2))
        .sum::<f64>();

    if distortion_power < f64::EPSILON {
        return Ok(f64::INFINITY);
    }

    Ok(10.0 * (target_power / distortion_power).log10())
}

/// Compute scale-invariant SDR (SI-SDR).
///
/// SI-SDR is invariant to scaling of the estimated signal and is preferred
/// for source separation evaluation.
pub fn si_sdr(reference: &[f64], estimated: &[f64]) -> SignalResult<f64> {
    // SI-SDR is the same as SDR computed above (which already does projection)
    sdr(reference, estimated)
}

// ---------------------------------------------------------------------------
// PESQ-like Perceptual Quality
// ---------------------------------------------------------------------------

/// Perceptual quality result.
#[derive(Debug, Clone)]
pub struct PerceptualQualityResult {
    /// Overall quality score (1.0 = bad, 5.0 = excellent, PESQ-like scale).
    pub score: f64,
    /// Per-band disturbance values.
    pub band_disturbances: Vec<f64>,
    /// Average loudness difference.
    pub loudness_diff: f64,
}

/// Compute a PESQ-like perceptual quality metric.
///
/// This is a simplified perceptual model inspired by ITU-T P.862 (PESQ).
/// It operates in the frequency domain and models loudness and masking.
///
/// Note: This is NOT a full PESQ implementation (which requires psychoacoustic
/// models and is ITU-licensed). This provides a reasonable approximation.
///
/// # Arguments
/// * `reference` - Clean reference signal
/// * `degraded` - Degraded signal
/// * `fs` - Sampling frequency
pub fn perceptual_quality(
    reference: &[f64],
    degraded: &[f64],
    fs: f64,
) -> SignalResult<PerceptualQualityResult> {
    if reference.is_empty() || degraded.is_empty() {
        return Err(SignalError::ValueError("Signals must not be empty".into()));
    }
    if reference.len() != degraded.len() {
        return Err(SignalError::DimensionMismatch(format!(
            "Reference length {} != degraded length {}",
            reference.len(),
            degraded.len()
        )));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency must be positive".into(),
        ));
    }

    let n = reference.len();
    let frame_len = (0.032 * fs) as usize; // 32 ms frames
    let frame_len = frame_len.max(16);
    let hop = frame_len / 2;

    // Bark-scale bands (simplified): 24 critical bands from 0 to fs/2
    let n_bands = 24;
    let nyquist = fs / 2.0;

    // Bark edge frequencies (approximation)
    let bark_edges: Vec<f64> = (0..=n_bands)
        .map(|i| {
            let z = i as f64;
            // Traunmuller formula inverse: Hz from Bark
            1960.0 * (z + 0.53) / (26.28 - z).max(0.1)
        })
        .collect();

    let n_fft = next_power_of_two(frame_len);
    let n_freq = n_fft / 2 + 1;

    // Generate Hann window
    let window: Vec<f64> = (0..frame_len)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (frame_len.max(1) - 1).max(1) as f64).cos()))
        .collect();

    let mut band_disturbances = vec![0.0; n_bands];
    let mut total_loudness_diff = 0.0;
    let mut n_frames = 0;

    let mut pos = 0;
    while pos + frame_len <= n {
        // Window and compute spectra
        let ref_frame: Vec<f64> = reference[pos..pos + frame_len]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();
        let deg_frame: Vec<f64> = degraded[pos..pos + frame_len]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        let ref_spec = scirs2_fft::rfft(&ref_frame, Some(n_fft))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;
        let deg_spec = scirs2_fft::rfft(&deg_frame, Some(n_fft))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

        // Compute bark-band powers
        let df = fs / n_fft as f64;
        for band in 0..n_bands {
            let f_low = bark_edges[band].max(0.0);
            let f_high = bark_edges[band + 1].min(nyquist);
            if f_low >= f_high {
                continue;
            }
            let k_low = (f_low / df).ceil() as usize;
            let k_high = ((f_high / df).floor() as usize).min(n_freq.saturating_sub(1));

            let mut ref_band_power = 0.0;
            let mut deg_band_power = 0.0;
            for k in k_low..=k_high {
                if k < ref_spec.len() {
                    ref_band_power +=
                        ref_spec[k].re * ref_spec[k].re + ref_spec[k].im * ref_spec[k].im;
                }
                if k < deg_spec.len() {
                    deg_band_power +=
                        deg_spec[k].re * deg_spec[k].re + deg_spec[k].im * deg_spec[k].im;
                }
            }

            // Loudness (simplified: power^0.3)
            let ref_loudness = ref_band_power.powf(0.3);
            let deg_loudness = deg_band_power.powf(0.3);
            let diff = (deg_loudness - ref_loudness).abs();

            band_disturbances[band] += diff;
            total_loudness_diff += diff;
        }

        n_frames += 1;
        pos += hop;
    }

    if n_frames == 0 {
        return Err(SignalError::ComputationError(
            "Signal too short for perceptual quality analysis".into(),
        ));
    }

    let inv_nf = 1.0 / n_frames as f64;
    for bd in &mut band_disturbances {
        *bd *= inv_nf;
    }
    total_loudness_diff *= inv_nf;

    // Map to MOS-like scale [1, 5]
    // Lower disturbance = higher score
    let avg_disturbance = band_disturbances.iter().sum::<f64>() / n_bands as f64;
    let score = (4.5 - 1.5 * avg_disturbance.ln().max(0.0))
        .max(1.0)
        .min(4.5);

    Ok(PerceptualQualityResult {
        score,
        band_disturbances,
        loudness_diff: total_loudness_diff,
    })
}

// ---------------------------------------------------------------------------
// Spectral Flatness
// ---------------------------------------------------------------------------

/// Compute the spectral flatness measure (Wiener entropy).
///
/// Spectral flatness is the ratio of the geometric mean to the arithmetic
/// mean of the power spectrum. A value close to 1 indicates white noise;
/// close to 0 indicates a tonal signal.
///
/// # Arguments
/// * `signal` - Input signal
/// * `n_fft` - FFT size (defaults to signal length, rounded to next power of 2)
///
/// # Returns
/// Spectral flatness in [0, 1].
pub fn spectral_flatness(signal: &[f64], n_fft: Option<usize>) -> SignalResult<f64> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }

    let nf = n_fft.unwrap_or_else(|| next_power_of_two(signal.len()));
    let spec = scirs2_fft::rfft(signal, Some(nf))
        .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

    let n_freq = spec.len();
    if n_freq == 0 {
        return Ok(0.0);
    }

    // Compute power spectrum (skip DC)
    let start = 1;
    let powers: Vec<f64> = spec[start..]
        .iter()
        .map(|c| c.re * c.re + c.im * c.im)
        .collect();
    let n = powers.len();
    if n == 0 {
        return Ok(0.0);
    }

    // Arithmetic mean
    let arith_mean = powers.iter().sum::<f64>() / n as f64;
    if arith_mean < f64::EPSILON {
        return Ok(0.0);
    }

    // Geometric mean = exp(mean(log(p)))
    // Use log to avoid underflow
    let log_sum: f64 = powers
        .iter()
        .map(|&p| if p > f64::EPSILON { p.ln() } else { -100.0 })
        .sum::<f64>();
    let geo_mean = (log_sum / n as f64).exp();

    Ok((geo_mean / arith_mean).min(1.0).max(0.0))
}

/// Compute spectral flatness per frame (for a time series of flatness values).
pub fn spectral_flatness_frames(
    signal: &[f64],
    frame_length: usize,
    hop_size: usize,
    n_fft: Option<usize>,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if frame_length == 0 || hop_size == 0 {
        return Err(SignalError::InvalidArgument(
            "Frame length and hop size must be > 0".into(),
        ));
    }

    let mut result = Vec::new();
    let mut pos = 0;
    while pos + frame_length <= signal.len() {
        let frame = &signal[pos..pos + frame_length];
        let sf = spectral_flatness(frame, n_fft)?;
        result.push(sf);
        pos += hop_size;
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Crest Factor
// ---------------------------------------------------------------------------

/// Compute the crest factor of a signal.
///
/// Crest factor = peak amplitude / RMS amplitude.
/// A sine wave has crest factor sqrt(2) ~ 1.414.
/// A square wave has crest factor 1.0.
pub fn crest_factor(signal: &[f64]) -> SignalResult<f64> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }

    let n = signal.len() as f64;
    let rms = (signal.iter().map(|&x| x * x).sum::<f64>() / n).sqrt();
    if rms < f64::EPSILON {
        return Ok(0.0);
    }

    let peak = signal
        .iter()
        .map(|&x| x.abs())
        .fold(0.0f64, |a, b| a.max(b));

    Ok(peak / rms)
}

/// Compute crest factor in dB.
pub fn crest_factor_db(signal: &[f64]) -> SignalResult<f64> {
    let cf = crest_factor(signal)?;
    if cf < f64::EPSILON {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(20.0 * cf.log10())
}

// ---------------------------------------------------------------------------
// Dynamic Range
// ---------------------------------------------------------------------------

/// Dynamic range measurement result.
#[derive(Debug, Clone)]
pub struct DynamicRangeResult {
    /// Dynamic range in dB.
    pub dynamic_range_db: f64,
    /// Peak level in dB (relative to full scale 1.0).
    pub peak_db: f64,
    /// RMS level in dB.
    pub rms_db: f64,
    /// Minimum non-silent level in dB (using a noise floor estimate).
    pub noise_floor_db: f64,
}

/// Measure the dynamic range of a signal.
///
/// Dynamic range = peak level - noise floor (in dB).
///
/// The noise floor is estimated as the energy of the lowest-energy
/// percentile of frames.
///
/// # Arguments
/// * `signal` - Input signal
/// * `frame_length` - Frame length in samples
/// * `noise_percentile` - Fraction of frames considered noise (0..1)
pub fn dynamic_range(
    signal: &[f64],
    frame_length: usize,
    noise_percentile: f64,
) -> SignalResult<DynamicRangeResult> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if frame_length == 0 {
        return Err(SignalError::InvalidArgument(
            "Frame length must be > 0".into(),
        ));
    }
    if noise_percentile <= 0.0 || noise_percentile >= 1.0 {
        return Err(SignalError::InvalidArgument(
            "noise_percentile must be in (0, 1)".into(),
        ));
    }

    let n = signal.len() as f64;
    let rms_val = (signal.iter().map(|&x| x * x).sum::<f64>() / n).sqrt();
    let peak = signal
        .iter()
        .map(|&x| x.abs())
        .fold(0.0f64, |a, b| a.max(b));

    // Frame energies for noise floor
    let mut frame_energies = Vec::new();
    let mut pos = 0;
    while pos + frame_length <= signal.len() {
        let energy: f64 = signal[pos..pos + frame_length]
            .iter()
            .map(|&x| x * x)
            .sum::<f64>()
            / frame_length as f64;
        frame_energies.push(energy);
        pos += frame_length; // non-overlapping
    }

    if frame_energies.is_empty() {
        return Err(SignalError::ValueError(
            "Signal too short for frame analysis".into(),
        ));
    }

    frame_energies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n_noise = (frame_energies.len() as f64 * noise_percentile).ceil() as usize;
    let n_noise = n_noise.max(1).min(frame_energies.len());
    let noise_floor_power = frame_energies[..n_noise].iter().sum::<f64>() / n_noise as f64;
    let noise_floor_rms = noise_floor_power.sqrt();

    let peak_db = if peak > f64::EPSILON {
        20.0 * peak.log10()
    } else {
        -120.0
    };
    let rms_db = if rms_val > f64::EPSILON {
        20.0 * rms_val.log10()
    } else {
        -120.0
    };
    let noise_floor_db = if noise_floor_rms > f64::EPSILON {
        20.0 * noise_floor_rms.log10()
    } else {
        -120.0
    };

    let dynamic_range_db = peak_db - noise_floor_db;

    Ok(DynamicRangeResult {
        dynamic_range_db,
        peak_db,
        rms_db,
        noise_floor_db,
    })
}

// ---------------------------------------------------------------------------
// Zero-Crossing Rate
// ---------------------------------------------------------------------------

/// Compute the zero-crossing rate of a signal.
///
/// ZCR = (number of sign changes) / (N - 1).
///
/// Returns a value in [0, 1].
pub fn zero_crossing_rate(signal: &[f64]) -> SignalResult<f64> {
    if signal.len() < 2 {
        return Err(SignalError::ValueError(
            "Signal must have at least 2 samples".into(),
        ));
    }

    let n = signal.len();
    let mut crossings = 0u64;
    for i in 1..n {
        if (signal[i] >= 0.0 && signal[i - 1] < 0.0) || (signal[i] < 0.0 && signal[i - 1] >= 0.0) {
            crossings += 1;
        }
    }

    Ok(crossings as f64 / (n - 1) as f64)
}

/// Compute zero-crossing rate per frame.
pub fn zero_crossing_rate_frames(
    signal: &[f64],
    frame_length: usize,
    hop_size: usize,
) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if frame_length < 2 || hop_size == 0 {
        return Err(SignalError::InvalidArgument(
            "Frame length must be >= 2 and hop size > 0".into(),
        ));
    }

    let mut result = Vec::new();
    let mut pos = 0;
    while pos + frame_length <= signal.len() {
        let zcr = zero_crossing_rate(&signal[pos..pos + frame_length])?;
        result.push(zcr);
        pos += hop_size;
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// SINAD and ENOB
// ---------------------------------------------------------------------------

/// Compute SINAD (Signal-to-Noise-and-Distortion ratio).
///
/// SINAD = 10 * log10(P_signal / (P_noise + P_distortion))
///
/// This is computed by finding the fundamental in the spectrum, then
/// computing the ratio of fundamental power to everything else.
pub fn sinad(signal: &[f64], fs: f64) -> SignalResult<f64> {
    if signal.is_empty() {
        return Err(SignalError::ValueError("Signal must not be empty".into()));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError(
            "Sampling frequency must be positive".into(),
        ));
    }

    let n = signal.len();
    let nfft = next_power_of_two(n);
    let spec = scirs2_fft::rfft(signal, Some(nfft))
        .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

    let n_freq = spec.len();
    if n_freq < 2 {
        return Err(SignalError::ComputationError("Spectrum too short".into()));
    }

    // Power spectrum (skip DC)
    let powers: Vec<f64> = spec.iter().map(|c| c.re * c.re + c.im * c.im).collect();

    // Find fundamental (highest peak, excluding DC)
    let mut max_power = 0.0;
    let mut max_idx = 1;
    for (i, &p) in powers.iter().enumerate().skip(1) {
        if p > max_power {
            max_power = p;
            max_idx = i;
        }
    }

    // Sum power around the fundamental (3 bins)
    let fund_start = max_idx.saturating_sub(1);
    let fund_end = (max_idx + 2).min(n_freq);
    let signal_power: f64 = powers[fund_start..fund_end].iter().sum();

    // Total power (excluding DC)
    let total_power: f64 = powers[1..].iter().sum();
    let noise_distortion_power = total_power - signal_power;

    if noise_distortion_power < f64::EPSILON {
        return Ok(f64::INFINITY);
    }

    Ok(10.0 * (signal_power / noise_distortion_power).log10())
}

/// Compute Effective Number of Bits (ENOB) from SINAD.
///
/// ENOB = (SINAD_dB - 1.76) / 6.02
pub fn enob(signal: &[f64], fs: f64) -> SignalResult<f64> {
    let sinad_db = sinad(signal, fs)?;
    Ok((sinad_db - 1.76) / 6.02)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn next_power_of_two(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_tone(n: usize, freq: f64, fs: f64, amplitude: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amplitude * (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    fn add_noise(signal: &[f64], noise_amplitude: f64) -> Vec<f64> {
        // Deterministic "noise" using a simple hash-like function
        signal
            .iter()
            .enumerate()
            .map(|(i, &s)| {
                let pseudo_noise = ((i as f64 * 0.618033988749895).fract() - 0.5) * 2.0;
                s + noise_amplitude * pseudo_noise
            })
            .collect()
    }

    #[test]
    fn test_snr_reference_high() {
        let reference = make_tone(1024, 100.0, 1000.0, 1.0);
        let degraded = add_noise(&reference, 0.01);
        let snr_db = snr_reference(&reference, &degraded).expect("snr_ref");
        // Very little noise => high SNR
        assert!(snr_db > 20.0, "Expected high SNR, got {}", snr_db);
    }

    #[test]
    fn test_snr_reference_low() {
        let reference = make_tone(1024, 100.0, 1000.0, 1.0);
        let degraded = add_noise(&reference, 1.0);
        let snr_db = snr_reference(&reference, &degraded).expect("snr_ref");
        // Lots of noise => low SNR
        assert!(snr_db < 20.0, "Expected low SNR, got {}", snr_db);
    }

    #[test]
    fn test_snr_from_noise_floor() {
        let signal = make_tone(512, 50.0, 500.0, 1.0);
        let snr_db = snr_from_noise_floor(&signal, 0.01).expect("snr_nf");
        // RMS of a sine of amplitude 1 is 1/sqrt(2) => power ~ 0.5
        // 10*log10(0.5/0.01) ~ 17 dB
        assert!(snr_db > 10.0, "Expected SNR > 10, got {}", snr_db);
    }

    #[test]
    fn test_snr_blind() {
        let signal = make_tone(2048, 100.0, 1000.0, 1.0);
        let config = BlindSnrConfig {
            frame_length: 128,
            hop_size: 64,
            noise_percentile: 0.1,
        };
        let result = snr_blind(&signal, &config);
        assert!(result.is_ok(), "Blind SNR should succeed");
    }

    #[test]
    fn test_segmental_snr() {
        let reference = make_tone(2048, 100.0, 1000.0, 1.0);
        let degraded = add_noise(&reference, 0.05);
        let seg_snr = segmental_snr(&reference, &degraded, 256, 128).expect("seg_snr");
        assert!(seg_snr > 0.0, "Segmental SNR should be positive");
    }

    #[test]
    fn test_sdr() {
        let reference = make_tone(1024, 100.0, 1000.0, 1.0);
        let estimated = add_noise(&reference, 0.1);
        let sdr_db = sdr(&reference, &estimated).expect("sdr");
        assert!(sdr_db > 10.0, "Expected SDR > 10, got {}", sdr_db);
    }

    #[test]
    fn test_sdr_perfect() {
        let reference = make_tone(512, 50.0, 500.0, 1.0);
        let sdr_db = sdr(&reference, &reference).expect("sdr");
        assert!(sdr_db > 100.0 || sdr_db == f64::INFINITY);
    }

    #[test]
    fn test_si_sdr() {
        let reference = make_tone(512, 50.0, 500.0, 1.0);
        let estimated = add_noise(&reference, 0.1);
        let result = si_sdr(&reference, &estimated);
        assert!(result.is_ok());
    }

    #[test]
    fn test_perceptual_quality() {
        let reference = make_tone(4096, 440.0, 16000.0, 1.0);
        let degraded = add_noise(&reference, 0.05);
        let result = perceptual_quality(&reference, &degraded, 16000.0).expect("pq");
        assert!(result.score >= 1.0 && result.score <= 5.0);
        assert_eq!(result.band_disturbances.len(), 24);
    }

    #[test]
    fn test_perceptual_quality_identical() {
        let reference = make_tone(4096, 440.0, 16000.0, 1.0);
        let result = perceptual_quality(&reference, &reference, 16000.0).expect("pq");
        // Identical signals => high score
        assert!(
            result.score >= 3.0,
            "Score should be high for identical signals: {}",
            result.score
        );
    }

    #[test]
    fn test_spectral_flatness_tone() {
        let signal = make_tone(1024, 100.0, 1000.0, 1.0);
        let sf = spectral_flatness(&signal, None).expect("sf");
        // A pure tone has low spectral flatness
        assert!(sf < 0.3, "Tone flatness should be low, got {}", sf);
    }

    #[test]
    fn test_spectral_flatness_noise() {
        // Generate pseudo-noise with better spectral properties
        // Use multiple incommensurate frequencies to approximate broadband noise
        let signal: Vec<f64> = (0..2048)
            .map(|i| {
                let t = i as f64;
                (t * 0.1).sin()
                    + (t * 0.317).sin()
                    + (t * 0.7123).sin()
                    + (t * 1.2347).sin()
                    + (t * 2.1987).sin()
                    + (t * 3.14159).sin()
                    + (t * 0.04517).sin()
                    + (t * 0.8765).sin()
            })
            .collect();
        let sf = spectral_flatness(&signal, None).expect("sf");
        // Multi-frequency signal should have higher flatness than a single pure tone
        // 8 frequencies spanning a range give moderate flatness
        let sf_tone =
            spectral_flatness(&make_tone(2048, 100.0, 2048.0, 1.0), None).expect("sf_tone");
        assert!(
            sf > sf_tone,
            "Multi-tone flatness ({}) should be > pure tone flatness ({})",
            sf,
            sf_tone
        );
    }

    #[test]
    fn test_spectral_flatness_frames() {
        let signal = make_tone(2048, 100.0, 1000.0, 1.0);
        let frames = spectral_flatness_frames(&signal, 256, 128, None).expect("sf frames");
        assert!(!frames.is_empty());
        for &f in &frames {
            assert!(f >= 0.0 && f <= 1.0);
        }
    }

    #[test]
    fn test_crest_factor_sine() {
        // Use a frequency/sampling combo where the peak is well-sampled
        // 100 samples/cycle ensures sin reaches very close to 1.0
        let signal = make_tone(10000, 100.0, 10000.0, 1.0);
        let cf = crest_factor(&signal).expect("cf");
        // Sine wave crest factor = sqrt(2) ~ 1.414
        assert!(
            (cf - 2.0_f64.sqrt()).abs() < 0.05,
            "Expected ~1.414, got {}",
            cf
        );
    }

    #[test]
    fn test_crest_factor_dc() {
        let signal = vec![5.0; 100];
        let cf = crest_factor(&signal).expect("cf");
        // DC signal: crest factor = 1.0
        assert!((cf - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_crest_factor_db() {
        let signal = make_tone(10000, 100.0, 10000.0, 1.0);
        let cf_db = crest_factor_db(&signal).expect("cf_db");
        // 20 * log10(sqrt(2)) ~ 3.01 dB
        assert!(
            (cf_db - 3.01).abs() < 0.1,
            "Expected ~3.01 dB, got {}",
            cf_db
        );
    }

    #[test]
    fn test_dynamic_range() {
        let signal = make_tone(4096, 100.0, 1000.0, 1.0);
        let result = dynamic_range(&signal, 256, 0.1).expect("dr");
        assert!(result.dynamic_range_db > 0.0);
        assert!(result.peak_db <= 0.1); // peak of amplitude 1 => ~0 dBFS
    }

    #[test]
    fn test_zero_crossing_rate_sine() {
        let fs = 1000.0;
        let freq = 100.0;
        let n = 1000;
        let signal = make_tone(n, freq, fs, 1.0);
        let zcr = zero_crossing_rate(&signal).expect("zcr");
        // 100 Hz sine at 1000 Hz sampling => ~200 crossings per 999 intervals ~ 0.2
        assert!((zcr - 0.2).abs() < 0.01, "Expected ~0.2, got {}", zcr);
    }

    #[test]
    fn test_zero_crossing_rate_dc() {
        let signal = vec![1.0; 100];
        let zcr = zero_crossing_rate(&signal).expect("zcr");
        assert!((zcr - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_crossing_rate_frames() {
        let signal = make_tone(2048, 100.0, 1000.0, 1.0);
        let frames = zero_crossing_rate_frames(&signal, 256, 128).expect("zcr frames");
        assert!(!frames.is_empty());
        for &z in &frames {
            assert!(z >= 0.0 && z <= 1.0);
        }
    }

    #[test]
    fn test_sinad() {
        // Use integer number of cycles to avoid spectral leakage:
        // 1000 samples at fs=1000 with freq=100 => exactly 100 cycles
        let signal = make_tone(1000, 100.0, 1000.0, 1.0);
        let sinad_db = sinad(&signal, 1000.0).expect("sinad");
        // Pure tone with perfect periodicity => high SINAD
        assert!(
            sinad_db > 5.0,
            "Expected SINAD > 5 dB for pure tone, got {}",
            sinad_db
        );
    }

    #[test]
    fn test_enob() {
        let signal = make_tone(1000, 100.0, 1000.0, 1.0);
        let enob_val = enob(&signal, 1000.0).expect("enob");
        // ENOB should be positive for a pure tone
        assert!(
            enob_val > 0.0,
            "Expected ENOB > 0 for pure tone, got {}",
            enob_val
        );
    }

    #[test]
    fn test_snr_errors() {
        assert!(snr_reference(&[], &[]).is_err());
        assert!(snr_reference(&[1.0], &[1.0, 2.0]).is_err());
        assert!(snr_from_noise_floor(&[], 1.0).is_err());
        assert!(snr_from_noise_floor(&[1.0], -1.0).is_err());
    }

    #[test]
    fn test_crest_factor_errors() {
        assert!(crest_factor(&[]).is_err());
    }

    #[test]
    fn test_zcr_errors() {
        assert!(zero_crossing_rate(&[1.0]).is_err());
    }
}
