//! Cepstral analysis module for signal processing
//!
//! This module provides cepstral analysis tools including:
//! - Real cepstrum computation
//! - Complex cepstrum computation
//! - Mel-Frequency Cepstral Coefficients (MFCC) extraction
//! - Mel filter bank generation
//! - Discrete Cosine Transform (Type-II) for MFCC
//!
//! Cepstral analysis is widely used in speech/audio processing for feature extraction,
//! pitch detection, echo analysis, and source-filter separation.

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Mel-frequency helpers
// ---------------------------------------------------------------------------

/// Convert frequency in Hz to Mel scale (HTK formula)
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert Mel scale value to frequency in Hz (HTK formula)
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

// ---------------------------------------------------------------------------
// Mel filter bank
// ---------------------------------------------------------------------------

/// Configuration for Mel filter bank generation
#[derive(Debug, Clone)]
pub struct MelFilterBankConfig {
    /// Number of Mel filters
    pub n_filters: usize,
    /// FFT size (number of frequency bins = fft_size / 2 + 1)
    pub fft_size: usize,
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Low frequency cutoff in Hz (default 0.0)
    pub low_freq: f64,
    /// High frequency cutoff in Hz (default sample_rate / 2)
    pub high_freq: Option<f64>,
}

impl MelFilterBankConfig {
    /// Create default config with the given number of filters, FFT size, and sample rate
    pub fn new(n_filters: usize, fft_size: usize, sample_rate: f64) -> Self {
        Self {
            n_filters,
            fft_size,
            sample_rate,
            low_freq: 0.0,
            high_freq: None,
        }
    }
}

/// Generate a Mel-spaced triangular filter bank
///
/// Returns a matrix of shape (n_filters, n_fft_bins) where n_fft_bins = fft_size / 2 + 1.
/// Each row is a triangular filter in the frequency domain.
///
/// # Arguments
///
/// * `config` - Mel filter bank configuration
///
/// # Returns
///
/// * `Vec<Vec<f64>>` - Filter bank matrix (n_filters x n_fft_bins)
pub fn mel_filter_bank(config: &MelFilterBankConfig) -> SignalResult<Vec<Vec<f64>>> {
    if config.n_filters == 0 {
        return Err(SignalError::ValueError(
            "Number of Mel filters must be positive".to_string(),
        ));
    }
    if config.fft_size == 0 {
        return Err(SignalError::ValueError(
            "FFT size must be positive".to_string(),
        ));
    }
    if config.sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "Sample rate must be positive".to_string(),
        ));
    }

    let high_freq = config.high_freq.unwrap_or(config.sample_rate / 2.0);
    if high_freq > config.sample_rate / 2.0 {
        return Err(SignalError::ValueError(
            "High frequency must not exceed Nyquist frequency".to_string(),
        ));
    }
    if config.low_freq >= high_freq {
        return Err(SignalError::ValueError(
            "Low frequency must be less than high frequency".to_string(),
        ));
    }

    let n_fft_bins = config.fft_size / 2 + 1;
    let mel_low = hz_to_mel(config.low_freq);
    let mel_high = hz_to_mel(high_freq);

    // Linearly spaced Mel points (n_filters + 2 to include edges)
    let n_points = config.n_filters + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_low + (mel_high - mel_low) * i as f64 / (n_points - 1) as f64)
        .collect();

    // Convert Mel points to frequency bin indices
    let bin_indices: Vec<f64> = mel_points
        .iter()
        .map(|&m| mel_to_hz(m) * config.fft_size as f64 / config.sample_rate)
        .collect();

    // Build triangular filters
    let mut filter_bank = vec![vec![0.0; n_fft_bins]; config.n_filters];
    for f in 0..config.n_filters {
        let left = bin_indices[f];
        let center = bin_indices[f + 1];
        let right = bin_indices[f + 2];

        for k in 0..n_fft_bins {
            let freq_bin = k as f64;
            if freq_bin >= left && freq_bin < center {
                let denom = center - left;
                if denom.abs() > 1e-12 {
                    filter_bank[f][k] = (freq_bin - left) / denom;
                }
            } else if freq_bin >= center && freq_bin <= right {
                let denom = right - center;
                if denom.abs() > 1e-12 {
                    filter_bank[f][k] = (right - freq_bin) / denom;
                }
            }
        }
    }

    Ok(filter_bank)
}

// ---------------------------------------------------------------------------
// Real cepstrum
// ---------------------------------------------------------------------------

/// Compute the real cepstrum of a signal
///
/// The real cepstrum is defined as the inverse FFT of the log-magnitude spectrum:
/// ```text
/// c[n] = IFFT(log(|FFT(x)|))
/// ```
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * Real cepstrum coefficients (same length as input)
pub fn real_cepstrum(signal: &[f64]) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Input signal must not be empty".to_string(),
        ));
    }

    let n = signal.len();

    // Compute FFT
    let spectrum = scirs2_fft::fft(signal, Some(n))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

    // Log magnitude spectrum (add small epsilon to avoid log(0))
    let log_magnitude: Vec<Complex64> = spectrum
        .iter()
        .map(|c| {
            let mag = c.norm().max(1e-20);
            Complex64::new(mag.ln(), 0.0)
        })
        .collect();

    // IFFT of log magnitude
    let cepstrum = scirs2_fft::ifft(&log_magnitude, Some(n))
        .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {}", e)))?;

    Ok(cepstrum.iter().map(|c| c.re).collect())
}

// ---------------------------------------------------------------------------
// Complex cepstrum
// ---------------------------------------------------------------------------

/// Compute the complex cepstrum of a signal
///
/// The complex cepstrum preserves phase information:
/// ```text
/// c[n] = IFFT(log(FFT(x)))
/// ```
/// where log is the complex logarithm with unwrapped phase.
///
/// # Arguments
///
/// * `signal` - Input signal
///
/// # Returns
///
/// * Tuple of (complex_cepstrum, unwrapped_phase) where complex_cepstrum is the
///   real part of the complex cepstrum and unwrapped_phase is the phase spectrum
pub fn complex_cepstrum(signal: &[f64]) -> SignalResult<(Vec<f64>, Vec<f64>)> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Input signal must not be empty".to_string(),
        ));
    }

    let n = signal.len();

    // Compute FFT
    let spectrum = scirs2_fft::fft(signal, Some(n))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

    // Compute magnitude and phase
    let magnitudes: Vec<f64> = spectrum.iter().map(|c| c.norm().max(1e-20)).collect();
    let phases: Vec<f64> = spectrum.iter().map(|c| c.im.atan2(c.re)).collect();

    // Unwrap phase
    let unwrapped_phase = unwrap_phase(&phases);

    // Complex log spectrum: log|X| + j * unwrapped_phase
    let log_spectrum: Vec<Complex64> = magnitudes
        .iter()
        .zip(unwrapped_phase.iter())
        .map(|(&mag, &phase)| Complex64::new(mag.ln(), phase))
        .collect();

    // IFFT to get complex cepstrum
    let cepstrum = scirs2_fft::ifft(&log_spectrum, Some(n))
        .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {}", e)))?;

    Ok((cepstrum.iter().map(|c| c.re).collect(), unwrapped_phase))
}

/// Unwrap phase angles to avoid discontinuities
fn unwrap_phase(phases: &[f64]) -> Vec<f64> {
    if phases.is_empty() {
        return Vec::new();
    }

    let mut unwrapped = vec![0.0; phases.len()];
    unwrapped[0] = phases[0];

    for i in 1..phases.len() {
        let mut diff = phases[i] - phases[i - 1];
        // Wrap diff into (-pi, pi]
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff <= -PI {
            diff += 2.0 * PI;
        }
        unwrapped[i] = unwrapped[i - 1] + diff;
    }

    unwrapped
}

// ---------------------------------------------------------------------------
// DCT-II (used by MFCC)
// ---------------------------------------------------------------------------

/// Compute the Type-II Discrete Cosine Transform (orthonormal)
///
/// DCT-II[k] = sqrt(2/N) * sum_{n=0}^{N-1} x[n] * cos(pi*(2n+1)*k / (2N))
/// with a normalization factor of sqrt(1/N) for k=0.
fn dct_ii(input: &[f64], n_output: usize) -> Vec<f64> {
    let n = input.len();
    if n == 0 {
        return vec![0.0; n_output];
    }

    let mut output = Vec::with_capacity(n_output);
    let norm = (2.0 / n as f64).sqrt();

    for k in 0..n_output {
        let mut sum = 0.0;
        for (idx, &x) in input.iter().enumerate() {
            sum += x * (PI * (2 * idx + 1) as f64 * k as f64 / (2.0 * n as f64)).cos();
        }
        // Apply orthonormal scaling
        if k == 0 {
            output.push(sum * (1.0 / n as f64).sqrt());
        } else {
            output.push(sum * norm);
        }
    }

    output
}

// ---------------------------------------------------------------------------
// MFCC
// ---------------------------------------------------------------------------

/// Configuration for MFCC extraction
#[derive(Debug, Clone)]
pub struct MfccConfig {
    /// Number of MFCC coefficients to return (default 13)
    pub n_mfcc: usize,
    /// Number of Mel filters (default 26)
    pub n_mels: usize,
    /// FFT size (default 512)
    pub fft_size: usize,
    /// Sample rate in Hz
    pub sample_rate: f64,
    /// Whether to include the 0th (energy) coefficient (default true)
    pub include_energy: bool,
    /// Pre-emphasis coefficient (0.0 = no pre-emphasis, typical = 0.97)
    pub pre_emphasis: f64,
    /// Low frequency cutoff in Hz (default 0)
    pub low_freq: f64,
    /// High frequency cutoff in Hz (default sample_rate/2)
    pub high_freq: Option<f64>,
}

impl MfccConfig {
    /// Create an MFCC config with the given sample rate and default parameters
    pub fn new(sample_rate: f64) -> Self {
        Self {
            n_mfcc: 13,
            n_mels: 26,
            fft_size: 512,
            sample_rate,
            include_energy: true,
            pre_emphasis: 0.97,
            low_freq: 0.0,
            high_freq: None,
        }
    }
}

/// Extract MFCC features from a signal (single frame)
///
/// Computes Mel-Frequency Cepstral Coefficients for a single frame of audio.
/// For multi-frame extraction, use windowed/overlapped frames externally.
///
/// # Arguments
///
/// * `signal` - Input signal frame
/// * `config` - MFCC configuration
///
/// # Returns
///
/// * Vector of MFCC coefficients (length = n_mfcc)
///
/// # Example
///
/// ```ignore
/// use scirs2_signal::cepstral::{mfcc_frame, MfccConfig};
///
/// let config = MfccConfig::new(16000.0);
/// let signal: Vec<f64> = (0..512).map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 16000.0).sin()).collect();
/// let coeffs = mfcc_frame(&signal, &config).expect("MFCC extraction failed");
/// ```
pub fn mfcc_frame(signal: &[f64], config: &MfccConfig) -> SignalResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Input signal must not be empty".to_string(),
        ));
    }
    if config.n_mfcc == 0 {
        return Err(SignalError::ValueError(
            "Number of MFCC coefficients must be positive".to_string(),
        ));
    }
    if config.sample_rate <= 0.0 {
        return Err(SignalError::ValueError(
            "Sample rate must be positive".to_string(),
        ));
    }

    // 1. Pre-emphasis
    let emphasized = if config.pre_emphasis > 0.0 {
        let mut emp = Vec::with_capacity(signal.len());
        emp.push(signal[0]);
        for i in 1..signal.len() {
            emp.push(signal[i] - config.pre_emphasis * signal[i - 1]);
        }
        emp
    } else {
        signal.to_vec()
    };

    // 2. Zero-pad or truncate to FFT size
    let fft_size = config.fft_size;
    let mut padded = vec![0.0; fft_size];
    let copy_len = emphasized.len().min(fft_size);
    padded[..copy_len].copy_from_slice(&emphasized[..copy_len]);

    // 3. FFT
    let spectrum = scirs2_fft::fft(&padded, Some(fft_size))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

    // 4. Power spectrum (first half + DC)
    let n_fft_bins = fft_size / 2 + 1;
    let power_spectrum: Vec<f64> = spectrum[..n_fft_bins]
        .iter()
        .map(|c| c.norm_sqr() / fft_size as f64)
        .collect();

    // 5. Generate Mel filter bank
    let mel_config = MelFilterBankConfig {
        n_filters: config.n_mels,
        fft_size,
        sample_rate: config.sample_rate,
        low_freq: config.low_freq,
        high_freq: config.high_freq,
    };
    let filters = mel_filter_bank(&mel_config)?;

    // 6. Apply Mel filter bank
    let mel_energies: Vec<f64> = filters
        .iter()
        .map(|filter_row| {
            let energy: f64 = filter_row
                .iter()
                .zip(power_spectrum.iter())
                .map(|(&f, &p)| f * p)
                .sum();
            // Log energy (with floor to avoid log(0))
            (energy.max(1e-20)).ln()
        })
        .collect();

    // 7. DCT-II to get cepstral coefficients
    let cepstral = dct_ii(&mel_energies, config.n_mfcc);

    // 8. Optionally replace 0th coeff with log total energy
    if config.include_energy {
        let mut result = cepstral;
        let total_energy: f64 = power_spectrum.iter().sum();
        result[0] = (total_energy.max(1e-20)).ln();
        Ok(result)
    } else {
        // Skip 0th coefficient
        if cepstral.len() > 1 {
            Ok(cepstral[1..].to_vec())
        } else {
            Ok(cepstral)
        }
    }
}

/// Convenience function: extract MFCCs from a signal with sensible defaults
///
/// # Arguments
///
/// * `signal` - Input signal frame (typically one windowed frame)
/// * `sample_rate` - Sample rate in Hz
/// * `n_mfcc` - Number of MFCC coefficients to compute
///
/// # Returns
///
/// * Vector of MFCC coefficients
pub fn mfcc(signal: &[f64], sample_rate: f64, n_mfcc: usize) -> SignalResult<Vec<f64>> {
    let config = MfccConfig {
        n_mfcc,
        sample_rate,
        ..MfccConfig::new(sample_rate)
    };
    mfcc_frame(signal, &config)
}

/// Extract MFCC features from a signal using framing and windowing
///
/// This function splits the signal into overlapping frames, applies a Hamming window,
/// and computes MFCC for each frame.
///
/// # Arguments
///
/// * `signal` - Full input signal
/// * `config` - MFCC configuration
/// * `frame_length` - Frame length in samples
/// * `hop_length` - Hop (step) length in samples
///
/// # Returns
///
/// * `Vec<Vec<f64>>` - MFCC matrix (n_frames x n_mfcc)
pub fn mfcc_extract(
    signal: &[f64],
    config: &MfccConfig,
    frame_length: usize,
    hop_length: usize,
) -> SignalResult<Vec<Vec<f64>>> {
    if signal.is_empty() {
        return Err(SignalError::ValueError(
            "Input signal must not be empty".to_string(),
        ));
    }
    if frame_length == 0 || hop_length == 0 {
        return Err(SignalError::ValueError(
            "Frame length and hop length must be positive".to_string(),
        ));
    }

    // Generate Hamming window
    let window: Vec<f64> = (0..frame_length)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (frame_length - 1).max(1) as f64).cos())
        .collect();

    let mut frames = Vec::new();
    let mut start = 0;

    while start + frame_length <= signal.len() {
        // Extract frame and apply window
        let frame: Vec<f64> = signal[start..start + frame_length]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect();

        let coeffs = mfcc_frame(&frame, config)?;
        frames.push(coeffs);

        start += hop_length;
    }

    if frames.is_empty() {
        return Err(SignalError::ValueError(
            "Signal too short for given frame length".to_string(),
        ));
    }

    Ok(frames)
}

/// Compute delta (first derivative) of cepstral coefficients
///
/// Uses the regression formula over a window of +/- `width` frames.
///
/// # Arguments
///
/// * `features` - MFCC feature matrix (n_frames x n_coeffs)
/// * `width` - Number of frames on each side for regression (default 2)
///
/// # Returns
///
/// * Delta features (same dimensions as input)
pub fn compute_deltas(features: &[Vec<f64>], width: usize) -> SignalResult<Vec<Vec<f64>>> {
    if features.is_empty() {
        return Err(SignalError::ValueError(
            "Feature matrix must not be empty".to_string(),
        ));
    }
    if width == 0 {
        return Err(SignalError::ValueError(
            "Delta width must be positive".to_string(),
        ));
    }

    let n_frames = features.len();
    let n_coeffs = features[0].len();

    // Denominator: 2 * sum(t^2) for t = 1..width
    let denom: f64 = 2.0 * (1..=width).map(|t| (t * t) as f64).sum::<f64>();
    if denom.abs() < 1e-20 {
        return Err(SignalError::ComputationError(
            "Delta denominator is zero".to_string(),
        ));
    }

    let mut deltas = vec![vec![0.0; n_coeffs]; n_frames];

    for t in 0..n_frames {
        for c in 0..n_coeffs {
            let mut numerator = 0.0;
            for w in 1..=width {
                let t_plus = (t + w).min(n_frames - 1);
                let t_minus = if t >= w { t - w } else { 0 };
                numerator += w as f64 * (features[t_plus][c] - features[t_minus][c]);
            }
            deltas[t][c] = numerator / denom;
        }
    }

    Ok(deltas)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ----- Mel scale tests -----

    #[test]
    fn test_hz_mel_conversion_roundtrip() {
        for freq in &[0.0, 100.0, 440.0, 1000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel(*freq);
            let recovered = mel_to_hz(mel);
            assert_relative_eq!(recovered, *freq, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_hz_to_mel_monotonic() {
        let freqs = [0.0, 100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
        for w in freqs.windows(2) {
            assert!(hz_to_mel(w[1]) > hz_to_mel(w[0]));
        }
    }

    #[test]
    fn test_mel_filter_bank_basic() {
        let config = MelFilterBankConfig::new(26, 512, 16000.0);
        let bank = mel_filter_bank(&config).expect("Filter bank creation failed");

        assert_eq!(bank.len(), 26);
        assert_eq!(bank[0].len(), 257); // 512/2 + 1

        // Each filter should have non-negative values
        for filter in &bank {
            assert!(filter.iter().all(|&v| v >= 0.0));
        }

        // Each filter should have at least one non-zero value
        for (i, filter) in bank.iter().enumerate() {
            let max_val: f64 = filter.iter().cloned().fold(0.0, f64::max);
            assert!(max_val > 0.0, "Filter {} is all zeros", i);
        }
    }

    #[test]
    fn test_mel_filter_bank_custom_range() {
        let config = MelFilterBankConfig {
            n_filters: 10,
            fft_size: 256,
            sample_rate: 8000.0,
            low_freq: 300.0,
            high_freq: Some(3400.0), // telephone band
        };
        let bank = mel_filter_bank(&config).expect("Filter bank creation failed");
        assert_eq!(bank.len(), 10);
    }

    #[test]
    fn test_mel_filter_bank_validation() {
        // Zero filters
        let config = MelFilterBankConfig::new(0, 512, 16000.0);
        assert!(mel_filter_bank(&config).is_err());

        // High freq above Nyquist
        let config = MelFilterBankConfig {
            n_filters: 10,
            fft_size: 512,
            sample_rate: 16000.0,
            low_freq: 0.0,
            high_freq: Some(9000.0), // above 8000 Nyquist
        };
        assert!(mel_filter_bank(&config).is_err());

        // Low freq >= high freq
        let config = MelFilterBankConfig {
            n_filters: 10,
            fft_size: 512,
            sample_rate: 16000.0,
            low_freq: 5000.0,
            high_freq: Some(3000.0),
        };
        assert!(mel_filter_bank(&config).is_err());
    }

    #[test]
    fn test_mel_filter_bank_overlapping() {
        let config = MelFilterBankConfig::new(10, 512, 16000.0);
        let bank = mel_filter_bank(&config).expect("Filter bank creation failed");

        // Adjacent filters should overlap
        for i in 0..bank.len() - 1 {
            let overlap: f64 = bank[i]
                .iter()
                .zip(bank[i + 1].iter())
                .map(|(&a, &b)| a.min(b))
                .sum();
            // Not all adjacent pairs will overlap due to spacing,
            // but most should have some overlap
            if overlap <= 0.0 {
                // At least check they are neighbors in frequency
                let peak_i = bank[i]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let peak_next = bank[i + 1]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                assert!(peak_next >= peak_i, "Filters not ordered by frequency");
            }
        }
    }

    // ----- Real cepstrum tests -----

    #[test]
    fn test_real_cepstrum_empty_input() {
        assert!(real_cepstrum(&[]).is_err());
    }

    #[test]
    fn test_real_cepstrum_impulse() {
        // Real cepstrum of a unit impulse should be near-zero everywhere
        let mut impulse = vec![0.0; 64];
        impulse[0] = 1.0;

        let cep = real_cepstrum(&impulse).expect("Real cepstrum failed");
        assert_eq!(cep.len(), 64);

        // For a pure impulse, the cepstrum should be near zero
        // (log(1) = 0 for all frequency bins)
        for (i, &c) in cep.iter().enumerate() {
            assert!(
                c.abs() < 1e-6,
                "Cepstrum coefficient {} = {} (expected ~0)",
                i,
                c
            );
        }
    }

    #[test]
    fn test_real_cepstrum_sinusoid() {
        let n = 256;
        let freq = 440.0;
        let sr = 8000.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect();

        let cep = real_cepstrum(&signal).expect("Real cepstrum failed");
        assert_eq!(cep.len(), n);
        // Just check it returned valid values
        assert!(cep.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn test_real_cepstrum_symmetry() {
        // Real cepstrum of a real signal should have real cepstrum that is symmetric
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin() + 0.5)
            .collect();

        let cep = real_cepstrum(&signal).expect("Real cepstrum failed");
        // Check approximate symmetry: c[k] ~ c[N-k]
        for k in 1..n / 2 {
            assert!(
                (cep[k] - cep[n - k]).abs() < 1e-6,
                "Cepstrum not symmetric at k={}: {} vs {}",
                k,
                cep[k],
                cep[n - k]
            );
        }
    }

    #[test]
    fn test_real_cepstrum_constant_signal() {
        let signal = vec![2.0; 64];
        let cep = real_cepstrum(&signal).expect("Real cepstrum failed");
        assert_eq!(cep.len(), 64);
        // For a constant signal, the spectrum is a single non-zero bin at DC.
        // The cepstrum[0] should contain log(amplitude * N), rest should be finite.
        assert!(cep[0].is_finite());
        // Cepstrum should be real and finite
        assert!(cep.iter().all(|c| c.is_finite()));
    }

    // ----- Complex cepstrum tests -----

    #[test]
    fn test_complex_cepstrum_empty_input() {
        assert!(complex_cepstrum(&[]).is_err());
    }

    #[test]
    fn test_complex_cepstrum_impulse() {
        let mut impulse = vec![0.0; 64];
        impulse[0] = 1.0;

        let (cep, phase) = complex_cepstrum(&impulse).expect("Complex cepstrum failed");
        assert_eq!(cep.len(), 64);
        assert_eq!(phase.len(), 64);

        // For a unit impulse, cepstrum should be near zero
        for &c in &cep {
            assert!(
                c.abs() < 1e-6,
                "Non-zero complex cepstrum for impulse: {}",
                c
            );
        }
    }

    #[test]
    fn test_complex_cepstrum_returns_finite() {
        let signal: Vec<f64> = (0..128)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / 128.0).sin() + 0.3)
            .collect();

        let (cep, phase) = complex_cepstrum(&signal).expect("Complex cepstrum failed");
        assert!(cep.iter().all(|c| c.is_finite()));
        assert!(phase.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn test_complex_cepstrum_delayed_impulse() {
        // A delayed impulse: x[n] = delta[n - d]
        let n = 64;
        let delay = 5;
        let mut signal = vec![0.0; n];
        signal[delay] = 1.0;

        let (cep, _phase) = complex_cepstrum(&signal).expect("Complex cepstrum failed");
        assert_eq!(cep.len(), n);
        assert!(cep.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn test_phase_unwrap_basic() {
        // Phases that jump across +-pi
        let phases = vec![0.0, 0.5, 1.0, 2.0, 3.0, -3.0, -2.0, -1.0];
        let unwrapped = unwrap_phase(&phases);
        assert_eq!(unwrapped.len(), phases.len());

        // Check that consecutive differences are in (-pi, pi]
        for w in unwrapped.windows(2) {
            let diff = w[1] - w[0];
            assert!(
                diff > -PI && diff <= PI,
                "Unwrapped phase diff out of range: {}",
                diff
            );
        }
    }

    #[test]
    fn test_phase_unwrap_empty() {
        let unwrapped = unwrap_phase(&[]);
        assert!(unwrapped.is_empty());
    }

    // ----- DCT-II tests -----

    #[test]
    fn test_dct_ii_zeros() {
        let zeros = vec![0.0; 8];
        let result = dct_ii(&zeros, 4);
        assert_eq!(result.len(), 4);
        for &v in &result {
            assert_relative_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_dct_ii_ones() {
        // DCT-II of a constant should concentrate energy in DC
        let ones = vec![1.0; 8];
        let result = dct_ii(&ones, 4);
        assert!(result[0].abs() > 0.0);
        // Higher order coefficients should be near zero for a constant signal
        for &v in &result[1..] {
            assert!(
                v.abs() < 1e-10,
                "Non-zero DCT coefficient for constant: {}",
                v
            );
        }
    }

    #[test]
    fn test_dct_ii_energy_preservation() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = dct_ii(&input, input.len());

        // Parseval's theorem: energy in time ~ energy in DCT domain
        let time_energy: f64 = input.iter().map(|x| x * x).sum();
        let dct_energy: f64 = result.iter().map(|x| x * x).sum();
        assert_relative_eq!(time_energy, dct_energy, epsilon = 1e-6);
    }

    // ----- MFCC tests -----

    #[test]
    fn test_mfcc_basic() {
        let n = 512;
        let sr = 16000.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();

        let coeffs = mfcc(&signal, sr, 13).expect("MFCC extraction failed");
        assert_eq!(coeffs.len(), 13);
        assert!(coeffs.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn test_mfcc_empty_signal() {
        assert!(mfcc(&[], 16000.0, 13).is_err());
    }

    #[test]
    fn test_mfcc_invalid_params() {
        let signal = vec![1.0; 256];
        assert!(mfcc(&signal, 0.0, 13).is_err()); // invalid sample rate
        assert!(mfcc(&signal, 16000.0, 0).is_err()); // zero coefficients
    }

    #[test]
    fn test_mfcc_different_signals_differ() {
        let sr = 16000.0;
        let n = 512;
        let sig1: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();
        let sig2: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 1000.0 * i as f64 / sr).sin())
            .collect();

        let mfcc1 = mfcc(&sig1, sr, 13).expect("MFCC failed for sig1");
        let mfcc2 = mfcc(&sig2, sr, 13).expect("MFCC failed for sig2");

        // MFCCs of different frequency signals should differ
        let diff: f64 = mfcc1
            .iter()
            .zip(mfcc2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 0.01,
            "MFCCs for different signals should differ, diff = {}",
            diff
        );
    }

    #[test]
    fn test_mfcc_with_pre_emphasis() {
        let sr = 16000.0;
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();

        let config_no_pre = MfccConfig {
            pre_emphasis: 0.0,
            ..MfccConfig::new(sr)
        };
        let config_with_pre = MfccConfig {
            pre_emphasis: 0.97,
            ..MfccConfig::new(sr)
        };

        let mfcc_no = mfcc_frame(&signal, &config_no_pre).expect("MFCC failed");
        let mfcc_yes = mfcc_frame(&signal, &config_with_pre).expect("MFCC failed");

        // They should differ due to pre-emphasis
        let diff: f64 = mfcc_no
            .iter()
            .zip(mfcc_yes.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-6, "Pre-emphasis should change MFCCs");
    }

    #[test]
    fn test_mfcc_frame_config() {
        let sr = 16000.0;
        let signal: Vec<f64> = (0..512)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();

        let config = MfccConfig {
            n_mfcc: 20,
            n_mels: 40,
            fft_size: 512,
            sample_rate: sr,
            include_energy: true,
            pre_emphasis: 0.97,
            low_freq: 80.0,
            high_freq: Some(7600.0),
        };
        let coeffs = mfcc_frame(&signal, &config).expect("MFCC frame failed");
        assert_eq!(coeffs.len(), 20);
    }

    // ----- MFCC extraction (multi-frame) tests -----

    #[test]
    fn test_mfcc_extract_basic() {
        let sr = 16000.0;
        let duration = 0.1; // 100ms
        let n = (sr * duration) as usize;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 440.0 * i as f64 / sr).sin())
            .collect();

        let config = MfccConfig::new(sr);
        let frame_length = 512;
        let hop_length = 160;

        let mfccs =
            mfcc_extract(&signal, &config, frame_length, hop_length).expect("MFCC extract failed");

        // Should produce at least one frame
        assert!(!mfccs.is_empty());
        // Each frame should have n_mfcc coefficients
        for frame in &mfccs {
            assert_eq!(frame.len(), config.n_mfcc);
        }
    }

    #[test]
    fn test_mfcc_extract_short_signal() {
        let config = MfccConfig::new(16000.0);
        let signal = vec![0.1; 100]; // shorter than frame_length

        assert!(mfcc_extract(&signal, &config, 512, 160).is_err());
    }

    #[test]
    fn test_mfcc_extract_frame_count() {
        let sr = 16000.0;
        let signal = vec![0.5; 2000];
        let config = MfccConfig::new(sr);
        let frame_length = 512;
        let hop_length = 256;

        let mfccs =
            mfcc_extract(&signal, &config, frame_length, hop_length).expect("MFCC extract failed");

        let expected_frames = (signal.len() - frame_length) / hop_length + 1;
        assert_eq!(mfccs.len(), expected_frames);
    }

    // ----- Delta tests -----

    #[test]
    fn test_compute_deltas_basic() {
        let features = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![3.0, 4.0],
            vec![4.0, 5.0],
            vec![5.0, 6.0],
        ];

        let deltas = compute_deltas(&features, 2).expect("Delta computation failed");
        assert_eq!(deltas.len(), 5);
        assert_eq!(deltas[0].len(), 2);

        // For a linearly increasing feature, delta should be constant
        // Check middle frames are approximately 1.0
        assert_relative_eq!(deltas[2][0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(deltas[2][1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_compute_deltas_constant_features() {
        let features = vec![vec![3.0, 7.0], vec![3.0, 7.0], vec![3.0, 7.0]];

        let deltas = compute_deltas(&features, 1).expect("Delta computation failed");
        for frame in &deltas {
            for &d in frame {
                assert_relative_eq!(d, 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_compute_deltas_empty() {
        assert!(compute_deltas(&[], 2).is_err());
    }

    #[test]
    fn test_compute_deltas_zero_width() {
        let features = vec![vec![1.0]];
        assert!(compute_deltas(&features, 0).is_err());
    }

    #[test]
    fn test_compute_deltas_double() {
        // Compute delta-delta (acceleration) on linearly increasing features
        let features = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
        ];
        // Delta of linear = constant
        let deltas = compute_deltas(&features, 2).expect("Delta failed");
        // Delta-delta of linear = 0
        let delta_deltas = compute_deltas(&deltas, 2).expect("Delta-delta failed");

        assert_eq!(delta_deltas.len(), 7);
        // Middle coefficients of delta-delta should be near zero for linear input
        assert_relative_eq!(delta_deltas[3][0], 0.0, epsilon = 1e-10);
    }
}
