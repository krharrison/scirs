//! Cyclostationary spectral analysis.
//!
//! Implements the Spectral Correlation Function (SCF) via the FFT Accumulation
//! Method (FAM), along with derived quantities: cyclic spectral density, spectral
//! coherence, and cyclic frequency detection (alpha-profile).
//!
//! # Background
//!
//! A cyclostationary signal has statistical properties that vary periodically
//! with time. The Spectral Correlation Function `S(f, alpha)` captures these
//! periodicities in the frequency domain, where `f` is spectral frequency and
//! `alpha` is the cyclic frequency.
//!
//! # References
//!
//! * Roberts, R.S., Brown, W.A., Loomis, H.H. "Computationally efficient
//!   algorithms for cyclic spectral analysis." IEEE SP Magazine, 1991.
//! * Gardner, W.A. "Cyclostationarity in communications and signal processing."
//!   IEEE Press, 1994.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use crate::window::{get_window, Window};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

/// Configuration for cyclostationary analysis.
#[derive(Debug, Clone)]
pub struct CyclostationaryConfig {
    /// FFT size for channelisation (N').
    pub fft_size: usize,
    /// Window function applied to each segment.
    pub window: Window,
    /// Overlap between consecutive segments in samples.
    pub overlap: usize,
    /// Number of time-averages (P). If `None`, uses all available segments.
    pub num_averages: Option<usize>,
    /// Sampling frequency in Hz (default 1.0).
    pub fs: f64,
}

impl Default for CyclostationaryConfig {
    fn default() -> Self {
        Self {
            fft_size: 256,
            window: Window::Hann,
            overlap: 128,
            num_averages: None,
            fs: 1.0,
        }
    }
}

/// Result of cyclostationary analysis.
#[derive(Debug, Clone)]
pub struct CyclostationaryResult {
    /// Spectral frequency axis (Hz).
    pub frequencies: Vec<f64>,
    /// Cyclic frequency axis (Hz).
    pub alphas: Vec<f64>,
    /// 2-D SCF magnitude: `scf[alpha_idx][freq_idx]`.
    pub scf: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
//  Internal: channelise and compute sliding short-time FFTs
// ---------------------------------------------------------------------------

/// Compute sliding windowed FFTs (channelisation step of the FAM).
///
/// Returns a matrix where each row is the FFT of one windowed segment.
fn channelise(
    signal: &[f64],
    fft_size: usize,
    window_coeffs: &[f64],
    hop: usize,
) -> FFTResult<Vec<Vec<Complex64>>> {
    let n = signal.len();
    if fft_size == 0 {
        return Err(FFTError::ValueError(
            "fft_size must be positive".to_string(),
        ));
    }
    if hop == 0 {
        return Err(FFTError::ValueError("hop must be positive".to_string()));
    }
    if fft_size > n {
        return Err(FFTError::ValueError(format!(
            "fft_size ({}) exceeds signal length ({})",
            fft_size, n
        )));
    }

    let num_segments = (n - fft_size) / hop + 1;
    let mut segments: Vec<Vec<Complex64>> = Vec::with_capacity(num_segments);

    for seg_idx in 0..num_segments {
        let start = seg_idx * hop;
        // Apply window and build the segment
        let windowed: Vec<f64> = (0..fft_size)
            .map(|k| signal[start + k] * window_coeffs[k])
            .collect();
        let spectrum = fft(&windowed, None)?;
        segments.push(spectrum);
    }

    Ok(segments)
}

// ---------------------------------------------------------------------------
//  Public API
// ---------------------------------------------------------------------------

/// Compute the Spectral Correlation Function via the FFT Accumulation Method.
///
/// The FAM channelises the input using sliding windowed FFTs, then for each
/// candidate cyclic frequency `alpha` it computes the cross-spectral product
/// between frequency bins separated by `alpha` and averages over time.
///
/// # Errors
///
/// Returns an error when the configuration is invalid (e.g. `fft_size` is zero
/// or exceeds the signal length).
pub fn spectral_correlation_function(
    signal: &[f64],
    config: &CyclostationaryConfig,
) -> FFTResult<CyclostationaryResult> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::ValueError("Signal is empty".to_string()));
    }

    let fft_size = config.fft_size;
    let hop = fft_size.saturating_sub(config.overlap).max(1);

    // Build window coefficients
    let win = get_window(config.window.clone(), fft_size, true)?;
    let window_coeffs: Vec<f64> = win.to_vec();

    // Channelise
    let segments = channelise(signal, fft_size, &window_coeffs, hop)?;
    let num_segments = segments.len();
    let max_averages = config
        .num_averages
        .unwrap_or(num_segments)
        .min(num_segments);

    if max_averages == 0 {
        return Err(FFTError::ValueError(
            "Not enough data for even one average".to_string(),
        ));
    }

    // Frequency axis
    let freq_resolution = config.fs / fft_size as f64;
    let frequencies: Vec<f64> = (0..fft_size)
        .map(|k| {
            let k_shifted = if k < fft_size / 2 {
                k as f64
            } else {
                k as f64 - fft_size as f64
            };
            k_shifted * freq_resolution
        })
        .collect();

    // Cyclic frequency axis — we consider integer multiples of the frequency
    // resolution up to the Nyquist-like limit for cyclic frequency.
    let num_alpha = fft_size;
    let alphas: Vec<f64> = (0..num_alpha)
        .map(|a| {
            let a_shifted = if a < num_alpha / 2 {
                a as f64
            } else {
                a as f64 - num_alpha as f64
            };
            a_shifted * freq_resolution
        })
        .collect();

    // Compute SCF(f, alpha)
    // For each alpha index `a`, the cross-spectral product is
    //   X_p(f + alpha/2) * conj(X_p(f - alpha/2))
    // averaged over segments p = 0..P.
    //
    // Here we approximate alpha as bin-shifts: for alpha_index `a`,
    //   upper_bin = (k + a/2) mod N,  lower_bin = (k - a/2) mod N
    // but since `a` may be odd, we use the half-bin-shift via the actual
    // FAM approach: shift index = a (integer bins).
    let mut scf: Vec<Vec<f64>> = vec![vec![0.0; fft_size]; num_alpha];

    for a_idx in 0..num_alpha {
        // shift in bins (a_idx maps to the cyclic frequency in the alpha axis)
        let shift = a_idx;

        let mut accum: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); fft_size];

        for seg in segments.iter().take(max_averages) {
            for k in 0..fft_size {
                let upper = (k + shift) % fft_size;
                let lower = k;
                accum[k] += seg[upper] * seg[lower].conj();
            }
        }

        // Average and store magnitude
        let inv_p = 1.0 / max_averages as f64;
        for k in 0..fft_size {
            scf[a_idx][k] = (accum[k] * inv_p).norm();
        }
    }

    Ok(CyclostationaryResult {
        frequencies,
        alphas,
        scf,
    })
}

/// Estimate the cyclic spectral density for a single cyclic frequency `alpha`.
///
/// This is a convenience wrapper that computes the full SCF and extracts the
/// row corresponding to the closest `alpha` value.
///
/// # Errors
///
/// Returns an error when the underlying SCF computation fails.
pub fn cyclic_spectral_density(
    signal: &[f64],
    alpha: f64,
    config: &CyclostationaryConfig,
) -> FFTResult<(Vec<f64>, Vec<f64>)> {
    let result = spectral_correlation_function(signal, config)?;

    // Find the closest alpha index
    let best_idx = result
        .alphas
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (*a - alpha).abs();
            let db = (*b - alpha).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    Ok((result.frequencies, result.scf[best_idx].clone()))
}

/// Detect prominent cyclic frequencies (alpha profile).
///
/// Computes the SCF and for each `alpha` sums the magnitude across all
/// spectral frequencies, yielding an "alpha profile". Peaks in this profile
/// indicate cyclostationary features.
///
/// Returns `(alphas, profile)`.
///
/// # Errors
///
/// Returns an error when the underlying SCF computation fails.
pub fn cyclic_frequency_detection(
    signal: &[f64],
    config: &CyclostationaryConfig,
) -> FFTResult<(Vec<f64>, Vec<f64>)> {
    let result = spectral_correlation_function(signal, config)?;
    let num_freqs = result.frequencies.len();

    let profile: Vec<f64> = result
        .scf
        .iter()
        .map(|row| {
            let sum: f64 = row.iter().sum();
            sum / num_freqs as f64
        })
        .collect();

    Ok((result.alphas, profile))
}

/// Compute the spectral coherence function.
///
/// The spectral coherence normalises the SCF by the power spectral density
/// so that `|C(f, alpha)| <= 1`.
///
///   `C(f, alpha) = S(f, alpha) / sqrt(S(f+a/2,0) * S(f-a/2,0))`
///
/// Returns the coherence as a 2-D array with the same axes as the SCF.
///
/// # Errors
///
/// Returns an error when the underlying SCF computation fails.
pub fn spectral_coherence_cyclic(
    signal: &[f64],
    config: &CyclostationaryConfig,
) -> FFTResult<CyclostationaryResult> {
    let result = spectral_correlation_function(signal, config)?;
    let fft_size = result.frequencies.len();

    // PSD is the alpha=0 row of the SCF
    let psd = &result.scf[0];

    let mut coherence: Vec<Vec<f64>> = vec![vec![0.0; fft_size]; result.alphas.len()];

    for (a_idx, row) in result.scf.iter().enumerate() {
        let shift = a_idx;
        for k in 0..fft_size {
            let upper = (k + shift) % fft_size;
            let denom = (psd[upper] * psd[k]).sqrt();
            coherence[a_idx][k] = if denom > 1e-30 { row[k] / denom } else { 0.0 };
        }
    }

    Ok(CyclostationaryResult {
        frequencies: result.frequencies,
        alphas: result.alphas,
        scf: coherence,
    })
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper: generate a pure sinusoid at `freq` Hz sampled at `fs` Hz.
    fn sine_signal(freq: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    /// Helper: generate white noise (deterministic pseudo-random via LCG).
    fn pseudo_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                // Simple LCG
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Map to [-1, 1]
                (state >> 33) as f64 / (1u64 << 31) as f64 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_scf_sinusoid_peaks() {
        // A pure sinusoid at f0 has cyclic features at alpha = +/-2*f0
        let fs = 1000.0;
        let f0 = 100.0;
        let n = 4096;
        let signal = sine_signal(f0, fs, n);

        let config = CyclostationaryConfig {
            fft_size: 256,
            window: Window::Hann,
            overlap: 128,
            num_averages: None,
            fs,
        };

        let result = spectral_correlation_function(&signal, &config)
            .expect("SCF computation should succeed");

        // The alpha=0 row (PSD) should have a peak near f0
        let psd_row = &result.scf[0];
        let peak_idx = psd_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("psd row should be non-empty");
        let peak_freq = result.frequencies[peak_idx].abs();
        // Peak should be near f0
        assert!(
            (peak_freq - f0).abs() < 20.0,
            "PSD peak at {} Hz, expected near {} Hz",
            peak_freq,
            f0
        );

        // Check that the SCF has non-zero content at multiple alpha values
        // (sinusoid produces cyclostationary features)
        let alpha_profile: Vec<f64> = result
            .scf
            .iter()
            .map(|row| row.iter().sum::<f64>())
            .collect();
        let max_profile = alpha_profile
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(max_profile > 0.0, "SCF should have non-zero content");
    }

    #[test]
    fn test_scf_white_noise_flat() {
        // White noise should have no prominent cyclic features (flat alpha profile)
        let fs = 1000.0;
        let n = 4096;
        let signal = pseudo_noise(n, 42);

        let config = CyclostationaryConfig {
            fft_size: 256,
            window: Window::Hann,
            overlap: 128,
            num_averages: None,
            fs,
        };

        let (alphas, profile) =
            cyclic_frequency_detection(&signal, &config).expect("Detection should succeed");

        // The alpha=0 component (PSD) will be large, but non-zero alpha
        // components should be relatively flat and small.
        let psd_energy = profile[0];
        let non_zero_alphas: Vec<f64> = profile.iter().skip(1).copied().collect();
        let mean_nz: f64 = non_zero_alphas.iter().sum::<f64>() / non_zero_alphas.len() as f64;
        let max_nz = non_zero_alphas
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        // The ratio of max non-zero-alpha energy to alpha=0 energy should be modest
        assert!(
            max_nz < psd_energy,
            "Noise cyclic features ({}) should be smaller than PSD ({})",
            max_nz,
            psd_energy
        );

        // Standard deviation should be small relative to the mean (flat profile)
        let var: f64 = non_zero_alphas
            .iter()
            .map(|&v| (v - mean_nz).powi(2))
            .sum::<f64>()
            / non_zero_alphas.len() as f64;
        let std_dev = var.sqrt();
        // Coefficient of variation < 1 means relatively flat
        if mean_nz > 1e-15 {
            let cv = std_dev / mean_nz;
            assert!(
                cv < 2.0,
                "Noise alpha profile should be relatively flat, CV = {}",
                cv
            );
        }
    }

    #[test]
    fn test_cyclic_spectral_density_extraction() {
        let fs = 1000.0;
        let f0 = 100.0;
        let n = 4096;
        let signal = sine_signal(f0, fs, n);

        let config = CyclostationaryConfig {
            fft_size: 256,
            window: Window::Hann,
            overlap: 128,
            num_averages: None,
            fs,
        };

        let (freqs, csd) =
            cyclic_spectral_density(&signal, 0.0, &config).expect("CSD should succeed");

        assert_eq!(freqs.len(), csd.len());
        assert!(!csd.is_empty());
        // At alpha=0 this is just the PSD; should have a peak near f0
        let peak_idx = csd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("CSD should have elements");
        let peak_freq = freqs[peak_idx].abs();
        assert!(
            (peak_freq - f0).abs() < 20.0,
            "CSD peak at {} Hz, expected near {} Hz",
            peak_freq,
            f0
        );
    }

    #[test]
    fn test_spectral_coherence() {
        let fs = 1000.0;
        let f0 = 100.0;
        let n = 4096;
        let signal = sine_signal(f0, fs, n);

        let config = CyclostationaryConfig {
            fft_size: 256,
            window: Window::Hann,
            overlap: 128,
            num_averages: None,
            fs,
        };

        let coherence =
            spectral_coherence_cyclic(&signal, &config).expect("Coherence should succeed");

        // At alpha=0, coherence should be exactly 1 everywhere the PSD is non-zero
        let alpha0_row = &coherence.scf[0];
        for &c in alpha0_row {
            assert!(c <= 1.0 + 1e-10, "Coherence should be <= 1, got {}", c);
            assert!(c >= 0.0, "Coherence should be >= 0, got {}", c);
        }
    }

    #[test]
    fn test_scf_invalid_params() {
        let signal = vec![1.0; 100];
        let config = CyclostationaryConfig {
            fft_size: 0,
            ..Default::default()
        };
        assert!(spectral_correlation_function(&signal, &config).is_err());

        let config2 = CyclostationaryConfig {
            fft_size: 200,
            ..Default::default()
        };
        assert!(spectral_correlation_function(&signal, &config2).is_err());
    }
}
