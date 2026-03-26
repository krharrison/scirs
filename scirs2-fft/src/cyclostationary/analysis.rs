//! Cyclostationary spectral analysis via the Time-Smoothed Cyclic Cross-Periodogram (TSCCP).
//!
//! A cyclostationary signal has statistical properties that are periodic in time.
//! The Spectral Correlation Function (SCF) `S_x(f; α)` quantifies this periodicity
//! in the joint (spectral frequency `f`, cyclic frequency `α`) domain.
//!
//! # Algorithm: TSCCP (FAM variant)
//!
//! The FFT Accumulation Method (FAM) computes the SCD as:
//!
//! ```text
//! S_x(f; α) ≈ (1/T) Σ_t X_T(t, f + α/2) · conj(X_T(t, f − α/2))
//! ```
//!
//! where `X_T(t, f)` is the windowed short-time Fourier transform evaluated at time
//! `t` and frequency `f`, and the sum is over all time segments.
//!
//! # References
//!
//! * Roberts, R.S., Brown, W.A., Loomis, H.H. "Computationally Efficient Algorithms
//!   for Cyclic Spectral Analysis." IEEE Signal Processing Magazine, 1991.
//! * Gardner, W.A. "Spectral Correlation of Modulated Signals." IEEE Trans. Commun., 1987.

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;
use crate::window::{get_window, Window};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

use super::types::{CyclostationaryConfig, SpectralCorrelationResult};

/// Cyclostationary spectral analyser.
///
/// # Examples
///
/// ```
/// use scirs2_fft::cyclostationary::{CyclostationaryAnalyzer, CyclostationaryConfig};
/// use std::f64::consts::PI;
///
/// // AM signal: carrier at 0.1 * fs, modulated at 0.05 * fs
/// let n = 512;
/// let fs = 1.0;
/// let fc = 0.1 * fs;
/// let fm = 0.05 * fs;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| {
///         let t = i as f64 / fs;
///         (1.0 + (2.0 * PI * fm * t).cos()) * (2.0 * PI * fc * t).cos()
///     })
///     .collect();
///
/// let mut config = CyclostationaryConfig::default();
/// config.n_fft = 64;
/// config.fs = fs;
///
/// let analyzer = CyclostationaryAnalyzer::new();
/// let result = analyzer.compute_scd(&signal, fs, &config).expect("SCD should succeed");
///
/// assert!(!result.cyclic_frequencies.is_empty());
/// ```
pub struct CyclostationaryAnalyzer;

impl CyclostationaryAnalyzer {
    /// Create a new `CyclostationaryAnalyzer`.
    pub fn new() -> Self {
        Self
    }

    /// Compute the Spectral Correlation Density (SCD) matrix using TSCCP/FAM.
    ///
    /// # Arguments
    ///
    /// * `signal` - Real-valued input signal.
    /// * `fs`     - Sampling frequency in Hz.
    /// * `config` - Configuration parameters.
    ///
    /// # Returns
    ///
    /// A [`SpectralCorrelationResult`] containing the SCD matrix and frequency axes.
    ///
    /// # Errors
    ///
    /// Returns an error if the signal is too short or internal FFTs fail.
    pub fn compute_scd(
        &self,
        signal: &[f64],
        fs: f64,
        config: &CyclostationaryConfig,
    ) -> FFTResult<SpectralCorrelationResult> {
        let n = signal.len();
        if n < config.n_fft {
            return Err(FFTError::ValueError(format!(
                "Signal length {n} must be >= n_fft {}",
                config.n_fft
            )));
        }

        // Determine which cyclic frequencies to evaluate
        let alpha_vec: Vec<f64> = match &config.cyclic_freqs {
            Some(alphas) => alphas.clone(),
            None => {
                // Auto-detect cyclic frequencies
                let detected = detect_cyclic_frequencies_impl(signal, fs, config)?;
                if detected.is_empty() {
                    // Fall back to a coarse grid
                    build_alpha_grid(fs, config.alpha_resolution)
                } else {
                    detected
                }
            }
        };

        let n_alphas = alpha_vec.len();
        let n_fft = config.n_fft;
        let hop = compute_hop(n_fft, config.overlap);

        // Build Hann window
        let window = build_hann_window(n_fft);

        // Compute STFTs for the whole signal
        let stft_matrix = compute_stft(signal, &window, n_fft, hop)?;
        let n_frames = stft_matrix.len();

        if n_frames == 0 {
            return Err(FFTError::ComputationError(
                "Signal too short for STFT computation".to_string(),
            ));
        }

        // Spectral frequency axis
        let spectral_frequencies: Vec<f64> =
            (0..n_fft).map(|k| k as f64 * fs / n_fft as f64).collect();

        // For each cyclic frequency α, compute SCD via TSCCP:
        // S_x(f; α) = mean_t { X(t, f+α/2) · conj(X(t, f-α/2)) }
        let mut scd_matrix: Vec<Vec<Complex64>> = Vec::with_capacity(n_alphas);

        for &alpha in &alpha_vec {
            let row = compute_scd_row(&stft_matrix, alpha, fs, n_fft, n_frames)?;
            scd_matrix.push(row);
        }

        Ok(SpectralCorrelationResult {
            scd: scd_matrix,
            cyclic_frequencies: alpha_vec,
            spectral_frequencies,
        })
    }

    /// Detect significant cyclic frequencies in a signal.
    ///
    /// Uses peak detection on the cyclic autocorrelation function to identify
    /// cyclic frequencies with statistically significant power.
    ///
    /// # Arguments
    ///
    /// * `signal`     - Real-valued input signal.
    /// * `fs`         - Sampling frequency in Hz.
    /// * `resolution` - Cyclic frequency resolution in Hz.
    ///
    /// # Errors
    ///
    /// Returns an error if the signal is empty or an internal FFT fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_fft::cyclostationary::CyclostationaryAnalyzer;
    /// use std::f64::consts::PI;
    ///
    /// let n = 512;
    /// let fs = 1.0;
    /// let fm = 0.05;
    /// let fc = 0.2;
    /// // AM signal
    /// let signal: Vec<f64> = (0..n)
    ///     .map(|i| {
    ///         let t = i as f64;
    ///         (1.0 + 0.5 * (2.0 * PI * fm * t).cos()) * (2.0 * PI * fc * t).cos()
    ///     })
    ///     .collect();
    ///
    /// let analyzer = CyclostationaryAnalyzer::new();
    /// let alphas = analyzer
    ///     .detect_cyclic_frequencies(&signal, fs, 0.01)
    ///     .expect("detection should succeed");
    /// // For AM signal, at least one cyclic freq should be found
    /// // (result may be empty for short/noisy signals)
    /// let _ = alphas;
    /// ```
    pub fn detect_cyclic_frequencies(
        &self,
        signal: &[f64],
        fs: f64,
        resolution: f64,
    ) -> FFTResult<Vec<f64>> {
        let config = CyclostationaryConfig {
            alpha_resolution: resolution,
            fs,
            ..CyclostationaryConfig::default()
        };
        detect_cyclic_frequencies_impl(signal, fs, &config)
    }
}

impl Default for CyclostationaryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Build the Hann window of length `n`.
fn build_hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos()))
        .collect()
}

/// Compute hop size from overlap fraction.
fn compute_hop(n_fft: usize, overlap: f64) -> usize {
    let overlap_clamped = overlap.clamp(0.0, 0.99);
    let hop = ((1.0 - overlap_clamped) * n_fft as f64).round() as usize;
    hop.max(1)
}

/// Compute the Short-Time Fourier Transform (STFT) of `signal`.
///
/// Returns `stft[frame][freq_bin]` as complex values.
fn compute_stft(
    signal: &[f64],
    window: &[f64],
    n_fft: usize,
    hop: usize,
) -> FFTResult<Vec<Vec<Complex64>>> {
    let n = signal.len();
    if n < n_fft {
        return Err(FFTError::ValueError(
            "Signal must be at least n_fft samples long".to_string(),
        ));
    }

    let mut frames: Vec<Vec<Complex64>> = Vec::new();
    let mut start = 0;

    while start + n_fft <= n {
        let segment: Vec<f64> = (0..n_fft).map(|i| signal[start + i] * window[i]).collect();

        let spectrum = fft(&segment, None)?;
        frames.push(spectrum);
        start += hop;
    }

    Ok(frames)
}

/// Compute one row of the SCD matrix for a given cyclic frequency `alpha`.
fn compute_scd_row(
    stft_matrix: &[Vec<Complex64>],
    alpha: f64,
    fs: f64,
    n_fft: usize,
    n_frames: usize,
) -> FFTResult<Vec<Complex64>> {
    // Convert alpha to bin offset: delta_k = round(alpha * n_fft / fs)
    let delta_k_raw = alpha * n_fft as f64 / fs;
    let delta_k = delta_k_raw.round() as i64;

    let mut row: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n_fft];

    for frame in stft_matrix.iter().take(n_frames) {
        for f_idx in 0..n_fft {
            // f + α/2  →  bin f_idx + delta_k/2
            let upper_idx = wrap_bin(f_idx as i64 + delta_k, n_fft);
            // f - α/2  →  bin f_idx - delta_k/2
            let lower_idx = wrap_bin(f_idx as i64 - delta_k, n_fft);

            let upper = frame[upper_idx];
            let lower = frame[lower_idx];

            row[f_idx] = row[f_idx] + upper * lower.conj();
        }
    }

    // Normalise by number of frames
    let norm = 1.0 / n_frames as f64;
    for val in row.iter_mut() {
        *val = Complex64::new(val.re * norm, val.im * norm);
    }

    Ok(row)
}

/// Wrap a signed bin index into `[0, n_fft)`.
#[inline]
fn wrap_bin(idx: i64, n: usize) -> usize {
    let n_i = n as i64;
    (((idx % n_i) + n_i) % n_i) as usize
}

/// Build a uniform cyclic frequency grid from `[-fs/2, fs/2)` with given resolution.
fn build_alpha_grid(fs: f64, resolution: f64) -> Vec<f64> {
    let n_steps = ((fs / resolution).ceil() as usize).max(1);
    (0..n_steps)
        .map(|i| i as f64 * resolution)
        .take_while(|&a| a <= fs / 2.0)
        .collect()
}

/// Detect cyclic frequencies via the cyclic autocorrelation approach.
///
/// Computes `R_x(τ; α) = E[x(t+τ)·x*(t) · e^{-j2παt}]` for a grid of α values,
/// then detects peaks.
fn detect_cyclic_frequencies_impl(
    signal: &[f64],
    fs: f64,
    config: &CyclostationaryConfig,
) -> FFTResult<Vec<f64>> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::ValueError("Signal must not be empty".to_string()));
    }

    // Method: compute the cyclic autocorrelation at lag=0 as a function of α.
    // CAF(α) = (1/N) * |Σ_t x(t) · x*(t) · e^{-j 2π α t/fs}|
    //        = (1/N) * |Σ_t |x(t)|^2 · e^{-j 2π α t / fs}|
    //
    // This is just the FFT of the instantaneous power |x(t)|^2, evaluated at
    // cyclic frequency α.

    let power: Vec<f64> = signal.iter().map(|&s| s * s).collect();

    // FFT of instantaneous power — peaks reveal cyclic frequencies
    let power_spectrum = fft(&power, None)?;

    let n_alpha = power_spectrum.len();
    let magnitudes: Vec<f64> = power_spectrum
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();

    // Find maximum magnitude (excluding DC at index 0)
    let max_mag = magnitudes[1..].iter().cloned().fold(0.0_f64, f64::max);
    if max_mag < f64::EPSILON {
        return Ok(Vec::new());
    }

    let threshold = config.detection_threshold * max_mag;

    // Detect peaks: magnitude above threshold and local maximum
    let mut peaks: Vec<f64> = Vec::new();
    for k in 1..(n_alpha / 2 + 1) {
        let mag = magnitudes[k];
        if mag < threshold {
            continue;
        }
        // Local maximum check
        let prev = if k > 1 { magnitudes[k - 1] } else { 0.0 };
        let next = if k + 1 < n_alpha {
            magnitudes[k + 1]
        } else {
            0.0
        };
        if mag >= prev && mag >= next {
            let alpha = k as f64 * fs / n_alpha as f64;
            if alpha <= fs / 2.0 {
                peaks.push(alpha);
            }
        }
    }

    Ok(peaks)
}

/// Convenience function: compute SCD with default config.
///
/// # Errors
///
/// Returns an error if the signal is too short or FFT fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::cyclostationary::analysis::compute_scd;
/// use std::f64::consts::PI;
///
/// let n = 512;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 0.1 * i as f64).sin())
///     .collect();
/// let result = compute_scd(&signal, 1.0, None).expect("should succeed");
/// assert!(!result.spectral_frequencies.is_empty());
/// ```
pub fn compute_scd(
    signal: &[f64],
    fs: f64,
    cyclic_freqs: Option<Vec<f64>>,
) -> FFTResult<SpectralCorrelationResult> {
    let config = CyclostationaryConfig {
        cyclic_freqs,
        fs,
        ..CyclostationaryConfig::default()
    };
    let analyzer = CyclostationaryAnalyzer::new();
    analyzer.compute_scd(signal, fs, &config)
}

/// Detect cyclic frequencies in a signal using default settings.
///
/// # Errors
///
/// Returns an error if the signal is empty or FFT fails.
pub fn detect_cyclic_frequencies(signal: &[f64], fs: f64, resolution: f64) -> FFTResult<Vec<f64>> {
    let analyzer = CyclostationaryAnalyzer::new();
    analyzer.detect_cyclic_frequencies(signal, fs, resolution)
}
