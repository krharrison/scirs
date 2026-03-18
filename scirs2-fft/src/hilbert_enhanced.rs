//! Enhanced Hilbert Transform module with Hilbert-Huang Transform (HHT)
//!
//! This module provides advanced Hilbert transform capabilities including:
//!
//! - **Empirical Mode Decomposition (EMD)**: Adaptive signal decomposition into
//!   Intrinsic Mode Functions (IMFs)
//! - **Hilbert-Huang Transform**: EMD + Hilbert spectral analysis for nonlinear
//!   and nonstationary time-frequency analysis
//! - **Marginal Hilbert spectrum**: Time-integrated spectral energy distribution
//! - **Instantaneous energy density**: Hilbert energy at each time-frequency point
//! - **Hilbert spectral analysis**: Full time-frequency-energy representation
//!
//! # Mathematical Background
//!
//! The Hilbert-Huang Transform (HHT) combines EMD with the Hilbert transform to
//! produce an adaptive time-frequency representation. Unlike STFT and wavelet
//! transforms, the HHT does not rely on predetermined basis functions.
//!
//! EMD decomposes a signal x(t) into a finite set of Intrinsic Mode Functions (IMFs)
//! c_i(t) plus a residual r(t):
//!   x(t) = sum_i c_i(t) + r(t)
//!
//! Each IMF satisfies two conditions:
//! 1. The number of extrema and zero crossings differ by at most one
//! 2. The mean of the upper and lower envelopes is approximately zero
//!
//! # References
//!
//! * Huang, N. E. et al. "The empirical mode decomposition and the Hilbert
//!   spectrum for nonlinear and non-stationary time series analysis."
//!   Proc. R. Soc. Lond. A, 454, 903-995, 1998.
//! * Huang, N. E. & Wu, Z. "A review on Hilbert-Huang Transform: Method and
//!   its applications to geophysical studies." Rev. Geophys., 46, 2008.

use crate::error::{FFTError, FFTResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ============================================================================
// EMD configuration and types
// ============================================================================

/// Configuration for Empirical Mode Decomposition
#[derive(Debug, Clone)]
pub struct EMDConfig {
    /// Maximum number of IMFs to extract
    pub max_imfs: usize,
    /// Maximum sifting iterations per IMF
    pub max_sift_iterations: usize,
    /// Cauchy convergence threshold for sifting
    pub sift_threshold: f64,
    /// Number of envelope evaluations for S-number stopping criterion
    pub s_number: usize,
    /// Method for envelope interpolation
    pub envelope_method: EnvelopeMethod,
}

impl Default for EMDConfig {
    fn default() -> Self {
        Self {
            max_imfs: 20,
            max_sift_iterations: 500,
            sift_threshold: 0.05,
            s_number: 4,
            envelope_method: EnvelopeMethod::CubicSpline,
        }
    }
}

/// Envelope interpolation method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnvelopeMethod {
    /// Cubic spline interpolation (standard)
    CubicSpline,
    /// Linear interpolation (faster but less smooth)
    Linear,
}

/// Result of Empirical Mode Decomposition
#[derive(Debug, Clone)]
pub struct EMDResult {
    /// Intrinsic Mode Functions (IMFs), ordered from highest to lowest frequency
    pub imfs: Vec<Vec<f64>>,
    /// Residual (monotonic trend)
    pub residual: Vec<f64>,
    /// Number of sifting iterations for each IMF
    pub iterations: Vec<usize>,
}

/// Result of Hilbert-Huang Transform
#[derive(Debug, Clone)]
pub struct HHTResult {
    /// IMFs from EMD
    pub imfs: Vec<Vec<f64>>,
    /// Instantaneous frequencies for each IMF (Hz)
    pub inst_frequencies: Vec<Vec<f64>>,
    /// Instantaneous amplitudes (envelopes) for each IMF
    pub inst_amplitudes: Vec<Vec<f64>>,
    /// Residual signal
    pub residual: Vec<f64>,
}

/// Result of Hilbert spectral analysis
#[derive(Debug, Clone)]
pub struct HilbertSpectrum {
    /// Time axis values
    pub times: Vec<f64>,
    /// Frequency axis values (Hz)
    pub frequencies: Vec<f64>,
    /// Energy density matrix (time x frequency)
    pub energy: Vec<Vec<f64>>,
}

// ============================================================================
// Empirical Mode Decomposition (EMD)
// ============================================================================

/// Perform Empirical Mode Decomposition (EMD) on a signal
///
/// Decomposes a signal into a set of Intrinsic Mode Functions (IMFs) and a
/// residual, using the sifting process.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `config` - Optional EMD configuration (uses defaults if None)
///
/// # Returns
///
/// EMDResult containing IMFs, residual, and iteration counts.
///
/// # Errors
///
/// Returns an error if the signal is too short for decomposition.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hilbert_enhanced::{emd, EMDConfig};
/// use std::f64::consts::PI;
///
/// let n = 256;
/// let signal: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 / 256.0;
///     (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 30.0 * t).sin()
/// }).collect();
///
/// let result = emd(&signal, None).expect("EMD should succeed");
/// assert!(!result.imfs.is_empty(), "Should extract at least one IMF");
/// ```
pub fn emd(signal: &[f64], config: Option<EMDConfig>) -> FFTResult<EMDResult> {
    let n = signal.len();
    if n < 4 {
        return Err(FFTError::ValueError(
            "Signal must have at least 4 samples for EMD".to_string(),
        ));
    }

    let cfg = config.unwrap_or_default();
    let mut imfs = Vec::new();
    let mut iterations_list = Vec::new();
    let mut residual = signal.to_vec();

    for _imf_idx in 0..cfg.max_imfs {
        // Check if residual is monotonic or has too few extrema
        let extrema = count_extrema(&residual);
        if extrema < 2 {
            break;
        }

        // Sifting process to extract one IMF
        let (imf, iters) = sift_imf(&residual, &cfg)?;

        // Subtract IMF from residual
        for i in 0..n {
            residual[i] -= imf[i];
        }

        imfs.push(imf);
        iterations_list.push(iters);

        // Check if residual is negligible or monotonic
        let residual_energy: f64 = residual.iter().map(|&v| v * v).sum();
        let signal_energy: f64 = signal.iter().map(|&v| v * v).sum();
        if signal_energy > 0.0 && residual_energy / signal_energy < 1e-12 {
            break;
        }

        // Check if residual has enough extrema to continue
        if count_extrema(&residual) < 2 {
            break;
        }
    }

    Ok(EMDResult {
        imfs,
        residual,
        iterations: iterations_list,
    })
}

/// Sifting process to extract one IMF from a signal
fn sift_imf(signal: &[f64], config: &EMDConfig) -> FFTResult<(Vec<f64>, usize)> {
    let n = signal.len();
    let mut h = signal.to_vec();
    let mut prev_h = h.clone();
    let mut s_count = 0;

    for iteration in 0..config.max_sift_iterations {
        // Find local maxima and minima
        let (max_pos, max_val) = find_local_maxima(&h);
        let (min_pos, min_val) = find_local_minima(&h);

        // Need at least 2 maxima and 2 minima for envelope interpolation
        if max_pos.len() < 2 || min_pos.len() < 2 {
            return Ok((h, iteration + 1));
        }

        // Compute upper and lower envelopes
        let upper_env = interpolate_envelope(&max_pos, &max_val, n, config.envelope_method)?;
        let lower_env = interpolate_envelope(&min_pos, &min_val, n, config.envelope_method)?;

        // Compute mean envelope
        let mean_env: Vec<f64> = upper_env
            .iter()
            .zip(lower_env.iter())
            .map(|(&u, &l)| (u + l) / 2.0)
            .collect();

        // Subtract mean from current candidate
        for i in 0..n {
            h[i] -= mean_env[i];
        }

        // Check Cauchy convergence criterion
        let diff_energy: f64 = h
            .iter()
            .zip(prev_h.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum();
        let h_energy: f64 = prev_h.iter().map(|&v| v * v).sum();

        if h_energy > 0.0 {
            let sd = diff_energy / h_energy;
            if sd < config.sift_threshold {
                // Check S-number criterion
                s_count += 1;
                if s_count >= config.s_number {
                    return Ok((h, iteration + 1));
                }
            } else {
                s_count = 0;
            }
        }

        prev_h.clone_from(&h);
    }

    Ok((h, config.max_sift_iterations))
}

/// Count the number of local extrema (maxima + minima) in a signal
fn count_extrema(signal: &[f64]) -> usize {
    if signal.len() < 3 {
        return 0;
    }

    let mut count = 0;
    for i in 1..signal.len() - 1 {
        if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1])
            || (signal[i] < signal[i - 1] && signal[i] < signal[i + 1])
        {
            count += 1;
        }
    }
    count
}

/// Find positions and values of local maxima
fn find_local_maxima(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut positions = Vec::new();
    let mut values = Vec::new();

    // Add endpoints for boundary handling
    positions.push(0.0);
    values.push(signal[0]);

    for i in 1..signal.len() - 1 {
        if signal[i] >= signal[i - 1] && signal[i] >= signal[i + 1] {
            // Exclude flat plateaus unless at actual peak
            if signal[i] > signal[i - 1] || signal[i] > signal[i + 1] {
                positions.push(i as f64);
                values.push(signal[i]);
            }
        }
    }

    // Add endpoint
    let last = signal.len() - 1;
    positions.push(last as f64);
    values.push(signal[last]);

    (positions, values)
}

/// Find positions and values of local minima
fn find_local_minima(signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut positions = Vec::new();
    let mut values = Vec::new();

    // Add endpoints
    positions.push(0.0);
    values.push(signal[0]);

    for i in 1..signal.len() - 1 {
        if signal[i] <= signal[i - 1]
            && signal[i] <= signal[i + 1]
            && (signal[i] < signal[i - 1] || signal[i] < signal[i + 1])
        {
            positions.push(i as f64);
            values.push(signal[i]);
        }
    }

    let last = signal.len() - 1;
    positions.push(last as f64);
    values.push(signal[last]);

    (positions, values)
}

/// Interpolate an envelope through given points
fn interpolate_envelope(
    positions: &[f64],
    values: &[f64],
    n: usize,
    method: EnvelopeMethod,
) -> FFTResult<Vec<f64>> {
    if positions.len() < 2 {
        return Err(FFTError::ValueError(
            "Need at least 2 points for envelope interpolation".to_string(),
        ));
    }

    match method {
        EnvelopeMethod::CubicSpline => cubic_spline_interpolate(positions, values, n),
        EnvelopeMethod::Linear => linear_interpolate(positions, values, n),
    }
}

/// Cubic spline interpolation (natural boundary conditions)
fn cubic_spline_interpolate(x_knots: &[f64], y_knots: &[f64], n_out: usize) -> FFTResult<Vec<f64>> {
    let m = x_knots.len();
    if m < 2 {
        return Err(FFTError::ValueError(
            "Need at least 2 knots for spline".to_string(),
        ));
    }

    if m == 2 {
        // Fall back to linear for just 2 points
        return linear_interpolate(x_knots, y_knots, n_out);
    }

    // Compute h[i] = x[i+1] - x[i]
    let mut h = Vec::with_capacity(m - 1);
    for i in 0..m - 1 {
        let hi = x_knots[i + 1] - x_knots[i];
        if hi <= 0.0 {
            // Non-monotonic x values; fall back to linear
            return linear_interpolate(x_knots, y_knots, n_out);
        }
        h.push(hi);
    }

    // Set up tridiagonal system for natural spline (c_0 = c_{m-1} = 0)
    let n_eqs = m - 2;
    if n_eqs == 0 {
        return linear_interpolate(x_knots, y_knots, n_out);
    }

    let mut diag = vec![0.0; n_eqs];
    let mut upper = vec![0.0; n_eqs.saturating_sub(1)];
    let mut lower = vec![0.0; n_eqs.saturating_sub(1)];
    let mut rhs = vec![0.0; n_eqs];

    for i in 0..n_eqs {
        diag[i] = 2.0 * (h[i] + h[i + 1]);
        rhs[i] = 3.0
            * ((y_knots[i + 2] - y_knots[i + 1]) / h[i + 1] - (y_knots[i + 1] - y_knots[i]) / h[i]);
    }

    let sub = n_eqs.saturating_sub(1);
    upper[..sub].copy_from_slice(&h[1..(sub + 1)]);
    lower[..sub].copy_from_slice(&h[1..(sub + 1)]);

    // Solve tridiagonal system (Thomas algorithm)
    let c_interior = solve_tridiagonal(&lower, &diag, &upper, &rhs)?;

    // Full c array with boundary conditions c[0] = c[m-1] = 0
    let mut c = vec![0.0; m];
    c[1..(n_eqs + 1)].copy_from_slice(&c_interior[..n_eqs]);

    // Compute b and d coefficients
    let mut b = vec![0.0; m - 1];
    let mut d = vec![0.0; m - 1];

    for i in 0..m - 1 {
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i]);
        b[i] = (y_knots[i + 1] - y_knots[i]) / h[i] - h[i] * (2.0 * c[i] + c[i + 1]) / 3.0;
    }

    // Evaluate spline at each output point
    let mut result = Vec::with_capacity(n_out);
    for t_idx in 0..n_out {
        let t = t_idx as f64;

        // Find the correct spline segment
        let seg = find_segment(x_knots, t);
        let dx = t - x_knots[seg];

        let val = y_knots[seg] + b[seg] * dx + c[seg] * dx * dx + d[seg] * dx * dx * dx;
        result.push(val);
    }

    Ok(result)
}

/// Find the segment index for interpolation
fn find_segment(x_knots: &[f64], t: f64) -> usize {
    if t <= x_knots[0] {
        return 0;
    }
    for i in 0..x_knots.len() - 1 {
        if t >= x_knots[i] && t < x_knots[i + 1] {
            return i;
        }
    }
    // Clamp to last segment
    x_knots.len().saturating_sub(2)
}

/// Solve tridiagonal system Ax = rhs using Thomas algorithm
fn solve_tridiagonal(
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> FFTResult<Vec<f64>> {
    let n = diag.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if n == 1 {
        if diag[0].abs() < 1e-15 {
            return Err(FFTError::ComputationError(
                "Singular tridiagonal system".to_string(),
            ));
        }
        return Ok(vec![rhs[0] / diag[0]]);
    }

    // Forward elimination
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    if diag[0].abs() < 1e-15 {
        return Err(FFTError::ComputationError(
            "Zero pivot in tridiagonal solve".to_string(),
        ));
    }

    c_prime[0] = upper[0] / diag[0];
    d_prime[0] = rhs[0] / diag[0];

    for i in 1..n {
        let l_val = if i > 0 && i - 1 < lower.len() {
            lower[i - 1]
        } else {
            0.0
        };
        let denom = diag[i] - l_val * c_prime[i - 1];
        if denom.abs() < 1e-15 {
            return Err(FFTError::ComputationError(
                "Near-singular tridiagonal system".to_string(),
            ));
        }
        c_prime[i] = if i < n - 1 && i < upper.len() {
            upper[i] / denom
        } else {
            0.0
        };
        d_prime[i] = (rhs[i] - l_val * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![0.0; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Ok(x)
}

/// Linear interpolation
fn linear_interpolate(x_knots: &[f64], y_knots: &[f64], n_out: usize) -> FFTResult<Vec<f64>> {
    let m = x_knots.len();
    let mut result = Vec::with_capacity(n_out);

    for t_idx in 0..n_out {
        let t = t_idx as f64;

        if t <= x_knots[0] {
            result.push(y_knots[0]);
        } else if t >= x_knots[m - 1] {
            result.push(y_knots[m - 1]);
        } else {
            let seg = find_segment(x_knots, t);
            let frac = (t - x_knots[seg]) / (x_knots[seg + 1] - x_knots[seg]);
            let val = y_knots[seg] + frac * (y_knots[seg + 1] - y_knots[seg]);
            result.push(val);
        }
    }

    Ok(result)
}

// ============================================================================
// Hilbert-Huang Transform (HHT)
// ============================================================================

/// Perform the Hilbert-Huang Transform on a signal
///
/// Combines EMD with Hilbert spectral analysis to produce a time-frequency
/// representation that adapts to the signal's local structure.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `fs` - Sampling frequency in Hz
/// * `config` - Optional EMD configuration
///
/// # Returns
///
/// HHTResult containing IMFs, instantaneous frequencies and amplitudes.
///
/// # Errors
///
/// Returns an error if the signal is too short or fs is non-positive.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hilbert_enhanced::{hht, EMDConfig};
/// use std::f64::consts::PI;
///
/// let fs = 256.0;
/// let n = 256;
/// let signal: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 / fs;
///     (2.0 * PI * 10.0 * t).sin() + 0.3 * (2.0 * PI * 40.0 * t).sin()
/// }).collect();
///
/// let result = hht(&signal, fs, None).expect("HHT should succeed");
/// assert!(!result.imfs.is_empty());
/// assert_eq!(result.inst_frequencies.len(), result.imfs.len());
/// ```
pub fn hht(signal: &[f64], fs: f64, config: Option<EMDConfig>) -> FFTResult<HHTResult> {
    if fs <= 0.0 {
        return Err(FFTError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    // Perform EMD
    let emd_result = emd(signal, config)?;

    // Compute Hilbert transform for each IMF
    let mut inst_frequencies = Vec::with_capacity(emd_result.imfs.len());
    let mut inst_amplitudes = Vec::with_capacity(emd_result.imfs.len());

    for imf in &emd_result.imfs {
        // Compute analytic signal
        let analytic = crate::hilbert::analytic_signal(imf)?;

        // Instantaneous amplitude (envelope)
        let amplitude: Vec<f64> = analytic.iter().map(|c| c.norm()).collect();

        // Instantaneous phase (unwrapped)
        let phase = unwrap_phase_vec(&analytic);

        // Instantaneous frequency via finite differences
        let mut freq = Vec::with_capacity(phase.len());
        for i in 0..phase.len() {
            if i == 0 {
                // Forward difference at start
                if phase.len() > 1 {
                    freq.push((phase[1] - phase[0]) * fs / (2.0 * PI));
                } else {
                    freq.push(0.0);
                }
            } else if i == phase.len() - 1 {
                // Backward difference at end
                freq.push((phase[i] - phase[i - 1]) * fs / (2.0 * PI));
            } else {
                // Central difference in the middle
                freq.push((phase[i + 1] - phase[i - 1]) * fs / (4.0 * PI));
            }
        }

        // Clamp frequencies to valid range [0, fs/2]
        let nyquist = fs / 2.0;
        for f in &mut freq {
            if *f < 0.0 {
                *f = 0.0;
            }
            if *f > nyquist {
                *f = nyquist;
            }
        }

        inst_frequencies.push(freq);
        inst_amplitudes.push(amplitude);
    }

    Ok(HHTResult {
        imfs: emd_result.imfs,
        inst_frequencies,
        inst_amplitudes,
        residual: emd_result.residual,
    })
}

/// Compute the Hilbert spectrum (time-frequency-energy representation)
///
/// Generates a 2D energy density map over time and frequency from the HHT.
///
/// # Arguments
///
/// * `hht_result` - Result from `hht()`
/// * `fs` - Sampling frequency
/// * `n_freq_bins` - Number of frequency bins
///
/// # Returns
///
/// HilbertSpectrum containing the time-frequency energy density.
///
/// # Errors
///
/// Returns an error if inputs are invalid.
pub fn hilbert_spectrum(
    hht_result: &HHTResult,
    fs: f64,
    n_freq_bins: usize,
) -> FFTResult<HilbertSpectrum> {
    if fs <= 0.0 {
        return Err(FFTError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }
    if n_freq_bins == 0 {
        return Err(FFTError::ValueError(
            "Number of frequency bins must be positive".to_string(),
        ));
    }

    let n_time = if let Some(first_imf) = hht_result.imfs.first() {
        first_imf.len()
    } else {
        return Ok(HilbertSpectrum {
            times: Vec::new(),
            frequencies: Vec::new(),
            energy: Vec::new(),
        });
    };

    let nyquist = fs / 2.0;
    let freq_step = nyquist / n_freq_bins as f64;

    // Build time and frequency axes
    let times: Vec<f64> = (0..n_time).map(|i| i as f64 / fs).collect();
    let frequencies: Vec<f64> = (0..n_freq_bins)
        .map(|k| (k as f64 + 0.5) * freq_step)
        .collect();

    // Build energy density matrix
    let mut energy = vec![vec![0.0; n_freq_bins]; n_time];

    for imf_idx in 0..hht_result.imfs.len() {
        let freqs = &hht_result.inst_frequencies[imf_idx];
        let amps = &hht_result.inst_amplitudes[imf_idx];

        for t in 0..n_time.min(freqs.len()).min(amps.len()) {
            let f = freqs[t];
            let a = amps[t];

            // Find the nearest frequency bin
            let bin = (f / freq_step).floor() as usize;
            if bin < n_freq_bins {
                energy[t][bin] += a * a;
            }
        }
    }

    Ok(HilbertSpectrum {
        times,
        frequencies,
        energy,
    })
}

/// Compute the marginal Hilbert spectrum
///
/// The marginal spectrum integrates the Hilbert spectrum over time,
/// giving the total energy distribution across frequencies.
///
/// # Arguments
///
/// * `hht_result` - Result from `hht()`
/// * `fs` - Sampling frequency
/// * `n_freq_bins` - Number of frequency bins
///
/// # Returns
///
/// Tuple of (frequencies, marginal_spectrum).
///
/// # Errors
///
/// Returns an error if inputs are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hilbert_enhanced::{hht, marginal_spectrum};
/// use std::f64::consts::PI;
///
/// let fs = 256.0;
/// let n = 256;
/// let signal: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 / fs;
///     (2.0 * PI * 10.0 * t).sin()
/// }).collect();
///
/// let result = hht(&signal, fs, None).expect("HHT should succeed");
/// let (freqs, spectrum) = marginal_spectrum(&result, fs, 128).expect("Spectrum should succeed");
/// assert_eq!(freqs.len(), 128);
/// assert_eq!(spectrum.len(), 128);
/// ```
pub fn marginal_spectrum(
    hht_result: &HHTResult,
    fs: f64,
    n_freq_bins: usize,
) -> FFTResult<(Vec<f64>, Vec<f64>)> {
    let hs = hilbert_spectrum(hht_result, fs, n_freq_bins)?;

    // Sum energy over time for each frequency bin
    let mut marginal = vec![0.0; n_freq_bins];
    let dt = if fs > 0.0 { 1.0 / fs } else { 1.0 };

    for time_slice in &hs.energy {
        for (k, &e) in time_slice.iter().enumerate() {
            marginal[k] += e * dt;
        }
    }

    Ok((hs.frequencies, marginal))
}

/// Compute the degree of stationarity using the Hilbert spectrum
///
/// Returns a value between 0 and 1 for each frequency bin, where 1
/// means the signal is stationary at that frequency and 0 means
/// highly non-stationary.
///
/// # Arguments
///
/// * `hht_result` - Result from `hht()`
/// * `fs` - Sampling frequency
/// * `n_freq_bins` - Number of frequency bins
///
/// # Returns
///
/// Tuple of (frequencies, degree_of_stationarity).
///
/// # Errors
///
/// Returns an error if inputs are invalid.
pub fn degree_of_stationarity(
    hht_result: &HHTResult,
    fs: f64,
    n_freq_bins: usize,
) -> FFTResult<(Vec<f64>, Vec<f64>)> {
    let hs = hilbert_spectrum(hht_result, fs, n_freq_bins)?;

    let n_time = hs.energy.len();
    if n_time == 0 {
        return Ok((hs.frequencies, vec![0.0; n_freq_bins]));
    }

    // For each frequency, compute the coefficient of variation of energy over time
    let mut stationarity = Vec::with_capacity(n_freq_bins);

    for k in 0..n_freq_bins {
        let energies: Vec<f64> = hs.energy.iter().map(|row| row[k]).collect();

        let mean = energies.iter().sum::<f64>() / n_time as f64;
        if mean < 1e-15 {
            stationarity.push(1.0); // No energy = perfectly stationary (trivially)
            continue;
        }

        let variance = energies
            .iter()
            .map(|&e| (e - mean) * (e - mean))
            .sum::<f64>()
            / n_time as f64;
        let cv = variance.sqrt() / mean;

        // Map coefficient of variation to [0, 1] stationarity measure
        let ds = 1.0 / (1.0 + cv);
        stationarity.push(ds);
    }

    Ok((hs.frequencies, stationarity))
}

/// Compute the instantaneous energy density of the signal
///
/// Returns the total instantaneous energy at each time point,
/// summed over all IMFs.
///
/// # Arguments
///
/// * `hht_result` - Result from `hht()`
///
/// # Returns
///
/// Vector of instantaneous energy at each time point.
pub fn instantaneous_energy(hht_result: &HHTResult) -> Vec<f64> {
    let n = if let Some(first) = hht_result.imfs.first() {
        first.len()
    } else {
        return Vec::new();
    };

    let mut energy = vec![0.0; n];

    for amps in &hht_result.inst_amplitudes {
        for (i, &a) in amps.iter().enumerate() {
            if i < n {
                energy[i] += a * a;
            }
        }
    }

    energy
}

/// Compute the mean frequency at each time point
///
/// Returns a weighted average of instantaneous frequencies across IMFs,
/// weighted by their instantaneous amplitudes.
///
/// # Arguments
///
/// * `hht_result` - Result from `hht()`
///
/// # Returns
///
/// Vector of mean instantaneous frequency at each time point.
pub fn mean_frequency(hht_result: &HHTResult) -> Vec<f64> {
    let n = if let Some(first) = hht_result.imfs.first() {
        first.len()
    } else {
        return Vec::new();
    };

    let mut weighted_freq = vec![0.0; n];
    let mut total_weight = vec![0.0; n];

    for imf_idx in 0..hht_result.imfs.len() {
        let freqs = &hht_result.inst_frequencies[imf_idx];
        let amps = &hht_result.inst_amplitudes[imf_idx];

        for i in 0..n.min(freqs.len()).min(amps.len()) {
            let weight = amps[i] * amps[i];
            weighted_freq[i] += freqs[i] * weight;
            total_weight[i] += weight;
        }
    }

    for i in 0..n {
        if total_weight[i] > 1e-15 {
            weighted_freq[i] /= total_weight[i];
        }
    }

    weighted_freq
}

// ============================================================================
// Ensemble EMD (EEMD)
// ============================================================================

/// Perform Ensemble EMD (EEMD) for more robust decomposition
///
/// EEMD adds white noise to the signal multiple times and averages the
/// resulting IMFs, which helps avoid mode mixing.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `n_ensembles` - Number of noise-added ensembles (typically 50-300)
/// * `noise_amplitude` - Standard deviation of added white noise (typically 0.1-0.5 of signal std)
/// * `config` - Optional EMD configuration
///
/// # Returns
///
/// EMDResult with the ensemble-averaged IMFs.
///
/// # Errors
///
/// Returns an error if parameters are invalid.
///
/// # Examples
///
/// ```
/// use scirs2_fft::hilbert_enhanced::{eemd, EMDConfig};
/// use std::f64::consts::PI;
///
/// let fs = 256.0;
/// let n = 256;
/// let signal: Vec<f64> = (0..n).map(|i| {
///     let t = i as f64 / fs;
///     (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 20.0 * t).sin()
/// }).collect();
///
/// let result = eemd(&signal, 10, 0.1, None).expect("EEMD should succeed");
/// assert!(!result.imfs.is_empty());
/// ```
pub fn eemd(
    signal: &[f64],
    n_ensembles: usize,
    noise_amplitude: f64,
    config: Option<EMDConfig>,
) -> FFTResult<EMDResult> {
    if n_ensembles == 0 {
        return Err(FFTError::ValueError(
            "Number of ensembles must be positive".to_string(),
        ));
    }
    if noise_amplitude < 0.0 {
        return Err(FFTError::ValueError(
            "Noise amplitude must be non-negative".to_string(),
        ));
    }

    let n = signal.len();
    let cfg = config.unwrap_or_default();

    // Compute signal standard deviation for noise scaling
    let mean = signal.iter().sum::<f64>() / n as f64;
    let variance = signal.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();
    let noise_std = noise_amplitude * std_dev;

    // Collect all ensemble results
    let mut max_imfs = 0;
    let mut all_results = Vec::with_capacity(n_ensembles);

    // Simple LCG random number generator for noise (pure Rust, no external deps)
    let mut rng_state: u64 = 42;

    for _ensemble in 0..n_ensembles {
        // Generate noisy signal
        let mut noisy_signal = signal.to_vec();
        for sample in &mut noisy_signal {
            // Box-Muller transform for Gaussian noise using simple LCG
            let u1 = lcg_next_f64(&mut rng_state);
            let u2 = lcg_next_f64(&mut rng_state);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            *sample += noise_std * z;
        }

        // Perform EMD on noisy signal
        let result = emd(&noisy_signal, Some(cfg.clone()))?;
        if result.imfs.len() > max_imfs {
            max_imfs = result.imfs.len();
        }
        all_results.push(result);
    }

    if max_imfs == 0 {
        return Ok(EMDResult {
            imfs: Vec::new(),
            residual: signal.to_vec(),
            iterations: Vec::new(),
        });
    }

    // Average IMFs across ensembles
    let mut avg_imfs = vec![vec![0.0; n]; max_imfs];
    let mut imf_counts = vec![0usize; max_imfs];

    for result in &all_results {
        for (i, imf) in result.imfs.iter().enumerate() {
            for (j, &val) in imf.iter().enumerate() {
                avg_imfs[i][j] += val;
            }
            imf_counts[i] += 1;
        }
    }

    for i in 0..max_imfs {
        if imf_counts[i] > 0 {
            let count = imf_counts[i] as f64;
            for val in &mut avg_imfs[i] {
                *val /= count;
            }
        }
    }

    // Compute residual
    let mut residual = signal.to_vec();
    for imf in &avg_imfs {
        for (i, &val) in imf.iter().enumerate() {
            residual[i] -= val;
        }
    }

    Ok(EMDResult {
        imfs: avg_imfs,
        residual,
        iterations: vec![0; max_imfs], // Not meaningful for EEMD
    })
}

/// Simple LCG random number generator returning f64 in (0, 1)
fn lcg_next_f64(state: &mut u64) -> f64 {
    // LCG with Numerical Recipes parameters
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Convert top bits to f64 in (0, 1)
    let val = ((*state >> 11) as f64) / ((1u64 << 53) as f64);
    if val <= 0.0 {
        f64::MIN_POSITIVE
    } else if val >= 1.0 {
        1.0 - f64::EPSILON
    } else {
        val
    }
}

// ============================================================================
// Advanced Hilbert analysis functions
// ============================================================================

/// Compute the Hilbert transform of a real signal and return the imaginary part only
///
/// The Hilbert transform H{x}(t) is the convolution of x(t) with 1/(pi*t).
/// For a cosine signal, it produces a sine signal.
///
/// # Arguments
///
/// * `signal` - Input real signal
///
/// # Returns
///
/// The Hilbert transform (imaginary part of the analytic signal).
///
/// # Errors
///
/// Returns an error if the signal is empty.
pub fn hilbert_transform(signal: &[f64]) -> FFTResult<Vec<f64>> {
    let analytic = crate::hilbert::analytic_signal(signal)?;
    Ok(analytic.iter().map(|c| c.im).collect())
}

/// Compute the analytic signal with padding to reduce edge effects
///
/// Uses mirror padding before computing the analytic signal, then trims
/// the result to the original length.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `pad_fraction` - Fraction of signal length to use as padding (0.0 to 0.5)
///
/// # Returns
///
/// Padded analytic signal of original length.
///
/// # Errors
///
/// Returns an error if inputs are invalid.
pub fn analytic_signal_padded(signal: &[f64], pad_fraction: f64) -> FFTResult<Vec<Complex64>> {
    if signal.is_empty() {
        return Err(FFTError::ValueError("Signal cannot be empty".to_string()));
    }
    if !(0.0..=0.5).contains(&pad_fraction) {
        return Err(FFTError::ValueError(
            "Pad fraction must be between 0.0 and 0.5".to_string(),
        ));
    }

    let n = signal.len();
    let pad_len = (n as f64 * pad_fraction).ceil() as usize;

    if pad_len == 0 {
        return crate::hilbert::analytic_signal(signal);
    }

    // Create mirror-padded signal
    let padded_len = n + 2 * pad_len;
    let mut padded = Vec::with_capacity(padded_len);

    // Left mirror padding
    for i in (0..pad_len).rev() {
        let idx = (i + 1).min(n - 1);
        padded.push(signal[idx]);
    }

    // Original signal
    padded.extend_from_slice(signal);

    // Right mirror padding
    for i in 0..pad_len {
        let idx = n.saturating_sub(2 + i);
        padded.push(signal[idx]);
    }

    // Compute analytic signal on padded version
    let analytic = crate::hilbert::analytic_signal(&padded)?;

    // Trim to original length
    Ok(analytic[pad_len..pad_len + n].to_vec())
}

/// Compute the Teager-Kaiser energy operator on a signal
///
/// The Teager energy operator provides a measure of the instantaneous energy:
///   Psi[x(n)] = x(n)^2 - x(n-1)*x(n+1)
///
/// For narrowband signals: `Psi[x] ~ A^2 * omega^2`
///
/// # Arguments
///
/// * `signal` - Input signal (length >= 3)
///
/// # Returns
///
/// Teager energy (length = n - 2)
///
/// # Errors
///
/// Returns an error if signal is too short.
pub fn teager_energy(signal: &[f64]) -> FFTResult<Vec<f64>> {
    if signal.len() < 3 {
        return Err(FFTError::ValueError(
            "Signal must have at least 3 samples for Teager energy".to_string(),
        ));
    }

    let n = signal.len();
    let mut energy = Vec::with_capacity(n - 2);

    for i in 1..n - 1 {
        let val = signal[i] * signal[i] - signal[i - 1] * signal[i + 1];
        energy.push(val);
    }

    Ok(energy)
}

/// Compute instantaneous frequency and amplitude using the Teager-Kaiser
/// energy separation algorithm (ESA)
///
/// Provides an alternative to Hilbert-based methods that's better suited
/// for AM-FM signals with rapidly varying frequency.
///
/// # Arguments
///
/// * `signal` - Input signal (length >= 5)
/// * `fs` - Sampling frequency
///
/// # Returns
///
/// Tuple of (instantaneous_frequency, instantaneous_amplitude).
/// Each has length n - 4 (due to the finite difference operations).
///
/// # Errors
///
/// Returns an error if signal is too short or fs is non-positive.
pub fn teager_esa(signal: &[f64], fs: f64) -> FFTResult<(Vec<f64>, Vec<f64>)> {
    if signal.len() < 5 {
        return Err(FFTError::ValueError(
            "Signal must have at least 5 samples for Teager ESA".to_string(),
        ));
    }
    if fs <= 0.0 {
        return Err(FFTError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    let n = signal.len();

    // Compute Teager energy of the signal
    let psi_x = teager_energy(signal)?;

    // Compute the forward difference of the signal (discrete derivative)
    let mut diff_signal = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        diff_signal.push(signal[i + 1] - signal[i]);
    }

    // Teager energy of the derivative
    let psi_dx = teager_energy(&diff_signal)?;

    // Energy Separation Algorithm
    let mut inst_freq = Vec::new();
    let mut inst_amp = Vec::new();

    // psi_x has length n-2 (indices 1..n-1 of original)
    // psi_dx has length n-3 (indices 1..n-2 of diff_signal, which is indices 2..n-1 of original)
    // We align them: psi_x[i] ~ sample i+1, psi_dx[i] ~ sample i+2
    let common_len = psi_x.len().min(psi_dx.len());

    for i in 0..common_len {
        let psi_x_val = psi_x[i];
        let psi_dx_val = psi_dx[i];

        if psi_x_val.abs() < 1e-15 || psi_dx_val < 0.0 {
            inst_freq.push(0.0);
            inst_amp.push(0.0);
            continue;
        }

        // Instantaneous frequency: omega = arccos(1 - psi_dx / (2 * psi_x))
        let ratio = psi_dx_val / (2.0 * psi_x_val);
        let cos_arg = 1.0 - ratio;

        if cos_arg.abs() > 1.0 {
            // Clamp to valid range
            let omega = if cos_arg > 1.0 { 0.0 } else { PI };
            inst_freq.push(omega * fs / (2.0 * PI));
            inst_amp.push(0.0);
        } else {
            let omega = cos_arg.acos();
            inst_freq.push(omega * fs / (2.0 * PI));

            // Instantaneous amplitude: A = sqrt(psi_x / sin^2(omega))
            let sin_omega = omega.sin();
            if sin_omega.abs() > 1e-10 {
                let a = (psi_x_val / (sin_omega * sin_omega)).sqrt();
                inst_amp.push(a);
            } else {
                inst_amp.push(psi_x_val.sqrt());
            }
        }
    }

    Ok((inst_freq, inst_amp))
}

// ============================================================================
// Utility functions
// ============================================================================

/// Unwrap phase from analytic signal (internal helper)
fn unwrap_phase_vec(analytic: &[Complex64]) -> Vec<f64> {
    if analytic.is_empty() {
        return Vec::new();
    }

    let mut phase = Vec::with_capacity(analytic.len());
    let mut prev_angle = analytic[0].im.atan2(analytic[0].re);
    phase.push(prev_angle);

    let mut cumulative = 0.0;

    for c in analytic.iter().skip(1) {
        let angle = c.im.atan2(c.re);
        let mut diff = angle - prev_angle;

        while diff > PI {
            diff -= 2.0 * PI;
            cumulative -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
            cumulative += 2.0 * PI;
        }

        phase.push(angle + cumulative);
        prev_angle = angle;
    }

    phase
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_emd_basic_two_tone() {
        // Two-tone signal: EMD should separate the components
        let n = 512;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / 512.0;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 30.0 * t).sin()
            })
            .collect();

        let result = emd(&signal, None).expect("EMD should succeed");
        assert!(!result.imfs.is_empty(), "Should extract at least one IMF");

        // The residual plus all IMFs should reconstruct the original
        let mut reconstructed = result.residual.clone();
        for imf in &result.imfs {
            for (i, &val) in imf.iter().enumerate() {
                reconstructed[i] += val;
            }
        }

        for i in 0..n {
            assert_abs_diff_eq!(reconstructed[i], signal[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_emd_monotonic_residual() {
        // A simple linear signal should not produce IMFs (or very few)
        let n = 128;
        let signal: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();

        let result = emd(&signal, None).expect("EMD on linear should succeed");
        // Most of the signal should be in the residual
        let residual_energy: f64 = result.residual.iter().map(|&v| v * v).sum();
        let signal_energy: f64 = signal.iter().map(|&v| v * v).sum();
        assert!(
            residual_energy > 0.5 * signal_energy,
            "Residual should capture most of the linear trend"
        );
    }

    #[test]
    fn test_emd_reconstruction_exact() {
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / 256.0;
                (2.0 * PI * 8.0 * t).sin() + 2.0 * t
            })
            .collect();

        let result = emd(&signal, None).expect("EMD should succeed");

        let mut reconstructed = result.residual.clone();
        for imf in &result.imfs {
            for (i, &v) in imf.iter().enumerate() {
                reconstructed[i] += v;
            }
        }

        for i in 0..n {
            assert_abs_diff_eq!(reconstructed[i], signal[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_hht_basic() {
        let fs = 256.0;
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 10.0 * t).sin()
            })
            .collect();

        let result = hht(&signal, fs, None).expect("HHT should succeed");
        assert!(!result.imfs.is_empty());
        assert_eq!(result.inst_frequencies.len(), result.imfs.len());
        assert_eq!(result.inst_amplitudes.len(), result.imfs.len());

        // First IMF should have frequency near 10 Hz in the middle
        if !result.inst_frequencies.is_empty() {
            let freqs = &result.inst_frequencies[0];
            let mid_start = n / 4;
            let mid_end = 3 * n / 4;
            let avg_freq: f64 =
                freqs[mid_start..mid_end].iter().sum::<f64>() / (mid_end - mid_start) as f64;
            assert!(
                (avg_freq - 10.0).abs() < 5.0,
                "Average freq should be near 10 Hz, got {avg_freq}"
            );
        }
    }

    #[test]
    fn test_hht_invalid_fs() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(hht(&signal, 0.0, None).is_err());
        assert!(hht(&signal, -1.0, None).is_err());
    }

    #[test]
    fn test_hilbert_spectrum_dimensions() {
        let fs = 256.0;
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 10.0 * t).sin()
            })
            .collect();

        let hht_result = hht(&signal, fs, None).expect("HHT should succeed");
        let n_freq_bins = 64;
        let hs = hilbert_spectrum(&hht_result, fs, n_freq_bins)
            .expect("Hilbert spectrum should succeed");

        assert_eq!(hs.times.len(), n);
        assert_eq!(hs.frequencies.len(), n_freq_bins);
        assert_eq!(hs.energy.len(), n);
        for row in &hs.energy {
            assert_eq!(row.len(), n_freq_bins);
        }
    }

    #[test]
    fn test_marginal_spectrum() {
        let fs = 256.0;
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 10.0 * t).sin()
            })
            .collect();

        let hht_result = hht(&signal, fs, None).expect("HHT should succeed");
        let (freqs, spectrum) =
            marginal_spectrum(&hht_result, fs, 64).expect("Marginal spectrum should succeed");

        assert_eq!(freqs.len(), 64);
        assert_eq!(spectrum.len(), 64);

        // Energy should be non-negative
        for &e in &spectrum {
            assert!(e >= 0.0, "Marginal spectrum energy must be non-negative");
        }

        // Total energy should be positive for a non-zero signal
        let total: f64 = spectrum.iter().sum();
        assert!(total > 0.0, "Total marginal energy should be positive");
    }

    #[test]
    fn test_instantaneous_energy() {
        let fs = 256.0;
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 10.0 * t).sin()
            })
            .collect();

        let hht_result = hht(&signal, fs, None).expect("HHT should succeed");
        let energy = instantaneous_energy(&hht_result);

        assert_eq!(energy.len(), n);
        // Energy should be non-negative
        for &e in &energy {
            assert!(e >= 0.0, "Instantaneous energy must be non-negative");
        }
    }

    #[test]
    fn test_eemd_basic() {
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / 256.0;
                (2.0 * PI * 5.0 * t).sin() + 0.5 * (2.0 * PI * 25.0 * t).sin()
            })
            .collect();

        let result = eemd(&signal, 5, 0.1, None).expect("EEMD should succeed");
        assert!(!result.imfs.is_empty(), "EEMD should produce IMFs");

        // Verify approximate reconstruction
        let mut reconstructed = result.residual.clone();
        for imf in &result.imfs {
            for (i, &val) in imf.iter().enumerate() {
                if i < reconstructed.len() {
                    reconstructed[i] += val;
                }
            }
        }

        // With noise, reconstruction won't be exact but should be close
        let error: f64 = reconstructed
            .iter()
            .zip(signal.iter())
            .map(|(&r, &s)| (r - s) * (r - s))
            .sum::<f64>()
            / n as f64;
        let signal_power: f64 = signal.iter().map(|&s| s * s).sum::<f64>() / n as f64;
        assert!(
            error < 0.5 * signal_power,
            "EEMD reconstruction error should be reasonable"
        );
    }

    #[test]
    fn test_hilbert_transform_cosine() {
        // Hilbert transform of cos(t) should give sin(t)
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / 256.0).cos())
            .collect();

        let ht = hilbert_transform(&signal).expect("Hilbert transform should succeed");
        assert_eq!(ht.len(), n);

        // In the middle, the Hilbert transform of cos should be approximately sin
        for i in n / 4..3 * n / 4 {
            let expected = (2.0 * PI * 5.0 * i as f64 / 256.0).sin();
            assert_abs_diff_eq!(ht[i], expected, epsilon = 0.15);
        }
    }

    #[test]
    fn test_analytic_signal_padded() {
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 10.0 * i as f64 / 128.0).cos())
            .collect();

        let result = analytic_signal_padded(&signal, 0.1).expect("Padded analytic should succeed");
        assert_eq!(result.len(), n);

        // Envelope should be approximately 1 for a pure cosine
        for i in n / 4..3 * n / 4 {
            let mag = result[i].norm();
            assert!(
                (mag - 1.0).abs() < 0.2,
                "Envelope should be near 1, got {mag} at index {i}"
            );
        }
    }

    #[test]
    fn test_teager_energy_sinusoid() {
        // For x(t) = A*sin(omega*t), Teager energy ~ A^2 * omega^2
        let n = 256;
        let omega = 2.0 * PI * 10.0 / 256.0;
        let amplitude = 2.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| amplitude * (omega * i as f64).sin())
            .collect();

        let energy = teager_energy(&signal).expect("Teager energy should succeed");
        assert_eq!(energy.len(), n - 2);

        // In the middle, energy should be approximately A^2 * sin^2(omega)
        let expected = amplitude * amplitude * omega.sin() * omega.sin();
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;
        let avg_energy: f64 =
            energy[mid_start..mid_end - 2].iter().sum::<f64>() / (mid_end - 2 - mid_start) as f64;

        assert!(
            (avg_energy - expected).abs() < expected * 0.3,
            "Average Teager energy {avg_energy:.4} should be near {expected:.4}"
        );
    }

    #[test]
    fn test_teager_esa_basic() {
        let fs = 1000.0;
        let n = 256;
        let freq = 50.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * freq * t).sin()
            })
            .collect();

        let (inst_freq, inst_amp) = teager_esa(&signal, fs).expect("Teager ESA should succeed");
        assert!(!inst_freq.is_empty());
        assert_eq!(inst_freq.len(), inst_amp.len());

        // Check that estimated frequency is reasonable in the middle
        let mid = inst_freq.len() / 2;
        let mid_range = mid / 2..mid + mid / 2;
        let avg_freq: f64 =
            inst_freq[mid_range.clone()].iter().sum::<f64>() / mid_range.len() as f64;
        assert!(
            (avg_freq - freq).abs() < 20.0,
            "Estimated freq {avg_freq:.1} should be near {freq}"
        );
    }

    #[test]
    fn test_mean_frequency() {
        let fs = 256.0;
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 15.0 * t).sin()
            })
            .collect();

        let hht_result = hht(&signal, fs, None).expect("HHT should succeed");
        let mf = mean_frequency(&hht_result);
        assert_eq!(mf.len(), n);

        // Mean frequency should be positive and within the valid range [0, fs/2]
        // EMD may decompose the signal differently, so we check the structure
        // rather than the exact value
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;
        let avg: f64 = mf[mid_start..mid_end].iter().sum::<f64>() / (mid_end - mid_start) as f64;

        // The mean frequency should be positive (signal has energy)
        assert!(avg > 0.0, "Mean frequency should be positive, got {avg:.1}");
        // And within the Nyquist range
        assert!(
            avg <= fs / 2.0,
            "Mean frequency {avg:.1} should be <= Nyquist ({:.1})",
            fs / 2.0
        );
    }

    #[test]
    fn test_degree_of_stationarity_stationary() {
        let fs = 256.0;
        let n = 256;
        // Purely stationary signal
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 10.0 * t).sin()
            })
            .collect();

        let hht_result = hht(&signal, fs, None).expect("HHT should succeed");
        let (freqs, ds) = degree_of_stationarity(&hht_result, fs, 64).expect("DoS should succeed");
        assert_eq!(freqs.len(), 64);
        assert_eq!(ds.len(), 64);

        // All stationarity values should be between 0 and 1
        for &val in &ds {
            assert!(
                (0.0..=1.0).contains(&val),
                "Stationarity should be in [0,1], got {val}"
            );
        }
    }

    #[test]
    fn test_error_handling() {
        // Too short signals
        assert!(emd(&[1.0, 2.0], None).is_err());
        assert!(teager_energy(&[1.0, 2.0]).is_err());
        assert!(teager_esa(&[1.0, 2.0, 3.0], 100.0).is_err());
        assert!(analytic_signal_padded(&[], 0.1).is_err());

        // Invalid parameters
        assert!(eemd(&[1.0; 10], 0, 0.1, None).is_err());
        assert!(eemd(&[1.0; 10], 5, -0.1, None).is_err());
        assert!(analytic_signal_padded(&[1.0], 0.6).is_err());
        assert!(hilbert_spectrum(
            &HHTResult {
                imfs: vec![vec![0.0; 10]],
                inst_frequencies: vec![vec![0.0; 10]],
                inst_amplitudes: vec![vec![0.0; 10]],
                residual: vec![0.0; 10],
            },
            0.0,
            64,
        )
        .is_err());
    }
}
