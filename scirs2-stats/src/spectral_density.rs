//! Spectral Density Estimation
//!
//! This module provides non-parametric spectral density estimation methods
//! for time series analysis:
//!
//! - **Periodogram**: raw spectral estimate (unsmoothed)
//! - **Welch's method**: averaged modified periodograms with overlapping segments
//! - **Bartlett's method**: averaged periodograms with non-overlapping segments
//! - **Cross-spectral density**: joint spectral analysis of two series
//! - **Coherence function**: squared coherence (normalized cross-spectrum magnitude)
//! - **Spectral Granger causality**: frequency-domain causality measure
//!
//! All methods use a pure-Rust DFT implementation (no external FFT crate required
//! for correctness; OxiFFT can be plugged in for performance).
//!
//! # References
//!
//! - Welch, P.D. (1967). The Use of Fast Fourier Transform for the Estimation
//!   of Power Spectra. IEEE Transactions on Audio and Electroacoustics.
//! - Bartlett, M.S. (1948). Smoothing Periodograms from Time-Series with
//!   Continuous Spectra. Nature.
//! - Geweke, J. (1982). Measurement of Linear Dependence and Feedback Between
//!   Multiple Time Series. JASA.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, ArrayView1};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a spectral density estimation
#[derive(Debug, Clone)]
pub struct SpectralDensityResult {
    /// Frequencies (in cycles per sample, [0, 0.5])
    pub frequencies: Array1<f64>,
    /// Power spectral density estimates
    pub psd: Array1<f64>,
    /// Number of segments used (for Welch/Bartlett)
    pub n_segments: usize,
    /// Effective bandwidth
    pub bandwidth: f64,
}

/// Result of a cross-spectral density estimation
#[derive(Debug, Clone)]
pub struct CrossSpectralResult {
    /// Frequencies
    pub frequencies: Array1<f64>,
    /// Cross-spectral density (real part)
    pub csd_real: Array1<f64>,
    /// Cross-spectral density (imaginary part)
    pub csd_imag: Array1<f64>,
    /// Magnitude of the cross-spectrum
    pub csd_magnitude: Array1<f64>,
    /// Phase of the cross-spectrum (radians)
    pub csd_phase: Array1<f64>,
    /// Power spectral density of x
    pub psd_x: Array1<f64>,
    /// Power spectral density of y
    pub psd_y: Array1<f64>,
}

/// Result of a coherence analysis
#[derive(Debug, Clone)]
pub struct CoherenceResult {
    /// Frequencies
    pub frequencies: Array1<f64>,
    /// Squared coherence (in [0, 1])
    pub coherence_sq: Array1<f64>,
    /// Phase spectrum (radians)
    pub phase: Array1<f64>,
    /// Gain spectrum (|Sxy| / Sxx)
    pub gain: Array1<f64>,
}

/// Result of spectral Granger causality analysis
#[derive(Debug, Clone)]
pub struct SpectralGrangerResult {
    /// Frequencies
    pub frequencies: Array1<f64>,
    /// Spectral Granger causality from x to y at each frequency
    pub causality_x_to_y: Array1<f64>,
    /// Spectral Granger causality from y to x at each frequency
    pub causality_y_to_x: Array1<f64>,
    /// Total spectral interdependence
    pub total_interdependence: Array1<f64>,
}

// ---------------------------------------------------------------------------
// Window functions
// ---------------------------------------------------------------------------

/// Window function types for spectral estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// Rectangular (no windowing)
    Rectangular,
    /// Hann (raised cosine)
    Hann,
    /// Hamming
    Hamming,
    /// Blackman
    Blackman,
    /// Bartlett (triangular)
    Bartlett,
    /// Tukey (tapered cosine) with alpha parameter stored separately
    Tukey,
}

/// Generate window coefficients
fn window_coefficients(window: Window, n: usize, alpha: f64) -> Array1<f64> {
    let nf = n as f64;
    Array1::from_vec(
        (0..n)
            .map(|i| {
                let t = i as f64;
                match window {
                    Window::Rectangular => 1.0,
                    Window::Hann => 0.5 * (1.0 - (2.0 * PI * t / (nf - 1.0)).cos()),
                    Window::Hamming => 0.54 - 0.46 * (2.0 * PI * t / (nf - 1.0)).cos(),
                    Window::Blackman => {
                        0.42 - 0.5 * (2.0 * PI * t / (nf - 1.0)).cos()
                            + 0.08 * (4.0 * PI * t / (nf - 1.0)).cos()
                    }
                    Window::Bartlett => {
                        if n <= 1 {
                            1.0
                        } else {
                            1.0 - (2.0 * t / (nf - 1.0) - 1.0).abs()
                        }
                    }
                    Window::Tukey => {
                        let a = alpha.max(0.0).min(1.0);
                        if a == 0.0 {
                            1.0
                        } else if a >= 1.0 {
                            0.5 * (1.0 - (2.0 * PI * t / (nf - 1.0)).cos())
                        } else {
                            let boundary = a * (nf - 1.0) / 2.0;
                            if t < boundary {
                                0.5 * (1.0 - (PI * t / boundary).cos())
                            } else if t > (nf - 1.0) - boundary {
                                0.5 * (1.0 - (PI * ((nf - 1.0) - t) / boundary).cos())
                            } else {
                                1.0
                            }
                        }
                    }
                }
            })
            .collect(),
    )
}

/// Window power (sum of squared coefficients / n), used for PSD normalization
fn window_power(w: &Array1<f64>) -> f64 {
    let n = w.len() as f64;
    if n == 0.0 {
        return 1.0;
    }
    w.iter().map(|&v| v * v).sum::<f64>() / n
}

// ---------------------------------------------------------------------------
// DFT helpers
// ---------------------------------------------------------------------------

/// Compute DFT of a real-valued signal, returning complex values for
/// non-negative frequencies only (N/2 + 1 values).
/// Returns (real_parts, imag_parts).
fn rfft(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let n_out = n / 2 + 1;
    let mut real = vec![0.0; n_out];
    let mut imag = vec![0.0; n_out];
    let nf = n as f64;
    for k in 0..n_out {
        let mut re = 0.0;
        let mut im = 0.0;
        for t in 0..n {
            let angle = 2.0 * PI * (k as f64) * (t as f64) / nf;
            re += x[t] * angle.cos();
            im -= x[t] * angle.sin();
        }
        real[k] = re;
        imag[k] = im;
    }
    (real, imag)
}

/// Compute the power spectral density from DFT coefficients.
/// Returns one-sided PSD (scaled by 2/N except at DC and Nyquist).
fn dft_to_psd(real: &[f64], imag: &[f64], n: usize, fs: f64, win_power: f64) -> Vec<f64> {
    let n_out = real.len();
    let scale = 1.0 / (fs * (n as f64) * win_power);
    let mut psd = vec![0.0; n_out];
    for k in 0..n_out {
        let power = real[k] * real[k] + imag[k] * imag[k];
        psd[k] = power * scale;
        // Double for one-sided (except DC and Nyquist)
        if k > 0 && k < n_out - 1 {
            psd[k] *= 2.0;
        }
    }
    psd
}

// ---------------------------------------------------------------------------
// Periodogram
// ---------------------------------------------------------------------------

/// Compute the periodogram (raw spectral estimate) of a time series.
///
/// # Arguments
/// * `x` - Time series data
/// * `window` - Window function to apply (default: `Hann`)
/// * `detrend` - If true, remove the mean before computing
///
/// # Example
/// ```
/// use scirs2_stats::spectral_density::{periodogram, Window};
/// use scirs2_core::ndarray::Array1;
///
/// // Sine wave at frequency 0.1 (cycles/sample)
/// let n = 256;
/// let x = Array1::from_vec((0..n).map(|i| {
///     (2.0 * std::f64::consts::PI * 0.1 * i as f64).sin()
/// }).collect());
/// let result = periodogram(&x.view(), Window::Hann, true).expect("periodogram failed");
/// assert_eq!(result.frequencies.len(), result.psd.len());
/// // Peak should be near frequency 0.1
/// let peak_idx = result.psd.iter()
///     .enumerate()
///     .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
///     .map(|(i, _)| i)
///     .unwrap_or(0);
/// assert!((result.frequencies[peak_idx] - 0.1).abs() < 0.02);
/// ```
pub fn periodogram(
    x: &ArrayView1<f64>,
    window: Window,
    detrend: bool,
) -> StatsResult<SpectralDensityResult> {
    let n = x.len();
    if n < 4 {
        return Err(StatsError::InsufficientData(
            "periodogram requires at least 4 data points".into(),
        ));
    }
    let fs = 1.0; // normalized sampling frequency
                  // Detrend (remove mean)
    let mean = if detrend {
        x.iter().sum::<f64>() / (n as f64)
    } else {
        0.0
    };
    // Apply window
    let w = window_coefficients(window, n, 0.5);
    let wp = window_power(&w);
    let windowed: Vec<f64> = (0..n).map(|i| (x[i] - mean) * w[i]).collect();

    let (real, imag) = rfft(&windowed);
    let psd_vec = dft_to_psd(&real, &imag, n, fs, wp);
    let n_out = psd_vec.len();
    let freqs = Array1::from_vec((0..n_out).map(|k| (k as f64) * fs / (n as f64)).collect());

    Ok(SpectralDensityResult {
        frequencies: freqs,
        psd: Array1::from_vec(psd_vec),
        n_segments: 1,
        bandwidth: fs / (n as f64),
    })
}

// ---------------------------------------------------------------------------
// Welch's method
// ---------------------------------------------------------------------------

/// Compute the power spectral density using Welch's method.
///
/// Divides the signal into overlapping segments, windows each, computes
/// modified periodograms, and averages them.
///
/// # Arguments
/// * `x` - Time series data
/// * `segment_length` - Length of each segment (None for `n/8` rounded to power of 2)
/// * `overlap` - Fraction of overlap between segments (default: 0.5)
/// * `window` - Window function
///
/// # Example
/// ```
/// use scirs2_stats::spectral_density::{welch, Window};
/// use scirs2_core::ndarray::Array1;
///
/// let n = 1024;
/// let x = Array1::from_vec((0..n).map(|i| {
///     (2.0 * std::f64::consts::PI * 0.25 * i as f64).sin() + ((i as f64) * 0.7).sin() * 0.1
/// }).collect());
/// let result = welch(&x.view(), Some(256), Some(0.5), Window::Hann)
///     .expect("Welch failed");
/// assert!(result.n_segments > 1);
/// ```
pub fn welch(
    x: &ArrayView1<f64>,
    segment_length: Option<usize>,
    overlap: Option<f64>,
    window: Window,
) -> StatsResult<SpectralDensityResult> {
    let n = x.len();
    if n < 8 {
        return Err(StatsError::InsufficientData(
            "Welch's method requires at least 8 data points".into(),
        ));
    }
    let seg_len = segment_length.unwrap_or_else(|| {
        // Default: n/8, but at least 8
        let target = n / 8;
        target.max(8).min(n)
    });
    if seg_len < 4 || seg_len > n {
        return Err(StatsError::InvalidArgument(format!(
            "segment_length must be in [4, {}], got {}",
            n, seg_len
        )));
    }
    let overlap_frac = overlap.unwrap_or(0.5).max(0.0).min(0.99);
    let step = ((seg_len as f64) * (1.0 - overlap_frac)).round() as usize;
    let step = step.max(1);

    let fs = 1.0;
    let w = window_coefficients(window, seg_len, 0.5);
    let wp = window_power(&w);

    let n_freq = seg_len / 2 + 1;
    let mut avg_psd = vec![0.0_f64; n_freq];
    let mut n_segments = 0_usize;

    let mut start = 0;
    while start + seg_len <= n {
        // Extract segment and detrend
        let mean: f64 = (start..start + seg_len).map(|i| x[i]).sum::<f64>() / (seg_len as f64);
        let windowed: Vec<f64> = (0..seg_len).map(|i| (x[start + i] - mean) * w[i]).collect();
        let (real, imag) = rfft(&windowed);
        let psd = dft_to_psd(&real, &imag, seg_len, fs, wp);
        for k in 0..n_freq {
            avg_psd[k] += psd[k];
        }
        n_segments += 1;
        start += step;
    }

    if n_segments == 0 {
        return Err(StatsError::ComputationError(
            "Welch: no segments could be formed".into(),
        ));
    }

    for k in 0..n_freq {
        avg_psd[k] /= n_segments as f64;
    }

    let freqs = Array1::from_vec(
        (0..n_freq)
            .map(|k| (k as f64) * fs / (seg_len as f64))
            .collect(),
    );

    Ok(SpectralDensityResult {
        frequencies: freqs,
        psd: Array1::from_vec(avg_psd),
        n_segments,
        bandwidth: fs / (seg_len as f64),
    })
}

// ---------------------------------------------------------------------------
// Bartlett's method
// ---------------------------------------------------------------------------

/// Compute the power spectral density using Bartlett's method.
///
/// Similar to Welch's method but with no overlap and a rectangular window.
///
/// # Arguments
/// * `x` - Time series data
/// * `n_segments` - Number of non-overlapping segments
///
/// # Example
/// ```
/// use scirs2_stats::spectral_density::bartlett;
/// use scirs2_core::ndarray::Array1;
///
/// let n = 256;
/// let x = Array1::from_vec((0..n).map(|i| {
///     (2.0 * std::f64::consts::PI * 0.1 * i as f64).sin()
/// }).collect());
/// let result = bartlett(&x.view(), 4).expect("Bartlett failed");
/// assert_eq!(result.n_segments, 4);
/// ```
pub fn bartlett(x: &ArrayView1<f64>, n_segments: usize) -> StatsResult<SpectralDensityResult> {
    let n = x.len();
    if n_segments == 0 || n_segments > n {
        return Err(StatsError::InvalidArgument(format!(
            "n_segments must be in [1, {}]",
            n
        )));
    }
    let seg_len = n / n_segments;
    if seg_len < 4 {
        return Err(StatsError::InsufficientData(
            "Bartlett: segments too short (< 4 points each)".into(),
        ));
    }
    // Bartlett = Welch with rectangular window and no overlap
    welch(x, Some(seg_len), Some(0.0), Window::Rectangular)
}

// ---------------------------------------------------------------------------
// Cross-spectral density
// ---------------------------------------------------------------------------

/// Compute the cross-spectral density of two time series.
///
/// Uses Welch's method to estimate the cross-spectrum Sxy(f) = E[X*(f) Y(f)].
///
/// # Arguments
/// * `x` - First time series
/// * `y` - Second time series
/// * `segment_length` - Segment length (None for auto)
/// * `overlap` - Overlap fraction (default 0.5)
/// * `window` - Window function
///
/// # Example
/// ```
/// use scirs2_stats::spectral_density::{cross_spectral_density, Window};
/// use scirs2_core::ndarray::Array1;
///
/// let n = 256;
/// let x = Array1::from_vec((0..n).map(|i| {
///     (2.0 * std::f64::consts::PI * 0.1 * i as f64).sin()
/// }).collect());
/// let y = Array1::from_vec((0..n).map(|i| {
///     (2.0 * std::f64::consts::PI * 0.1 * i as f64 + 0.5).sin()
/// }).collect());
/// let result = cross_spectral_density(&x.view(), &y.view(), Some(64), Some(0.5), Window::Hann)
///     .expect("CSD failed");
/// assert_eq!(result.frequencies.len(), result.csd_magnitude.len());
/// ```
pub fn cross_spectral_density(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    segment_length: Option<usize>,
    overlap: Option<f64>,
    window: Window,
) -> StatsResult<CrossSpectralResult> {
    let n = x.len();
    if n != y.len() {
        return Err(StatsError::DimensionMismatch(format!(
            "x and y must have the same length (got {} and {})",
            n,
            y.len()
        )));
    }
    if n < 8 {
        return Err(StatsError::InsufficientData(
            "cross-spectral density requires at least 8 data points".into(),
        ));
    }
    let seg_len = segment_length.unwrap_or_else(|| (n / 8).max(8).min(n));
    if seg_len < 4 || seg_len > n {
        return Err(StatsError::InvalidArgument(format!(
            "segment_length must be in [4, {}]",
            n
        )));
    }
    let overlap_frac = overlap.unwrap_or(0.5).max(0.0).min(0.99);
    let step = ((seg_len as f64) * (1.0 - overlap_frac)).round() as usize;
    let step = step.max(1);

    let fs = 1.0;
    let w = window_coefficients(window, seg_len, 0.5);
    let wp = window_power(&w);

    let n_freq = seg_len / 2 + 1;
    let mut avg_csd_re = vec![0.0_f64; n_freq];
    let mut avg_csd_im = vec![0.0_f64; n_freq];
    let mut avg_psd_x = vec![0.0_f64; n_freq];
    let mut avg_psd_y = vec![0.0_f64; n_freq];
    let mut n_seg = 0_usize;

    let mut start = 0;
    while start + seg_len <= n {
        let x_mean: f64 = (start..start + seg_len).map(|i| x[i]).sum::<f64>() / (seg_len as f64);
        let y_mean: f64 = (start..start + seg_len).map(|i| y[i]).sum::<f64>() / (seg_len as f64);

        let wx: Vec<f64> = (0..seg_len)
            .map(|i| (x[start + i] - x_mean) * w[i])
            .collect();
        let wy: Vec<f64> = (0..seg_len)
            .map(|i| (y[start + i] - y_mean) * w[i])
            .collect();

        let (xr, xi) = rfft(&wx);
        let (yr, yi) = rfft(&wy);

        let scale = 1.0 / (fs * (seg_len as f64) * wp);
        for k in 0..n_freq {
            // Cross: conj(X) * Y = (xr - j*xi_neg)(yr + j*yi) but xi stored as -sin
            // X* = (xr, -xi), Y = (yr, yi)
            // X* * Y = (xr*yr + xi*yi) + j*(xr*yi - xi*yr)
            // But our rfft stores imag as -sin component, so conj(X) has imag = +xi
            let csd_re = (xr[k] * yr[k] + xi[k] * yi[k]) * scale;
            let csd_im = (xr[k] * yi[k] - xi[k] * yr[k]) * scale;
            let psd_x = (xr[k] * xr[k] + xi[k] * xi[k]) * scale;
            let psd_y = (yr[k] * yr[k] + yi[k] * yi[k]) * scale;
            let double = if k > 0 && k < n_freq - 1 { 2.0 } else { 1.0 };
            avg_csd_re[k] += csd_re * double;
            avg_csd_im[k] += csd_im * double;
            avg_psd_x[k] += psd_x * double;
            avg_psd_y[k] += psd_y * double;
        }
        n_seg += 1;
        start += step;
    }

    if n_seg == 0 {
        return Err(StatsError::ComputationError(
            "no segments formed for cross-spectral density".into(),
        ));
    }

    let ns = n_seg as f64;
    let mut magnitude = vec![0.0_f64; n_freq];
    let mut phase = vec![0.0_f64; n_freq];
    for k in 0..n_freq {
        avg_csd_re[k] /= ns;
        avg_csd_im[k] /= ns;
        avg_psd_x[k] /= ns;
        avg_psd_y[k] /= ns;
        magnitude[k] = (avg_csd_re[k] * avg_csd_re[k] + avg_csd_im[k] * avg_csd_im[k]).sqrt();
        phase[k] = avg_csd_im[k].atan2(avg_csd_re[k]);
    }

    let freqs = Array1::from_vec(
        (0..n_freq)
            .map(|k| (k as f64) * fs / (seg_len as f64))
            .collect(),
    );

    Ok(CrossSpectralResult {
        frequencies: freqs,
        csd_real: Array1::from_vec(avg_csd_re),
        csd_imag: Array1::from_vec(avg_csd_im),
        csd_magnitude: Array1::from_vec(magnitude),
        csd_phase: Array1::from_vec(phase),
        psd_x: Array1::from_vec(avg_psd_x),
        psd_y: Array1::from_vec(avg_psd_y),
    })
}

// ---------------------------------------------------------------------------
// Coherence function
// ---------------------------------------------------------------------------

/// Compute the squared coherence and phase spectrum between two series.
///
/// The squared coherence is |Sxy(f)|^2 / (Sxx(f) * Syy(f)), ranging in [0, 1].
/// A value near 1 indicates strong linear relationship at that frequency.
///
/// # Arguments
/// * `x` - First time series
/// * `y` - Second time series
/// * `segment_length` - Segment length for Welch (None for auto)
/// * `overlap` - Overlap fraction (default 0.5)
/// * `window` - Window function
///
/// # Example
/// ```
/// use scirs2_stats::spectral_density::{coherence, Window};
/// use scirs2_core::ndarray::Array1;
///
/// let n = 256;
/// let x = Array1::from_vec((0..n).map(|i| {
///     (2.0 * std::f64::consts::PI * 0.1 * i as f64).sin()
/// }).collect());
/// // y is a phase-shifted version of x => high coherence
/// let y = Array1::from_vec((0..n).map(|i| {
///     (2.0 * std::f64::consts::PI * 0.1 * i as f64 + 1.0).sin()
/// }).collect());
/// let result = coherence(&x.view(), &y.view(), Some(64), Some(0.5), Window::Hann)
///     .expect("coherence failed");
/// // At the signal frequency, coherence should be high
/// let peak_idx = result.coherence_sq.iter()
///     .enumerate()
///     .skip(1)
///     .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
///     .map(|(i, _)| i)
///     .unwrap_or(0);
/// assert!(result.coherence_sq[peak_idx] > 0.5);
/// ```
pub fn coherence(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    segment_length: Option<usize>,
    overlap: Option<f64>,
    window: Window,
) -> StatsResult<CoherenceResult> {
    let csd = cross_spectral_density(x, y, segment_length, overlap, window)?;
    let n_freq = csd.frequencies.len();
    let mut coh_sq = Array1::<f64>::zeros(n_freq);
    let mut phase = Array1::<f64>::zeros(n_freq);
    let mut gain = Array1::<f64>::zeros(n_freq);

    for k in 0..n_freq {
        let sxy_sq = csd.csd_real[k] * csd.csd_real[k] + csd.csd_imag[k] * csd.csd_imag[k];
        let denom = csd.psd_x[k] * csd.psd_y[k];
        coh_sq[k] = if denom > 1e-30 {
            (sxy_sq / denom).min(1.0)
        } else {
            0.0
        };
        phase[k] = csd.csd_phase[k];
        gain[k] = if csd.psd_x[k] > 1e-30 {
            csd.csd_magnitude[k] / csd.psd_x[k]
        } else {
            0.0
        };
    }

    Ok(CoherenceResult {
        frequencies: csd.frequencies,
        coherence_sq: coh_sq,
        phase,
        gain,
    })
}

// ---------------------------------------------------------------------------
// Spectral Granger causality helper
// ---------------------------------------------------------------------------

/// Compute a spectral Granger causality measure between two series.
///
/// This is a frequency-domain decomposition of Granger causality based on
/// comparing the spectral density of the restricted model (univariate AR) with
/// the full model (bivariate VAR).
///
/// The measure at each frequency f is:
///   GC_{x->y}(f) = ln(S_y(f) / S_y|x(f))
///
/// where S_y is the spectrum of y from a univariate AR, and S_y|x is the
/// spectrum of y from the bivariate VAR residuals.
///
/// # Arguments
/// * `x` - First time series (potential cause)
/// * `y` - Second time series (potential effect)
/// * `max_lags` - Maximum number of AR/VAR lags
/// * `segment_length` - Segment length for spectral estimation
///
/// # Example
/// ```
/// use scirs2_stats::spectral_density::spectral_granger_causality;
/// use scirs2_core::ndarray::Array1;
///
/// let n = 200;
/// // x leads y by a few samples
/// let x = Array1::from_vec((0..n).map(|i| ((i as f64) * 0.3).sin()).collect());
/// let mut y_vec = vec![0.0_f64; n];
/// for i in 3..n {
///     y_vec[i] = 0.7 * x[i-3] + ((i as f64) * 0.5).sin() * 0.3;
/// }
/// let y = Array1::from_vec(y_vec);
/// let result = spectral_granger_causality(&x.view(), &y.view(), 5, Some(64))
///     .expect("spectral GC failed");
/// assert_eq!(result.frequencies.len(), result.causality_x_to_y.len());
/// ```
pub fn spectral_granger_causality(
    x: &ArrayView1<f64>,
    y: &ArrayView1<f64>,
    max_lags: usize,
    segment_length: Option<usize>,
) -> StatsResult<SpectralGrangerResult> {
    let n = x.len();
    if n != y.len() {
        return Err(StatsError::DimensionMismatch(
            "x and y must have the same length".into(),
        ));
    }
    if n < max_lags + 10 {
        return Err(StatsError::InsufficientData(
            "insufficient data for spectral Granger causality".into(),
        ));
    }

    // Fit univariate AR(p) for y
    let resid_y_only = fit_ar_residuals(y, max_lags)?;
    // Fit bivariate VAR(p) for (x->y direction): y_t = sum a_i*y_{t-i} + b_i*x_{t-i} + e_t
    let resid_y_full = fit_var_residuals(x, y, max_lags)?;
    // Similarly for x direction
    let resid_x_only = fit_ar_residuals(x, max_lags)?;
    let resid_x_full = fit_var_residuals(y, x, max_lags)?;

    // Compute spectral densities of residuals
    let seg_len = segment_length.unwrap_or_else(|| (n / 8).max(8).min(n));
    let spec_y_only = welch(&resid_y_only.view(), Some(seg_len), Some(0.5), Window::Hann)?;
    let spec_y_full = welch(&resid_y_full.view(), Some(seg_len), Some(0.5), Window::Hann)?;
    let spec_x_only = welch(&resid_x_only.view(), Some(seg_len), Some(0.5), Window::Hann)?;
    let spec_x_full = welch(&resid_x_full.view(), Some(seg_len), Some(0.5), Window::Hann)?;

    let n_freq = spec_y_only.psd.len().min(spec_y_full.psd.len());
    let n_freq = n_freq.min(spec_x_only.psd.len()).min(spec_x_full.psd.len());

    let mut gc_x_to_y = Array1::<f64>::zeros(n_freq);
    let mut gc_y_to_x = Array1::<f64>::zeros(n_freq);
    let mut total = Array1::<f64>::zeros(n_freq);

    for k in 0..n_freq {
        let ratio_xy = spec_y_only.psd[k] / spec_y_full.psd[k].max(1e-30);
        gc_x_to_y[k] = ratio_xy.max(1.0).ln();
        let ratio_yx = spec_x_only.psd[k] / spec_x_full.psd[k].max(1e-30);
        gc_y_to_x[k] = ratio_yx.max(1.0).ln();
        total[k] = gc_x_to_y[k] + gc_y_to_x[k];
    }

    let freqs = Array1::from_vec((0..n_freq).map(|k| (k as f64) / (seg_len as f64)).collect());

    Ok(SpectralGrangerResult {
        frequencies: freqs,
        causality_x_to_y: gc_x_to_y,
        causality_y_to_x: gc_y_to_x,
        total_interdependence: total,
    })
}

/// Fit a univariate AR(p) model and return residuals.
fn fit_ar_residuals(y: &ArrayView1<f64>, p: usize) -> StatsResult<Array1<f64>> {
    let n = y.len();
    if n <= p + 1 {
        return Err(StatsError::InsufficientData(
            "too few observations for AR model".into(),
        ));
    }
    let n_eff = n - p;
    // Design: [y_{t-1}, y_{t-2}, ..., y_{t-p}, 1]
    let n_reg = p + 1;
    let mut design = scirs2_core::ndarray::Array2::<f64>::zeros((n_eff, n_reg));
    let dep = Array1::from_vec((p..n).map(|i| y[i]).collect());
    for i in 0..n_eff {
        for lag in 1..=p {
            design[[i, lag - 1]] = y[p + i - lag];
        }
        design[[i, p]] = 1.0; // constant
    }
    let ols = crate::stationarity::ols_regression(&dep.view(), &design)?;
    Ok(ols.residuals)
}

/// Fit a bivariate VAR equation: z_t = sum a_i*z_{t-i} + b_i*cause_{t-i} + c + e_t
/// Returns the residuals for the effect variable.
fn fit_var_residuals(
    cause: &ArrayView1<f64>,
    effect: &ArrayView1<f64>,
    p: usize,
) -> StatsResult<Array1<f64>> {
    let n = cause.len();
    if n <= p + 1 {
        return Err(StatsError::InsufficientData(
            "too few observations for VAR model".into(),
        ));
    }
    let n_eff = n - p;
    let n_reg = 2 * p + 1; // p lags of effect + p lags of cause + constant
    let mut design = scirs2_core::ndarray::Array2::<f64>::zeros((n_eff, n_reg));
    let dep = Array1::from_vec((p..n).map(|i| effect[i]).collect());
    for i in 0..n_eff {
        let mut col = 0;
        for lag in 1..=p {
            design[[i, col]] = effect[p + i - lag];
            col += 1;
        }
        for lag in 1..=p {
            design[[i, col]] = cause[p + i - lag];
            col += 1;
        }
        design[[i, col]] = 1.0;
    }
    let ols = crate::stationarity::ols_regression(&dep.view(), &design)?;
    Ok(ols.residuals)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    fn make_sine(n: usize, freq: f64) -> Array1<f64> {
        Array1::from_vec(
            (0..n)
                .map(|i| (2.0 * PI * freq * (i as f64)).sin())
                .collect(),
        )
    }

    fn make_noise(n: usize) -> Array1<f64> {
        Array1::from_vec(
            (0..n)
                .map(|i| ((i as f64) * 2.7 + 0.3).sin() * 0.5)
                .collect(),
        )
    }

    #[test]
    fn test_periodogram_pure_sine() {
        let x = make_sine(256, 0.1);
        let result = periodogram(&x.view(), Window::Hann, true);
        assert!(result.is_ok());
        let r = result.expect("periodogram should succeed");
        assert_eq!(r.frequencies.len(), 129); // 256/2 + 1
        assert_eq!(r.psd.len(), 129);
        // Find peak
        let peak_idx = r
            .psd
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert!((r.frequencies[peak_idx] - 0.1).abs() < 0.02);
    }

    #[test]
    fn test_periodogram_rectangular() {
        let x = make_sine(128, 0.2);
        let result = periodogram(&x.view(), Window::Rectangular, false);
        assert!(result.is_ok());
    }

    #[test]
    fn test_periodogram_blackman() {
        let x = make_sine(128, 0.15);
        let result = periodogram(&x.view(), Window::Blackman, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_periodogram_insufficient() {
        let x = Array1::from_vec(vec![1.0, 2.0]);
        let result = periodogram(&x.view(), Window::Hann, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_welch_basic() {
        let x = make_sine(512, 0.1);
        let result = welch(&x.view(), Some(128), Some(0.5), Window::Hann);
        assert!(result.is_ok());
        let r = result.expect("Welch should succeed");
        assert!(r.n_segments > 1);
        assert_eq!(r.psd.len(), 65); // 128/2 + 1
    }

    #[test]
    fn test_welch_auto_segment() {
        let x = make_sine(1024, 0.25);
        let result = welch(&x.view(), None, None, Window::Hamming);
        assert!(result.is_ok());
        let r = result.expect("Welch auto should succeed");
        assert!(r.n_segments >= 1);
    }

    #[test]
    fn test_welch_peak_detection() {
        let x = make_sine(1024, 0.1);
        let r = welch(&x.view(), Some(256), Some(0.5), Window::Hann).expect("Welch should succeed");
        let peak_idx = r
            .psd
            .iter()
            .enumerate()
            .skip(1)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert!((r.frequencies[peak_idx] - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_bartlett_basic() {
        let x = make_sine(256, 0.1);
        let result = bartlett(&x.view(), 4);
        assert!(result.is_ok());
        let r = result.expect("Bartlett should succeed");
        assert!(r.n_segments >= 1);
    }

    #[test]
    fn test_bartlett_invalid_segments() {
        let x = make_sine(16, 0.1);
        let result = bartlett(&x.view(), 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_spectral_density_basic() {
        let x = make_sine(256, 0.1);
        let y = make_sine(256, 0.1); // same frequency
        let result =
            cross_spectral_density(&x.view(), &y.view(), Some(64), Some(0.5), Window::Hann);
        assert!(result.is_ok());
        let r = result.expect("CSD should succeed");
        assert_eq!(r.csd_magnitude.len(), r.frequencies.len());
    }

    #[test]
    fn test_cross_spectral_density_different_freqs() {
        let x = make_sine(256, 0.1);
        let y = make_sine(256, 0.3);
        let result =
            cross_spectral_density(&x.view(), &y.view(), Some(64), Some(0.5), Window::Hann);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cross_spectral_density_length_mismatch() {
        let x = make_sine(100, 0.1);
        let y = make_sine(200, 0.1);
        let result = cross_spectral_density(&x.view(), &y.view(), None, None, Window::Hann);
        assert!(result.is_err());
    }

    #[test]
    fn test_coherence_same_signal() {
        let x = make_sine(256, 0.1);
        let result = coherence(&x.view(), &x.view(), Some(64), Some(0.5), Window::Hann);
        assert!(result.is_ok());
        let r = result.expect("coherence should succeed");
        // Coherence of a signal with itself should be very high
        let max_coh = r
            .coherence_sq
            .iter()
            .skip(1)
            .cloned()
            .fold(0.0_f64, f64::max);
        assert!(max_coh > 0.9);
    }

    #[test]
    fn test_coherence_values_bounded() {
        let x = make_sine(256, 0.1);
        let y = make_noise(256);
        let r = coherence(&x.view(), &y.view(), Some(64), Some(0.5), Window::Hann)
            .expect("coherence should succeed");
        for &c in r.coherence_sq.iter() {
            assert!(c >= 0.0, "coherence must be >= 0, got {}", c);
            assert!(c <= 1.0 + 1e-10, "coherence must be <= 1, got {}", c);
        }
    }

    #[test]
    fn test_spectral_granger_causality() {
        let n = 200;
        let x = make_sine(n, 0.1);
        let mut y_vec = vec![0.0_f64; n];
        for i in 3..n {
            y_vec[i] = 0.7 * x[i - 3] + ((i as f64) * 0.5).sin() * 0.3;
        }
        let y = Array1::from_vec(y_vec);
        let result = spectral_granger_causality(&x.view(), &y.view(), 5, Some(32));
        assert!(result.is_ok());
        let r = result.expect("spectral GC should succeed");
        assert_eq!(r.causality_x_to_y.len(), r.frequencies.len());
        // All GC values should be non-negative
        for &gc in r.causality_x_to_y.iter() {
            assert!(gc >= 0.0, "GC should be non-negative, got {}", gc);
        }
    }

    #[test]
    fn test_window_coefficients_hann() {
        let w = window_coefficients(Window::Hann, 8, 0.5);
        assert_eq!(w.len(), 8);
        // Hann window starts and ends at 0
        assert!((w[0]).abs() < 1e-10);
        assert!((w[7]).abs() < 1e-10);
    }

    #[test]
    fn test_window_coefficients_rectangular() {
        let w = window_coefficients(Window::Rectangular, 10, 0.5);
        for &v in w.iter() {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_window_coefficients_bartlett() {
        let w = window_coefficients(Window::Bartlett, 5, 0.5);
        // Bartlett is triangular, peaks at center
        assert!(w[2] > w[0]);
        assert!(w[2] > w[4]);
    }

    #[test]
    fn test_psd_non_negative() {
        let x = make_noise(128);
        let r = periodogram(&x.view(), Window::Hann, true).expect("periodogram should succeed");
        for &p in r.psd.iter() {
            assert!(p >= 0.0, "PSD must be non-negative, got {}", p);
        }
    }

    #[test]
    fn test_spectral_granger_insufficient() {
        let x = Array1::from_vec(vec![1.0; 5]);
        let y = Array1::from_vec(vec![2.0; 5]);
        let result = spectral_granger_causality(&x.view(), &y.view(), 10, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_welch_overlap_zero() {
        let x = make_sine(256, 0.1);
        let result = welch(&x.view(), Some(64), Some(0.0), Window::Hann);
        assert!(result.is_ok());
        let r = result.expect("Welch with 0 overlap should succeed");
        assert_eq!(r.n_segments, 4); // 256/64 = 4
    }

    #[test]
    fn test_tukey_window() {
        let w = window_coefficients(Window::Tukey, 100, 0.5);
        assert_eq!(w.len(), 100);
        // Middle should be 1.0
        assert!((w[50] - 1.0).abs() < 0.01);
    }
}
