//! Enhanced spectral analysis: Welch PSD, coherence, mel spectrogram, MFCC.
//!
//! This module extends `scirs2-fft` with higher-level spectral analysis tools
//! closely modelled on SciPy's `scipy.signal` module:
//!
//! * [`welch`]               — Power Spectral Density via Welch's averaged periodogram.
//! * [`periodogram`]         — Single-segment PSD estimate.
//! * [`coherence`]           — Magnitude-squared coherence between two signals.
//! * [`spectrogram_full`]    — STFT spectrogram with magnitude, phase and dB outputs.
//! * [`mel_filterbank`]      — Mel-frequency triangular filterbank matrix.
//! * [`mel_spectrogram`]     — Spectrogram projected onto the mel scale.
//! * [`log_mel_spectrogram`] — Log-power mel spectrogram.
//! * [`mfcc`]                — Mel-Frequency Cepstral Coefficients.

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;
use crate::helper::next_fast_len;
use crate::window::{get_window, Window};
use scirs2_core::ndarray::{Array2, Axis};
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  Public configuration types
// ─────────────────────────────────────────────────────────────────────────────

/// Window type selector for the enhanced spectral API.
///
/// Maps to the underlying `crate::window::Window` variants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Hann (raised cosine) window — good general-purpose choice.
    Hann,
    /// Hamming window — slightly less spectral leakage than Hann.
    Hamming,
    /// Blackman window — very low side-lobes at the cost of wider main lobe.
    Blackman,
    /// Bartlett (triangular) window.
    Bartlett,
    /// Rectangular (no windowing) — maximum frequency resolution.
    Rectangular,
}

impl WindowType {
    fn to_window(self) -> Window {
        match self {
            WindowType::Hann => Window::Hann,
            WindowType::Hamming => Window::Hamming,
            WindowType::Blackman => Window::Blackman,
            WindowType::Bartlett => Window::Bartlett,
            WindowType::Rectangular => Window::Rectangular,
        }
    }
}

/// PSD scaling convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PsdScaling {
    /// Power spectral *density* (V²/Hz).  Divides by `fs * W²`.
    Density,
    /// Power *spectrum* (V²).  Divides by `W²`.
    Spectrum,
}

/// Result of a PSD or coherence computation.
#[derive(Debug, Clone)]
pub struct PsdResult {
    /// Frequency axis (Hz).
    pub frequencies: Vec<f64>,
    /// Power (or coherence) values.
    pub power: Vec<f64>,
}

/// Configuration for Welch's method.
#[derive(Debug, Clone)]
pub struct WelchConfig {
    /// Window function.
    pub window: WindowType,
    /// Window / segment length in samples.
    pub window_len: usize,
    /// Number of samples shared between consecutive segments.
    pub overlap: usize,
    /// FFT length (defaults to `window_len`).
    pub n_fft: Option<usize>,
    /// PSD scaling convention.
    pub scaling: PsdScaling,
}

impl Default for WelchConfig {
    fn default() -> Self {
        Self {
            window: WindowType::Hann,
            window_len: 256,
            overlap: 128,
            n_fft: None,
            scaling: PsdScaling::Density,
        }
    }
}

/// Configuration for the enhanced STFT spectrogram.
#[derive(Debug, Clone)]
pub struct SpectrogramConfig {
    /// Window length in samples.
    pub window_len: usize,
    /// Hop (step) length in samples.
    pub hop_length: usize,
    /// FFT size (must be >= `window_len`).
    pub n_fft: usize,
    /// Window function.
    pub window: WindowType,
    /// If true, pad both ends so the first/last frames are centred on the
    /// first/last sample.
    pub center: bool,
}

impl Default for SpectrogramConfig {
    fn default() -> Self {
        Self {
            window_len: 256,
            hop_length: 128,
            n_fft: 256,
            window: WindowType::Hann,
            center: true,
        }
    }
}

/// Result of the enhanced STFT spectrogram.
#[derive(Debug, Clone)]
pub struct SpectrogramResult {
    /// Time axis (s).
    pub times: Vec<f64>,
    /// Frequency axis (Hz).
    pub frequencies: Vec<f64>,
    /// Magnitude matrix (n_freq × n_time).
    pub magnitude: Array2<f64>,
    /// Phase matrix in radians (n_freq × n_time).
    pub phase: Array2<f64>,
    /// Power in dB re max power (n_freq × n_time).
    pub power_db: Array2<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a window and compute the one-sided FFT magnitude² spectrum of a
/// segment.  Returns a vector of length `n_fft/2 + 1`.
fn segment_power(
    segment: &[f64],
    win: &[f64],
    n_fft: usize,
) -> FFTResult<Vec<f64>> {
    // Apply window.
    let mut windowed = vec![0.0_f64; n_fft];
    for (i, (&s, &w)) in segment.iter().zip(win.iter()).enumerate() {
        if i < n_fft {
            windowed[i] = s * w;
        }
    }
    let spectrum = fft(&windowed, None)?;
    let n_out = n_fft / 2 + 1;
    Ok(spectrum[..n_out].iter().map(|c| c.norm_sqr()).collect())
}

/// Compute the cross-power spectrum (one-sided, length `n_fft/2+1`) between
/// two windowed segments as a vector of Complex64.
fn segment_cross_power(
    seg_x: &[f64],
    seg_y: &[f64],
    win: &[f64],
    n_fft: usize,
) -> FFTResult<Vec<scirs2_core::numeric::Complex64>> {
    let mut wx = vec![0.0_f64; n_fft];
    let mut wy = vec![0.0_f64; n_fft];
    for (i, ((&sx, &sy), &w)) in seg_x.iter().zip(seg_y.iter()).zip(win.iter()).enumerate() {
        if i < n_fft {
            wx[i] = sx * w;
            wy[i] = sy * w;
        }
    }
    let x_freq = fft(&wx, None)?;
    let y_freq = fft(&wy, None)?;
    let n_out = n_fft / 2 + 1;
    Ok(x_freq[..n_out]
        .iter()
        .zip(y_freq[..n_out].iter())
        .map(|(&xf, &yf)| xf.conj() * yf)
        .collect())
}

/// Build the list of segment start indices.
fn segment_starts(sig_len: usize, win_len: usize, step: usize) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut pos = 0usize;
    while pos + win_len <= sig_len {
        starts.push(pos);
        pos += step;
    }
    starts
}

// ─────────────────────────────────────────────────────────────────────────────
//  welch
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the Power Spectral Density (PSD) using Welch's method.
///
/// The signal is divided into overlapping segments; each is windowed and
/// FFT'd; the resulting power spectra are averaged.
///
/// # Arguments
///
/// * `x`      – Real input signal.
/// * `fs`     – Sampling frequency (Hz).
/// * `config` – Algorithm parameters.
///
/// # Errors
///
/// Returns an error if the signal is too short for even one segment.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectrogram_enhanced::{welch, WelchConfig};
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let signal: Vec<f64> = (0..2048)
///     .map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin())
///     .collect();
/// let psd = welch(&signal, fs, WelchConfig::default()).expect("welch");
/// assert_eq!(psd.frequencies.len(), psd.power.len());
/// // Frequency resolution = fs / n_fft
/// let df = psd.frequencies[1] - psd.frequencies[0];
/// assert!((df - fs / WelchConfig::default().window_len as f64).abs() < 1e-6);
/// ```
pub fn welch(x: &[f64], fs: f64, config: WelchConfig) -> FFTResult<PsdResult> {
    let win_len = config.window_len;
    let n_fft = config.n_fft.unwrap_or(win_len);
    let step = if config.overlap >= win_len {
        return Err(FFTError::ValueError(
            "welch: overlap must be < window_len".into(),
        ));
    } else {
        win_len - config.overlap
    };

    if x.len() < win_len {
        return Err(FFTError::ValueError(format!(
            "welch: signal length {} < window_len {}",
            x.len(),
            win_len
        )));
    }

    let win = get_window(config.window.to_window(), win_len, true)?;
    let win_vec: Vec<f64> = win.to_vec();
    let win_sum_sq: f64 = win_vec.iter().map(|&w| w * w).sum();

    let scale = match config.scaling {
        PsdScaling::Density => 1.0 / (fs * win_sum_sq),
        PsdScaling::Spectrum => 1.0 / win_sum_sq,
    };

    let starts = segment_starts(x.len(), win_len, step);
    let n_out = n_fft / 2 + 1;
    let mut psd_acc = vec![0.0_f64; n_out];
    let n_seg = starts.len();

    for &s in &starts {
        let seg = &x[s..s + win_len];
        let pw = segment_power(seg, &win_vec, n_fft)?;
        for (acc, &p) in psd_acc.iter_mut().zip(pw.iter()) {
            *acc += p;
        }
    }

    // Average and scale; double the non-DC, non-Nyquist bins.
    let n_seg_f = n_seg as f64;
    let mut power: Vec<f64> = psd_acc
        .iter()
        .enumerate()
        .map(|(k, &p)| {
            let p_avg = p / n_seg_f * scale;
            // One-sided: double all except DC (k=0) and Nyquist (k = n_fft/2).
            if k == 0 || k == n_fft / 2 {
                p_avg
            } else {
                2.0 * p_avg
            }
        })
        .collect();

    // Clamp numerical noise.
    power.iter_mut().for_each(|p| {
        if *p < 0.0 {
            *p = 0.0;
        }
    });

    let frequencies: Vec<f64> = (0..n_out).map(|k| k as f64 * fs / n_fft as f64).collect();

    Ok(PsdResult { frequencies, power })
}

// ─────────────────────────────────────────────────────────────────────────────
//  periodogram
// ─────────────────────────────────────────────────────────────────────────────

/// Single-segment (whole-signal) periodogram.
///
/// Equivalent to `welch` with one segment and no overlap.
///
/// # Errors
///
/// Returns an error if the input is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectrogram_enhanced::{periodogram, WindowType};
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let sig: Vec<f64> = (0..512).map(|i| (2.0*PI*100.0*i as f64/fs).sin()).collect();
/// let psd = periodogram(&sig, fs, WindowType::Hann).expect("periodogram");
/// assert!(!psd.power.is_empty());
/// ```
pub fn periodogram(x: &[f64], fs: f64, window: WindowType) -> FFTResult<PsdResult> {
    if x.is_empty() {
        return Err(FFTError::ValueError("periodogram: x is empty".into()));
    }
    let n = x.len();
    let n_fft = next_fast_len(n, true);
    let win = get_window(window.to_window(), n, true)?;
    let win_vec: Vec<f64> = win.to_vec();
    let win_sum_sq: f64 = win_vec.iter().map(|&w| w * w).sum();
    let scale = 1.0 / (fs * win_sum_sq);

    let pw = segment_power(x, &win_vec, n_fft)?;
    let n_out = n_fft / 2 + 1;
    let power: Vec<f64> = pw
        .iter()
        .enumerate()
        .map(|(k, &p)| {
            let ps = p * scale;
            if k == 0 || k == n_fft / 2 {
                ps
            } else {
                2.0 * ps
            }
        })
        .collect();
    let frequencies: Vec<f64> = (0..n_out).map(|k| k as f64 * fs / n_fft as f64).collect();
    Ok(PsdResult { frequencies, power })
}

// ─────────────────────────────────────────────────────────────────────────────
//  coherence
// ─────────────────────────────────────────────────────────────────────────────

/// Magnitude-squared coherence between two signals.
///
/// `Cxy(f) = |Pxy(f)|² / (Pxx(f) · Pyy(f))`
///
/// Values range from 0 (incoherent) to 1 (fully coherent).
///
/// # Arguments
///
/// * `x`, `y`      – Input signals (must have the same length).
/// * `fs`          – Sampling frequency.
/// * `window_len`  – Segment length (samples).
/// * `overlap`     – Overlap between segments (samples).
///
/// # Errors
///
/// Returns an error if signal lengths differ, overlap is too large, or an FFT
/// fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectrogram_enhanced::coherence;
///
/// let n = 1024usize;
/// let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
/// // Identical signals → coherence ≈ 1.0 everywhere.
/// let coh = coherence(&x, &x, 1000.0, 256, 128).expect("coherence");
/// for &c in &coh.power {
///     assert!(c >= 0.0 && c <= 1.0 + 1e-9);
/// }
/// ```
pub fn coherence(
    x: &[f64],
    y: &[f64],
    fs: f64,
    window_len: usize,
    overlap: usize,
) -> FFTResult<PsdResult> {
    if x.len() != y.len() {
        return Err(FFTError::DimensionError(format!(
            "coherence: x.len()={} != y.len()={}",
            x.len(),
            y.len()
        )));
    }
    if overlap >= window_len {
        return Err(FFTError::ValueError(
            "coherence: overlap must be < window_len".into(),
        ));
    }
    if x.len() < window_len {
        return Err(FFTError::ValueError(
            "coherence: signal too short for one window".into(),
        ));
    }

    let n_fft = window_len;
    let step = window_len - overlap;
    let win = get_window(Window::Hann, window_len, true)?;
    let win_vec: Vec<f64> = win.to_vec();

    let starts = segment_starts(x.len(), window_len, step);
    let n_out = n_fft / 2 + 1;

    let mut pxx = vec![0.0_f64; n_out];
    let mut pyy = vec![0.0_f64; n_out];
    let mut pxy_re = vec![0.0_f64; n_out];
    let mut pxy_im = vec![0.0_f64; n_out];

    for &s in &starts {
        let sx = &x[s..s + window_len];
        let sy = &y[s..s + window_len];

        let px = segment_power(sx, &win_vec, n_fft)?;
        let py = segment_power(sy, &win_vec, n_fft)?;
        let cp = segment_cross_power(sx, sy, &win_vec, n_fft)?;

        for k in 0..n_out {
            pxx[k] += px[k];
            pyy[k] += py[k];
            pxy_re[k] += cp[k].re;
            pxy_im[k] += cp[k].im;
        }
    }

    let n_seg = starts.len() as f64;
    let coh: Vec<f64> = (0..n_out)
        .map(|k| {
            let pxx_k = pxx[k] / n_seg;
            let pyy_k = pyy[k] / n_seg;
            let re = pxy_re[k] / n_seg;
            let im = pxy_im[k] / n_seg;
            let pxy_sq = re * re + im * im;
            let denom = pxx_k * pyy_k;
            if denom > 0.0 {
                (pxy_sq / denom).min(1.0)
            } else {
                0.0
            }
        })
        .collect();

    let frequencies: Vec<f64> = (0..n_out).map(|k| k as f64 * fs / n_fft as f64).collect();
    Ok(PsdResult {
        frequencies,
        power: coh,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
//  spectrogram_full  (new enhanced API)
// ─────────────────────────────────────────────────────────────────────────────

/// Enhanced STFT spectrogram with magnitude, phase and power_db outputs.
///
/// # Arguments
///
/// * `x`      – Real input signal.
/// * `fs`     – Sampling frequency (Hz).
/// * `config` – Spectrogram parameters.
///
/// # Returns
///
/// A [`SpectrogramResult`] containing time/frequency axes plus magnitude,
/// phase and power_dB matrices of shape `(n_freq, n_time)` where
/// `n_freq = n_fft / 2 + 1`.
///
/// # Errors
///
/// Returns an error if the configuration is invalid or an FFT fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectrogram_enhanced::{spectrogram_full, SpectrogramConfig};
///
/// let fs = 1000.0_f64;
/// let sig: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.1).sin()).collect();
/// let result = spectrogram_full(&sig, fs, SpectrogramConfig::default())
///     .expect("spectrogram_full");
/// let nf = result.frequencies.len();
/// let nt = result.times.len();
/// assert_eq!(result.magnitude.shape(), &[nf, nt]);
/// assert_eq!(result.phase.shape(), &[nf, nt]);
/// assert_eq!(result.power_db.shape(), &[nf, nt]);
/// ```
pub fn spectrogram_full(
    x: &[f64],
    fs: f64,
    config: SpectrogramConfig,
) -> FFTResult<SpectrogramResult> {
    let win_len = config.window_len;
    let hop = config.hop_length;
    let n_fft = config.n_fft;

    if win_len == 0 {
        return Err(FFTError::ValueError(
            "spectrogram_full: window_len must be > 0".into(),
        ));
    }
    if hop == 0 {
        return Err(FFTError::ValueError(
            "spectrogram_full: hop_length must be > 0".into(),
        ));
    }
    if n_fft < win_len {
        return Err(FFTError::ValueError(
            "spectrogram_full: n_fft must be >= window_len".into(),
        ));
    }

    let win = get_window(config.window.to_window(), win_len, true)?;
    let win_vec: Vec<f64> = win.to_vec();

    // Optionally pad so frames are centred on the first/last sample.
    let (sig, offset) = if config.center {
        let pad = n_fft / 2;
        let mut padded = vec![0.0_f64; pad + x.len() + pad];
        padded[pad..pad + x.len()].copy_from_slice(x);
        (padded, pad as f64 / fs)
    } else {
        (x.to_vec(), 0.0_f64)
    };

    let n_fft_fast = next_fast_len(n_fft, true);
    let n_freq = n_fft_fast / 2 + 1;

    // Build frame start positions.
    let mut starts = Vec::new();
    let mut pos = 0usize;
    while pos + win_len <= sig.len() {
        starts.push(pos);
        pos += hop;
    }

    let n_time = starts.len();
    if n_time == 0 {
        return Err(FFTError::ValueError(
            "spectrogram_full: signal too short for one frame".into(),
        ));
    }

    let mut magnitude = Array2::<f64>::zeros((n_freq, n_time));
    let mut phase = Array2::<f64>::zeros((n_freq, n_time));

    for (ti, &s) in starts.iter().enumerate() {
        let seg = &sig[s..s + win_len];
        // Apply window and zero-pad to n_fft_fast.
        let mut frame = vec![0.0_f64; n_fft_fast];
        for (k, (&sv, &wv)) in seg.iter().zip(win_vec.iter()).enumerate() {
            frame[k] = sv * wv;
        }
        let spectrum = fft(&frame, None)?;
        for fi in 0..n_freq {
            magnitude[[fi, ti]] = spectrum[fi].norm();
            phase[[fi, ti]] = spectrum[fi].im.atan2(spectrum[fi].re);
        }
    }

    // Compute power_db relative to the global maximum magnitude².
    let max_power = {
        let mut mx = 0.0_f64;
        for &v in magnitude.iter() {
            let p = v * v;
            if p > mx {
                mx = p;
            }
        }
        mx
    };
    let ref_power = if max_power > 0.0 {
        max_power
    } else {
        1.0
    };

    let mut power_db = Array2::<f64>::zeros((n_freq, n_time));
    for r in 0..n_freq {
        for c in 0..n_time {
            let p = magnitude[[r, c]] * magnitude[[r, c]];
            power_db[[r, c]] = if p > 0.0 {
                10.0 * (p / ref_power).log10()
            } else {
                -120.0
            };
        }
    }

    // Build time axis.
    let times: Vec<f64> = starts
        .iter()
        .map(|&s| s as f64 / fs - offset)
        .collect();

    let frequencies: Vec<f64> = (0..n_freq)
        .map(|k| k as f64 * fs / n_fft_fast as f64)
        .collect();

    Ok(SpectrogramResult {
        times,
        frequencies,
        magnitude,
        phase,
        power_db,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
//  mel_filterbank
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a frequency (Hz) to the mel scale.
#[inline]
fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert a mel value to Hz.
#[inline]
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

/// Build a mel-scale triangular filterbank matrix.
///
/// Returns a matrix of shape `(n_mels, n_fft/2 + 1)`.  Each row is one
/// mel filter, mapping from the one-sided linear frequency axis to a mel band.
///
/// # Arguments
///
/// * `n_mels` – Number of mel filters.
/// * `n_fft`  – FFT size (the filterbank width is `n_fft/2 + 1`).
/// * `fs`     – Sampling frequency (Hz).
/// * `f_min`  – Lowest frequency (Hz).
/// * `f_max`  – Highest frequency (Hz, must be ≤ `fs/2`).
///
/// # Errors
///
/// Returns an error if arguments are invalid (e.g. `f_max > fs/2`).
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectrogram_enhanced::mel_filterbank;
///
/// let fb = mel_filterbank(40, 512, 16000.0, 0.0, 8000.0)
///     .expect("filterbank");
/// assert_eq!(fb.shape(), &[40, 257]);
/// ```
pub fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    fs: f64,
    f_min: f64,
    f_max: f64,
) -> FFTResult<Array2<f64>> {
    if n_mels == 0 {
        return Err(FFTError::ValueError(
            "mel_filterbank: n_mels must be > 0".into(),
        ));
    }
    if n_fft == 0 {
        return Err(FFTError::ValueError(
            "mel_filterbank: n_fft must be > 0".into(),
        ));
    }
    if f_max > fs / 2.0 + 1e-9 {
        return Err(FFTError::ValueError(format!(
            "mel_filterbank: f_max ({f_max}) must be <= fs/2 ({})",
            fs / 2.0
        )));
    }
    if f_min < 0.0 {
        return Err(FFTError::ValueError(
            "mel_filterbank: f_min must be >= 0".into(),
        ));
    }

    let n_bins = n_fft / 2 + 1;

    // Linear frequency axis.
    let freqs: Vec<f64> = (0..n_bins).map(|k| k as f64 * fs / n_fft as f64).collect();

    // Mel-scale centre frequencies including two boundary points.
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels + 1) as f64))
        .collect();

    let mut fb = Array2::<f64>::zeros((n_mels, n_bins));

    for m in 0..n_mels {
        let f_m_minus = mel_points[m];
        let f_m = mel_points[m + 1];
        let f_m_plus = mel_points[m + 2];

        for (k, &f_k) in freqs.iter().enumerate() {
            let val = if f_k >= f_m_minus && f_k < f_m {
                (f_k - f_m_minus) / (f_m - f_m_minus)
            } else if f_k >= f_m && f_k <= f_m_plus {
                (f_m_plus - f_k) / (f_m_plus - f_m)
            } else {
                0.0
            };
            fb[[m, k]] = val;
        }
    }

    Ok(fb)
}

// ─────────────────────────────────────────────────────────────────────────────
//  mel_spectrogram
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the mel-frequency spectrogram.
///
/// Applies the STFT and projects the power spectrum onto a mel filterbank.
///
/// # Returns
///
/// A matrix of shape `(n_mels, n_time)`.
///
/// # Errors
///
/// Returns an error if any sub-computation fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectrogram_enhanced::{mel_spectrogram, SpectrogramConfig};
///
/// let fs = 22050.0_f64;
/// let sig: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.05).sin()).collect();
/// let mel = mel_spectrogram(&sig, fs, 40, SpectrogramConfig::default())
///     .expect("mel_spectrogram");
/// assert_eq!(mel.shape()[0], 40);
/// ```
pub fn mel_spectrogram(
    x: &[f64],
    fs: f64,
    n_mels: usize,
    config: SpectrogramConfig,
) -> FFTResult<Array2<f64>> {
    let n_fft = config.n_fft;
    let f_max = fs / 2.0;
    let fb = mel_filterbank(n_mels, n_fft, fs, 0.0, f_max)?;

    let result = spectrogram_full(x, fs, config)?;
    let power = result.magnitude.mapv(|v| v * v); // (n_freq, n_time)

    let n_time = power.ncols();
    let n_bins = n_fft / 2 + 1;
    // fb: (n_mels, n_bins); power: (n_bins_fast, n_time)
    // Use the first n_bins rows of power (power may be slightly larger due to
    // next_fast_len).
    let n_freq_actual = power.nrows().min(n_bins);
    let mut mel_spec = Array2::<f64>::zeros((n_mels, n_time));

    for m in 0..n_mels {
        for t in 0..n_time {
            let mut acc = 0.0_f64;
            for k in 0..n_freq_actual {
                acc += fb[[m, k]] * power[[k, t]];
            }
            mel_spec[[m, t]] = acc;
        }
    }

    Ok(mel_spec)
}

// ─────────────────────────────────────────────────────────────────────────────
//  log_mel_spectrogram
// ─────────────────────────────────────────────────────────────────────────────

/// Log-power mel spectrogram (first step of MFCC computation).
///
/// Returns `10 * log10(mel_power + ε)` with `ε = 1e-10` to avoid `-inf`.
///
/// # Returns
///
/// A matrix of shape `(n_mels, n_time)`.
///
/// # Errors
///
/// Propagates errors from [`mel_spectrogram`].
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectrogram_enhanced::{log_mel_spectrogram, SpectrogramConfig};
///
/// let fs = 22050.0_f64;
/// let sig: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.05).sin()).collect();
/// let log_mel = log_mel_spectrogram(&sig, fs, 40, SpectrogramConfig::default())
///     .expect("log_mel");
/// assert_eq!(log_mel.shape()[0], 40);
/// ```
pub fn log_mel_spectrogram(
    x: &[f64],
    fs: f64,
    n_mels: usize,
    config: SpectrogramConfig,
) -> FFTResult<Array2<f64>> {
    let mel = mel_spectrogram(x, fs, n_mels, config)?;
    Ok(mel.mapv(|v| 10.0 * (v + 1e-10_f64).log10()))
}

// ─────────────────────────────────────────────────────────────────────────────
//  mfcc
// ─────────────────────────────────────────────────────────────────────────────

/// Mel-Frequency Cepstral Coefficients (MFCCs).
///
/// Applies the discrete cosine transform (DCT-II) to the log-mel spectrogram
/// rows, then returns the first `n_mfcc` coefficients.
///
/// # Returns
///
/// A matrix of shape `(n_mfcc, n_time)`.
///
/// # Errors
///
/// Returns an error if `n_mfcc > n_mels` or any sub-computation fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectrogram_enhanced::{mfcc, SpectrogramConfig};
///
/// let fs = 22050.0_f64;
/// let sig: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.05).sin()).collect();
/// let coeffs = mfcc(&sig, fs, 13, 40, SpectrogramConfig::default())
///     .expect("mfcc");
/// assert_eq!(coeffs.shape()[0], 13);
/// ```
pub fn mfcc(
    x: &[f64],
    fs: f64,
    n_mfcc: usize,
    n_mels: usize,
    config: SpectrogramConfig,
) -> FFTResult<Array2<f64>> {
    if n_mfcc == 0 {
        return Err(FFTError::ValueError("mfcc: n_mfcc must be > 0".into()));
    }
    if n_mfcc > n_mels {
        return Err(FFTError::ValueError(format!(
            "mfcc: n_mfcc ({n_mfcc}) must be <= n_mels ({n_mels})"
        )));
    }

    let log_mel = log_mel_spectrogram(x, fs, n_mels, config)?; // (n_mels, n_time)
    let n_time = log_mel.ncols();

    // Apply DCT-II along the mel axis for each time frame.
    // DCT-II[k] = Σ_{n=0}^{N-1} x[n] * cos(π*k*(n+0.5)/N)
    let mut out = Array2::<f64>::zeros((n_mfcc, n_time));

    for t in 0..n_time {
        let col: Vec<f64> = log_mel.column(t).to_vec();
        let n = col.len() as f64;
        for k in 0..n_mfcc {
            let mut sum = 0.0_f64;
            for (i, &v) in col.iter().enumerate() {
                sum += v * (PI * k as f64 * (i as f64 + 0.5) / n).cos();
            }
            out[[k, t]] = sum;
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_wave(freq: f64, fs: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    // ── Welch ────────────────────────────────────────────────────────────────

    #[test]
    fn test_welch_frequency_resolution() {
        let fs = 1000.0_f64;
        let cfg = WelchConfig {
            window_len: 256,
            overlap: 128,
            n_fft: None,
            ..Default::default()
        };
        let sig = sine_wave(100.0, fs, 2048);
        let psd = welch(&sig, fs, cfg.clone()).expect("welch");
        // Frequency resolution should be fs / window_len.
        let df = psd.frequencies[1] - psd.frequencies[0];
        let expected_df = fs / cfg.window_len as f64;
        assert!(
            (df - expected_df).abs() < 1e-6,
            "expected df={expected_df}, got {df}"
        );
        assert_eq!(psd.frequencies.len(), psd.power.len());
    }

    #[test]
    fn test_welch_peak_at_signal_frequency() {
        let fs = 1000.0_f64;
        let f0 = 100.0_f64;
        let sig = sine_wave(f0, fs, 4096);
        let psd = welch(&sig, fs, WelchConfig::default()).expect("welch_peak");
        // Find bin closest to f0.
        let peak_idx = psd
            .frequencies
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| {
                (a - f0)
                    .abs()
                    .partial_cmp(&(b - f0).abs())
                    .expect("cmp ok")
            })
            .map(|(i, _)| i)
            .expect("min_by");
        let peak_power = psd.power[peak_idx];
        let mean_power = psd.power.iter().sum::<f64>() / psd.power.len() as f64;
        assert!(
            peak_power > 5.0 * mean_power,
            "peak={peak_power} mean={mean_power}"
        );
    }

    #[test]
    fn test_periodogram_output_non_negative() {
        let fs = 1000.0_f64;
        let sig = sine_wave(50.0, fs, 512);
        let psd = periodogram(&sig, fs, WindowType::Hann).expect("periodogram");
        for &p in &psd.power {
            assert!(p >= 0.0, "negative power: {p}");
        }
    }

    // ── Coherence ────────────────────────────────────────────────────────────

    #[test]
    fn test_coherence_identical_signals() {
        let n = 2048usize;
        let x = sine_wave(100.0, 1000.0, n);
        let coh = coherence(&x, &x, 1000.0, 256, 128).expect("coh_identical");
        // DC bin may be degenerate; check all other bins.
        for (k, &c) in coh.power.iter().enumerate() {
            assert!(
                c >= 0.0 && c <= 1.0 + 1e-9,
                "coherence[{k}]={c} out of [0,1]"
            );
        }
        // At the signal frequency the coherence should be close to 1.
        let f0_idx = coh
            .frequencies
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| {
                (a - 100.0)
                    .abs()
                    .partial_cmp(&(b - 100.0).abs())
                    .expect("cmp ok")
            })
            .map(|(i, _)| i)
            .expect("idx");
        assert!(
            coh.power[f0_idx] > 0.9,
            "coherence at f0 = {}",
            coh.power[f0_idx]
        );
    }

    #[test]
    fn test_coherence_length_mismatch_error() {
        let x = vec![1.0_f64; 512];
        let y = vec![1.0_f64; 256];
        assert!(coherence(&x, &y, 1000.0, 256, 128).is_err());
    }

    // ── spectrogram_full ─────────────────────────────────────────────────────

    #[test]
    fn test_spectrogram_full_shapes() {
        let fs = 1000.0_f64;
        let sig = sine_wave(100.0, fs, 2000);
        let cfg = SpectrogramConfig {
            window_len: 256,
            hop_length: 128,
            n_fft: 256,
            window: WindowType::Hann,
            center: false,
        };
        let res = spectrogram_full(&sig, fs, cfg.clone()).expect("spec_full");
        let n_fft_fast = next_fast_len(cfg.n_fft, true);
        let expected_n_freq = n_fft_fast / 2 + 1;
        assert_eq!(res.frequencies.len(), expected_n_freq);
        let n_time = res.times.len();
        assert_eq!(res.magnitude.shape(), &[expected_n_freq, n_time]);
        assert_eq!(res.phase.shape(), &[expected_n_freq, n_time]);
        assert_eq!(res.power_db.shape(), &[expected_n_freq, n_time]);
    }

    #[test]
    fn test_spectrogram_full_power_db_range() {
        let fs = 1000.0_f64;
        let sig = sine_wave(200.0, fs, 2000);
        let res =
            spectrogram_full(&sig, fs, SpectrogramConfig::default()).expect("spec_db");
        // All dB values should be non-positive (reference is max power).
        for &v in res.power_db.iter() {
            assert!(v <= 0.01, "power_db > 0: {v}");
        }
    }

    // ── mel_filterbank ───────────────────────────────────────────────────────

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = mel_filterbank(40, 512, 16000.0, 0.0, 8000.0).expect("fb");
        assert_eq!(fb.shape(), &[40, 257]);
    }

    #[test]
    fn test_mel_filterbank_non_negative() {
        let fb = mel_filterbank(40, 512, 16000.0, 0.0, 8000.0).expect("fb_nonneg");
        for &v in fb.iter() {
            assert!(v >= 0.0, "filter weight < 0: {v}");
        }
    }

    #[test]
    fn test_mel_filterbank_invalid_fmax() {
        assert!(mel_filterbank(40, 512, 16000.0, 0.0, 9000.0).is_err());
    }

    // ── mel_spectrogram ──────────────────────────────────────────────────────

    #[test]
    fn test_mel_spectrogram_shape() {
        let fs = 22050.0_f64;
        let sig: Vec<f64> = (0..8192).map(|i| (i as f64 * 0.05).sin()).collect();
        let mel = mel_spectrogram(&sig, fs, 40, SpectrogramConfig::default())
            .expect("mel_spec");
        assert_eq!(mel.shape()[0], 40);
        assert!(mel.shape()[1] > 0);
    }

    #[test]
    fn test_mel_spectrogram_non_negative() {
        let fs = 22050.0_f64;
        let sig: Vec<f64> = (0..8192).map(|i| (i as f64 * 0.05).sin()).collect();
        let mel = mel_spectrogram(&sig, fs, 40, SpectrogramConfig::default())
            .expect("mel_nonneg");
        for &v in mel.iter() {
            assert!(v >= 0.0, "mel < 0: {v}");
        }
    }

    // ── MFCC ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_mfcc_shape() {
        let fs = 22050.0_f64;
        let sig: Vec<f64> = (0..8192).map(|i| (i as f64 * 0.05).sin()).collect();
        let coeffs = mfcc(&sig, fs, 13, 40, SpectrogramConfig::default()).expect("mfcc_shape");
        assert_eq!(coeffs.shape()[0], 13);
        assert!(coeffs.shape()[1] > 0);
    }

    #[test]
    fn test_mfcc_invalid_n_mfcc() {
        let fs = 22050.0_f64;
        let sig: Vec<f64> = vec![0.0; 4096];
        // n_mfcc > n_mels should be an error.
        assert!(mfcc(&sig, fs, 50, 40, SpectrogramConfig::default()).is_err());
    }

    // ── log_mel ───────────────────────────────────────────────────────────────

    #[test]
    fn test_log_mel_spectrogram_shape() {
        let fs = 22050.0_f64;
        let sig: Vec<f64> = (0..4096).map(|i| (i as f64 * 0.05).sin()).collect();
        let log_mel =
            log_mel_spectrogram(&sig, fs, 40, SpectrogramConfig::default()).expect("log_mel");
        assert_eq!(log_mel.shape()[0], 40);
    }
}
