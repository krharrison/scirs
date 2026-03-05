//! Advanced spectral analysis utilities.
//!
//! This module provides spectral estimation methods complementing the basic
//! periodogram already in `spectrogram_enhanced`:
//!
//! * [`welch_psd`]        — Welch's averaged periodogram PSD estimate.
//! * [`bartlett_psd`]     — Bartlett's method (Welch with no overlap).
//! * [`lomb_scargle`]     — Lomb-Scargle periodogram for unevenly sampled data.
//! * [`multitaper_psd`]   — Thomson's multitaper PSD using DPSS tapers.
//! * [`coherence`]        — Magnitude-squared coherence between two signals.
//!
//! # References
//!
//! * Welch, P.D. "The use of fast Fourier transform for the estimation of
//!   power spectra." IEEE Trans. Audio Electroacoust., 1967.
//! * Scargle, J.D. "Studies in astronomical time series analysis. II."
//!   Astrophys. J., 263, 835-853, 1982.
//! * Thomson, D.J. "Spectrum estimation and harmonic analysis."
//!   Proc. IEEE, 70(9), 1055-1096, 1982.

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;
use crate::helper::next_fast_len;
use crate::window_functions::dpss;
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n as f64 - 1.0)).cos()))
        .collect()
}

/// Apply a window to a segment and zero-pad to `fft_len`.
fn windowed_fft(segment: &[f64], window: &[f64], fft_len: usize) -> FFTResult<Vec<Complex64>> {
    let seg_len = segment.len().min(window.len());
    let mut buf = vec![0.0_f64; fft_len];
    for i in 0..seg_len {
        buf[i] = segment[i] * window[i];
    }
    fft(&buf, None)
}

/// Build the one-sided frequency axis for `nfft` points sampled at `fs`.
fn rfft_freqs(nfft: usize, fs: f64) -> Vec<f64> {
    let n_freqs = nfft / 2 + 1;
    (0..n_freqs)
        .map(|k| k as f64 * fs / nfft as f64)
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  welch_psd
// ─────────────────────────────────────────────────────────────────────────────

/// Welch's method for Power Spectral Density estimation.
///
/// Splits the signal into overlapping segments, applies a Hann window,
/// computes the periodogram of each segment, and averages the results.
///
/// # Arguments
///
/// * `signal`      – Real-valued input signal.
/// * `window_size` – Length of each segment in samples.
/// * `overlap`     – Number of overlapping samples between adjacent segments.
/// * `fs`          – Sampling frequency (Hz).
///
/// # Returns
///
/// `(frequencies, psd)` where `frequencies` is the one-sided frequency axis
/// and `psd` is the power spectral density in units of V²/Hz.
///
/// # Errors
///
/// Returns an error if the signal is empty, `window_size` is zero, or
/// `overlap >= window_size`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_analysis::welch_psd;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0;
/// let n = 4096;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
///     .collect();
/// let (freqs, psd) = welch_psd(&signal, 256, 128, fs).expect("welch_psd");
/// assert_eq!(freqs.len(), psd.len());
/// assert!(freqs.len() == 256 / 2 + 1);
/// ```
pub fn welch_psd(
    signal: &[f64],
    window_size: usize,
    overlap: usize,
    fs: f64,
) -> FFTResult<(Vec<f64>, Vec<f64>)> {
    if signal.is_empty() {
        return Err(FFTError::ValueError("welch_psd: signal is empty".into()));
    }
    if window_size == 0 {
        return Err(FFTError::ValueError(
            "welch_psd: window_size must be positive".into(),
        ));
    }
    if overlap >= window_size {
        return Err(FFTError::ValueError(
            "welch_psd: overlap must be < window_size".into(),
        ));
    }

    let nfft = next_fast_len(window_size, true);
    let win = hann_window(window_size);
    let win_power = win.iter().map(|&w| w * w).sum::<f64>();

    let hop = window_size - overlap;
    let n_freqs = nfft / 2 + 1;
    let mut psd_acc = vec![0.0_f64; n_freqs];
    let mut n_segments = 0usize;

    let mut start = 0usize;
    while start + window_size <= signal.len() {
        let seg = &signal[start..start + window_size];
        let spectrum = windowed_fft(seg, &win, nfft)?;

        // One-sided periodogram
        for k in 0..n_freqs {
            let power = spectrum[k].norm_sqr();
            psd_acc[k] += if k == 0 || k == nfft / 2 {
                power
            } else {
                2.0 * power // double for two-sided → one-sided
            };
        }

        n_segments += 1;
        start += hop;
    }

    if n_segments == 0 {
        return Err(FFTError::ValueError(
            "welch_psd: signal is shorter than window_size".into(),
        ));
    }

    // Normalize: average over segments, then divide by (fs * win_power)
    let scale = fs * win_power * n_segments as f64 * nfft as f64 * nfft as f64;
    let psd: Vec<f64> = psd_acc.iter().map(|&p| p / scale).collect();

    let freqs = rfft_freqs(nfft, fs);
    Ok((freqs, psd))
}

// ─────────────────────────────────────────────────────────────────────────────
//  bartlett_psd
// ─────────────────────────────────────────────────────────────────────────────

/// Bartlett's method: PSD estimation using non-overlapping segments.
///
/// Equivalent to [`welch_psd`] with `overlap = 0` and a rectangular window.
/// Each segment is multiplied by a Bartlett (triangular) window before the FFT.
///
/// # Arguments
///
/// * `signal`      – Real-valued input signal.
/// * `segment_len` – Length of each non-overlapping segment.
/// * `fs`          – Sampling frequency (Hz).
///
/// # Returns
///
/// `(frequencies, psd)` — one-sided frequency axis and PSD.
///
/// # Errors
///
/// Returns an error if the signal is empty or `segment_len` is zero.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_analysis::bartlett_psd;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0;
/// let signal: Vec<f64> = (0..4096)
///     .map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin())
///     .collect();
/// let (freqs, psd) = bartlett_psd(&signal, 256, fs).expect("bartlett");
/// assert_eq!(freqs.len(), psd.len());
/// ```
pub fn bartlett_psd(signal: &[f64], segment_len: usize, fs: f64) -> FFTResult<(Vec<f64>, Vec<f64>)> {
    welch_psd(signal, segment_len, 0, fs)
}

// ─────────────────────────────────────────────────────────────────────────────
//  lomb_scargle
// ─────────────────────────────────────────────────────────────────────────────

/// Lomb-Scargle periodogram for unevenly sampled data.
///
/// Computes the normalised Lomb-Scargle power at each requested frequency.
/// The normalisation follows Scargle (1982) so the output is dimensionless and
/// lies in [0, 1] when the signal is pure noise.
///
/// # Arguments
///
/// * `t`     – Sample times (arbitrary units, need not be equally spaced).
/// * `y`     – Observed values at times `t`.
/// * `freqs` – Angular frequencies (radians per unit time) at which to evaluate
///             the periodogram.
///
/// # Returns
///
/// Vector of normalised Lomb-Scargle powers, one per entry in `freqs`.
///
/// # Errors
///
/// Returns an error if `t` and `y` have different lengths, `t` has fewer than
/// 2 points, or `freqs` is empty.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_analysis::lomb_scargle;
/// use std::f64::consts::PI;
///
/// // Unevenly spaced samples of a 1 Hz sine wave
/// let t: Vec<f64> = vec![0.0, 0.15, 0.35, 0.48, 0.72, 0.91, 1.10, 1.45, 1.70, 1.95];
/// let y: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * 1.0 * ti).sin()).collect();
/// let freqs = vec![2.0 * PI * 0.5, 2.0 * PI * 1.0, 2.0 * PI * 2.0];
/// let power = lomb_scargle(&t, &y, &freqs).expect("lomb_scargle");
/// assert_eq!(power.len(), freqs.len());
/// // Power should peak near 2*pi*1.0
/// assert!(power[1] > power[0]);
/// ```
pub fn lomb_scargle(t: &[f64], y: &[f64], freqs: &[f64]) -> FFTResult<Vec<f64>> {
    if t.len() != y.len() {
        return Err(FFTError::DimensionError(format!(
            "lomb_scargle: t.len()={} != y.len()={}",
            t.len(),
            y.len()
        )));
    }
    if t.len() < 2 {
        return Err(FFTError::ValueError(
            "lomb_scargle: need at least 2 data points".into(),
        ));
    }
    if freqs.is_empty() {
        return Err(FFTError::ValueError(
            "lomb_scargle: freqs must be non-empty".into(),
        ));
    }

    let n = t.len();
    // Subtract the mean from y
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let y_c: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();
    let y_var = y_c.iter().map(|&yi| yi * yi).sum::<f64>() / n as f64;

    if y_var < 1e-30 {
        // Constant signal — no spectral power.
        return Ok(vec![0.0; freqs.len()]);
    }

    let mut power = Vec::with_capacity(freqs.len());

    for &omega in freqs {
        // Phase offset τ: tan(2ωτ) = Σ sin(2ωt) / Σ cos(2ωt)
        let sum_sin2 = t.iter().map(|&ti| (2.0 * omega * ti).sin()).sum::<f64>();
        let sum_cos2 = t.iter().map(|&ti| (2.0 * omega * ti).cos()).sum::<f64>();
        let tau = (0.5 / omega) * sum_sin2.atan2(sum_cos2);

        // Lomb-Scargle power
        let c_sq: f64 = t
            .iter()
            .zip(y_c.iter())
            .map(|(&ti, &yi)| yi * (omega * (ti - tau)).cos())
            .sum::<f64>()
            .powi(2);

        let s_sq: f64 = t
            .iter()
            .zip(y_c.iter())
            .map(|(&ti, &yi)| yi * (omega * (ti - tau)).sin())
            .sum::<f64>()
            .powi(2);

        let denom_cos: f64 = t
            .iter()
            .map(|&ti| (omega * (ti - tau)).cos().powi(2))
            .sum::<f64>();

        let denom_sin: f64 = t
            .iter()
            .map(|&ti| (omega * (ti - tau)).sin().powi(2))
            .sum::<f64>();

        let p = if denom_cos < 1e-30 || denom_sin < 1e-30 {
            0.0
        } else {
            (c_sq / denom_cos + s_sq / denom_sin) / (2.0 * y_var)
        };

        power.push(p);
    }

    Ok(power)
}

// ─────────────────────────────────────────────────────────────────────────────
//  multitaper_psd
// ─────────────────────────────────────────────────────────────────────────────

/// Thomson's multitaper PSD estimation using DPSS Slepian tapers.
///
/// Uses `k` Slepian (DPSS) tapers with time-bandwidth product `nw` to estimate
/// the PSD.  The tapers are combined with equal weights (simple average).
///
/// # Arguments
///
/// * `signal` – Real-valued input signal.
/// * `nw`     – Time-bandwidth product (typical values: 2.0, 2.5, 3.0, 4.0).
/// * `k`      – Number of tapers (must satisfy `k < 2*nw`; typically `k = 2*nw - 1`).
/// * `fs`     – Sampling frequency (Hz).
///
/// # Returns
///
/// `(frequencies, psd)` — one-sided frequency axis and power spectral density.
///
/// # Errors
///
/// Returns an error if the signal is shorter than 2 samples, `nw <= 0`,
/// `k == 0`, or `k >= 2*nw` (tapers would be too ill-conditioned).
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_analysis::multitaper_psd;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let n = 512;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
///     .collect();
/// let (freqs, psd) = multitaper_psd(&signal, 4.0, 7, fs).expect("multitaper");
/// assert_eq!(freqs.len(), psd.len());
/// // There should be a strong peak near 100 Hz
/// let peak_idx = psd
///     .iter()
///     .enumerate()
///     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
///     .map(|(i, _)| i)
///     .unwrap_or(0);
/// let peak_freq = freqs[peak_idx];
/// assert!((peak_freq - 100.0).abs() < 5.0, "peak at {peak_freq} Hz");
/// ```
pub fn multitaper_psd(
    signal: &[f64],
    nw: f64,
    k: usize,
    fs: f64,
) -> FFTResult<(Vec<f64>, Vec<f64>)> {
    if signal.len() < 2 {
        return Err(FFTError::ValueError(
            "multitaper_psd: signal must have at least 2 samples".into(),
        ));
    }
    if nw <= 0.0 {
        return Err(FFTError::ValueError(
            "multitaper_psd: nw must be positive".into(),
        ));
    }
    if k == 0 {
        return Err(FFTError::ValueError(
            "multitaper_psd: k must be at least 1".into(),
        ));
    }
    let k_max = (2.0 * nw - 1.0).floor() as usize;
    if k > k_max {
        return Err(FFTError::ValueError(format!(
            "multitaper_psd: k={k} exceeds 2*nw-1={k_max}; tapers would be ill-conditioned"
        )));
    }

    let n = signal.len();
    let nfft = next_fast_len(n, true);
    let n_freqs = nfft / 2 + 1;

    // Compute k DPSS tapers
    let tapers = dpss(n, nw, k)?;

    let mut psd_acc = vec![0.0_f64; n_freqs];

    for taper in &tapers {
        // Apply taper and FFT
        let mut buf = vec![0.0_f64; nfft];
        for (i, (&s, &w)) in signal.iter().zip(taper.iter()).enumerate() {
            buf[i] = s * w;
        }
        let spectrum = fft(&buf, None)?;

        // Accumulate one-sided power
        for k_idx in 0..n_freqs {
            let power = spectrum[k_idx].norm_sqr();
            psd_acc[k_idx] += if k_idx == 0 || k_idx == nfft / 2 {
                power
            } else {
                2.0 * power
            };
        }
    }

    // Average over tapers and normalise
    let scale = fs * (n as f64) * (n as f64) * tapers.len() as f64;
    let psd: Vec<f64> = psd_acc.iter().map(|&p| p / scale).collect();

    let freqs = rfft_freqs(nfft, fs);
    Ok((freqs, psd))
}

// ─────────────────────────────────────────────────────────────────────────────
//  coherence
// ─────────────────────────────────────────────────────────────────────────────

/// Magnitude-squared coherence between two signals via Welch's method.
///
/// The coherence `C_xy(f) = |P_xy(f)|² / (P_xx(f) · P_yy(f))` measures the
/// linear relationship between `x` and `y` as a function of frequency.
/// Values lie in [0, 1].
///
/// # Arguments
///
/// * `x`  – First real-valued signal.
/// * `y`  – Second real-valued signal (same length as `x`).
/// * `fs` – Sampling frequency (Hz).
///
/// # Returns
///
/// `(frequencies, coherence)` — one-sided frequency axis and coherence values.
///
/// # Errors
///
/// Returns an error if the signals have different lengths, are empty, or an
/// FFT call fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::spectral_analysis::coherence;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0_f64;
/// let n = 2048;
/// // Identical signals → coherence ≈ 1 everywhere
/// let x: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
///     .collect();
/// let (freqs, coh) = coherence(&x, &x, fs).expect("coherence");
/// assert_eq!(freqs.len(), coh.len());
/// // Near the signal frequency the coherence should be very close to 1
/// let peak = coh.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
/// assert!(peak > 0.99);
/// ```
pub fn coherence(x: &[f64], y: &[f64], fs: f64) -> FFTResult<(Vec<f64>, Vec<f64>)> {
    if x.is_empty() {
        return Err(FFTError::ValueError("coherence: x is empty".into()));
    }
    if x.len() != y.len() {
        return Err(FFTError::DimensionError(format!(
            "coherence: x.len()={} != y.len()={}",
            x.len(),
            y.len()
        )));
    }

    let window_size = 256.min(x.len());
    let overlap = window_size / 2;
    let nfft = next_fast_len(window_size, true);
    let win = hann_window(window_size);
    let hop = window_size - overlap;
    let n_freqs = nfft / 2 + 1;

    let mut pxx = vec![0.0_f64; n_freqs]; // Auto-spectrum x
    let mut pyy = vec![0.0_f64; n_freqs]; // Auto-spectrum y
    let mut pxy = vec![Complex64::new(0.0, 0.0); n_freqs]; // Cross-spectrum
    let mut n_segments = 0usize;

    let mut start = 0usize;
    while start + window_size <= x.len() {
        let sx = windowed_fft(&x[start..start + window_size], &win, nfft)?;
        let sy = windowed_fft(&y[start..start + window_size], &win, nfft)?;

        for k in 0..n_freqs {
            let scale = if k == 0 || k == nfft / 2 { 1.0 } else { 2.0 };
            pxx[k] += scale * sx[k].norm_sqr();
            pyy[k] += scale * sy[k].norm_sqr();
            let cross = sx[k].conj() * sy[k];
            pxy[k] = Complex64::new(
                pxy[k].re + scale * cross.re,
                pxy[k].im + scale * cross.im,
            );
        }

        n_segments += 1;
        start += hop;
    }

    if n_segments == 0 {
        return Err(FFTError::ValueError(
            "coherence: signal is shorter than window_size".into(),
        ));
    }

    // Coherence: |Pxy|² / (Pxx * Pyy)
    let coh: Vec<f64> = (0..n_freqs)
        .map(|k| {
            let denom = pxx[k] * pyy[k];
            if denom < 1e-60 {
                0.0
            } else {
                pxy[k].norm_sqr() / denom
            }
        })
        .collect();

    let freqs = rfft_freqs(nfft, fs);
    Ok((freqs, coh))
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn sine_wave(freq_hz: f64, n: usize, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq_hz * i as f64 / fs).sin())
            .collect()
    }

    #[test]
    fn test_welch_psd_output_shape() {
        let sig = sine_wave(100.0, 4096, 1000.0);
        let (freqs, psd) = welch_psd(&sig, 256, 128, 1000.0).expect("welch");
        assert_eq!(freqs.len(), psd.len());
        assert_eq!(freqs.len(), 256 / 2 + 1);
    }

    #[test]
    fn test_welch_psd_peak_near_signal_freq() {
        let fs = 1000.0_f64;
        let sig = sine_wave(100.0, 8192, fs);
        let (freqs, psd) = welch_psd(&sig, 512, 256, fs).expect("welch");
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("comparison"))
            .map(|(i, _)| i)
            .expect("peak");
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - 100.0).abs() < 5.0,
            "Peak at {peak_freq} Hz, expected ~100 Hz"
        );
    }

    #[test]
    fn test_bartlett_psd_no_overlap() {
        let fs = 1000.0_f64;
        let sig = sine_wave(200.0, 4096, fs);
        let (f1, p1) = welch_psd(&sig, 256, 0, fs).expect("welch no-overlap");
        let (f2, p2) = bartlett_psd(&sig, 256, fs).expect("bartlett");
        assert_eq!(f1.len(), f2.len());
        for (a, b) in p1.iter().zip(p2.iter()) {
            assert!((a - b).abs() < 1e-20, "bartlett mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_lomb_scargle_peak_at_signal_freq() {
        let t: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect(); // 0..4.9 s
        let y: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * 1.0 * ti).sin()).collect();
        // Probe frequencies around 1 Hz
        let freqs: Vec<f64> = (1..=20)
            .map(|k| 2.0 * PI * k as f64 * 0.5)
            .collect();
        let power = lomb_scargle(&t, &y, &freqs).expect("lomb");
        // Highest power should be at 2π*1.0 (index 1 = 2*pi*0.5*2 = 2pi*1.0, k=2 → index 1)
        let peak_idx = power
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("cmp"))
            .map(|(i, _)| i)
            .expect("peak");
        // freqs[peak_idx] / (2π) should be close to 1.0 Hz
        let peak_hz = freqs[peak_idx] / (2.0 * PI);
        assert!(
            (peak_hz - 1.0).abs() < 0.6,
            "LS peak at {peak_hz} Hz, expected ~1 Hz"
        );
    }

    #[test]
    fn test_multitaper_psd_shape() {
        let sig = sine_wave(100.0, 512, 1000.0);
        let (freqs, psd) = multitaper_psd(&sig, 4.0, 7, 1000.0).expect("multitaper");
        assert_eq!(freqs.len(), psd.len());
    }

    #[test]
    fn test_multitaper_psd_peak() {
        let fs = 1000.0_f64;
        let sig = sine_wave(100.0, 512, fs);
        let (freqs, psd) = multitaper_psd(&sig, 4.0, 7, fs).expect("multitaper");
        let peak_idx = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("cmp"))
            .map(|(i, _)| i)
            .expect("peak");
        let peak_freq = freqs[peak_idx];
        assert!(
            (peak_freq - 100.0).abs() < 15.0,
            "Multitaper peak at {peak_freq}, expected ~100 Hz"
        );
    }

    #[test]
    fn test_coherence_identical_signals() {
        let fs = 1000.0_f64;
        let sig = sine_wave(100.0, 4096, fs);
        let (freqs, coh) = coherence(&sig, &sig, fs).expect("coherence");
        assert_eq!(freqs.len(), coh.len());
        let peak = coh.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(peak > 0.99, "coherence peak={peak}, expected near 1");
    }

    #[test]
    fn test_coherence_unrelated_signals() {
        let fs = 1000.0_f64;
        let n = 4096;
        let x = sine_wave(100.0, n, fs);
        let y = sine_wave(300.0, n, fs);
        let (freqs, coh) = coherence(&x, &y, fs).expect("coherence");
        assert_eq!(freqs.len(), coh.len());
        // No coherence — values should be small except near harmonics
        for &c in &coh {
            assert!(c >= 0.0 - 1e-9 && c <= 1.0 + 1e-9, "coherence out of [0,1]: {c}");
        }
    }

    #[test]
    fn test_lomb_scargle_constant_signal_zero_power() {
        let t: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let y = vec![3.14_f64; 20];
        let freqs = vec![2.0 * PI * 1.0, 2.0 * PI * 2.0];
        let power = lomb_scargle(&t, &y, &freqs).expect("constant");
        for &p in &power {
            assert!(p.abs() < 1e-10, "constant signal should have zero LS power: {p}");
        }
    }
}
