//! Reassigned Spectrogram and Synchrosqueezed STFT
//!
//! Provides sharpened time-frequency representations by relocating each STFT
//! coefficient to its instantaneous group delay / instantaneous frequency.
//!
//! ## Reassignment
//!
//! The reassignment method (Kodera, Gendrin & de Villedary 1978; Auger &
//! Flandrin 1995) computes three STFTs:
//!
//! - S_h:  standard STFT with window h
//! - S_Th: STFT with time-ramp window Th(n) = n · h(n)
//! - S_Dh: STFT with derivative window Dh(n) = h'(n)
//!
//! and reassigns each TF point to:
//!
//! ```text
//! t̂ = t + Re{ S_Th / S_h }          (group delay / reassigned time)
//! f̂ = f - Im{ S_Dh / S_h } / (2π)   (instantaneous frequency)
//! ```
//!
//! ## Synchrosqueezing
//!
//! Synchrosqueezing (Daubechies & Maes 1996; Thakur et al. 2013) squeezes the
//! STFT energy in the frequency direction only, preserving the time axis and
//! enabling perfect reconstruction.
//!
//! # References
//!
//! - Auger, F. & Flandrin, P. (1995). "Improving the readability of time-frequency
//!   and time-scale representations by the reassignment method." IEEE Trans. Signal
//!   Process., 43(5), 1068-1089.
//! - Daubechies, I., Lu, J. & Wu, H.-T. (2011). "Synchrosqueezed wavelet
//!   transforms: An empirical mode decomposition-like tool." Applied and
//!   Computational Harmonic Analysis, 30(2), 243-261.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        let mut p = 1;
        while p < n {
            p <<= 1;
        }
        p
    }
}

/// In-place radix-2 DIT FFT.
fn fft_inplace(buf: &mut [Complex64]) {
    let n = buf.len();
    if n <= 1 {
        return;
    }
    // Bit-reversal
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            buf.swap(i, j);
        }
    }
    let mut len = 2usize;
    while len <= n {
        let ang = -2.0 * PI / len as f64;
        let wlen = Complex64::new(ang.cos(), ang.sin());
        let mut i = 0;
        while i < n {
            let mut w = Complex64::new(1.0, 0.0);
            for k in 0..(len / 2) {
                let u = buf[i + k];
                let v = buf[i + k + len / 2] * w;
                buf[i + k] = u + v;
                buf[i + k + len / 2] = u - v;
                w *= wlen;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Compute one-frame STFT (complex spectrum, length nfft).
fn stft_frame(signal: &[f64], center: usize, window: &[f64], nfft: usize) -> Vec<Complex64> {
    let win_len = window.len();
    let half = win_len / 2;
    let mut buf = vec![Complex64::new(0.0, 0.0); nfft];
    for k in 0..win_len {
        let sig_idx = center as isize - half as isize + k as isize;
        if sig_idx >= 0 && (sig_idx as usize) < signal.len() {
            buf[k] = Complex64::new(signal[sig_idx as usize] * window[k], 0.0);
        }
    }
    fft_inplace(&mut buf);
    buf
}

// ---------------------------------------------------------------------------
// Reassigned Spectrogram
// ---------------------------------------------------------------------------

/// Compute the reassigned spectrogram.
///
/// Returns `(power, times, freqs)` where `power` is a 2D array of shape
/// `(n_freq_bins, n_frames)`.  Each sample of `power` represents the
/// accumulated squared magnitude of the STFT reassigned to the instantaneous
/// frequency and group delay at that point.
///
/// # Arguments
///
/// * `x`           – Input signal (real-valued).
/// * `window_size` – Length of the analysis window.
/// * `hop_size`    – Frame shift in samples.
/// * `fs`          – Sampling frequency in Hz.
///
/// # Returns
///
/// `(power, times_s, freqs_hz)` where `power` has shape
/// `(n_freq_bins, n_frames)`.
///
/// # Errors
///
/// Returns `SignalError::ValueError` for invalid inputs.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::reassignment::reassigned_spectrogram;
/// use std::f64::consts::PI;
///
/// let fs = 1000.0f64;
/// let n = 512usize;
/// let x: Vec<f64> = (0..n).map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin()).collect();
/// let (power, times, freqs) = reassigned_spectrogram(&x, 64, 16, fs).expect("failed");
/// assert_eq!(power.nrows(), freqs.len());
/// assert_eq!(power.ncols(), times.len());
/// ```
pub fn reassigned_spectrogram(
    x: &[f64],
    window_size: usize,
    hop_size: usize,
    fs: f64,
) -> SignalResult<(Array2<f64>, Vec<f64>, Vec<f64>)> {
    if x.is_empty() {
        return Err(SignalError::ValueError("x must not be empty".to_string()));
    }
    if window_size < 2 {
        return Err(SignalError::ValueError(
            "window_size must be >= 2".to_string(),
        ));
    }
    if hop_size == 0 {
        return Err(SignalError::ValueError("hop_size must be > 0".to_string()));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }

    let nfft = next_pow2(window_size);
    let n_freq = nfft / 2 + 1;

    // Build three windows
    let h = hann_window(window_size);
    // Time-ramp window: Th(n) = n * h(n), centred at 0
    let half = window_size as f64 / 2.0;
    let th: Vec<f64> = h
        .iter()
        .enumerate()
        .map(|(k, &hk)| (k as f64 - half) * hk)
        .collect();
    // Derivative window: Dh(n) = h'(n) via finite difference
    let dh: Vec<f64> = {
        let mut d = vec![0.0f64; window_size];
        if window_size > 1 {
            d[0] = h[1] - h[0];
            for k in 1..(window_size - 1) {
                d[k] = (h[k + 1] - h[k - 1]) / 2.0;
            }
            d[window_size - 1] = h[window_size - 1] - h[window_size - 2];
        }
        d
    };

    // Compute frame centres
    let n = x.len();
    let frame_centres: Vec<usize> = {
        let mut v = Vec::new();
        let mut c = window_size / 2;
        while c + window_size / 2 <= n {
            v.push(c);
            c += hop_size;
        }
        if v.is_empty() {
            v.push(window_size / 2);
        }
        v
    };
    let n_frames = frame_centres.len();

    // Accumulator for reassigned power
    let mut power = Array2::<f64>::zeros((n_freq, n_frames));

    let dt = 1.0 / fs;
    let df = fs / nfft as f64;

    for (t_frame, &center) in frame_centres.iter().enumerate() {
        let t_center_s = center as f64 * dt;

        // Compute the three STFT frames
        let sh = stft_frame(x, center, &h, nfft);
        let sth = stft_frame(x, center, &th, nfft);
        let sdh = stft_frame(x, center, &dh, nfft);

        for k in 0..n_freq {
            let sh_k = sh[k];
            let mag2 = sh_k.re * sh_k.re + sh_k.im * sh_k.im;
            if mag2 < 1e-30 {
                continue;
            }

            // Reassigned time: t̂ = t_center + Re{ S_Th(t,k) / S_h(t,k) }
            let ratio_th = sth[k] / sh_k;
            let t_hat_s = t_center_s + ratio_th.re * dt;

            // Reassigned frequency: f̂ = k*df - Im{ S_Dh / S_h } / (2π)
            let ratio_dh = sdh[k] / sh_k;
            let f_hat = k as f64 * df - ratio_dh.im * fs / (2.0 * PI * nfft as f64);

            // Map back to grid indices
            let f_bin = (f_hat / df).round() as isize;
            let t_bin = (t_hat_s / (hop_size as f64 * dt)).round() as isize;

            let f_bin = f_bin.clamp(0, (n_freq - 1) as isize) as usize;
            let t_bin = t_bin.clamp(0, (n_frames - 1) as isize) as usize;

            power[[f_bin, t_bin]] += mag2;
        }
    }

    let times: Vec<f64> = frame_centres.iter().map(|&c| c as f64 * dt).collect();
    let freqs: Vec<f64> = (0..n_freq).map(|k| k as f64 * df).collect();

    Ok((power, times, freqs))
}

// ---------------------------------------------------------------------------
// Synchrosqueezed STFT
// ---------------------------------------------------------------------------

/// Compute the synchrosqueezed STFT (SST).
///
/// The SST squeezes the STFT in the frequency direction according to the
/// instantaneous frequency estimate, concentrating the energy around the
/// instantaneous frequency trajectory.
///
/// # Arguments
///
/// * `x`           – Input signal.
/// * `window_size` – Analysis window length.
/// * `hop_size`    – Frame hop in samples.
/// * `fs`          – Sampling frequency in Hz.
///
/// # Returns
///
/// `(sst_matrix, times, freqs)` where `sst_matrix` has shape
/// `(n_freq_bins, n_frames)` and contains complex SST coefficients.
///
/// # Example
///
/// ```
/// use scirs2_signal::spectral::reassignment::synchrosqueezed_stft;
///
/// let fs = 1000.0f64;
/// let n = 512usize;
/// let x: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 100.0 * i as f64 / fs).sin()).collect();
/// let (sst, times, freqs) = synchrosqueezed_stft(&x, 64, 16, fs).expect("sst failed");
/// assert_eq!(sst.nrows(), freqs.len());
/// assert_eq!(sst.ncols(), times.len());
/// ```
pub fn synchrosqueezed_stft(
    x: &[f64],
    window_size: usize,
    hop_size: usize,
    fs: f64,
) -> SignalResult<(Array2<Complex64>, Vec<f64>, Vec<f64>)> {
    if x.is_empty() {
        return Err(SignalError::ValueError("x must not be empty".to_string()));
    }
    if window_size < 2 {
        return Err(SignalError::ValueError(
            "window_size must be >= 2".to_string(),
        ));
    }
    if hop_size == 0 {
        return Err(SignalError::ValueError("hop_size must be > 0".to_string()));
    }
    if fs <= 0.0 {
        return Err(SignalError::ValueError("fs must be positive".to_string()));
    }

    let nfft = next_pow2(window_size);
    let n_freq = nfft / 2 + 1;

    let h = hann_window(window_size);
    // Derivative window for IF estimation
    let dh: Vec<f64> = {
        let mut d = vec![0.0f64; window_size];
        if window_size > 1 {
            d[0] = h[1] - h[0];
            for k in 1..(window_size - 1) {
                d[k] = (h[k + 1] - h[k - 1]) / 2.0;
            }
            d[window_size - 1] = h[window_size - 1] - h[window_size - 2];
        }
        d
    };

    let n = x.len();
    let frame_centres: Vec<usize> = {
        let mut v = Vec::new();
        let mut c = window_size / 2;
        while c + window_size / 2 <= n {
            v.push(c);
            c += hop_size;
        }
        if v.is_empty() {
            v.push(window_size / 2);
        }
        v
    };
    let n_frames = frame_centres.len();

    let mut sst = Array2::<Complex64>::zeros((n_freq, n_frames));
    let df = fs / nfft as f64;
    let dt = 1.0 / fs;

    for (t_frame, &center) in frame_centres.iter().enumerate() {
        let sh = stft_frame(x, center, &h, nfft);
        let sdh = stft_frame(x, center, &dh, nfft);

        for k in 0..n_freq {
            let sh_k = sh[k];
            let mag2 = sh_k.re * sh_k.re + sh_k.im * sh_k.im;
            if mag2 < 1e-30 {
                continue;
            }

            // Instantaneous frequency estimate
            let ratio_dh = sdh[k] / sh_k;
            // ω̂ = Re{ -j * S_Dh / S_h } / (Δt)  (in rad/s)
            // f̂ = Re{ S_Dh * conj(S_h) } / (|S_h|^2 * 2π * Δt)
            // Using the phase derivative form:
            let if_est = k as f64 * df - ratio_dh.im * fs / (2.0 * PI * nfft as f64);

            // Bin the contribution to the nearest frequency bin
            let l = (if_est / df).round() as isize;
            let l = l.clamp(0, (n_freq - 1) as isize) as usize;

            // SST coefficient: transfer STFT value to the squeezed bin
            sst[[l, t_frame]] += sh_k * dt;
        }
    }

    let times: Vec<f64> = frame_centres.iter().map(|&c| c as f64 * dt).collect();
    let freqs: Vec<f64> = (0..n_freq).map(|k| k as f64 * df).collect();

    Ok((sst, times, freqs))
}

// ---------------------------------------------------------------------------
// Window function
// ---------------------------------------------------------------------------

fn hann_window(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f64 / (n - 1) as f64).cos()))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_chirp(n: usize, fs: f64) -> Vec<f64> {
        // Linear chirp from 50 Hz to 200 Hz
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let f = 50.0 + 150.0 * t;
                (2.0 * PI * f * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_reassigned_spectrogram_shape() {
        let fs = 1000.0;
        let x = test_chirp(512, fs);
        let (power, times, freqs) =
            reassigned_spectrogram(&x, 64, 16, fs).expect("reassigned_spectrogram failed");
        assert_eq!(power.nrows(), freqs.len());
        assert_eq!(power.ncols(), times.len());
        assert!(freqs.len() > 0);
        assert!(times.len() > 0);
    }

    #[test]
    fn test_reassigned_spectrogram_non_negative() {
        let fs = 1000.0;
        let x: Vec<f64> = (0..256)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / fs).sin())
            .collect();
        let (power, _, _) =
            reassigned_spectrogram(&x, 64, 16, fs).expect("reassigned_spectrogram failed");
        assert!(power.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_reassigned_spectrogram_invalid() {
        assert!(reassigned_spectrogram(&[], 64, 16, 1000.0).is_err());
        assert!(reassigned_spectrogram(&[1.0; 256], 0, 16, 1000.0).is_err());
        assert!(reassigned_spectrogram(&[1.0; 256], 64, 0, 1000.0).is_err());
        assert!(reassigned_spectrogram(&[1.0; 256], 64, 16, -1.0).is_err());
    }

    #[test]
    fn test_synchrosqueezed_stft_shape() {
        let fs = 1000.0;
        let x: Vec<f64> = (0..512)
            .map(|i| (2.0 * PI * 80.0 * i as f64 / fs).sin())
            .collect();
        let (sst, times, freqs) =
            synchrosqueezed_stft(&x, 64, 16, fs).expect("synchrosqueezed_stft failed");
        assert_eq!(sst.nrows(), freqs.len());
        assert_eq!(sst.ncols(), times.len());
    }

    #[test]
    fn test_synchrosqueezed_stft_energy_concentration() {
        // A pure sinusoid should concentrate energy in a narrow frequency band
        let fs = 1000.0;
        let f0 = 100.0f64;
        let n = 512;
        let x: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * f0 * i as f64 / fs).sin())
            .collect();
        let (sst, _times, freqs) = synchrosqueezed_stft(&x, 64, 16, fs).expect("sst failed");

        // Find the frequency bin with maximum total energy
        let energy_per_freq: Vec<f64> = (0..sst.nrows())
            .map(|k| {
                (0..sst.ncols())
                    .map(|t| {
                        let c = sst[[k, t]];
                        c.re * c.re + c.im * c.im
                    })
                    .sum::<f64>()
            })
            .collect();

        let peak_bin = energy_per_freq
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        assert!(
            (freqs[peak_bin] - f0).abs() < 30.0,
            "SST peak at {} Hz, expected near {f0} Hz",
            freqs[peak_bin]
        );
    }
}
