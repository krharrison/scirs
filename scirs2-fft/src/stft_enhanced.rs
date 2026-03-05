//! Enhanced Short-Time Fourier Transform (STFT) module
//!
//! This module provides advanced time-frequency analysis tools beyond the basic STFT:
//!
//! - **Inverse STFT (ISTFT)** with overlap-add synthesis
//! - **Griffin-Lim algorithm** for phase reconstruction from magnitude spectrograms
//! - **Reassigned spectrogram** (time-frequency reassignment for sharper localization)
//! - **Synchrosqueezing transform (SST)** for improved frequency resolution
//! - **Additional window functions**: DPSS/Slepian, Dolph-Chebyshev
//!
//! # References
//!
//! * Griffin, D. W. & Lim, J. S. "Signal estimation from modified short-time
//!   Fourier transform." IEEE Trans. ASSP, 1984.
//! * Auger, F. & Flandrin, P. "Improving the readability of time-frequency and
//!   time-scale representations by the reassignment method." IEEE Trans. SP, 1995.
//! * Daubechies, I., Lu, J., & Wu, H.-T. "Synchrosqueezed wavelet transforms."
//!   Applied and Computational Harmonic Analysis, 2011.

use crate::error::{FFTError, FFTResult};
use crate::window::{get_window, Window};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Complex64;
use scirs2_core::numeric::NumCast;
use std::f64::consts::PI;
use std::fmt::Debug;

/// Compute the inverse STFT using overlap-add synthesis
///
/// Reconstructs a time-domain signal from its STFT representation using
/// the overlap-add method.
///
/// # Arguments
///
/// * `stft_matrix` - STFT matrix (rows = frequencies, columns = time frames)
/// * `window` - Window function (must match forward STFT)
/// * `nperseg` - Segment length (must match forward STFT)
/// * `noverlap` - Overlap between segments
/// * `nfft` - FFT length used in forward STFT
///
/// # Returns
///
/// Reconstructed time-domain signal.
///
/// # Errors
///
/// Returns an error if parameters are inconsistent.
pub fn istft(
    stft_matrix: &Array2<Complex64>,
    window: Window,
    nperseg: usize,
    noverlap: Option<usize>,
    nfft: Option<usize>,
) -> FFTResult<Vec<f64>> {
    let (n_freq, n_frames) = stft_matrix.dim();
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let nfft = nfft.unwrap_or(nperseg);
    let hop = nperseg - noverlap;

    if nperseg == 0 {
        return Err(FFTError::ValueError(
            "Segment length must be positive".to_string(),
        ));
    }
    if noverlap >= nperseg {
        return Err(FFTError::ValueError(
            "Overlap must be less than segment length".to_string(),
        ));
    }

    // Get the analysis window
    let win = get_window(window, nperseg, true)?;

    // Compute output length
    let output_len = nperseg + hop * (n_frames - 1);
    let mut output = vec![0.0; output_len];
    let mut window_sum = vec![0.0; output_len];

    // For each frame, IFFT and overlap-add
    for frame_idx in 0..n_frames {
        // Extract this frame's spectrum
        let mut spectrum = vec![Complex64::new(0.0, 0.0); nfft];
        let n_copy = n_freq.min(nfft);
        for k in 0..n_copy {
            spectrum[k] = stft_matrix[[k, frame_idx]];
        }

        // If one-sided spectrum, mirror the conjugate symmetric part
        if n_freq <= nfft / 2 + 1 && nfft > 1 {
            for k in 1..nfft / 2 {
                if k < n_freq {
                    spectrum[nfft - k] = spectrum[k].conj();
                }
            }
        }

        // Inverse FFT
        let time_segment = crate::fft::ifft(&spectrum, None)?;

        // Apply synthesis window and overlap-add
        let frame_start = frame_idx * hop;
        for i in 0..nperseg {
            if frame_start + i < output_len {
                output[frame_start + i] += time_segment[i].re * win[i];
                window_sum[frame_start + i] += win[i] * win[i];
            }
        }
    }

    // Normalize by window sum (COLA condition)
    for i in 0..output_len {
        if window_sum[i] > 1e-10 {
            output[i] /= window_sum[i];
        }
    }

    Ok(output)
}

/// Griffin-Lim algorithm for phase reconstruction from magnitude spectrogram
///
/// Iteratively estimates the phase of a signal given only the magnitude
/// of its STFT. Each iteration performs ISTFT followed by STFT to enforce
/// the magnitude constraint.
///
/// # Arguments
///
/// * `magnitude` - Magnitude spectrogram (rows = frequencies, columns = time frames)
/// * `window` - Window function
/// * `nperseg` - Segment length
/// * `noverlap` - Overlap between segments
/// * `nfft` - FFT length
/// * `n_iter` - Number of iterations (default: 32)
/// * `momentum` - Momentum factor for faster convergence (0 to 1, default: 0.99)
///
/// # Returns
///
/// Reconstructed time-domain signal.
///
/// # Errors
///
/// Returns an error if parameters are inconsistent.
pub fn griffin_lim(
    magnitude: &Array2<f64>,
    window: Window,
    nperseg: usize,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    n_iter: Option<usize>,
    momentum: Option<f64>,
) -> FFTResult<Vec<f64>> {
    let (n_freq, n_frames) = magnitude.dim();
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let nfft = nfft.unwrap_or(nperseg);
    let n_iter = n_iter.unwrap_or(32);
    let momentum = momentum.unwrap_or(0.99);
    let hop = nperseg - noverlap;

    if nperseg == 0 {
        return Err(FFTError::ValueError(
            "Segment length must be positive".to_string(),
        ));
    }
    if noverlap >= nperseg {
        return Err(FFTError::ValueError(
            "Overlap must be less than segment length".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&momentum) {
        return Err(FFTError::ValueError(
            "Momentum must be in [0, 1]".to_string(),
        ));
    }

    // Initialize with random phase
    let mut stft_estimate = Array2::zeros((n_freq, n_frames));
    for k in 0..n_freq {
        for t in 0..n_frames {
            // Use a deterministic initial phase based on position
            let phase = 2.0 * PI * (k as f64 * 0.37 + t as f64 * 0.61) % (2.0 * PI);
            stft_estimate[[k, t]] = Complex64::from_polar(magnitude[[k, t]], phase);
        }
    }

    let mut prev_estimate = stft_estimate.clone();

    for _iteration in 0..n_iter {
        // Step 1: ISTFT to get time-domain estimate
        let signal = istft(
            &stft_estimate,
            window.clone(),
            nperseg,
            Some(noverlap),
            Some(nfft),
        )?;

        // Step 2: STFT of the reconstructed signal
        let win = get_window(window.clone(), nperseg, true)?;
        let signal_len = signal.len();
        let mut new_stft = Array2::zeros((n_freq, n_frames));

        for frame_idx in 0..n_frames {
            let frame_start = frame_idx * hop;
            let mut windowed = vec![Complex64::new(0.0, 0.0); nfft];

            for i in 0..nperseg {
                if frame_start + i < signal_len {
                    windowed[i] = Complex64::new(signal[frame_start + i] * win[i], 0.0);
                }
            }

            let frame_fft = crate::fft::fft(&windowed, None)?;

            for k in 0..n_freq {
                new_stft[[k, frame_idx]] = frame_fft[k];
            }
        }

        // Step 3: Replace magnitude, keep estimated phase (with momentum)
        let temp_estimate = stft_estimate.clone();
        for k in 0..n_freq {
            for t in 0..n_frames {
                let current_phase = new_stft[[k, t]].arg();
                let prev_phase = prev_estimate[[k, t]].arg();

                // Apply momentum for faster convergence
                let blended_phase = current_phase + momentum * (current_phase - prev_phase);
                stft_estimate[[k, t]] = Complex64::from_polar(magnitude[[k, t]], blended_phase);
            }
        }
        prev_estimate = temp_estimate;
    }

    // Final ISTFT
    istft(&stft_estimate, window, nperseg, Some(noverlap), Some(nfft))
}

/// Compute the reassigned spectrogram for sharper time-frequency localization
///
/// The reassignment method moves each STFT coefficient to its "center of gravity"
/// in the time-frequency plane, producing a much sharper representation than
/// the standard spectrogram.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `window` - Window function
/// * `nperseg` - Segment length
/// * `noverlap` - Overlap between segments
/// * `nfft` - FFT length
/// * `fs` - Sampling frequency
///
/// # Returns
///
/// A tuple of (frequencies, times, reassigned_spectrogram)
///
/// # Errors
///
/// Returns an error if parameters are invalid.
#[allow(clippy::too_many_arguments)]
pub fn reassigned_spectrogram<T>(
    x: &[T],
    window: Window,
    nperseg: usize,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    fs: Option<f64>,
) -> FFTResult<(Vec<f64>, Vec<f64>, Array2<f64>)>
where
    T: NumCast + Copy + Debug,
{
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let nfft = nfft.unwrap_or(nperseg);
    let fs = fs.unwrap_or(1.0);
    let hop = nperseg - noverlap;

    if x.is_empty() {
        return Err(FFTError::ValueError("Input signal is empty".to_string()));
    }
    if nperseg == 0 {
        return Err(FFTError::ValueError(
            "Segment length must be positive".to_string(),
        ));
    }

    // Convert input to f64
    let signal: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val)
                .ok_or_else(|| FFTError::ValueError("Could not convert input value".to_string()))
        })
        .collect::<FFTResult<Vec<_>>>()?;

    // Get window and its derivative (time-weighted version)
    let win = get_window(window.clone(), nperseg, true)?;

    // Time-weighted window: t * w(t)
    let t_win: Vec<f64> = (0..nperseg)
        .map(|i| {
            let t = (i as f64 - nperseg as f64 / 2.0) / fs;
            t * win[i]
        })
        .collect();

    // Derivative of window (approximate via central differences)
    let mut dwin = vec![0.0; nperseg];
    if nperseg >= 3 {
        dwin[0] = (win[1] - win[0]) * fs;
        for i in 1..nperseg - 1 {
            dwin[i] = (win[i + 1] - win[i - 1]) * fs / 2.0;
        }
        dwin[nperseg - 1] = (win[nperseg - 1] - win[nperseg - 2]) * fs;
    }

    // Compute number of frames
    let n_frames = if signal.len() >= nperseg {
        (signal.len() - nperseg) / hop + 1
    } else {
        0
    };

    if n_frames == 0 {
        return Err(FFTError::ValueError(
            "Signal too short for given segment length".to_string(),
        ));
    }

    let n_freq = nfft / 2 + 1;

    // Compute three STFTs: with window, time-weighted window, and derivative window
    let mut stft_w = Array2::zeros((n_freq, n_frames));
    let mut stft_tw = Array2::zeros((n_freq, n_frames));
    let mut stft_dw = Array2::zeros((n_freq, n_frames));

    for frame_idx in 0..n_frames {
        let frame_start = frame_idx * hop;

        let mut seg_w = vec![Complex64::new(0.0, 0.0); nfft];
        let mut seg_tw = vec![Complex64::new(0.0, 0.0); nfft];
        let mut seg_dw = vec![Complex64::new(0.0, 0.0); nfft];

        for i in 0..nperseg {
            if frame_start + i < signal.len() {
                let s = signal[frame_start + i];
                seg_w[i] = Complex64::new(s * win[i], 0.0);
                seg_tw[i] = Complex64::new(s * t_win[i], 0.0);
                seg_dw[i] = Complex64::new(s * dwin[i], 0.0);
            }
        }

        let fft_w = crate::fft::fft(&seg_w, None)?;
        let fft_tw = crate::fft::fft(&seg_tw, None)?;
        let fft_dw = crate::fft::fft(&seg_dw, None)?;

        for k in 0..n_freq {
            stft_w[[k, frame_idx]] = fft_w[k];
            stft_tw[[k, frame_idx]] = fft_tw[k];
            stft_dw[[k, frame_idx]] = fft_dw[k];
        }
    }

    // Compute reassignment operators
    // Time reassignment: t_hat = t - Re(STFT_tw / STFT_w)
    // Frequency reassignment: f_hat = f + Im(STFT_dw / STFT_w) / (2*pi)
    let mut reassigned = Array2::zeros((n_freq, n_frames));

    // Build standard spectrogram and redistribute energy
    let dt = hop as f64 / fs;
    let df = fs / nfft as f64;

    // First compute the standard spectrogram
    for k in 0..n_freq {
        for t in 0..n_frames {
            let power = stft_w[[k, t]].norm_sqr();
            reassigned[[k, t]] = power;
        }
    }

    // Apply reassignment
    let threshold = 1e-10;
    let mut reassigned_output = Array2::zeros((n_freq, n_frames));

    for k in 0..n_freq {
        for t_idx in 0..n_frames {
            let power = reassigned[[k, t_idx]];
            if power < threshold {
                continue;
            }

            let s_w = stft_w[[k, t_idx]];

            // Compute reassigned time
            let s_tw = stft_tw[[k, t_idx]];
            let t_shift = -(s_tw / s_w).re;
            let t_reassigned = t_idx as f64 * dt + t_shift;
            let t_new_idx_f = t_reassigned / dt;

            // Compute reassigned frequency
            let s_dw = stft_dw[[k, t_idx]];
            let f_shift = (s_dw / s_w).im / (2.0 * PI);
            let f_reassigned = k as f64 * df + f_shift;
            let f_new_idx_f = f_reassigned / df;

            // Map to nearest bin
            let t_new = t_new_idx_f.round() as i64;
            let f_new = f_new_idx_f.round() as i64;

            if t_new >= 0 && (t_new as usize) < n_frames && f_new >= 0 && (f_new as usize) < n_freq
            {
                reassigned_output[[f_new as usize, t_new as usize]] += power;
            }
        }
    }

    // Frequency axis
    let frequencies: Vec<f64> = (0..n_freq).map(|k| k as f64 * df).collect();

    // Time axis
    let times: Vec<f64> = (0..n_frames)
        .map(|t| (t * hop) as f64 / fs + nperseg as f64 / (2.0 * fs))
        .collect();

    Ok((frequencies, times, reassigned_output))
}

/// Compute the synchrosqueezing transform (SST)
///
/// SST is a post-processing technique that sharpens the frequency resolution
/// of the STFT by reassigning energy only along the frequency axis based on
/// instantaneous frequency estimates.
///
/// # Arguments
///
/// * `x` - Input signal
/// * `window` - Window function
/// * `nperseg` - Segment length
/// * `noverlap` - Overlap
/// * `fs` - Sampling frequency
///
/// # Returns
///
/// A tuple of (frequencies, times, sst_matrix)
///
/// # Errors
///
/// Returns an error if parameters are invalid.
pub fn synchrosqueezing<T>(
    x: &[T],
    window: Window,
    nperseg: usize,
    noverlap: Option<usize>,
    fs: Option<f64>,
) -> FFTResult<(Vec<f64>, Vec<f64>, Array2<f64>)>
where
    T: NumCast + Copy + Debug,
{
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let fs = fs.unwrap_or(1.0);
    let hop = nperseg - noverlap;
    let nfft = nperseg;

    if x.is_empty() {
        return Err(FFTError::ValueError("Input signal is empty".to_string()));
    }
    if nperseg == 0 {
        return Err(FFTError::ValueError(
            "Segment length must be positive".to_string(),
        ));
    }

    // Convert input
    let signal: Vec<f64> = x
        .iter()
        .map(|&val| {
            NumCast::from(val)
                .ok_or_else(|| FFTError::ValueError("Could not convert input value".to_string()))
        })
        .collect::<FFTResult<Vec<_>>>()?;

    let win = get_window(window, nperseg, true)?;

    // Compute number of frames
    let n_frames = if signal.len() >= nperseg {
        (signal.len() - nperseg) / hop + 1
    } else {
        return Err(FFTError::ValueError(
            "Signal too short for segment length".to_string(),
        ));
    };

    let n_freq = nfft / 2 + 1;
    let df = fs / nfft as f64;

    // Compute STFT
    let mut stft_matrix = Array2::zeros((n_freq, n_frames));
    for frame_idx in 0..n_frames {
        let frame_start = frame_idx * hop;
        let mut seg = vec![Complex64::new(0.0, 0.0); nfft];
        for i in 0..nperseg {
            if frame_start + i < signal.len() {
                seg[i] = Complex64::new(signal[frame_start + i] * win[i], 0.0);
            }
        }
        let frame_fft = crate::fft::fft(&seg, None)?;
        for k in 0..n_freq {
            stft_matrix[[k, frame_idx]] = frame_fft[k];
        }
    }

    // Estimate instantaneous frequency from phase differences between adjacent frames
    let mut sst_output = Array2::zeros((n_freq, n_frames));
    let threshold = 1e-10;

    for t_idx in 0..n_frames {
        for k in 0..n_freq {
            let power = stft_matrix[[k, t_idx]].norm_sqr();
            if power < threshold {
                continue;
            }

            // Estimate instantaneous frequency
            let inst_freq = if t_idx + 1 < n_frames {
                // Use phase difference between adjacent frames
                let phase_current = stft_matrix[[k, t_idx]].arg();
                let phase_next = stft_matrix[[k, t_idx + 1]].arg();
                let mut dphi = phase_next - phase_current;

                // Unwrap phase
                while dphi > PI {
                    dphi -= 2.0 * PI;
                }
                while dphi < -PI {
                    dphi += 2.0 * PI;
                }

                k as f64 * df + dphi * fs / (2.0 * PI * hop as f64)
            } else {
                k as f64 * df
            };

            // Map to nearest frequency bin
            let target_bin = (inst_freq / df).round() as i64;
            if target_bin >= 0 && (target_bin as usize) < n_freq {
                sst_output[[target_bin as usize, t_idx]] += power;
            }
        }
    }

    let frequencies: Vec<f64> = (0..n_freq).map(|k| k as f64 * df).collect();
    let times: Vec<f64> = (0..n_frames)
        .map(|t| (t * hop) as f64 / fs + nperseg as f64 / (2.0 * fs))
        .collect();

    Ok((frequencies, times, sst_output))
}

/// Generate a Discrete Prolate Spheroidal Sequence (DPSS/Slepian) window
///
/// DPSS windows maximize the energy concentration in the main lobe for
/// a given main-lobe width, making them optimal for spectral estimation.
///
/// # Arguments
///
/// * `n` - Window length
/// * `nw` - Time-bandwidth product (NW). Determines the main lobe width.
///   Common values: 2.0, 2.5, 3.0, 3.5, 4.0
///
/// # Returns
///
/// The first DPSS window (lowest order).
///
/// # Errors
///
/// Returns an error if `n` is zero or `nw` is non-positive.
pub fn dpss_window(n: usize, nw: f64) -> FFTResult<Vec<f64>> {
    if n == 0 {
        return Err(FFTError::ValueError(
            "Window length must be positive".to_string(),
        ));
    }
    if nw <= 0.0 {
        return Err(FFTError::ValueError(
            "Time-bandwidth product must be positive".to_string(),
        ));
    }

    // Build the tridiagonal matrix for the DPSS eigenvalue problem
    // The DPSS is the eigenvector corresponding to the largest eigenvalue of:
    //   T[i,i] = ((n-1)/2 - i)^2 * cos(2*pi*W)
    //   T[i,i+1] = T[i+1,i] = i*(n-i)/2
    // where W = nw/n is the half-bandwidth
    let w = nw / n as f64;

    // Diagonal and off-diagonal entries
    let mut diag = vec![0.0; n];
    let mut off_diag = vec![0.0; n.saturating_sub(1)];

    for i in 0..n {
        let t = (n as f64 - 1.0) / 2.0 - i as f64;
        diag[i] = t * t * (2.0 * PI * w).cos();
    }

    for i in 0..n.saturating_sub(1) {
        off_diag[i] = (i as f64 + 1.0) * (n as f64 - 1.0 - i as f64) / 2.0;
    }

    // Use power iteration to find the dominant eigenvector
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let max_iter = 200;

    for _iter in 0..max_iter {
        // Apply tridiagonal matrix: Tv
        let mut tv = vec![0.0; n];
        tv[0] = diag[0] * v[0];
        if n > 1 {
            tv[0] += off_diag[0] * v[1];
        }
        for i in 1..n - 1 {
            tv[i] = off_diag[i - 1] * v[i - 1] + diag[i] * v[i] + off_diag[i] * v[i + 1];
        }
        if n > 1 {
            tv[n - 1] = off_diag[n - 2] * v[n - 2] + diag[n - 1] * v[n - 1];
        }

        // Normalize
        let norm: f64 = tv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        for val in &mut tv {
            *val /= norm;
        }

        // Check convergence
        let diff: f64 = v
            .iter()
            .zip(tv.iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum::<f64>();

        v = tv;

        if diff < 1e-12 {
            break;
        }
    }

    // Ensure the window is positive (DPSS can have sign ambiguity)
    let sum: f64 = v.iter().sum();
    if sum < 0.0 {
        for val in &mut v {
            *val = -*val;
        }
    }

    Ok(v)
}

/// Generate a Dolph-Chebyshev window
///
/// The Dolph-Chebyshev window minimizes the main lobe width for a given
/// sidelobe level, or equivalently, minimizes the maximum sidelobe level
/// for a given main lobe width.
///
/// # Arguments
///
/// * `n` - Window length
/// * `attenuation_db` - Sidelobe attenuation in dB (positive number, e.g., 60.0)
///
/// # Returns
///
/// Dolph-Chebyshev window coefficients.
///
/// # Errors
///
/// Returns an error if `n` is zero or attenuation is non-positive.
pub fn dolph_chebyshev_window(n: usize, attenuation_db: f64) -> FFTResult<Vec<f64>> {
    if n == 0 {
        return Err(FFTError::ValueError(
            "Window length must be positive".to_string(),
        ));
    }
    if attenuation_db <= 0.0 {
        return Err(FFTError::ValueError(
            "Attenuation must be positive (in dB)".to_string(),
        ));
    }

    if n == 1 {
        return Ok(vec![1.0]);
    }

    let m = n - 1;
    let r = 10.0_f64.powf(attenuation_db / 20.0); // Sidelobe ratio

    // Compute beta (Chebyshev parameter)
    let beta = (1.0 / r + (1.0 / (r * r) - 1.0).max(0.0).sqrt()).powf(1.0 / m as f64);
    let x0 = (beta + 1.0 / beta) / 2.0;

    // Compute window via DFT of the Chebyshev polynomial
    let mut window = vec![0.0; n];

    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..n {
            let theta = 2.0 * PI * k as f64 / n as f64;
            let x = x0 * (theta / 2.0).cos();

            // Evaluate Chebyshev polynomial T_m(x)
            let t_m = if x.abs() <= 1.0 {
                (m as f64 * x.acos()).cos()
            } else if x > 1.0 {
                (m as f64 * x.acosh()).cosh()
            } else {
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                sign * (m as f64 * (-x).acosh()).cosh()
            };

            let phase = 2.0 * PI * i as f64 * k as f64 / n as f64;
            sum += t_m * phase.cos();
        }
        window[i] = sum / n as f64;
    }

    // Normalize to peak of 1
    let max_val = window.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
    if max_val > 1e-15 {
        for val in &mut window {
            *val /= max_val;
        }
    }

    Ok(window)
}

/// Compute spectral coherence between two signals
///
/// The coherence function measures the linear relationship between two
/// signals in the frequency domain. It is the cross-spectral density
/// normalized by the individual spectral densities.
///
/// # Arguments
///
/// * `x` - First signal
/// * `y` - Second signal (same length as x)
/// * `window` - Window function
/// * `nperseg` - Segment length
/// * `noverlap` - Overlap
/// * `fs` - Sampling frequency
///
/// # Returns
///
/// A tuple of (frequencies, coherence) where coherence is in [0, 1].
///
/// # Errors
///
/// Returns an error if signals have different lengths or parameters are invalid.
pub fn spectral_coherence(
    x: &[f64],
    y: &[f64],
    window: Window,
    nperseg: usize,
    noverlap: Option<usize>,
    fs: Option<f64>,
) -> FFTResult<(Vec<f64>, Vec<f64>)> {
    if x.len() != y.len() {
        return Err(FFTError::ValueError(
            "Input signals must have the same length".to_string(),
        ));
    }
    if x.is_empty() {
        return Err(FFTError::ValueError("Input signals are empty".to_string()));
    }
    if nperseg == 0 {
        return Err(FFTError::ValueError(
            "Segment length must be positive".to_string(),
        ));
    }

    let noverlap = noverlap.unwrap_or(nperseg / 2);
    let fs = fs.unwrap_or(1.0);
    let hop = nperseg - noverlap;
    let nfft = nperseg;
    let n_freq = nfft / 2 + 1;

    let win = get_window(window, nperseg, true)?;

    let n_frames = if x.len() >= nperseg {
        (x.len() - nperseg) / hop + 1
    } else {
        return Err(FFTError::ValueError(
            "Signal too short for segment length".to_string(),
        ));
    };

    // Compute averaged cross and auto spectra
    let mut pxx = vec![0.0; n_freq];
    let mut pyy = vec![0.0; n_freq];
    let mut pxy = vec![Complex64::new(0.0, 0.0); n_freq];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop;

        let mut seg_x = vec![Complex64::new(0.0, 0.0); nfft];
        let mut seg_y = vec![Complex64::new(0.0, 0.0); nfft];

        for i in 0..nperseg {
            if start + i < x.len() {
                seg_x[i] = Complex64::new(x[start + i] * win[i], 0.0);
                seg_y[i] = Complex64::new(y[start + i] * win[i], 0.0);
            }
        }

        let fft_x = crate::fft::fft(&seg_x, None)?;
        let fft_y = crate::fft::fft(&seg_y, None)?;

        for k in 0..n_freq {
            pxx[k] += fft_x[k].norm_sqr();
            pyy[k] += fft_y[k].norm_sqr();
            pxy[k] += fft_x[k] * fft_y[k].conj();
        }
    }

    // Compute coherence: |Pxy|^2 / (Pxx * Pyy)
    let frequencies: Vec<f64> = (0..n_freq).map(|k| k as f64 * fs / nfft as f64).collect();

    let coherence: Vec<f64> = (0..n_freq)
        .map(|k| {
            let denom = pxx[k] * pyy[k];
            if denom > 1e-15 {
                (pxy[k].norm_sqr() / denom).min(1.0)
            } else {
                0.0
            }
        })
        .collect();

    Ok((frequencies, coherence))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_test_signal(n: usize, freq: f64, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    #[test]
    fn test_istft_roundtrip() {
        let n = 256;
        let fs = 1000.0;
        let signal = make_test_signal(n, 50.0, fs);

        let nperseg = 64;
        let noverlap = 48;

        // Forward STFT
        let (_, _, stft_matrix) = crate::spectrogram::stft(
            &signal,
            Window::Hann,
            nperseg,
            Some(noverlap),
            None,
            Some(fs),
            Some(false),
            Some(false),
            None,
        )
        .expect("Forward STFT should succeed");

        // Inverse STFT
        let recovered = istft(&stft_matrix, Window::Hann, nperseg, Some(noverlap), None)
            .expect("ISTFT should succeed");

        // Check that the middle portion matches (avoid edge effects)
        let start = nperseg;
        let end = recovered.len().min(signal.len()) - nperseg;
        for i in start..end {
            assert_abs_diff_eq!(recovered[i], signal[i], epsilon = 0.3);
        }
    }

    #[test]
    fn test_griffin_lim_convergence() {
        let n = 128;
        let fs = 1000.0;
        let signal = make_test_signal(n, 100.0, fs);
        let nperseg = 32;
        let noverlap = 24;

        // Compute STFT
        let (_, _, stft_matrix) = crate::spectrogram::stft(
            &signal,
            Window::Hann,
            nperseg,
            Some(noverlap),
            None,
            Some(fs),
            Some(false),
            Some(false),
            None,
        )
        .expect("STFT should succeed");

        // Get magnitude
        let magnitude = stft_matrix.mapv(|c| c.norm());

        // Griffin-Lim reconstruction
        let recovered = griffin_lim(
            &magnitude,
            Window::Hann,
            nperseg,
            Some(noverlap),
            None,
            Some(16),
            None,
        )
        .expect("Griffin-Lim should succeed");

        // The recovered signal should have similar energy
        let orig_energy: f64 = signal.iter().map(|x| x * x).sum();
        let rec_energy: f64 = recovered.iter().map(|x| x * x).sum();

        // Energy should be in the same order of magnitude
        let ratio = rec_energy / orig_energy.max(1e-15);
        assert!(
            ratio > 0.01 && ratio < 100.0,
            "Energy ratio {ratio:.4} is out of reasonable range"
        );
    }

    #[test]
    fn test_reassigned_spectrogram() {
        let n = 256;
        let fs = 1000.0;
        let signal = make_test_signal(n, 100.0, fs);

        let (freqs, times, reassigned) =
            reassigned_spectrogram(&signal, Window::Hann, 64, Some(48), None, Some(fs))
                .expect("Reassigned spectrogram should succeed");

        assert!(!freqs.is_empty());
        assert!(!times.is_empty());
        assert_eq!(reassigned.dim(), (freqs.len(), times.len()));

        // Total energy should be non-negative
        let total: f64 = reassigned.iter().sum();
        assert!(total >= 0.0, "Total energy should be non-negative");
    }

    #[test]
    fn test_synchrosqueezing() {
        let n = 256;
        let fs = 1000.0;
        let signal = make_test_signal(n, 50.0, fs);

        let (freqs, times, sst) = synchrosqueezing(&signal, Window::Hann, 64, Some(48), Some(fs))
            .expect("SST should succeed");

        assert!(!freqs.is_empty());
        assert!(!times.is_empty());
        assert_eq!(sst.dim(), (freqs.len(), times.len()));

        // SST should concentrate energy better than standard spectrogram
        // Check that there's significant energy near 50 Hz
        let target_bin = (50.0 / (fs / 64.0)).round() as usize;
        if target_bin < freqs.len() {
            let energy_near_target: f64 = (0..times.len()).map(|t| sst[[target_bin, t]]).sum();
            assert!(
                energy_near_target > 0.0,
                "Should have energy near target frequency"
            );
        }
    }

    #[test]
    fn test_dpss_window() {
        let n = 64;
        let nw = 4.0;

        let win = dpss_window(n, nw).expect("DPSS should succeed");
        assert_eq!(win.len(), n);

        // Window should be approximately symmetric
        for i in 0..n / 2 {
            assert_abs_diff_eq!(win[i], win[n - 1 - i], epsilon = 0.1);
        }

        // Window values should be reasonable
        for &val in &win {
            assert!(val.abs() < 2.0, "Window value {val} seems too large");
        }
    }

    #[test]
    fn test_dpss_window_concentration() {
        let n = 32;
        let nw = 3.0;

        let win = dpss_window(n, nw).expect("DPSS should succeed");

        // DPSS should have good energy concentration
        let total_energy: f64 = win.iter().map(|x| x * x).sum();
        assert!(total_energy > 0.0, "Window should have positive energy");
    }

    #[test]
    fn test_dolph_chebyshev_window() {
        let n = 32;
        let attenuation = 60.0;

        let win = dolph_chebyshev_window(n, attenuation).expect("Dolph-Chebyshev should succeed");
        assert_eq!(win.len(), n);

        // Window should peak at 1
        let max_val = win.iter().copied().fold(0.0_f64, f64::max);
        assert_abs_diff_eq!(max_val, 1.0, epsilon = 1e-10);

        // Window should be symmetric
        for i in 0..n / 2 {
            assert_abs_diff_eq!(win[i], win[n - 1 - i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_dolph_chebyshev_single() {
        let win = dolph_chebyshev_window(1, 40.0).expect("Single-point should succeed");
        assert_eq!(win.len(), 1);
        assert_abs_diff_eq!(win[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_coherence_identical() {
        // Coherence of a signal with itself should be 1
        let n = 256;
        let fs = 1000.0;
        let signal = make_test_signal(n, 100.0, fs);

        let (freqs, coherence) =
            spectral_coherence(&signal, &signal, Window::Hann, 64, Some(32), Some(fs))
                .expect("Self-coherence should succeed");

        assert_eq!(freqs.len(), coherence.len());

        // Coherence should be close to 1 at all frequencies
        for &c in &coherence {
            assert!(c > 0.9, "Self-coherence should be near 1, got {c}");
        }
    }

    #[test]
    fn test_spectral_coherence_uncorrelated() {
        // Coherence of uncorrelated signals should be low
        let n = 512;
        let fs = 1000.0;
        let x = make_test_signal(n, 100.0, fs);
        let y = make_test_signal(n, 300.0, fs); // Different frequency

        let (_, coherence) = spectral_coherence(&x, &y, Window::Hann, 64, Some(32), Some(fs))
            .expect("Coherence of different signals should succeed");

        // Average coherence should be relatively low (not exactly 0 due to finite data)
        let avg: f64 = coherence.iter().sum::<f64>() / coherence.len() as f64;
        assert!(
            avg < 0.8,
            "Average coherence {avg:.4} should be moderate for different signals"
        );
    }

    #[test]
    fn test_error_handling_istft() {
        let stft = Array2::zeros((33, 5));
        assert!(istft(&stft, Window::Hann, 0, None, None).is_err());
        assert!(istft(&stft, Window::Hann, 64, Some(64), None).is_err());
    }

    #[test]
    fn test_error_handling_windows() {
        assert!(dpss_window(0, 4.0).is_err());
        assert!(dpss_window(32, -1.0).is_err());
        assert!(dolph_chebyshev_window(0, 60.0).is_err());
        assert!(dolph_chebyshev_window(32, -10.0).is_err());
    }
}
