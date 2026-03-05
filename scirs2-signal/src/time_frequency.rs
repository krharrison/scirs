//! Time-Frequency Analysis
//!
//! This module provides quadratic (bilinear) time-frequency distributions
//! from Cohen's class and related methods:
//!
//! - **Wigner-Ville Distribution (WVD)**: The fundamental bilinear TF representation
//! - **Pseudo Wigner-Ville Distribution (PWVD)**: Windowed WVD to reduce cross-terms
//! - **Smoothed Pseudo Wigner-Ville (SPWVD)**: Smoothed in both time and frequency
//! - **Choi-Williams Distribution (CWD)**: Exponential kernel to suppress cross-terms
//! - **Cohen's class**: General parameterised kernel TF distributions
//! - **Instantaneous frequency estimation**: From analytic signal phase
//! - **Reassignment methods**: Sharpen any TF distribution towards true ridges

use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Wigner-Ville Distribution
// ---------------------------------------------------------------------------

/// Result of a time-frequency distribution computation.
#[derive(Debug, Clone)]
pub struct TfDistribution {
    /// 2-D distribution: `data[time_idx * n_freq + freq_idx]`.
    pub data: Vec<f64>,
    /// Number of time points.
    pub n_time: usize,
    /// Number of frequency bins.
    pub n_freq: usize,
    /// Time axis values (in sample indices or seconds if `fs` provided).
    pub time_axis: Vec<f64>,
    /// Frequency axis values (normalised or in Hz if `fs` provided).
    pub freq_axis: Vec<f64>,
}

impl TfDistribution {
    /// Access value at `(t_idx, f_idx)`.
    pub fn at(&self, t_idx: usize, f_idx: usize) -> Option<f64> {
        if t_idx < self.n_time && f_idx < self.n_freq {
            Some(self.data[t_idx * self.n_freq + f_idx])
        } else {
            None
        }
    }
}

/// Compute the analytic signal via FFT (doubling positive-frequency content).
fn analytic_signal(x: &[f64]) -> SignalResult<Vec<Complex64>> {
    let n = x.len();
    let spectrum = scirs2_fft::fft(x, None)
        .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

    let mut h = vec![Complex64::new(0.0, 0.0); n];
    if n == 0 {
        return Ok(h);
    }
    h[0] = spectrum[0];
    let half = if n % 2 == 0 { n / 2 } else { (n + 1) / 2 };
    for i in 1..half {
        h[i] = spectrum[i] * 2.0;
    }
    if n % 2 == 0 {
        h[half] = spectrum[half];
    }

    let result = scirs2_fft::ifft(&h, None)
        .map_err(|e| SignalError::ComputationError(format!("IFFT error: {e}")))?;
    Ok(result)
}

/// Compute the Wigner-Ville Distribution of a real or complex signal.
///
/// The WVD is the Fourier transform of the instantaneous autocorrelation:
///
/// `W(t, f) = sum_tau  z(t+tau) * conj(z(t-tau)) * exp(-j2pi f tau)`
///
/// # Arguments
/// * `signal` - Real-valued input signal
/// * `n_freq` - Number of frequency bins (defaults to signal length, must be even)
/// * `fs` - Sampling frequency (defaults to 1.0)
///
/// # Returns
/// A `TfDistribution` with the WVD.
pub fn wigner_ville(
    signal: &[f64],
    n_freq: Option<usize>,
    fs: Option<f64>,
) -> SignalResult<TfDistribution> {
    let n = signal.len();
    if n < 2 {
        return Err(SignalError::ValueError(
            "Signal must have at least 2 samples".into(),
        ));
    }
    let fs_val = fs.unwrap_or(1.0);
    let nf = n_freq.unwrap_or(next_power_of_two(n));
    let nf = if nf % 2 != 0 { nf + 1 } else { nf };

    // Compute analytic signal
    let z = analytic_signal(signal)?;

    wvd_from_analytic(&z, nf, fs_val)
}

/// Core WVD computation from an analytic signal.
fn wvd_from_analytic(z: &[Complex64], n_freq: usize, fs: f64) -> SignalResult<TfDistribution> {
    let n = z.len();
    let n_time = n;

    let mut data = vec![0.0; n_time * n_freq];
    let mut row_buf = vec![Complex64::new(0.0, 0.0); n_freq];

    for t in 0..n_time {
        // Maximum lag
        let tau_max = t.min(n - 1 - t).min(n_freq / 2 - 1);

        // Fill the instantaneous autocorrelation
        for v in row_buf.iter_mut() {
            *v = Complex64::new(0.0, 0.0);
        }

        for tau in 0..=tau_max {
            let val = z[t + tau] * z[t.wrapping_sub(tau).min(n - 1)].conj();
            if tau < n_freq {
                row_buf[tau] = val;
            }
            if tau > 0 && (n_freq - tau) < n_freq {
                row_buf[n_freq - tau] = val.conj();
            }
        }

        // Proper conjugate symmetry for t-tau underflow
        if t < tau_max {
            // Already handled above via wrapping_sub/min
        }

        // FFT to get frequency distribution at this time
        let row_slice: Vec<Complex64> = row_buf.clone();
        let freq_row = scirs2_fft::fft(&row_slice, Some(n_freq))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

        for f in 0..n_freq {
            data[t * n_freq + f] = freq_row[f].re;
        }
    }

    let time_axis: Vec<f64> = (0..n_time).map(|t| t as f64 / fs).collect();
    let freq_axis: Vec<f64> = (0..n_freq).map(|f| f as f64 * fs / n_freq as f64).collect();

    Ok(TfDistribution {
        data,
        n_time,
        n_freq,
        time_axis,
        freq_axis,
    })
}

// ---------------------------------------------------------------------------
// Pseudo Wigner-Ville Distribution
// ---------------------------------------------------------------------------

/// Compute the Pseudo Wigner-Ville Distribution (frequency-smoothed WVD).
///
/// Uses a window `h` in the lag domain to reduce cross-term artefacts.
///
/// # Arguments
/// * `signal` - Real-valued input
/// * `window` - Lag-domain window (symmetric, length should be odd)
/// * `n_freq` - Number of frequency bins
/// * `fs` - Sampling frequency
pub fn pseudo_wigner_ville(
    signal: &[f64],
    window: &[f64],
    n_freq: Option<usize>,
    fs: Option<f64>,
) -> SignalResult<TfDistribution> {
    let n = signal.len();
    if n < 2 {
        return Err(SignalError::ValueError(
            "Signal must have at least 2 samples".into(),
        ));
    }
    if window.is_empty() {
        return Err(SignalError::ValueError("Window must not be empty".into()));
    }
    let fs_val = fs.unwrap_or(1.0);
    let nf = n_freq.unwrap_or(next_power_of_two(n));
    let nf = if nf % 2 != 0 { nf + 1 } else { nf };

    let z = analytic_signal(signal)?;
    let n_time = n;
    let half_win = window.len() / 2;

    let mut data = vec![0.0; n_time * nf];
    let mut row_buf = vec![Complex64::new(0.0, 0.0); nf];

    for t in 0..n_time {
        for v in row_buf.iter_mut() {
            *v = Complex64::new(0.0, 0.0);
        }

        let tau_max = t.min(n - 1 - t).min(half_win).min(nf / 2 - 1);

        for tau in 0..=tau_max {
            let w = if tau < window.len() {
                window[half_win.wrapping_add(tau).min(window.len() - 1)]
                    * window[half_win.wrapping_sub(tau).min(window.len() - 1)]
            } else {
                0.0
            };
            let val = z[t + tau] * z[t.wrapping_sub(tau).min(n - 1)].conj() * w;
            if tau < nf {
                row_buf[tau] = val;
            }
            if tau > 0 && (nf - tau) < nf {
                row_buf[nf - tau] = val.conj();
            }
        }

        let freq_row = scirs2_fft::fft(&row_buf.clone(), Some(nf))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

        for f in 0..nf {
            data[t * nf + f] = freq_row[f].re;
        }
    }

    let time_axis: Vec<f64> = (0..n_time).map(|t| t as f64 / fs_val).collect();
    let freq_axis: Vec<f64> = (0..nf).map(|f| f as f64 * fs_val / nf as f64).collect();

    Ok(TfDistribution {
        data,
        n_time,
        n_freq: nf,
        time_axis,
        freq_axis,
    })
}

// ---------------------------------------------------------------------------
// Smoothed Pseudo Wigner-Ville Distribution
// ---------------------------------------------------------------------------

/// Compute the Smoothed Pseudo Wigner-Ville Distribution.
///
/// Applies both a time-domain smoothing window `g` and a frequency-domain
/// (lag) window `h`.
///
/// # Arguments
/// * `signal` - Real-valued input
/// * `time_window` - Time-domain smoothing window
/// * `freq_window` - Frequency (lag) domain window
/// * `n_freq` - Number of frequency bins
/// * `fs` - Sampling frequency
pub fn smoothed_pseudo_wigner_ville(
    signal: &[f64],
    time_window: &[f64],
    freq_window: &[f64],
    n_freq: Option<usize>,
    fs: Option<f64>,
) -> SignalResult<TfDistribution> {
    let n = signal.len();
    if n < 2 {
        return Err(SignalError::ValueError(
            "Signal must have at least 2 samples".into(),
        ));
    }
    if time_window.is_empty() || freq_window.is_empty() {
        return Err(SignalError::ValueError("Windows must not be empty".into()));
    }
    let fs_val = fs.unwrap_or(1.0);
    let nf = n_freq.unwrap_or(next_power_of_two(n));
    let nf = if nf % 2 != 0 { nf + 1 } else { nf };

    let z = analytic_signal(signal)?;
    let n_time = n;
    let half_tw = time_window.len() / 2;
    let half_fw = freq_window.len() / 2;

    let mut data = vec![0.0; n_time * nf];
    let mut row_buf = vec![Complex64::new(0.0, 0.0); nf];

    for t in 0..n_time {
        for v in row_buf.iter_mut() {
            *v = Complex64::new(0.0, 0.0);
        }

        let tau_max = half_fw.min(nf / 2 - 1);

        for tau in 0..=tau_max {
            let fw = if tau <= half_fw && (half_fw + tau) < freq_window.len() {
                let idx_p = half_fw + tau;
                let idx_m = if tau <= half_fw { half_fw - tau } else { 0 };
                freq_window[idx_p.min(freq_window.len() - 1)]
                    * freq_window[idx_m.min(freq_window.len() - 1)]
            } else {
                0.0
            };

            // Time smoothing
            let mut accum = Complex64::new(0.0, 0.0);
            let u_min = if half_tw > t { 0 } else { t - half_tw };
            let u_max = (t + half_tw).min(n - 1);
            for u in u_min..=u_max {
                let tw_idx = (u as i64 - t as i64 + half_tw as i64) as usize;
                let tw = if tw_idx < time_window.len() {
                    time_window[tw_idx]
                } else {
                    0.0
                };
                let idx_p = u + tau;
                let idx_m = u.wrapping_sub(tau);
                if idx_p < n && idx_m < n {
                    accum += z[idx_p] * z[idx_m].conj() * tw;
                }
            }

            let val = accum * fw;
            if tau < nf {
                row_buf[tau] = val;
            }
            if tau > 0 && (nf - tau) < nf {
                row_buf[nf - tau] = val.conj();
            }
        }

        let freq_row = scirs2_fft::fft(&row_buf.clone(), Some(nf))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

        for f in 0..nf {
            data[t * nf + f] = freq_row[f].re;
        }
    }

    let time_axis: Vec<f64> = (0..n_time).map(|t| t as f64 / fs_val).collect();
    let freq_axis: Vec<f64> = (0..nf).map(|f| f as f64 * fs_val / nf as f64).collect();

    Ok(TfDistribution {
        data,
        n_time,
        n_freq: nf,
        time_axis,
        freq_axis,
    })
}

// ---------------------------------------------------------------------------
// Choi-Williams Distribution
// ---------------------------------------------------------------------------

/// Compute the Choi-Williams Distribution.
///
/// The CWD uses an exponential kernel `exp(-xi^2 * tau^2 / sigma)` to
/// suppress cross-term interference while preserving auto-terms.
///
/// # Arguments
/// * `signal` - Real-valued input
/// * `sigma` - Kernel bandwidth (larger = more smoothing). Typical: 1.0
/// * `n_freq` - Number of frequency bins
/// * `fs` - Sampling frequency
pub fn choi_williams(
    signal: &[f64],
    sigma: f64,
    n_freq: Option<usize>,
    fs: Option<f64>,
) -> SignalResult<TfDistribution> {
    let n = signal.len();
    if n < 2 {
        return Err(SignalError::ValueError(
            "Signal must have at least 2 samples".into(),
        ));
    }
    if sigma <= 0.0 {
        return Err(SignalError::ValueError("sigma must be positive".into()));
    }
    let fs_val = fs.unwrap_or(1.0);
    let nf = n_freq.unwrap_or(next_power_of_two(n));
    let nf = if nf % 2 != 0 { nf + 1 } else { nf };

    let z = analytic_signal(signal)?;
    let n_time = n;

    let mut data = vec![0.0; n_time * nf];
    let mut row_buf = vec![Complex64::new(0.0, 0.0); nf];

    for t in 0..n_time {
        for v in row_buf.iter_mut() {
            *v = Complex64::new(0.0, 0.0);
        }

        let tau_max = t.min(n - 1 - t).min(nf / 2 - 1);

        for tau in 0..=tau_max {
            if tau == 0 {
                // Kernel at tau=0 is 1 for all xi
                let val = z[t] * z[t].conj();
                row_buf[0] = val;
                continue;
            }

            // Sum over time shifts xi with exponential kernel
            let tau_sq = (tau as f64) * (tau as f64);
            let mut accum = Complex64::new(0.0, 0.0);
            let xi_max = ((5.0 * (sigma / tau_sq).sqrt()).ceil() as usize).min(n - 1);

            for xi_abs in 0..=xi_max {
                let xi_f = xi_abs as f64;
                let kernel_val = (-(xi_f * xi_f) * tau_sq / sigma).exp()
                    / (4.0 * PI * tau_sq / sigma).sqrt().max(f64::EPSILON);

                // Positive xi
                let idx_p = t as i64 + xi_abs as i64;
                let idx_m_p = t as i64 + xi_abs as i64;
                if idx_p + tau as i64 >= 0
                    && (idx_p + tau as i64) < n as i64
                    && (idx_p - tau as i64) >= 0
                    && (idx_p - tau as i64) < n as i64
                {
                    let r_val =
                        z[(idx_p + tau as i64) as usize] * z[(idx_p - tau as i64) as usize].conj();
                    accum += r_val * kernel_val;
                }

                // Negative xi (avoid double counting xi=0)
                if xi_abs > 0 {
                    let idx_n = t as i64 - xi_abs as i64;
                    if idx_n + tau as i64 >= 0
                        && (idx_n + tau as i64) < n as i64
                        && (idx_n - tau as i64) >= 0
                        && (idx_n - tau as i64) < n as i64
                    {
                        let r_val = z[(idx_n + tau as i64) as usize]
                            * z[(idx_n - tau as i64) as usize].conj();
                        accum += r_val * kernel_val;
                    }
                }
            }

            if tau < nf {
                row_buf[tau] = accum;
            }
            if tau > 0 && (nf - tau) < nf {
                row_buf[nf - tau] = accum.conj();
            }
        }

        let freq_row = scirs2_fft::fft(&row_buf.clone(), Some(nf))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

        for f in 0..nf {
            data[t * nf + f] = freq_row[f].re;
        }
    }

    let time_axis: Vec<f64> = (0..n_time).map(|t| t as f64 / fs_val).collect();
    let freq_axis: Vec<f64> = (0..nf).map(|f| f as f64 * fs_val / nf as f64).collect();

    Ok(TfDistribution {
        data,
        n_time,
        n_freq: nf,
        time_axis,
        freq_axis,
    })
}

// ---------------------------------------------------------------------------
// Cohen's class (general kernel)
// ---------------------------------------------------------------------------

/// A kernel function for Cohen's class distribution.
///
/// Takes `(xi, tau)` and returns the kernel value (real-valued).
pub type CohenKernelFn = fn(f64, f64) -> f64;

/// Predefined kernel: Wigner-Ville (identity kernel).
pub fn kernel_wigner_ville(_xi: f64, _tau: f64) -> f64 {
    1.0
}

/// Predefined kernel: Choi-Williams with parameter sigma.
pub fn kernel_choi_williams_factory(sigma: f64) -> impl Fn(f64, f64) -> f64 {
    move |xi: f64, tau: f64| {
        let tau_sq = tau * tau;
        if tau_sq < f64::EPSILON {
            return 1.0;
        }
        (-xi * xi * tau_sq / sigma).exp()
    }
}

/// Predefined kernel: Born-Jordan.
pub fn kernel_born_jordan(xi: f64, tau: f64) -> f64 {
    let tau_abs = tau.abs();
    if tau_abs < f64::EPSILON {
        return 1.0;
    }
    let arg = PI * xi * tau;
    if arg.abs() < f64::EPSILON {
        1.0
    } else {
        arg.sin() / arg
    }
}

/// Compute a Cohen's class TF distribution with an arbitrary kernel.
///
/// `kernel_fn` receives `(xi, tau)` both in sample units.
///
/// This is the general form; specific distributions (WVD, CWD) are special
/// cases with particular kernels.
pub fn cohens_class(
    signal: &[f64],
    kernel_fn: CohenKernelFn,
    n_freq: Option<usize>,
    fs: Option<f64>,
) -> SignalResult<TfDistribution> {
    let n = signal.len();
    if n < 2 {
        return Err(SignalError::ValueError(
            "Signal must have at least 2 samples".into(),
        ));
    }
    let fs_val = fs.unwrap_or(1.0);
    let nf = n_freq.unwrap_or(next_power_of_two(n));
    let nf = if nf % 2 != 0 { nf + 1 } else { nf };

    let z = analytic_signal(signal)?;
    let n_time = n;

    let mut data = vec![0.0; n_time * nf];
    let mut row_buf = vec![Complex64::new(0.0, 0.0); nf];

    for t in 0..n_time {
        for v in row_buf.iter_mut() {
            *v = Complex64::new(0.0, 0.0);
        }

        let tau_max = t.min(n - 1 - t).min(nf / 2 - 1);

        for tau in 0..=tau_max {
            // Sum over xi
            let mut accum = Complex64::new(0.0, 0.0);
            let xi_range = n.min(20); // limit computation range

            for xi_idx in 0..xi_range {
                let xi = xi_idx as f64 - xi_range as f64 / 2.0;
                let u = t as i64 + xi as i64;
                if u + tau as i64 >= 0
                    && (u + tau as i64) < n as i64
                    && u - tau as i64 >= 0
                    && (u - tau as i64) < n as i64
                {
                    let kval = kernel_fn(xi, tau as f64);
                    let r = z[(u + tau as i64) as usize] * z[(u - tau as i64) as usize].conj();
                    accum += r * kval;
                }
            }

            if tau < nf {
                row_buf[tau] = accum;
            }
            if tau > 0 && (nf - tau) < nf {
                row_buf[nf - tau] = accum.conj();
            }
        }

        let freq_row = scirs2_fft::fft(&row_buf.clone(), Some(nf))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

        for f in 0..nf {
            data[t * nf + f] = freq_row[f].re;
        }
    }

    let time_axis: Vec<f64> = (0..n_time).map(|t| t as f64 / fs_val).collect();
    let freq_axis: Vec<f64> = (0..nf).map(|f| f as f64 * fs_val / nf as f64).collect();

    Ok(TfDistribution {
        data,
        n_time,
        n_freq: nf,
        time_axis,
        freq_axis,
    })
}

// ---------------------------------------------------------------------------
// Instantaneous Frequency Estimation
// ---------------------------------------------------------------------------

/// Estimate the instantaneous frequency of a real-valued signal.
///
/// Computes the analytic signal and differentiates its phase.
///
/// # Arguments
/// * `signal` - Real-valued input
/// * `fs` - Sampling frequency (defaults to 1.0)
///
/// # Returns
/// A vector of instantaneous frequency values (length = signal.len() - 1).
pub fn instantaneous_frequency(signal: &[f64], fs: Option<f64>) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    if n < 3 {
        return Err(SignalError::ValueError(
            "Signal must have at least 3 samples".into(),
        ));
    }
    let fs_val = fs.unwrap_or(1.0);
    let z = analytic_signal(signal)?;

    // Phase unwrapping and differentiation
    let phases: Vec<f64> = z.iter().map(|c| c.im.atan2(c.re)).collect();

    // Unwrap phase
    let mut unwrapped = vec![0.0; n];
    unwrapped[0] = phases[0];
    for i in 1..n {
        let mut diff = phases[i] - phases[i - 1];
        // Unwrap
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
        }
        unwrapped[i] = unwrapped[i - 1] + diff;
    }

    // Instantaneous frequency = d(phase)/dt / (2*pi)
    let mut inst_freq = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let dphi = unwrapped[i + 1] - unwrapped[i];
        let freq = dphi * fs_val / (2.0 * PI);
        inst_freq.push(freq);
    }

    Ok(inst_freq)
}

/// Estimate instantaneous amplitude (envelope) of a real-valued signal.
///
/// Returns the magnitude of the analytic signal.
pub fn instantaneous_amplitude(signal: &[f64]) -> SignalResult<Vec<f64>> {
    if signal.len() < 2 {
        return Err(SignalError::ValueError(
            "Signal must have at least 2 samples".into(),
        ));
    }
    let z = analytic_signal(signal)?;
    Ok(z.iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect())
}

// ---------------------------------------------------------------------------
// Reassignment Methods
// ---------------------------------------------------------------------------

/// Reassigned spectrogram coordinates for sharpening a TF distribution.
#[derive(Debug, Clone)]
pub struct ReassignedTfDistribution {
    /// Original distribution.
    pub original: TfDistribution,
    /// Reassigned distribution (same dimensions).
    pub reassigned: TfDistribution,
    /// Reassigned time coordinates `[n_time * n_freq]`.
    pub reassigned_times: Vec<f64>,
    /// Reassigned frequency coordinates `[n_time * n_freq]`.
    pub reassigned_freqs: Vec<f64>,
}

/// Compute a reassigned spectrogram.
///
/// The reassignment method computes the local group delay and instantaneous
/// frequency from the STFT and uses them to redistribute energy to the
/// correct TF location.
///
/// # Arguments
/// * `signal` - Real-valued input
/// * `window` - Analysis window
/// * `hop_size` - Hop size in samples
/// * `n_freq` - FFT size (>= window length)
/// * `fs` - Sampling frequency
pub fn reassigned_spectrogram(
    signal: &[f64],
    window: &[f64],
    hop_size: usize,
    n_freq: Option<usize>,
    fs: Option<f64>,
) -> SignalResult<ReassignedTfDistribution> {
    let n = signal.len();
    let win_len = window.len();
    if n < win_len {
        return Err(SignalError::ValueError(
            "Signal must be at least as long as the window".into(),
        ));
    }
    if hop_size == 0 {
        return Err(SignalError::InvalidArgument("hop_size must be > 0".into()));
    }
    let fs_val = fs.unwrap_or(1.0);
    let nf = n_freq.unwrap_or(next_power_of_two(win_len));
    let nf = if nf < win_len {
        next_power_of_two(win_len)
    } else {
        nf
    };
    let nf = if nf % 2 != 0 { nf + 1 } else { nf };

    // Derivative window: d_window[k] = k * window[k]
    let d_window: Vec<f64> = window
        .iter()
        .enumerate()
        .map(|(k, &w)| (k as f64 - win_len as f64 / 2.0) * w)
        .collect();

    // Time-weighted window: t_window[k] = k * window[k] (for group delay)
    let t_window: Vec<f64> = window
        .iter()
        .enumerate()
        .map(|(k, &w)| k as f64 * w)
        .collect();

    let n_time = (n.saturating_sub(win_len)) / hop_size + 1;
    let n_freq_out = nf / 2 + 1;

    let mut orig_data = vec![0.0; n_time * n_freq_out];
    let mut reass_data = vec![0.0; n_time * n_freq_out];
    let mut reass_times = vec![0.0; n_time * n_freq_out];
    let mut reass_freqs = vec![0.0; n_time * n_freq_out];

    for frame in 0..n_time {
        let start = frame * hop_size;
        let end = (start + win_len).min(n);
        let actual_len = end - start;

        // Windowed frame
        let mut windowed = vec![0.0; nf];
        let mut d_windowed = vec![0.0; nf];
        let mut t_windowed = vec![0.0; nf];
        for k in 0..actual_len {
            windowed[k] = signal[start + k] * window[k];
            d_windowed[k] = signal[start + k] * d_window[k];
            t_windowed[k] = signal[start + k] * t_window[k];
        }

        // FFTs
        let stft_frame = scirs2_fft::fft(&windowed, Some(nf))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;
        let d_stft = scirs2_fft::fft(&d_windowed, Some(nf))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;
        let t_stft = scirs2_fft::fft(&t_windowed, Some(nf))
            .map_err(|e| SignalError::ComputationError(format!("FFT error: {e}")))?;

        let center_time = start as f64 + win_len as f64 / 2.0;

        for f in 0..n_freq_out {
            let mag_sq = stft_frame[f].re * stft_frame[f].re + stft_frame[f].im * stft_frame[f].im;
            orig_data[frame * n_freq_out + f] = mag_sq;

            if mag_sq > f64::EPSILON {
                // Reassigned time: t_hat = t + Re(T_STFT / STFT)
                let t_ratio = t_stft[f] * stft_frame[f].conj() / mag_sq;
                let time_reass = (center_time + t_ratio.re) / fs_val;

                // Reassigned frequency: f_hat = f - Im(D_STFT / STFT) / (2*pi)
                let d_ratio = d_stft[f] * stft_frame[f].conj() / mag_sq;
                let freq_orig = f as f64 * fs_val / nf as f64;
                let freq_reass = freq_orig - d_ratio.im * fs_val / (2.0 * PI);

                reass_times[frame * n_freq_out + f] = time_reass;
                reass_freqs[frame * n_freq_out + f] = freq_reass;
            } else {
                reass_times[frame * n_freq_out + f] = center_time / fs_val;
                reass_freqs[frame * n_freq_out + f] = f as f64 * fs_val / nf as f64;
            }
        }
    }

    // Accumulate reassigned energy
    let time_axis: Vec<f64> = (0..n_time)
        .map(|t| (t * hop_size) as f64 / fs_val + (win_len as f64 / 2.0) / fs_val)
        .collect();
    let freq_axis: Vec<f64> = (0..n_freq_out)
        .map(|f| f as f64 * fs_val / nf as f64)
        .collect();

    // Map reassigned energy to nearest bin
    let dt = if n_time > 1 {
        time_axis[1] - time_axis[0]
    } else {
        1.0 / fs_val
    };
    let df = if n_freq_out > 1 {
        freq_axis[1] - freq_axis[0]
    } else {
        fs_val / nf as f64
    };

    for frame in 0..n_time {
        for f in 0..n_freq_out {
            let idx = frame * n_freq_out + f;
            let energy = orig_data[idx];
            if energy < f64::EPSILON {
                continue;
            }
            let rt = reass_times[idx];
            let rf = reass_freqs[idx];

            // Find nearest bin
            let t_bin = if dt > 0.0 {
                let tb = ((rt - time_axis[0]) / dt).round() as i64;
                tb.max(0).min(n_time as i64 - 1) as usize
            } else {
                frame
            };
            let f_bin = if df > 0.0 {
                let fb = ((rf - freq_axis[0]) / df).round() as i64;
                fb.max(0).min(n_freq_out as i64 - 1) as usize
            } else {
                f
            };

            reass_data[t_bin * n_freq_out + f_bin] += energy;
        }
    }

    let original = TfDistribution {
        data: orig_data,
        n_time,
        n_freq: n_freq_out,
        time_axis: time_axis.clone(),
        freq_axis: freq_axis.clone(),
    };
    let reassigned = TfDistribution {
        data: reass_data,
        n_time,
        n_freq: n_freq_out,
        time_axis,
        freq_axis,
    };

    Ok(ReassignedTfDistribution {
        original,
        reassigned,
        reassigned_times: reass_times,
        reassigned_freqs: reass_freqs,
    })
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

/// Generate a Hann window of given length (useful for PWVD / SPWVD).
pub fn hann_window(length: usize) -> Vec<f64> {
    (0..length)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (length.max(1) - 1).max(1) as f64).cos()))
        .collect()
}

/// Generate a Gaussian window of given length and standard deviation.
pub fn gaussian_window(length: usize, std_dev: f64) -> Vec<f64> {
    let center = (length as f64 - 1.0) / 2.0;
    (0..length)
        .map(|i| {
            let x = (i as f64 - center) / std_dev;
            (-0.5 * x * x).exp()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_chirp(n: usize, f0: f64, f1: f64, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let t_max = (n - 1) as f64 / fs;
                let freq = f0 + (f1 - f0) * t / (2.0 * t_max);
                (2.0 * PI * freq * t).sin()
            })
            .collect()
    }

    fn make_tone(n: usize, freq: f64, fs: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    #[test]
    fn test_wigner_ville_basic() {
        let signal = make_tone(64, 8.0, 64.0);
        let result = wigner_ville(&signal, Some(64), Some(64.0));
        assert!(result.is_ok());
        let tf = result.expect("wvd");
        assert_eq!(tf.n_time, 64);
        assert_eq!(tf.n_freq, 64);
        // Energy should be concentrated near frequency 8 Hz
        // Check the middle time point
        let t_mid = 32;
        let f_bin = 8; // 8 Hz at 64 Hz sampling with 64 bins = bin 8
        let val = tf.at(t_mid, f_bin).unwrap_or(0.0);
        assert!(
            val.abs() > 0.0,
            "WVD should have energy at the signal frequency"
        );
    }

    #[test]
    fn test_pseudo_wigner_ville() {
        let signal = make_tone(64, 8.0, 64.0);
        let window = hann_window(15);
        let result = pseudo_wigner_ville(&signal, &window, Some(64), Some(64.0));
        assert!(result.is_ok());
        let tf = result.expect("pwvd");
        assert_eq!(tf.n_time, 64);
    }

    #[test]
    fn test_smoothed_pseudo_wigner_ville() {
        let signal = make_tone(64, 8.0, 64.0);
        let tw = hann_window(11);
        let fw = hann_window(11);
        let result = smoothed_pseudo_wigner_ville(&signal, &tw, &fw, Some(64), Some(64.0));
        assert!(result.is_ok());
        let tf = result.expect("spwvd");
        assert_eq!(tf.n_time, 64);
    }

    #[test]
    fn test_choi_williams() {
        let signal = make_tone(64, 8.0, 64.0);
        let result = choi_williams(&signal, 1.0, Some(64), Some(64.0));
        assert!(result.is_ok());
        let tf = result.expect("cwd");
        assert_eq!(tf.n_time, 64);
    }

    #[test]
    fn test_cohens_class_wvd_kernel() {
        let signal = make_tone(32, 4.0, 32.0);
        let result = cohens_class(&signal, kernel_wigner_ville, Some(32), Some(32.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_cohens_class_born_jordan() {
        let signal = make_tone(32, 4.0, 32.0);
        let result = cohens_class(&signal, kernel_born_jordan, Some(32), Some(32.0));
        assert!(result.is_ok());
    }

    #[test]
    fn test_instantaneous_frequency_constant() {
        let fs = 256.0;
        let freq = 20.0;
        let n = 256;
        let signal = make_tone(n, freq, fs);
        let inst_freq = instantaneous_frequency(&signal, Some(fs)).expect("inst freq");
        assert_eq!(inst_freq.len(), n - 1);
        // In the middle portion the frequency should be close to 20 Hz
        let mid_start = n / 4;
        let mid_end = 3 * n / 4;
        let avg_freq: f64 =
            inst_freq[mid_start..mid_end].iter().sum::<f64>() / (mid_end - mid_start) as f64;
        assert!(
            (avg_freq - freq).abs() < 1.0,
            "Expected ~{} Hz, got {} Hz",
            freq,
            avg_freq
        );
    }

    #[test]
    fn test_instantaneous_amplitude() {
        let n = 256;
        let signal = make_tone(n, 10.0, 256.0);
        let envelope = instantaneous_amplitude(&signal).expect("envelope");
        assert_eq!(envelope.len(), n);
        // Envelope of a pure tone should be approximately constant and ~1
        let mid = &envelope[n / 4..3 * n / 4];
        let avg: f64 = mid.iter().sum::<f64>() / mid.len() as f64;
        assert!(
            (avg - 1.0).abs() < 0.15,
            "Expected envelope ~1.0, got {}",
            avg
        );
    }

    #[test]
    fn test_reassigned_spectrogram() {
        let n = 256;
        let fs = 256.0;
        let signal = make_tone(n, 20.0, fs);
        let window = hann_window(64);
        let result = reassigned_spectrogram(&signal, &window, 32, Some(128), Some(fs));
        assert!(result.is_ok());
        let rs = result.expect("reassigned");
        assert!(rs.original.n_time > 0);
        assert!(rs.reassigned.n_time > 0);
        assert_eq!(
            rs.reassigned_times.len(),
            rs.original.n_time * rs.original.n_freq
        );
    }

    #[test]
    fn test_hann_window() {
        let w = hann_window(7);
        assert_eq!(w.len(), 7);
        // Endpoints should be 0
        assert!(w[0].abs() < 1e-10);
        assert!(w[6].abs() < 1e-10);
        // Centre should be 1
        assert!((w[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gaussian_window() {
        let w = gaussian_window(11, 2.0);
        assert_eq!(w.len(), 11);
        // Centre should be 1
        assert!((w[5] - 1.0).abs() < 1e-10);
        // Symmetric
        for i in 0..5 {
            assert!((w[i] - w[10 - i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_chirp_wvd() {
        let signal = make_chirp(128, 5.0, 50.0, 256.0);
        let result = wigner_ville(&signal, Some(128), Some(256.0));
        assert!(result.is_ok());
        let tf = result.expect("chirp wvd");
        // Just verify dimensions
        assert_eq!(tf.n_time, 128);
        assert_eq!(tf.n_freq, 128);
    }

    #[test]
    fn test_tf_distribution_at() {
        let tf = TfDistribution {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            n_time: 2,
            n_freq: 3,
            time_axis: vec![0.0, 1.0],
            freq_axis: vec![0.0, 1.0, 2.0],
        };
        assert_eq!(tf.at(0, 0), Some(1.0));
        assert_eq!(tf.at(0, 2), Some(3.0));
        assert_eq!(tf.at(1, 1), Some(5.0));
        assert_eq!(tf.at(2, 0), None);
    }

    #[test]
    fn test_wvd_errors() {
        assert!(wigner_ville(&[], None, None).is_err());
        assert!(wigner_ville(&[1.0], None, None).is_err());
    }

    #[test]
    fn test_instantaneous_freq_errors() {
        assert!(instantaneous_frequency(&[1.0, 2.0], None).is_err());
    }
}
