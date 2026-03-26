//! Wigner-Ville Distribution (WVD) and Pseudo-WVD (PWVD) computation.
//!
//! The Wigner-Ville distribution is a time-frequency energy distribution that
//! provides perfect time-frequency localisation for linear chirps and mono-component
//! signals.  For multi-component signals it exhibits cross-terms (interference
//! terms) between distinct signal components.
//!
//! The Pseudo-WVD (PWVD) smooths the instantaneous autocorrelation with a window
//! function prior to the Fourier transform, trading cross-term reduction for some
//! loss of resolution.
//!
//! # Mathematical Background
//!
//! Given the analytic signal `z(t)`, the WVD is defined as:
//!
//! ```text
//! W(t, f) = ∫ z(t + τ/2) · conj(z(t − τ/2)) · e^{−j 2π f τ} dτ
//! ```
//!
//! In discrete form:
//!
//! ```text
//! W[n, k] = 2 · Re{ Σ_{m} z[n+m] · conj(z[n−m]) · e^{−j 2π k m / N} }
//! ```
//!
//! # References
//!
//! * Claasen, T.A.C.M., Mecklenbraüker, W.F.G. "The Wigner distribution —
//!   A tool for time-frequency signal analysis." Philips J. Res., 1980.
//! * Cohen, L. "Time-Frequency Analysis." Prentice-Hall, 1995.

use crate::error::{FFTError, FFTResult};
use crate::fft::fft;
use crate::hilbert::analytic_signal;
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

use super::types::{WvdConfig, WvdResult};

/// Wigner-Ville Distribution (WVD) calculator.
///
/// # Examples
///
/// ```
/// use scirs2_fft::wigner_ville::{WignerVille, WvdConfig};
/// use std::f64::consts::PI;
///
/// let n = 64;
/// let fs = 64.0;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 4.0 * i as f64 / fs).sin())
///     .collect();
///
/// let wv = WignerVille::new();
/// let config = WvdConfig::default();
/// let result = wv.compute_wvd(&signal, fs, &config).expect("WVD should succeed");
///
/// assert_eq!(result.wvd.len(), n);
/// assert!(!result.frequencies.is_empty());
/// ```
pub struct WignerVille;

impl WignerVille {
    /// Create a new `WignerVille` calculator.
    pub fn new() -> Self {
        Self
    }

    /// Compute the Wigner-Ville Distribution of `signal`.
    ///
    /// # Arguments
    ///
    /// * `signal` - Real-valued input signal.
    /// * `fs`     - Sampling frequency in Hz.
    /// * `config` - Configuration parameters.
    ///
    /// # Returns
    ///
    /// A [`WvdResult`] with the distribution matrix, time axis, and frequency axis.
    ///
    /// # Errors
    ///
    /// Returns an error if the signal is empty or if an internal FFT fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_fft::wigner_ville::{WignerVille, WvdConfig};
    /// use std::f64::consts::PI;
    ///
    /// let n = 32;
    /// let fs = 32.0;
    /// let signal: Vec<f64> = (0..n)
    ///     .map(|i| (2.0 * PI * 4.0 * i as f64 / fs).cos())
    ///     .collect();
    ///
    /// let wv = WignerVille::new();
    /// let config = WvdConfig::default();
    /// let result = wv.compute_wvd(&signal, fs, &config).expect("WVD should succeed");
    /// assert_eq!(result.wvd.len(), n);
    /// assert_eq!(result.wvd[0].len(), result.frequencies.len());
    /// ```
    pub fn compute_wvd(&self, signal: &[f64], fs: f64, config: &WvdConfig) -> FFTResult<WvdResult> {
        compute_wvd_impl(signal, fs, config, false)
    }

    /// Compute the Pseudo-WVD (PWVD) of `signal`.
    ///
    /// The PWVD applies a Gaussian window of half-length `config.smooth_window`
    /// to the instantaneous autocorrelation before the FFT, reducing cross-terms.
    /// If `config.smooth_window == 0` the result is identical to the plain WVD.
    ///
    /// # Arguments
    ///
    /// * `signal` - Real-valued input signal.
    /// * `fs`     - Sampling frequency in Hz.
    /// * `config` - Configuration parameters (use `smooth_window > 0` for PWVD).
    ///
    /// # Errors
    ///
    /// Returns an error if the signal is empty or an internal FFT fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_fft::wigner_ville::{WignerVille, WvdConfig};
    /// use std::f64::consts::PI;
    ///
    /// let n = 32;
    /// let fs = 32.0;
    /// let signal: Vec<f64> = (0..n)
    ///     .map(|i| (2.0 * PI * 3.0 * i as f64 / fs).sin())
    ///     .collect();
    ///
    /// let wv = WignerVille::new();
    /// let mut config = WvdConfig::default();
    /// config.smooth_window = 5;
    ///
    /// let result = wv.compute_pwvd(&signal, fs, &config).expect("PWVD should succeed");
    /// assert_eq!(result.wvd.len(), n);
    /// ```
    pub fn compute_pwvd(
        &self,
        signal: &[f64],
        fs: f64,
        config: &WvdConfig,
    ) -> FFTResult<WvdResult> {
        compute_wvd_impl(signal, fs, config, true)
    }
}

impl Default for WignerVille {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Core implementation
// ---------------------------------------------------------------------------

/// Shared implementation for WVD and PWVD.
fn compute_wvd_impl(
    signal: &[f64],
    fs: f64,
    config: &WvdConfig,
    apply_smooth: bool,
) -> FFTResult<WvdResult> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::ValueError("Signal must not be empty".to_string()));
    }
    if fs <= 0.0 {
        return Err(FFTError::ValueError(
            "Sampling frequency must be positive".to_string(),
        ));
    }

    // Determine FFT size (number of frequency bins)
    let n_freqs = config.n_freqs.unwrap_or(n);
    if n_freqs == 0 {
        return Err(FFTError::ValueError("n_freqs must be > 0".to_string()));
    }

    // Compute analytic signal if requested (suppresses negative-frequency artefacts)
    let z: Vec<Complex64> = if config.analytic {
        analytic_signal(signal)?
    } else {
        signal.iter().map(|&s| Complex64::new(s, 0.0)).collect()
    };

    // Build smooth window (Gaussian) for PWVD
    let smooth_len = if apply_smooth {
        config.smooth_window
    } else {
        0
    };
    let gaussian_window: Vec<f64> = build_gaussian_window(smooth_len);

    // Allocate output: [time][freq]
    let mut wvd_matrix: Vec<Vec<f64>> = Vec::with_capacity(n);

    for t in 0..n {
        // Build instantaneous autocorrelation R(t, τ) for τ in -τ_max..τ_max
        // τ_max is limited by signal boundaries.
        let tau_max = t.min(n - 1 - t);

        // Buffer of length n_freqs for the FFT (zero-padded if needed)
        let mut acf_buf: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n_freqs];

        for tau in 0..=tau_max.min(n_freqs / 2) {
            // R(t, τ) = z(t+τ) · conj(z(t-τ))
            let r_pos = z[t + tau] * z[t - tau].conj();
            let r_neg = if tau > 0 {
                z[t - tau] * z[t + tau].conj()
            } else {
                r_pos
            };

            // Apply smooth window weight
            let win_weight = if smooth_len > 0 {
                let w_idx = tau.min(gaussian_window.len().saturating_sub(1));
                gaussian_window[w_idx]
            } else {
                1.0
            };

            // Place in buffer (symmetric about τ=0 for proper FFT)
            let buf_idx_pos = tau % n_freqs;
            let buf_idx_neg = if tau > 0 {
                (n_freqs - tau % n_freqs) % n_freqs
            } else {
                0
            };

            acf_buf[buf_idx_pos] = acf_buf[buf_idx_pos] + r_pos * win_weight;
            if tau > 0 && buf_idx_neg < n_freqs {
                acf_buf[buf_idx_neg] = acf_buf[buf_idx_neg] + r_neg * win_weight;
            }
        }

        // FFT over τ to get W(t, f)
        let wvd_row_complex = fft_complex(&acf_buf)?;

        // The WVD is real-valued (take real part)
        let row: Vec<f64> = wvd_row_complex.iter().map(|c| c.re).collect();
        wvd_matrix.push(row);
    }

    // Build time and frequency axes
    let times: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
    let frequencies: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * fs / n_freqs as f64)
        .collect();

    Ok(WvdResult {
        wvd: wvd_matrix,
        times,
        frequencies,
    })
}

/// Compute FFT of a complex-valued slice.
fn fft_complex(signal: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    // Convert to (re, im) pairs as f64 slices and call the underlying FFT.
    // We use the real-valued fft on the real parts and handle the imaginary parts
    // manually via the FFT of the imaginary parts.
    // More efficiently: use a single real FFT of length 2n or call the complex FFT.
    // Since the crate's fft() only accepts real inputs, we implement the complex FFT
    // via the two-real-FFT trick:
    //   F{z} = F{re} + j * F{im}
    let n = signal.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    let re_part: Vec<f64> = signal.iter().map(|c| c.re).collect();
    let im_part: Vec<f64> = signal.iter().map(|c| c.im).collect();

    let fft_re = crate::fft::fft(&re_part, None)?;
    let fft_im = crate::fft::fft(&im_part, None)?;

    // F{z} = F{x} + j * F{y}
    // where z = x + jy, so F{z}[k] = F{x}[k] + j*F{y}[k]
    let result: Vec<Complex64> = fft_re
        .iter()
        .zip(fft_im.iter())
        .map(|(r, i)| {
            // F{x}[k] + j * F{y}[k]
            // = (r.re + j*r.im) + j*(i.re + j*i.im)
            // = r.re + j*r.im + j*i.re - i.im
            // = (r.re - i.im) + j*(r.im + i.re)
            Complex64::new(r.re - i.im, r.im + i.re)
        })
        .collect();

    Ok(result)
}

/// Build a Gaussian window of half-length `half_len` centered at 0.
///
/// Returns `[w(0), w(1), ..., w(half_len)]` where `w(τ) = exp(-τ² / (2σ²))`.
/// If `half_len == 0`, returns `[1.0]`.
fn build_gaussian_window(half_len: usize) -> Vec<f64> {
    if half_len == 0 {
        return vec![1.0];
    }

    let sigma = half_len as f64 / 2.0;
    let sigma2 = 2.0 * sigma * sigma;

    (0..=half_len)
        .map(|tau| {
            let t = tau as f64;
            (-t * t / sigma2).exp()
        })
        .collect()
}

/// Convenience function: compute WVD with default settings.
///
/// # Errors
///
/// Returns an error if the signal is empty or internal FFT fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::wigner_ville::distribution::compute_wvd;
/// use std::f64::consts::PI;
///
/// let n = 32;
/// let signal: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
///     .collect();
/// let result = compute_wvd(&signal, 1.0).expect("should succeed");
/// assert_eq!(result.wvd.len(), n);
/// ```
pub fn compute_wvd(signal: &[f64], fs: f64) -> FFTResult<WvdResult> {
    let config = WvdConfig::default();
    let wv = WignerVille::new();
    wv.compute_wvd(signal, fs, &config)
}

/// Convenience function: compute PWVD with a given smooth window.
///
/// # Errors
///
/// Returns an error if the signal is empty or internal FFT fails.
pub fn compute_pwvd(signal: &[f64], fs: f64, smooth_window: usize) -> FFTResult<WvdResult> {
    let config = WvdConfig {
        smooth_window,
        ..WvdConfig::default()
    };
    let wv = WignerVille::new();
    wv.compute_pwvd(signal, fs, &config)
}
