//! FFT-based correlation, convolution enhancements, and related utilities.
//!
//! This module provides high-performance correlation and convolution algorithms
//! built on top of FFT, as well as 2D correlation, normalized cross-correlation,
//! phase correlation for image alignment, and autocorrelation.
//!
//! # Functions
//!
//! * [`fft_correlate`] — Full/Same/Valid cross-correlation via FFT.
//! * [`fft_convolve`] — Fast linear convolution with mode selection.
//! * [`fft_correlate2d`] — 2-D cross-correlation for flat arrays.
//! * [`normalized_cross_correlation`] — Normalized cross-correlation (NCC).
//! * [`phase_correlation`] — Image alignment via phase correlation.
//! * [`autocorrelation_fft`] — Autocorrelation in O(N log N).

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use crate::helper::next_fast_len;
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  Mode enumerations
// ─────────────────────────────────────────────────────────────────────────────

/// Output size mode for [`fft_correlate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationMode {
    /// Full output: length `a.len() + b.len() - 1`.
    Full,
    /// Trimmed to `max(a.len(), b.len())`.
    Same,
    /// Only the fully-overlapping region.
    Valid,
}

/// Output size mode for [`fft_convolve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolutionMode {
    /// Full output: length `a.len() + b.len() - 1`.
    Full,
    /// Trimmed to `max(a.len(), b.len())`.
    Same,
    /// Only the fully-overlapping region.
    Valid,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Zero-pad a slice to length `n`.
#[inline]
fn zero_pad(x: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n];
    let copy = x.len().min(n);
    out[..copy].copy_from_slice(&x[..copy]);
    out
}

/// Extract real parts of a complex vector.
#[inline]
fn real_parts(v: &[Complex64]) -> Vec<f64> {
    v.iter().map(|c| c.re).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  fft_correlate
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-correlation of two real signals via FFT.
///
/// Computes `corr(a, b)[τ] = Σ_t  a[t] · b[t + τ]`.
///
/// In the frequency domain this is `IFFT(conj(FFT(a)) · FFT(b))`.
///
/// # Arguments
///
/// * `a`    – First signal.
/// * `b`    – Second signal.
/// * `mode` – Output size selection.
///
/// # Errors
///
/// Returns an error if either input is empty or an FFT call fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::correlation::{fft_correlate, CorrelationMode};
///
/// let a = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let corr = fft_correlate(&a, &a, CorrelationMode::Full).expect("autocorrelation");
/// // The peak of autocorrelation is at lag 0 (index a.len()-1 in Full mode).
/// let centre = a.len() - 1;
/// assert!(corr[centre] >= corr[centre + 1]);
/// ```
pub fn fft_correlate(a: &[f64], b: &[f64], mode: CorrelationMode) -> FFTResult<Vec<f64>> {
    if a.is_empty() {
        return Err(FFTError::ValueError("fft_correlate: a is empty".into()));
    }
    if b.is_empty() {
        return Err(FFTError::ValueError("fft_correlate: b is empty".into()));
    }

    let full_len = a.len() + b.len() - 1;
    let fft_len = next_fast_len(full_len, true);

    let a_pad = zero_pad(a, fft_len);
    let b_pad = zero_pad(b, fft_len);

    let a_freq = fft(&a_pad, None)?;
    let b_freq = fft(&b_pad, None)?;

    // corr(a,b) = IFFT(conj(A) * B)
    let product: Vec<Complex64> = a_freq
        .iter()
        .zip(b_freq.iter())
        .map(|(&af, &bf)| af.conj() * bf)
        .collect();

    let corr_c = ifft(&product, None)?;

    // Rearrange from circular representation to standard lag ordering:
    //   positive lags (0 .. b.len()-1)  → stored at indices 0 .. b.len()-1
    //   negative lags (-(a.len()-1)..-1) → stored at indices fft_len-a.len()+1 .. fft_len-1
    let pos_lags = b.len();
    let neg_lags = a.len() - 1;
    let mut full = vec![0.0_f64; full_len];

    for i in 0..pos_lags {
        full[neg_lags + i] = corr_c[i].re;
    }
    for i in 0..neg_lags {
        let src = fft_len - neg_lags + i;
        full[i] = corr_c[src].re;
    }

    let result = trim_output_corr(&full, a.len(), b.len(), full_len, mode);
    Ok(result)
}

/// Trim the full cross-correlation output according to the requested mode.
fn trim_output_corr(
    full: &[f64],
    a_len: usize,
    b_len: usize,
    full_len: usize,
    mode: CorrelationMode,
) -> Vec<f64> {
    match mode {
        CorrelationMode::Full => full.to_vec(),
        CorrelationMode::Same => {
            let out_len = a_len.max(b_len);
            let offset = (full_len - out_len) / 2;
            full[offset..offset + out_len].to_vec()
        }
        CorrelationMode::Valid => {
            let out_len = if a_len >= b_len {
                a_len - b_len + 1
            } else {
                b_len - a_len + 1
            };
            if out_len == 0 {
                return Vec::new();
            }
            let offset = (full_len - out_len) / 2;
            full[offset..offset + out_len].to_vec()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  fft_convolve
// ─────────────────────────────────────────────────────────────────────────────

/// Linear convolution of two real signals via FFT with mode selection.
///
/// # Arguments
///
/// * `a`    – First signal.
/// * `b`    – Second signal (impulse response).
/// * `mode` – Output size selection.
///
/// # Errors
///
/// Returns an error if either input is empty or an FFT call fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::correlation::{fft_convolve, ConvolutionMode};
///
/// let x = vec![1.0_f64, 2.0, 3.0];
/// let h = vec![1.0_f64, 1.0];
/// let y = fft_convolve(&x, &h, ConvolutionMode::Full).expect("convolution");
/// assert_eq!(y.len(), x.len() + h.len() - 1);
/// ```
pub fn fft_convolve(a: &[f64], b: &[f64], mode: ConvolutionMode) -> FFTResult<Vec<f64>> {
    if a.is_empty() {
        return Err(FFTError::ValueError("fft_convolve: a is empty".into()));
    }
    if b.is_empty() {
        return Err(FFTError::ValueError("fft_convolve: b is empty".into()));
    }

    let full_len = a.len() + b.len() - 1;
    let fft_len = next_fast_len(full_len, true);

    let a_pad = zero_pad(a, fft_len);
    let b_pad = zero_pad(b, fft_len);

    let a_freq = fft(&a_pad, None)?;
    let b_freq = fft(&b_pad, None)?;

    let product: Vec<Complex64> = a_freq
        .iter()
        .zip(b_freq.iter())
        .map(|(&af, &bf)| af * bf)
        .collect();

    let y_c = ifft(&product, None)?;
    let mut full = real_parts(&y_c);
    full.truncate(full_len);

    let result = trim_output_conv(&full, a.len(), b.len(), full_len, mode);
    Ok(result)
}

/// Trim the full convolution output according to the requested mode.
fn trim_output_conv(
    full: &[f64],
    a_len: usize,
    b_len: usize,
    full_len: usize,
    mode: ConvolutionMode,
) -> Vec<f64> {
    match mode {
        ConvolutionMode::Full => full.to_vec(),
        ConvolutionMode::Same => {
            let out_len = a_len.max(b_len);
            let offset = (full_len - out_len) / 2;
            full[offset..offset + out_len].to_vec()
        }
        ConvolutionMode::Valid => {
            let out_len = if a_len >= b_len {
                a_len - b_len + 1
            } else {
                b_len - a_len + 1
            };
            if out_len == 0 {
                return Vec::new();
            }
            let offset = (full_len - out_len) / 2;
            full[offset..offset + out_len].to_vec()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  fft_correlate2d
// ─────────────────────────────────────────────────────────────────────────────

/// 2-D cross-correlation via FFT.
///
/// Both `a` and `b` are provided as flat row-major arrays with dimensions
/// `(rows × cols)`.  The output is a full correlation grid of size
/// `(2*rows-1) × (2*cols-1)`.
///
/// # Arguments
///
/// * `a`    – First 2-D signal (flat, row-major).
/// * `b`    – Second 2-D signal (flat, row-major).
/// * `rows` – Number of rows in `a` and `b`.
/// * `cols` – Number of columns in `a` and `b`.
///
/// # Errors
///
/// Returns an error if the slice lengths are inconsistent or an FFT fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::correlation::fft_correlate2d;
///
/// let image = vec![1.0_f64; 4 * 4];   // 4×4 all-ones
/// let kernel = vec![1.0_f64; 2 * 2];  // 2×2 all-ones
/// let out = fft_correlate2d(&image, &kernel, 4, 4).expect("2D correlation");
/// assert_eq!(out.len(), (2 * 4 - 1) * (2 * 4 - 1));
/// ```
pub fn fft_correlate2d(
    a: &[f64],
    b: &[f64],
    rows: usize,
    cols: usize,
) -> FFTResult<Vec<f64>> {
    if a.len() != rows * cols {
        return Err(FFTError::DimensionError(format!(
            "fft_correlate2d: a.len()={} != rows*cols={}",
            a.len(),
            rows * cols
        )));
    }
    if b.len() != rows * cols {
        return Err(FFTError::DimensionError(format!(
            "fft_correlate2d: b.len()={} != rows*cols={}",
            b.len(),
            rows * cols
        )));
    }
    if rows == 0 || cols == 0 {
        return Err(FFTError::ValueError(
            "fft_correlate2d: rows and cols must be non-zero".into(),
        ));
    }

    // Full output dimensions
    let out_rows = 2 * rows - 1;
    let out_cols = 2 * cols - 1;

    // Padded FFT dimensions
    let fft_rows = next_fast_len(out_rows, true);
    let fft_cols = next_fast_len(out_cols, true);
    let fft_size = fft_rows * fft_cols;

    // Build zero-padded complex arrays (row-major)
    let mut a_c = vec![Complex64::new(0.0, 0.0); fft_size];
    let mut b_c = vec![Complex64::new(0.0, 0.0); fft_size];

    for r in 0..rows {
        for c in 0..cols {
            let src_idx = r * cols + c;
            let dst_idx = r * fft_cols + c;
            a_c[dst_idx] = Complex64::new(a[src_idx], 0.0);
            b_c[dst_idx] = Complex64::new(b[src_idx], 0.0);
        }
    }

    // 2-D FFT: row-wise then column-wise
    fft2d_rows_then_cols(&mut a_c, fft_rows, fft_cols, false)?;
    fft2d_rows_then_cols(&mut b_c, fft_rows, fft_cols, false)?;

    // Cross-correlation in frequency domain: conj(A) * B
    for (ac, bc) in a_c.iter_mut().zip(b_c.iter()) {
        *ac = ac.conj() * bc;
    }

    // Inverse 2-D FFT
    fft2d_rows_then_cols(&mut a_c, fft_rows, fft_cols, true)?;

    // Extract the full output; the correlation is circular so we need to
    // rearrange negative lags which wrap around.
    // Standard full cross-correlation has lags from -(rows-1) to +(rows-1).
    // In circular FFT output, negative row lags are in fft_rows-rows+1..fft_rows
    // and negative col lags are in fft_cols-cols+1..fft_cols.
    let neg_r = rows - 1;
    let neg_c = cols - 1;
    let mut output = vec![0.0_f64; out_rows * out_cols];

    for or_ in 0..out_rows {
        let src_r = if or_ < rows {
            // Positive row lag (or_ - neg_r >= 0 → store at or_)
            neg_r + or_ - neg_r + neg_r
        } else {
            // Negative row lag
            fft_rows - (out_rows - or_)
        };
        // Simpler indexing: map output row to source row in circular buffer
        let fr = if or_ < rows { or_ } else { fft_rows - out_rows + or_ };
        for oc in 0..out_cols {
            let fc = if oc < cols { oc } else { fft_cols - out_cols + oc };
            output[or_ * out_cols + oc] = a_c[fr * fft_cols + fc].re;
        }
        let _ = src_r; // suppress unused warning
    }

    Ok(output)
}

/// Apply a 2-D FFT (or IFFT) in-place on a flat complex array.
/// Data is stored row-major with dimensions `(fft_rows × fft_cols)`.
fn fft2d_rows_then_cols(
    data: &mut [Complex64],
    fft_rows: usize,
    fft_cols: usize,
    inverse: bool,
) -> FFTResult<()> {
    // Transform each row
    for r in 0..fft_rows {
        let start = r * fft_cols;
        let row: Vec<Complex64> = data[start..start + fft_cols].to_vec();
        let transformed = if inverse {
            ifft(&row, None)?
        } else {
            fft(&row, None)?
        };
        data[start..start + fft_cols].copy_from_slice(&transformed);
    }

    // Transform each column
    let mut col_buf = vec![Complex64::new(0.0, 0.0); fft_rows];
    for c in 0..fft_cols {
        for r in 0..fft_rows {
            col_buf[r] = data[r * fft_cols + c];
        }
        let transformed = if inverse {
            ifft(&col_buf, None)?
        } else {
            fft(&col_buf, None)?
        };
        for r in 0..fft_rows {
            data[r * fft_cols + c] = transformed[r];
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
//  normalized_cross_correlation
// ─────────────────────────────────────────────────────────────────────────────

/// Normalized cross-correlation (NCC) between a template and a signal.
///
/// The output is the standard NCC, where each lag's correlation is divided
/// by the product of the RMS of the template and the RMS of the corresponding
/// overlapping window of the signal.  Values lie in `[-1, +1]`.
///
/// # Arguments
///
/// * `template` – Short reference pattern.
/// * `signal`   – Longer signal to search in.
///
/// # Returns
///
/// Vector of NCC values at each lag (Same mode: length `signal.len()`).
///
/// # Errors
///
/// Returns an error if either input is empty or an FFT call fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::correlation::normalized_cross_correlation;
///
/// let template = vec![1.0_f64, 2.0, 3.0];
/// let signal: Vec<f64> = (0..20).map(|i| (i as f64).sin()).collect();
/// let ncc = normalized_cross_correlation(&template, &signal).expect("ncc");
/// // All NCC values should be in [-1, 1]
/// for v in &ncc {
///     assert!(v.abs() <= 1.0 + 1e-9);
/// }
/// ```
pub fn normalized_cross_correlation(template: &[f64], signal: &[f64]) -> FFTResult<Vec<f64>> {
    if template.is_empty() {
        return Err(FFTError::ValueError(
            "normalized_cross_correlation: template is empty".into(),
        ));
    }
    if signal.is_empty() {
        return Err(FFTError::ValueError(
            "normalized_cross_correlation: signal is empty".into(),
        ));
    }

    let t_len = template.len();
    let s_len = signal.len();

    // Template statistics
    let t_mean = template.iter().sum::<f64>() / t_len as f64;
    let t_centered: Vec<f64> = template.iter().map(|&x| x - t_mean).collect();
    let t_rms = (t_centered.iter().map(|&x| x * x).sum::<f64>() / t_len as f64).sqrt();

    if t_rms < 1e-15 {
        // Template is constant — correlation is undefined; return zeros.
        return Ok(vec![0.0; s_len]);
    }

    // Compute raw cross-correlation (Full mode).
    let full_len = s_len + t_len - 1;
    let fft_len = next_fast_len(full_len, true);

    let t_pad = zero_pad(&t_centered, fft_len);
    let s_pad = zero_pad(signal, fft_len);

    let t_freq = fft(&t_pad, None)?;
    let s_freq = fft(&s_pad, None)?;

    let product: Vec<Complex64> = t_freq
        .iter()
        .zip(s_freq.iter())
        .map(|(&tf, &sf)| tf.conj() * sf)
        .collect();

    let corr_c = ifft(&product, None)?;

    // Rearrange to standard lag ordering (Same mode: length s_len)
    let pos_lags = s_len;
    let neg_lags = t_len - 1;
    let mut full = vec![0.0_f64; full_len];
    for i in 0..pos_lags {
        full[neg_lags + i] = corr_c[i].re;
    }
    for i in 0..neg_lags {
        let src = fft_len - neg_lags + i;
        full[i] = corr_c[src].re;
    }

    // Trim to Same mode (length = s_len)
    let out_len = s_len;
    let offset = (full_len - out_len) / 2;
    let raw_ncc = &full[offset..offset + out_len];

    // Normalize: for each lag, compute the RMS of the overlapping signal window.
    // The sliding window of size t_len over signal, centered at each lag.
    // We use the "valid" normalization: the denominator is t_rms * local_signal_rms.
    let mut ncc = vec![0.0_f64; out_len];

    // Precompute signal mean/variance for sliding windows using cumulative sums.
    let mut cum_sum = vec![0.0_f64; s_len + 1];
    let mut cum_sq = vec![0.0_f64; s_len + 1];
    for i in 0..s_len {
        cum_sum[i + 1] = cum_sum[i] + signal[i];
        cum_sq[i + 1] = cum_sq[i] + signal[i] * signal[i];
    }

    for (lag_idx, (&raw, ncc_out)) in raw_ncc.iter().zip(ncc.iter_mut()).enumerate() {
        // Which signal sample does the centre of the template align with?
        let center_s = lag_idx as i64 + (t_len as i64) / 2 - (out_len as i64 - s_len as i64) / 2;

        // Window start/end in signal (allow partial overlap, pad the rest with 0)
        let win_start = center_s - t_len as i64 / 2;
        let win_end = win_start + t_len as i64;

        let clamped_start = win_start.max(0) as usize;
        let clamped_end = win_end.min(s_len as i64) as usize;

        if clamped_start >= clamped_end {
            *ncc_out = 0.0;
            continue;
        }

        let win_sum = cum_sum[clamped_end] - cum_sum[clamped_start];
        let win_sq = cum_sq[clamped_end] - cum_sq[clamped_start];
        let win_n = (clamped_end - clamped_start) as f64;

        let mean_w = win_sum / win_n;
        let var_w = (win_sq / win_n - mean_w * mean_w).max(0.0);
        let rms_w = var_w.sqrt();

        let denom = t_rms * rms_w * t_len as f64;
        if denom < 1e-15 {
            *ncc_out = 0.0;
        } else {
            *ncc_out = raw / denom;
        }
    }

    Ok(ncc)
}

// ─────────────────────────────────────────────────────────────────────────────
//  phase_correlation
// ─────────────────────────────────────────────────────────────────────────────

/// Phase correlation for estimating the translational shift between two signals.
///
/// Computes the normalized cross-power spectrum and returns the estimated shift
/// (in samples) and the peak confidence value (0..1).
///
/// For 1-D signals the shift estimate is the lag at which the phase-correlation
/// peak occurs. The second return value is the normalized peak magnitude.
///
/// # Arguments
///
/// * `a` – First signal.
/// * `b` – Second signal (same length as `a`).
///
/// # Returns
///
/// `(shift, confidence)` where `shift` is the estimated lag in samples (positive
/// means `b` is delayed relative to `a`) and `confidence` is the normalized peak
/// value of the phase-correlation function.
///
/// # Errors
///
/// Returns an error if the inputs have different lengths, are empty, or an FFT
/// call fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::correlation::phase_correlation;
///
/// // A sine wave and a version delayed by 5 samples.
/// use std::f64::consts::PI;
/// let n = 128;
/// let a: Vec<f64> = (0..n).map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin()).collect();
/// let delay = 5usize;
/// let b: Vec<f64> = (0..n)
///     .map(|i| (2.0 * PI * 5.0 * ((i + n - delay) % n) as f64 / n as f64).sin())
///     .collect();
///
/// let (shift, _conf) = phase_correlation(&a, &b).expect("phase_correlation");
/// // Detected shift should be close to `delay` or `n - delay` (circular).
/// assert!((shift.round() as i64 - delay as i64).abs() <= 2);
/// ```
pub fn phase_correlation(a: &[f64], b: &[f64]) -> FFTResult<(f64, f64)> {
    if a.is_empty() {
        return Err(FFTError::ValueError(
            "phase_correlation: a is empty".into(),
        ));
    }
    if a.len() != b.len() {
        return Err(FFTError::DimensionError(format!(
            "phase_correlation: a.len()={} != b.len()={}",
            a.len(),
            b.len()
        )));
    }

    let n = a.len();

    let a_freq = fft(a, None)?;
    let b_freq = fft(b, None)?;

    // Normalized cross-power spectrum: (A * conj(B)) / |A * conj(B)|
    let cross: Vec<Complex64> = a_freq
        .iter()
        .zip(b_freq.iter())
        .map(|(&af, &bf)| {
            let c = af * bf.conj();
            let mag = c.norm();
            if mag < 1e-30 {
                Complex64::new(0.0, 0.0)
            } else {
                c / mag
            }
        })
        .collect();

    let phase_corr_c = ifft(&cross, None)?;

    // Find peak
    let mut peak_val = f64::NEG_INFINITY;
    let mut peak_idx = 0usize;
    for (i, c) in phase_corr_c.iter().enumerate() {
        if c.re > peak_val {
            peak_val = c.re;
            peak_idx = i;
        }
    }

    // Sub-pixel refinement via parabolic interpolation around the peak
    let shift = if peak_idx > 0 && peak_idx < n - 1 {
        let y_m1 = phase_corr_c[peak_idx - 1].re;
        let y_0 = phase_corr_c[peak_idx].re;
        let y_p1 = phase_corr_c[peak_idx + 1].re;
        let denom = 2.0 * y_0 - y_m1 - y_p1;
        let delta = if denom.abs() > 1e-15 {
            0.5 * (y_p1 - y_m1) / denom
        } else {
            0.0
        };
        peak_idx as f64 + delta
    } else {
        peak_idx as f64
    };

    // Normalize confidence: peak / n (since IFFT is normalized by 1/n)
    let confidence = peak_val / n as f64;

    Ok((shift, confidence))
}

// ─────────────────────────────────────────────────────────────────────────────
//  autocorrelation_fft
// ─────────────────────────────────────────────────────────────────────────────

/// Autocorrelation of a real signal via FFT — O(N log N).
///
/// Returns the **biased** two-sided autocorrelation of length `2*n-1` where the
/// zero-lag coefficient is at index `n-1`.  The biased estimator divides by `n`
/// at every lag.
///
/// # Arguments
///
/// * `signal` – Input real-valued signal.
///
/// # Returns
///
/// Biased two-sided autocorrelation vector of length `2*signal.len()-1`.
///
/// # Errors
///
/// Returns an error if the signal is empty or an FFT call fails.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::correlation::autocorrelation_fft;
///
/// let signal = vec![1.0_f64, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
/// let acf = autocorrelation_fft(&signal).expect("autocorrelation");
/// // Peak at zero lag (centre of the output)
/// let centre = signal.len() - 1;
/// assert!(acf[centre] >= acf[centre + 1]);
/// // Symmetry
/// assert!((acf[centre - 2] - acf[centre + 2]).abs() < 1e-10);
/// ```
pub fn autocorrelation_fft(signal: &[f64]) -> FFTResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(FFTError::ValueError(
            "autocorrelation_fft: signal is empty".into(),
        ));
    }

    let n = signal.len();
    let full_len = 2 * n - 1;
    let fft_len = next_fast_len(full_len, true);

    let padded = zero_pad(signal, fft_len);
    let s_freq = fft(&padded, None)?;

    // Autocorrelation in frequency domain: |S|²
    let power: Vec<Complex64> = s_freq.iter().map(|&sf| Complex64::new(sf.norm_sqr(), 0.0)).collect();

    let acf_c = ifft(&power, None)?;

    // Rearrange: positive lags 0..n-1 are at indices 0..n-1,
    //            negative lags -(n-1)..-1 are at fft_len-n+1..fft_len-1.
    let mut full = vec![0.0_f64; full_len];
    for i in 0..n {
        full[n - 1 + i] = acf_c[i].re / n as f64;
    }
    for i in 0..n - 1 {
        let src = fft_len - (n - 1) + i;
        full[i] = acf_c[src].re / n as f64;
    }

    Ok(full)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Spectral coherence helpers (used by spectral_analysis)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute one segment's FFT and return its complex spectrum.
pub(crate) fn segment_fft(segment: &[f64], window: &[f64], fft_len: usize) -> FFTResult<Vec<Complex64>> {
    if segment.len() != window.len() {
        return Err(FFTError::DimensionError(
            "segment and window must have the same length".into(),
        ));
    }
    let mut padded = vec![0.0_f64; fft_len];
    for (i, (&s, &w)) in segment.iter().zip(window.iter()).enumerate() {
        padded[i] = s * w;
    }
    fft(&padded, None)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_correlate_autocorr_peak() {
        let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.3).sin()).collect();
        let corr = fft_correlate(&x, &x, CorrelationMode::Full).expect("autocorr");
        let centre = x.len() - 1;
        let peak = corr[centre];
        for (k, &v) in corr.iter().enumerate() {
            assert!(
                peak >= v - 1e-9,
                "Peak at centre ({peak}) should be >= corr[{k}]={v}"
            );
        }
    }

    #[test]
    fn test_fft_correlate_modes() {
        let a = vec![1.0_f64; 10];
        let b = vec![1.0_f64; 5];
        let full = fft_correlate(&a, &b, CorrelationMode::Full).expect("full");
        let same = fft_correlate(&a, &b, CorrelationMode::Same).expect("same");
        let valid = fft_correlate(&a, &b, CorrelationMode::Valid).expect("valid");
        assert_eq!(full.len(), a.len() + b.len() - 1);
        assert_eq!(same.len(), a.len().max(b.len()));
        assert_eq!(valid.len(), a.len() - b.len() + 1);
    }

    #[test]
    fn test_fft_convolve_full_length() {
        let a = vec![1.0_f64, 2.0, 3.0];
        let b = vec![1.0_f64, 1.0];
        let y = fft_convolve(&a, &b, ConvolutionMode::Full).expect("convolve");
        assert_eq!(y.len(), a.len() + b.len() - 1);
    }

    #[test]
    fn test_fft_convolve_impulse() {
        let x = vec![1.0_f64, -2.0, 3.5, 0.0, 4.0];
        let h = vec![1.0_f64];
        let y = fft_convolve(&x, &h, ConvolutionMode::Full).expect("impulse");
        assert_eq!(y.len(), x.len());
        for (a, b) in x.iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_convolve_modes_length() {
        let a = vec![1.0_f64; 10];
        let b = vec![1.0_f64; 4];
        let full = fft_convolve(&a, &b, ConvolutionMode::Full).expect("full");
        let same = fft_convolve(&a, &b, ConvolutionMode::Same).expect("same");
        let valid = fft_convolve(&a, &b, ConvolutionMode::Valid).expect("valid");
        assert_eq!(full.len(), a.len() + b.len() - 1);
        assert_eq!(same.len(), a.len().max(b.len()));
        assert_eq!(valid.len(), a.len() - b.len() + 1);
    }

    #[test]
    fn test_fft_correlate2d_size() {
        let rows = 4;
        let cols = 4;
        let a = vec![1.0_f64; rows * cols];
        let b = vec![1.0_f64; rows * cols];
        let out = fft_correlate2d(&a, &b, rows, cols).expect("2D corr");
        let exp_rows = 2 * rows - 1;
        let exp_cols = 2 * cols - 1;
        assert_eq!(out.len(), exp_rows * exp_cols);
    }

    #[test]
    fn test_normalized_cross_correlation_range() {
        let template = vec![1.0_f64, 2.0, 3.0, 2.0, 1.0];
        let signal: Vec<f64> = (0..40).map(|i| (i as f64 * 0.2).sin()).collect();
        let ncc = normalized_cross_correlation(&template, &signal).expect("ncc");
        for &v in &ncc {
            assert!(v.abs() <= 1.0 + 1e-6, "NCC out of range: {v}");
        }
    }

    #[test]
    fn test_autocorrelation_fft_symmetry() {
        let signal = vec![1.0_f64, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let acf = autocorrelation_fft(&signal).expect("acf");
        assert_eq!(acf.len(), 2 * signal.len() - 1);
        let centre = signal.len() - 1;
        // Symmetry: acf[centre-k] ≈ acf[centre+k]
        for k in 1..centre {
            assert!(
                (acf[centre - k] - acf[centre + k]).abs() < 1e-9,
                "asymmetry at lag {k}: {} vs {}",
                acf[centre - k],
                acf[centre + k]
            );
        }
    }

    #[test]
    fn test_autocorrelation_fft_peak_at_zero_lag() {
        let signal: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
        let acf = autocorrelation_fft(&signal).expect("acf");
        let centre = signal.len() - 1;
        let peak = acf[centre];
        for (k, &v) in acf.iter().enumerate() {
            assert!(
                peak >= v - 1e-9,
                "Zero-lag should be max; peak={peak} but acf[{k}]={v}"
            );
        }
    }

    #[test]
    fn test_phase_correlation_detects_shift() {
        use std::f64::consts::PI;
        let n = 64;
        let a: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let delay = 5usize;
        let b: Vec<f64> = (0..n)
            .map(|i| a[(i + n - delay) % n])
            .collect();
        let (shift, _conf) = phase_correlation(&a, &b).expect("phase_correlation");
        // shift should be delay or n - delay (both are correct circular representations)
        let round_shift = shift.round() as usize;
        let detected = round_shift.min(n - round_shift);
        assert!(
            detected == delay || (detected as i64 - delay as i64).abs() <= 1,
            "Expected shift ≈ {delay}, got {shift}"
        );
    }

    #[test]
    fn test_fft_correlate_known_lag() {
        // corr([1, 0, 0], [0, 0, 1]) — peak should be at lag +2
        let a = vec![1.0_f64, 0.0, 0.0];
        let b = vec![0.0_f64, 0.0, 1.0];
        let corr = fft_correlate(&a, &b, CorrelationMode::Full).expect("known");
        // Full length = 5; centre (lag 0) is at index 2; lag +2 is at index 4
        assert!(
            corr[4].abs() > 0.9,
            "Expected peak at lag +2 (index 4): {:?}",
            corr
        );
    }
}
