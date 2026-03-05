//! FFT-based convolution and correlation with full/same/valid output modes.
//!
//! This module provides a comprehensive set of convolution primitives built on
//! top of the FFT, including:
//!
//! * [`fft_convolve_mode`]  — linear convolution with selectable output mode
//! * [`fft_correlate_mode`] — cross-correlation with selectable output mode
//! * [`oa_overlap_add`]     — overlap-add for streaming / long-signal convolution
//! * [`os_overlap_save`]    — overlap-save (alias overlap-discard) method
//! * [`fft_convolve2d_mode`]— 2-D convolution via 2-D FFT with output modes
//!
//! ## Output modes
//!
//! | Mode    | Output length (1-D)              | Description                           |
//! |---------|----------------------------------|---------------------------------------|
//! | `Full`  | `x.len() + h.len() - 1`         | All non-zero lags                     |
//! | `Same`  | `max(x.len(), h.len())`          | Centre-cropped to match the larger    |
//! | `Valid` | `max(x.len(), h.len()) - min(x.len(), h.len()) + 1` | Only fully-overlapping part |
//!
//! ## Overlap-add / Overlap-save
//!
//! Both methods decompose a long signal `x` into short blocks of length `L`,
//! convolve each block with `h`, and recombine — reducing memory requirements
//! and enabling causal / real-time filtering.
//!
//! For a filter of length M the optimal block size is typically chosen as the
//! largest power-of-two ≤ 8 M (a heuristic minimising per-sample FFT cost).
//!
//! # References
//!
//! * Oppenheim, A. V.; Schafer, R. W. *Discrete-Time Signal Processing*,
//!   3rd ed., Prentice-Hall, 2010. §§ 8.6, 8.7.
//! * Smith, J. O. *Mathematics of the Discrete Fourier Transform (DFT)*.
//!   W3K Publishing, 2007.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use crate::helper::next_fast_len;
use scirs2_core::ndarray::{Array2, Axis};
use scirs2_core::numeric::Complex64;

// ─────────────────────────────────────────────────────────────────────────────
//  ConvMode
// ─────────────────────────────────────────────────────────────────────────────

/// Output mode for convolution and correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvMode {
    /// Full linear output of length `x.len() + h.len() - 1`.
    Full,
    /// Output trimmed to `max(x.len(), h.len())` (centred).
    Same,
    /// Only the portion with complete overlap (`|x| - |h| + 1` if `|x| >= |h|`).
    Valid,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Zero-pad a real slice to length `n` (returns owned Vec).
fn zero_pad_real(x: &[f64], n: usize) -> Vec<f64> {
    let mut v = vec![0.0_f64; n];
    let copy_len = x.len().min(n);
    v[..copy_len].copy_from_slice(&x[..copy_len]);
    v
}

/// Lift a real slice to complex.
fn to_complex(x: &[f64]) -> Vec<Complex64> {
    x.iter().map(|&r| Complex64::new(r, 0.0)).collect()
}

/// Extract real parts.
fn real_parts(v: &[Complex64]) -> Vec<f64> {
    v.iter().map(|c| c.re).collect()
}

/// Element-wise complex multiplication.
fn pointwise_mul(a: &[Complex64], b: &[Complex64]) -> Vec<Complex64> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

/// Apply output mode cropping.
fn apply_mode(full: &[f64], lx: usize, lh: usize, mode: ConvMode) -> Vec<f64> {
    let full_len = lx + lh - 1;
    debug_assert_eq!(full.len(), full_len);

    match mode {
        ConvMode::Full => full.to_vec(),
        ConvMode::Same => {
            let out_len = lx.max(lh);
            let start = (full_len - out_len) / 2;
            full[start..start + out_len].to_vec()
        }
        ConvMode::Valid => {
            if lx >= lh {
                let out_len = lx - lh + 1;
                let start = lh - 1;
                full[start..start + out_len].to_vec()
            } else {
                // Swap roles: valid output uses the longer as signal
                let out_len = lh - lx + 1;
                let start = lx - 1;
                full[start..start + out_len].to_vec()
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  fft_convolve_mode
// ─────────────────────────────────────────────────────────────────────────────

/// Linear convolution of two real signals via FFT with selectable output mode.
///
/// # Arguments
///
/// * `x` - First real signal.
/// * `h` - Second real signal (typically the filter impulse response).
/// * `mode` - Output mode: [`ConvMode::Full`], [`ConvMode::Same`], or [`ConvMode::Valid`].
///
/// # Returns
///
/// `Vec<f64>` with length determined by `mode`.
///
/// # Errors
///
/// Returns an error if either input is empty or if the underlying FFT fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::convolution_fft::{fft_convolve_mode, ConvMode};
///
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let h = vec![1.0_f64, -1.0];
///
/// let full = fft_convolve_mode(&x, &h, ConvMode::Full).expect("convolve full");
/// assert_eq!(full.len(), x.len() + h.len() - 1); // 5
///
/// let same = fft_convolve_mode(&x, &h, ConvMode::Same).expect("convolve same");
/// assert_eq!(same.len(), x.len().max(h.len())); // 4
///
/// let valid = fft_convolve_mode(&x, &h, ConvMode::Valid).expect("convolve valid");
/// assert_eq!(valid.len(), x.len() - h.len() + 1); // 3
/// ```
pub fn fft_convolve_mode(x: &[f64], h: &[f64], mode: ConvMode) -> FFTResult<Vec<f64>> {
    if x.is_empty() {
        return Err(FFTError::ValueError("fft_convolve_mode: x is empty".into()));
    }
    if h.is_empty() {
        return Err(FFTError::ValueError("fft_convolve_mode: h is empty".into()));
    }

    let full_len = x.len() + h.len() - 1;
    let fft_len = next_fast_len(full_len, true);

    let x_pad = zero_pad_real(x, fft_len);
    let h_pad = zero_pad_real(h, fft_len);

    let x_fft = fft(&to_complex(&x_pad), None)?;
    let h_fft = fft(&to_complex(&h_pad), None)?;

    let product = pointwise_mul(&x_fft, &h_fft);
    let conv_raw = ifft(&product, None)?;

    let full: Vec<f64> = real_parts(&conv_raw[..full_len]);

    Ok(apply_mode(&full, x.len(), h.len(), mode))
}

// ─────────────────────────────────────────────────────────────────────────────
//  fft_correlate_mode
// ─────────────────────────────────────────────────────────────────────────────

/// Cross-correlation of two real signals via FFT with selectable output mode.
///
/// The cross-correlation is defined as:
///
/// ```text
/// (x ⋆ y)[τ] = sum_n  x[n] · y[n + τ]
/// ```
///
/// This is equivalent to convolution of `x` with the time-reversed `y`.
///
/// # Arguments
///
/// * `x` - Reference signal.
/// * `y` - Template signal.
/// * `mode` - Output mode.
///
/// # Returns
///
/// `Vec<f64>` of cross-correlation values.
///
/// # Errors
///
/// Returns an error if either input is empty.
///
/// # Examples
///
/// ```
/// use scirs2_fft::convolution_fft::{fft_correlate_mode, ConvMode};
///
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let y = vec![1.0_f64, 2.0, 3.0, 4.0];
///
/// // Auto-correlation peak at zero lag
/// let corr = fft_correlate_mode(&x, &y, ConvMode::Full).expect("correlate");
/// let peak_idx = corr.iter()
///     .enumerate()
///     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
///     .map(|(i, _)| i)
///     .expect("max");
/// assert_eq!(peak_idx, x.len() - 1); // zero-lag at centre of full output
/// ```
pub fn fft_correlate_mode(x: &[f64], y: &[f64], mode: ConvMode) -> FFTResult<Vec<f64>> {
    if x.is_empty() {
        return Err(FFTError::ValueError("fft_correlate_mode: x is empty".into()));
    }
    if y.is_empty() {
        return Err(FFTError::ValueError("fft_correlate_mode: y is empty".into()));
    }

    // Cross-correlation = convolution with time-reversed y.
    let y_rev: Vec<f64> = y.iter().copied().rev().collect();
    fft_convolve_mode(x, &y_rev, mode)
}

// ─────────────────────────────────────────────────────────────────────────────
//  oa_overlap_add
// ─────────────────────────────────────────────────────────────────────────────

/// Convolve a (potentially long) signal with a (short) filter using the
/// **overlap-add** (OLA) method.
///
/// The signal is partitioned into non-overlapping blocks of length `block_size`.
/// Each block is zero-padded to `block_size + h.len() - 1` (rounded up to the
/// next efficient FFT length), multiplied in frequency domain with the filter's
/// FFT, inverse-transformed, and added into the output at the correct position.
///
/// # Arguments
///
/// * `x` - Input signal (arbitrary length).
/// * `h` - Filter impulse response (length ≤ `block_size`).
/// * `block_size` - Block length (must be ≥ 1; if 0 an error is returned).
///
/// # Returns
///
/// `Vec<f64>` of length `x.len() + h.len() - 1`.
///
/// # Errors
///
/// Returns an error if any input is empty or if `block_size` is 0.
///
/// # Examples
///
/// ```
/// use scirs2_fft::convolution_fft::oa_overlap_add;
///
/// let x: Vec<f64> = (0..64).map(|k| k as f64).collect();
/// let h = vec![0.25_f64, 0.5, 0.25]; // simple averaging filter
///
/// let out = oa_overlap_add(&x, &h, 16).expect("overlap_add failed");
/// assert_eq!(out.len(), x.len() + h.len() - 1);
/// ```
pub fn oa_overlap_add(x: &[f64], h: &[f64], block_size: usize) -> FFTResult<Vec<f64>> {
    if x.is_empty() {
        return Err(FFTError::ValueError("oa_overlap_add: x is empty".into()));
    }
    if h.is_empty() {
        return Err(FFTError::ValueError("oa_overlap_add: h is empty".into()));
    }
    if block_size == 0 {
        return Err(FFTError::ValueError("oa_overlap_add: block_size must be > 0".into()));
    }

    let m = h.len(); // filter length
    let out_len = x.len() + m - 1;
    let fft_len = next_fast_len(block_size + m - 1, true);

    // Pre-compute H[k] = FFT(h, fft_len).
    let h_pad = zero_pad_real(h, fft_len);
    let h_fft = fft(&to_complex(&h_pad), None)?;

    let mut output = vec![0.0_f64; out_len];

    let mut start = 0usize;
    while start < x.len() {
        let end = (start + block_size).min(x.len());
        let block = &x[start..end];

        // Zero-pad block to fft_len.
        let x_pad = zero_pad_real(block, fft_len);
        let x_fft = fft(&to_complex(&x_pad), None)?;

        // Multiply and inverse-transform.
        let product = pointwise_mul(&x_fft, &h_fft);
        let conv_raw = ifft(&product, None)?;

        // The linear convolution of this block is fft_len samples,
        // but only block.len() + m - 1 are non-zero.
        let block_conv_len = block.len() + m - 1;

        // Add into output at position `start`.
        for i in 0..block_conv_len {
            let out_idx = start + i;
            if out_idx < output.len() {
                output[out_idx] += conv_raw[i].re;
            }
        }

        start += block_size;
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
//  os_overlap_save
// ─────────────────────────────────────────────────────────────────────────────

/// Convolve a (potentially long) signal with a (short) filter using the
/// **overlap-save** (OLS) method (also called overlap-discard).
///
/// The signal is extended with `M − 1` zeros at the start (M = filter length),
/// then processed in overlapping blocks of length `block_size + M - 1`.  The
/// first `M − 1` samples of each block's convolution are discarded; only the
/// last `block_size` samples are valid.
///
/// # Arguments
///
/// * `x` - Input signal.
/// * `h` - Filter impulse response.
/// * `block_size` - Number of **new** input samples per block (must be > 0).
///
/// # Returns
///
/// `Vec<f64>` of length `x.len() + h.len() - 1` (same as overlap-add).
///
/// # Errors
///
/// Returns an error if inputs are empty or block_size is 0.
///
/// # Examples
///
/// ```
/// use scirs2_fft::convolution_fft::os_overlap_save;
///
/// let x: Vec<f64> = (0..64).map(|k| k as f64).collect();
/// let h = vec![0.25_f64, 0.5, 0.25];
///
/// let out = os_overlap_save(&x, &h, 16).expect("overlap_save failed");
/// assert_eq!(out.len(), x.len() + h.len() - 1);
/// ```
pub fn os_overlap_save(x: &[f64], h: &[f64], block_size: usize) -> FFTResult<Vec<f64>> {
    if x.is_empty() {
        return Err(FFTError::ValueError("os_overlap_save: x is empty".into()));
    }
    if h.is_empty() {
        return Err(FFTError::ValueError("os_overlap_save: h is empty".into()));
    }
    if block_size == 0 {
        return Err(FFTError::ValueError("os_overlap_save: block_size must be > 0".into()));
    }

    let m = h.len(); // filter length
    let out_len = x.len() + m - 1;
    // Each FFT block length = block_size + M - 1 (includes M-1 history samples).
    let fft_len = next_fast_len(block_size + m - 1, true);

    // Pre-compute H[k].
    let h_pad = zero_pad_real(h, fft_len);
    let h_fft = fft(&to_complex(&h_pad), None)?;

    // Prepend M-1 zeros to x (initial history).
    let mut padded_x = vec![0.0_f64; m - 1];
    padded_x.extend_from_slice(x);

    let mut output = vec![0.0_f64; out_len];
    let mut out_pos = 0usize;

    // Step through padded_x in steps of block_size.
    let mut block_start = 0usize;
    while block_start + m - 1 < padded_x.len() && out_pos < out_len {
        let block_end = (block_start + fft_len).min(padded_x.len());
        // Take fft_len samples from padded_x (zero-pad at the end if necessary).
        let mut buf = vec![0.0_f64; fft_len];
        let take = (block_end - block_start).min(fft_len);
        buf[..take].copy_from_slice(&padded_x[block_start..block_start + take]);

        let buf_fft = fft(&to_complex(&buf), None)?;
        let product = pointwise_mul(&buf_fft, &h_fft);
        let conv_raw = ifft(&product, None)?;

        // Discard first M-1 samples; save block_size samples starting at index M-1.
        let valid_start = m - 1;
        let valid_count = block_size.min(out_len - out_pos);
        for i in 0..valid_count {
            if valid_start + i < conv_raw.len() {
                output[out_pos + i] = conv_raw[valid_start + i].re;
            }
        }
        out_pos += valid_count;
        block_start += block_size;
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
//  fft_convolve2d_mode
// ─────────────────────────────────────────────────────────────────────────────

/// 2-D linear convolution via 2-D FFT with selectable output mode.
///
/// The 2-D convolution is computed by zero-padding both matrices to a common
/// size (the sum of their shapes minus 1 along each axis, rounded up to the
/// next efficient FFT length), multiplying in the 2-D frequency domain, and
/// inverse-transforming.
///
/// # Arguments
///
/// * `x` - 2-D input signal (Array2<f64>).
/// * `h` - 2-D filter kernel (Array2<f64>).
/// * `mode` - Output mode (`Full`, `Same`, `Valid`).
///
/// # Returns
///
/// An `Array2<f64>` whose shape depends on `mode`.
///
/// # Errors
///
/// Returns an error if either array is empty (has a zero dimension) or if the
/// underlying FFTs fail.
///
/// # Examples
///
/// ```
/// use scirs2_fft::convolution_fft::{fft_convolve2d_mode, ConvMode};
/// use scirs2_core::ndarray::Array2;
///
/// let x = Array2::from_shape_vec((3, 3), vec![1.0_f64; 9]).expect("shape");
/// let h = Array2::from_shape_vec((2, 2), vec![0.25_f64; 4]).expect("shape");
///
/// let out = fft_convolve2d_mode(&x, &h, ConvMode::Full).expect("conv2d full");
/// assert_eq!(out.shape(), &[4, 4]); // 3+2-1 = 4
/// ```
pub fn fft_convolve2d_mode(
    x: &Array2<f64>,
    h: &Array2<f64>,
    mode: ConvMode,
) -> FFTResult<Array2<f64>> {
    let (xr, xc) = x.dim();
    let (hr, hc) = h.dim();

    if xr == 0 || xc == 0 {
        return Err(FFTError::ValueError("fft_convolve2d_mode: x has zero dimension".into()));
    }
    if hr == 0 || hc == 0 {
        return Err(FFTError::ValueError("fft_convolve2d_mode: h has zero dimension".into()));
    }

    let full_rows = xr + hr - 1;
    let full_cols = xc + hc - 1;
    let fft_rows = next_fast_len(full_rows, true);
    let fft_cols = next_fast_len(full_cols, true);
    let fft_size = fft_rows * fft_cols;

    // Flatten, zero-pad and row-major index into fft_rows × fft_cols.
    let mut x_flat = vec![Complex64::new(0.0, 0.0); fft_size];
    let mut h_flat = vec![Complex64::new(0.0, 0.0); fft_size];

    for i in 0..xr {
        for j in 0..xc {
            x_flat[i * fft_cols + j] = Complex64::new(x[[i, j]], 0.0);
        }
    }
    for i in 0..hr {
        for j in 0..hc {
            h_flat[i * fft_cols + j] = Complex64::new(h[[i, j]], 0.0);
        }
    }

    // 2-D FFT = row-wise FFT then column-wise FFT.
    let x_2d_fft = fft2d_complex(&x_flat, fft_rows, fft_cols)?;
    let h_2d_fft = fft2d_complex(&h_flat, fft_rows, fft_cols)?;

    // Element-wise multiply.
    let product = pointwise_mul(&x_2d_fft, &h_2d_fft);

    // 2-D IFFT.
    let result_flat = ifft2d_complex(&product, fft_rows, fft_cols)?;

    // Extract full-convolution region.
    let mut full = Array2::zeros((full_rows, full_cols));
    for i in 0..full_rows {
        for j in 0..full_cols {
            full[[i, j]] = result_flat[i * fft_cols + j].re;
        }
    }

    // Apply mode cropping.
    let out = match mode {
        ConvMode::Full => full,
        ConvMode::Same => {
            let out_r = xr.max(hr);
            let out_c = xc.max(hc);
            let sr = (full_rows - out_r) / 2;
            let sc = (full_cols - out_c) / 2;
            full.slice(scirs2_core::ndarray::s![sr..sr + out_r, sc..sc + out_c]).to_owned()
        }
        ConvMode::Valid => {
            // Valid mode: only fully-overlapping region
            let (out_r, out_c, sr, sc) = if xr >= hr && xc >= hc {
                (xr - hr + 1, xc - hc + 1, hr - 1, hc - 1)
            } else if hr >= xr && hc >= xc {
                (hr - xr + 1, hc - xc + 1, xr - 1, xc - 1)
            } else {
                return Err(FFTError::ValueError(
                    "fft_convolve2d_mode: Valid mode requires one array to fully contain the other in both dimensions".into()
                ));
            };
            full.slice(scirs2_core::ndarray::s![sr..sr + out_r, sc..sc + out_c]).to_owned()
        }
    };

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal 2-D FFT helpers
// ─────────────────────────────────────────────────────────────────────────────

/// 2-D FFT of a flattened row-major complex buffer.
fn fft2d_complex(data: &[Complex64], rows: usize, cols: usize) -> FFTResult<Vec<Complex64>> {
    let mut buf = data.to_vec();

    // Row-wise FFT.
    for i in 0..rows {
        let row = buf[i * cols..(i + 1) * cols].to_vec();
        let row_fft = fft(&row, None)?;
        buf[i * cols..(i + 1) * cols].copy_from_slice(&row_fft);
    }

    // Column-wise FFT.
    for j in 0..cols {
        let col: Vec<Complex64> = (0..rows).map(|i| buf[i * cols + j]).collect();
        let col_fft = fft(&col, None)?;
        for i in 0..rows {
            buf[i * cols + j] = col_fft[i];
        }
    }

    Ok(buf)
}

/// 2-D IFFT of a flattened row-major complex buffer.
fn ifft2d_complex(data: &[Complex64], rows: usize, cols: usize) -> FFTResult<Vec<Complex64>> {
    let mut buf = data.to_vec();

    // Row-wise IFFT.
    for i in 0..rows {
        let row = buf[i * cols..(i + 1) * cols].to_vec();
        let row_ifft = ifft(&row, None)?;
        buf[i * cols..(i + 1) * cols].copy_from_slice(&row_ifft);
    }

    // Column-wise IFFT.
    for j in 0..cols {
        let col: Vec<Complex64> = (0..rows).map(|i| buf[i * cols + j]).collect();
        let col_ifft = ifft(&col, None)?;
        for i in 0..rows {
            buf[i * cols + j] = col_ifft[i];
        }
    }

    Ok(buf)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Brute-force linear convolution for verification.
    fn direct_convolve(x: &[f64], h: &[f64]) -> Vec<f64> {
        let out_len = x.len() + h.len() - 1;
        let mut out = vec![0.0_f64; out_len];
        for (i, &xi) in x.iter().enumerate() {
            for (j, &hj) in h.iter().enumerate() {
                out[i + j] += xi * hj;
            }
        }
        out
    }

    fn assert_f64_slice_eq(a: &[f64], b: &[f64], tol: f64, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            assert_relative_eq!(ai, bi, epsilon = tol, var_name = format!("{label}[{i}]"));
        }
    }

    // ── fft_convolve_mode ────────────────────────────────────────────────────

    #[test]
    fn test_convolve_full_matches_direct() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let h = vec![1.0_f64, 0.5];
        let expected = direct_convolve(&x, &h);
        let got = fft_convolve_mode(&x, &h, ConvMode::Full).expect("convolve");
        assert_f64_slice_eq(&got, &expected, 1e-9, "full");
    }

    #[test]
    fn test_convolve_same_length() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let h = vec![1.0_f64, -1.0, 1.0];
        let out = fft_convolve_mode(&x, &h, ConvMode::Same).expect("same");
        assert_eq!(out.len(), x.len().max(h.len()));
    }

    #[test]
    fn test_convolve_valid_length() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let h = vec![1.0_f64, -1.0, 1.0];
        let out = fft_convolve_mode(&x, &h, ConvMode::Valid).expect("valid");
        assert_eq!(out.len(), x.len() - h.len() + 1); // 5 - 3 + 1 = 3
    }

    #[test]
    fn test_convolve_impulse() {
        // Convolving with a delta should reproduce x.
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let delta = vec![1.0_f64];
        let out = fft_convolve_mode(&x, &delta, ConvMode::Full).expect("impulse");
        assert_f64_slice_eq(&out, &x, 1e-9, "impulse");
    }

    #[test]
    fn test_convolve_empty_error() {
        assert!(fft_convolve_mode(&[], &[1.0_f64], ConvMode::Full).is_err());
        assert!(fft_convolve_mode(&[1.0_f64], &[], ConvMode::Full).is_err());
    }

    // ── fft_correlate_mode ───────────────────────────────────────────────────

    #[test]
    fn test_correlate_autocorrelation_peak() {
        let x = vec![1.0_f64, 2.0, 3.0, 2.0, 1.0];
        let corr = fft_correlate_mode(&x, &x, ConvMode::Full).expect("autocorr");
        // Zero-lag is at index x.len() - 1.
        let zero_lag_idx = x.len() - 1;
        let max_idx = corr
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .expect("max");
        assert_eq!(max_idx, zero_lag_idx);
    }

    #[test]
    fn test_correlate_same_length() {
        let x = vec![1.0_f64, 0.0, -1.0, 0.0, 1.0];
        let y = vec![1.0_f64, 0.0, -1.0];
        let out = fft_correlate_mode(&x, &y, ConvMode::Same).expect("correlate same");
        assert_eq!(out.len(), x.len().max(y.len()));
    }

    // ── oa_overlap_add ───────────────────────────────────────────────────────

    #[test]
    fn test_overlap_add_matches_direct() {
        let x: Vec<f64> = (0..64).map(|k| k as f64 * 0.1).collect();
        let h = vec![0.25_f64, 0.5, 0.25];
        let direct = direct_convolve(&x, &h);
        let ola = oa_overlap_add(&x, &h, 16).expect("ola");
        assert_f64_slice_eq(&ola, &direct, 1e-9, "ola");
    }

    #[test]
    fn test_overlap_add_small_block() {
        let x: Vec<f64> = (0..20).map(|k| k as f64).collect();
        let h = vec![1.0_f64, -2.0, 1.0];
        let direct = direct_convolve(&x, &h);
        let ola = oa_overlap_add(&x, &h, 4).expect("ola small block");
        assert_f64_slice_eq(&ola, &direct, 1e-9, "ola small");
    }

    #[test]
    fn test_overlap_add_empty_error() {
        assert!(oa_overlap_add(&[], &[1.0_f64], 8).is_err());
        assert!(oa_overlap_add(&[1.0_f64], &[], 8).is_err());
        assert!(oa_overlap_add(&[1.0_f64], &[1.0_f64], 0).is_err());
    }

    // ── os_overlap_save ──────────────────────────────────────────────────────

    #[test]
    fn test_overlap_save_matches_direct() {
        let x: Vec<f64> = (0..64).map(|k| (k as f64).sin()).collect();
        let h = vec![0.25_f64, 0.5, 0.25];
        let direct = direct_convolve(&x, &h);
        let ols = os_overlap_save(&x, &h, 16).expect("ols");
        assert_f64_slice_eq(&ols, &direct, 1e-9, "ols");
    }

    #[test]
    fn test_overlap_save_length() {
        let x: Vec<f64> = (0..100).map(|k| k as f64).collect();
        let h = vec![1.0_f64, 1.0, 1.0, 1.0];
        let out = os_overlap_save(&x, &h, 16).expect("ols len");
        assert_eq!(out.len(), x.len() + h.len() - 1);
    }

    #[test]
    fn test_overlap_save_empty_error() {
        assert!(os_overlap_save(&[], &[1.0_f64], 8).is_err());
        assert!(os_overlap_save(&[1.0_f64], &[], 8).is_err());
    }

    // ── fft_convolve2d_mode ──────────────────────────────────────────────────

    #[test]
    fn test_conv2d_full_shape() {
        let x = Array2::from_shape_vec((4, 5), vec![1.0_f64; 20]).expect("x shape");
        let h = Array2::from_shape_vec((2, 3), vec![1.0_f64; 6]).expect("h shape");
        let out = fft_convolve2d_mode(&x, &h, ConvMode::Full).expect("conv2d full");
        assert_eq!(out.shape(), &[5, 7]); // (4+2-1)=5, (5+3-1)=7
    }

    #[test]
    fn test_conv2d_same_shape() {
        let x = Array2::from_shape_vec((4, 5), vec![1.0_f64; 20]).expect("x shape");
        let h = Array2::from_shape_vec((3, 3), vec![1.0_f64; 9]).expect("h shape");
        let out = fft_convolve2d_mode(&x, &h, ConvMode::Same).expect("conv2d same");
        assert_eq!(out.shape(), &[4, 5]); // max(4,3)=4, max(5,3)=5
    }

    #[test]
    fn test_conv2d_identity_kernel() {
        let x = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0_f64],
        )
        .expect("shape");
        // 1×1 identity kernel
        let h = Array2::from_shape_vec((1, 1), vec![1.0_f64]).expect("shape");
        let out = fft_convolve2d_mode(&x, &h, ConvMode::Full).expect("conv2d id");
        assert_eq!(out.shape(), &[3, 3]);
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(out[[i, j]], x[[i, j]], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_conv2d_empty_error() {
        let empty_r = Array2::<f64>::zeros((0, 3));
        let h = Array2::from_shape_vec((2, 2), vec![1.0_f64; 4]).expect("h");
        assert!(fft_convolve2d_mode(&empty_r, &h, ConvMode::Full).is_err());
    }
}
