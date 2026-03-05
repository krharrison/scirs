//! Multi-dimensional FFT utilities and signal processing operations.
//!
//! This module provides comprehensive N-dimensional FFT operations and
//! FFT-based signal processing functions built on top of the core 1D FFT:
//!
//! * [`fft2d_real`]          — 2D FFT of a real-valued flat array.
//! * [`ifft2d_real`]         — 2D IFFT recovering real output.
//! * [`fft3d_real`]          — 3D FFT of a real-valued flat array.
//! * [`fftn_real`]           — N-dimensional FFT.
//! * [`ifftn_real`]          — N-dimensional IFFT.
//! * [`rfft_simple`]         — Real-to-complex FFT (n/2+1 outputs).
//! * [`irfft_simple`]        — Inverse real FFT.
//! * [`cross_power_spectrum`] — Cross-power spectrum for phase correlation.
//! * [`phase_correlation_2d`] — Sub-pixel 2D image registration.
//! * [`fft_correlate_full`]  — FFT-based cross-correlation (full/same/valid).
//! * [`fft_convolve_1d`]     — FFT-based 1D convolution.
//!
//! # Coordinate conventions
//!
//! Arrays are stored in row-major (C) order.  For a 2D array of shape
//! `(rows, cols)`, element `(r, c)` is at index `r * cols + c`.
//!
//! For 3D arrays of shape `(depth, rows, cols)`, element `(d, r, c)` is at
//! index `d * rows * cols + r * cols + c`.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use crate::rfft::{irfft, rfft};
use scirs2_core::numeric::Complex64;

// ─────────────────────────────────────────────────────────────────────────────
//  Correlation mode
// ─────────────────────────────────────────────────────────────────────────────

/// Mode controlling the output length of cross-correlation / convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultidimCorrelationMode {
    /// Full cross-correlation of length `len(a) + len(b) - 1`.
    Full,
    /// Central part matching the length of the longer input.
    Same,
    /// Only the valid (non-zero-padded) part, length `|len(a) - len(b)| + 1`.
    Valid,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p = p.saturating_mul(2);
    }
    p
}

/// Perform a 1D FFT on columns of a row-major complex matrix.
///
/// `data` has shape `(rows, cols)`.  Transforms each column in-place.
/// Since the input may be complex (from a prior row FFT), we split real/imag
/// parts, apply 1D FFT to each, then recombine.
fn fft_columns(data: &mut Vec<Complex64>, rows: usize, cols: usize) -> FFTResult<()> {
    for c in 0..cols {
        // Extract column c: split into real and imaginary parts
        let col_re: Vec<f64> = (0..rows).map(|r| data[r * cols + c].re).collect();
        let col_im: Vec<f64> = (0..rows).map(|r| data[r * cols + c].im).collect();

        // FFT of each part (using linearity of FFT for complex input)
        let fft_re = fft(&col_re, Some(rows))?;
        let fft_im = fft(&col_im, Some(rows))?;

        // Recombine: FFT(re + j·im) = FFT(re) + j·FFT(im)
        for r in 0..rows {
            data[r * cols + c] = Complex64::new(
                fft_re[r].re - fft_im[r].im,
                fft_re[r].im + fft_im[r].re,
            );
        }
    }
    Ok(())
}

/// Perform a 1D IFFT on columns.
fn ifft_columns(data: &mut Vec<Complex64>, rows: usize, cols: usize) -> FFTResult<()> {
    for c in 0..cols {
        let col: Vec<Complex64> = (0..rows).map(|r| data[r * cols + c]).collect();
        let ifft_col = ifft(&col, None)?;
        for r in 0..rows {
            data[r * cols + c] = ifft_col[r];
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
//  2D FFT on flat arrays
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 2D FFT of a real-valued flat array stored in row-major order.
///
/// The transform is computed by:
/// 1. Applying a 1D FFT to each row.
/// 2. Applying a 1D FFT to each column of the result.
///
/// # Arguments
///
/// * `data` – Row-major flat array of shape `(rows, cols)`.
/// * `rows` – Number of rows.
/// * `cols` – Number of columns.
///
/// # Returns
///
/// Complex spectrum as a flat row-major array of length `rows * cols`.
/// Each element is `[re, im]`.
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] if `data.len() != rows * cols`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::fft2d_flat;
///
/// let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
/// let spectrum = fft2d_flat(&data, 4, 4).expect("fft2d_flat");
/// assert_eq!(spectrum.len(), 16);
/// ```
pub fn fft2d_flat(data: &[f64], rows: usize, cols: usize) -> FFTResult<Vec<[f64; 2]>> {
    if data.len() != rows * cols {
        return Err(FFTError::InvalidInput(format!(
            "fft2d_flat: data.len()={} != rows*cols={}",
            data.len(),
            rows * cols
        )));
    }

    // Step 1: FFT each row
    let mut row_ffts: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); rows * cols];
    for r in 0..rows {
        let row: Vec<f64> = data[r * cols..(r + 1) * cols].to_vec();
        let fft_row = fft(&row, Some(cols))?;
        for c in 0..cols {
            row_ffts[r * cols + c] = fft_row[c];
        }
    }

    // Step 2: FFT each column
    fft_columns(&mut row_ffts, rows, cols)?;

    // Convert to output format
    Ok(row_ffts.iter().map(|c| [c.re, c.im]).collect())
}

/// Compute the 2D IFFT from a complex spectrum and return the real part.
///
/// # Arguments
///
/// * `spectrum` – Row-major flat complex spectrum `[re, im]` of shape `(rows, cols)`.
/// * `rows`     – Number of rows.
/// * `cols`     – Number of columns.
///
/// # Returns
///
/// Real-valued output (imaginary parts are discarded after IFFT).
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] for size mismatches.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::{fft2d_flat, ifft2d_flat};
///
/// let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
/// let spectrum = fft2d_flat(&data, 4, 4).expect("fft2d_flat");
/// let recovered = ifft2d_flat(&spectrum, 4, 4).expect("ifft2d_flat");
/// assert_eq!(recovered.len(), 16);
/// for (a, b) in data.iter().zip(recovered.iter()) {
///     assert!((a - b).abs() < 1e-9, "Reconstruction error");
/// }
/// ```
pub fn ifft2d_flat(spectrum: &[[f64; 2]], rows: usize, cols: usize) -> FFTResult<Vec<f64>> {
    if spectrum.len() != rows * cols {
        return Err(FFTError::InvalidInput(format!(
            "ifft2d_flat: spectrum.len()={} != rows*cols={}",
            spectrum.len(),
            rows * cols
        )));
    }

    // Convert to Complex64
    let mut data: Vec<Complex64> = spectrum
        .iter()
        .map(|&[re, im]| Complex64::new(re, im))
        .collect();

    // IFFT each column first
    ifft_columns(&mut data, rows, cols)?;

    // IFFT each row (preserve exact size)
    let mut result = vec![0.0f64; rows * cols];
    for r in 0..rows {
        let row: Vec<Complex64> = data[r * cols..(r + 1) * cols].to_vec();
        let ifft_row = ifft(&row, Some(cols))?;
        for c in 0..cols {
            result[r * cols + c] = ifft_row[c].re;
        }
    }

    Ok(result)
}

/// 2D FFT on a Vec-of-Vec input (rows × cols).
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::fft2d_vec;
///
/// let data: Vec<Vec<f64>> = vec![
///     vec![1.0, 2.0, 3.0, 4.0],
///     vec![5.0, 6.0, 7.0, 8.0],
/// ];
/// let spectrum = fft2d_vec(&data).expect("fft2d_vec");
/// assert_eq!(spectrum.len(), 2);
/// assert_eq!(spectrum[0].len(), 4);
/// ```
pub fn fft2d_vec(data: &[Vec<f64>]) -> FFTResult<Vec<Vec<[f64; 2]>>> {
    let rows = data.len();
    if rows == 0 {
        return Ok(vec![]);
    }
    let cols = data[0].len();
    for row in data {
        if row.len() != cols {
            return Err(FFTError::InvalidInput(
                "fft2d_vec: rows have different lengths".into(),
            ));
        }
    }

    let flat: Vec<f64> = data.iter().flat_map(|r| r.iter().copied()).collect();
    let flat_out = fft2d_flat(&flat, rows, cols)?;

    let mut out = vec![vec![[0.0f64; 2]; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r][c] = flat_out[r * cols + c];
        }
    }
    Ok(out)
}

/// 2D IFFT from Vec-of-Vec complex spectrum.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::{fft2d_vec, ifft2d_vec};
///
/// let data: Vec<Vec<f64>> = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// let spectrum = fft2d_vec(&data).expect("fft2d_vec");
/// let recovered = ifft2d_vec(&spectrum).expect("ifft2d_vec");
/// for (ri, row) in recovered.iter().enumerate() {
///     for (ci, &v) in row.iter().enumerate() {
///         assert!((v - data[ri][ci]).abs() < 1e-9);
///     }
/// }
/// ```
pub fn ifft2d_vec(spectrum: &[Vec<[f64; 2]>]) -> FFTResult<Vec<Vec<f64>>> {
    let rows = spectrum.len();
    if rows == 0 {
        return Ok(vec![]);
    }
    let cols = spectrum[0].len();

    let flat: Vec<[f64; 2]> = spectrum.iter().flat_map(|r| r.iter().copied()).collect();
    let flat_out = ifft2d_flat(&flat, rows, cols)?;

    let mut out = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r][c] = flat_out[r * cols + c];
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  3D FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 3D FFT of a flat real-valued array.
///
/// Shape: `(depth, rows, cols)`, stored in row-major order.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::fft3d_flat;
///
/// let shape = (2, 3, 4);
/// let data: Vec<f64> = (0..(2*3*4)).map(|i| i as f64).collect();
/// let spectrum = fft3d_flat(&data, shape.0, shape.1, shape.2).expect("fft3d_flat");
/// assert_eq!(spectrum.len(), 2 * 3 * 4);
/// ```
pub fn fft3d_flat(
    data: &[f64],
    depth: usize,
    rows: usize,
    cols: usize,
) -> FFTResult<Vec<[f64; 2]>> {
    let total = depth * rows * cols;
    if data.len() != total {
        return Err(FFTError::InvalidInput(format!(
            "fft3d_flat: data.len()={} != {}*{}*{}={}",
            data.len(),
            depth,
            rows,
            cols,
            total
        )));
    }

    // Step 1: 2D FFT on each depth slice
    let slice_size = rows * cols;
    let mut complex_data: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); total];

    for d in 0..depth {
        let slice = &data[d * slice_size..(d + 1) * slice_size];
        let slice_fft = fft2d_flat(slice, rows, cols)?;
        for i in 0..slice_size {
            complex_data[d * slice_size + i] = Complex64::new(slice_fft[i][0], slice_fft[i][1]);
        }
    }

    // Step 2: FFT along the depth dimension for each (row, col) position
    for r in 0..rows {
        for c in 0..cols {
            let depth_slice: Vec<f64> = (0..depth)
                .map(|d| complex_data[d * slice_size + r * cols + c].re)
                .collect();
            let depth_im: Vec<f64> = (0..depth)
                .map(|d| complex_data[d * slice_size + r * cols + c].im)
                .collect();

            let fft_re = fft(&depth_slice, Some(depth))?;
            let fft_im = fft(&depth_im, Some(depth))?;

            for d in 0..depth {
                complex_data[d * slice_size + r * cols + c] = Complex64::new(
                    fft_re[d].re - fft_im[d].im,
                    fft_re[d].im + fft_im[d].re,
                );
            }
        }
    }

    Ok(complex_data.iter().map(|c| [c.re, c.im]).collect())
}

/// 3D FFT on a Vec-of-Vec-of-Vec input `(depth × rows × cols)`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::fft3d_vec;
///
/// let data: Vec<Vec<Vec<f64>>> = vec![
///     vec![vec![1.0, 2.0], vec![3.0, 4.0]],
///     vec![vec![5.0, 6.0], vec![7.0, 8.0]],
/// ];
/// let spectrum = fft3d_vec(&data).expect("fft3d_vec");
/// assert_eq!(spectrum.len(), 2);
/// assert_eq!(spectrum[0].len(), 2);
/// assert_eq!(spectrum[0][0].len(), 2);
/// ```
pub fn fft3d_vec(data: &[Vec<Vec<f64>>]) -> FFTResult<Vec<Vec<Vec<[f64; 2]>>>> {
    let depth = data.len();
    if depth == 0 {
        return Ok(vec![]);
    }
    let rows = data[0].len();
    let cols = if rows > 0 { data[0][0].len() } else { 0 };

    let flat: Vec<f64> = data
        .iter()
        .flat_map(|d| d.iter().flat_map(|r| r.iter().copied()))
        .collect();
    let flat_out = fft3d_flat(&flat, depth, rows, cols)?;
    let slice_size = rows * cols;

    let mut out = vec![vec![vec![[0.0f64; 2]; cols]; rows]; depth];
    for d in 0..depth {
        for r in 0..rows {
            for c in 0..cols {
                out[d][r][c] = flat_out[d * slice_size + r * cols + c];
            }
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  N-dimensional FFT (general)
// ─────────────────────────────────────────────────────────────────────────────

/// N-dimensional FFT of a real-valued flat array.
///
/// The transform is computed axis by axis (row-column decomposition).
/// `shape` gives the size along each dimension in row-major order.
///
/// # Arguments
///
/// * `data`  – Real-valued flat array; length must equal the product of `shape`.
/// * `shape` – Size of each dimension.
///
/// # Returns
///
/// Complex spectrum as `[re, im]` pairs, same flat layout as input.
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] for size mismatches.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::fftn_real;
///
/// let shape = vec![4, 4];
/// let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
/// let spectrum = fftn_real(&data, &shape).expect("fftn_real");
/// assert_eq!(spectrum.len(), 16);
/// ```
pub fn fftn_real(data: &[f64], shape: &[usize]) -> FFTResult<Vec<[f64; 2]>> {
    let total: usize = shape.iter().product();
    if total == 0 {
        return Ok(vec![]);
    }
    if data.len() != total {
        return Err(FFTError::InvalidInput(format!(
            "fftn_real: data.len()={} != product(shape)={}",
            data.len(),
            total
        )));
    }
    if shape.is_empty() {
        return Err(FFTError::InvalidInput("fftn_real: shape must not be empty".into()));
    }

    // Convert to complex
    let mut buf: Vec<Complex64> = data.iter().map(|&x| Complex64::new(x, 0.0)).collect();

    // Transform each axis
    ndim_fft_inplace(&mut buf, shape, false)?;

    Ok(buf.iter().map(|c| [c.re, c.im]).collect())
}

/// N-dimensional IFFT returning the real part.
///
/// # Arguments
///
/// * `spectrum` – Complex spectrum as `[re, im]` pairs.
/// * `shape`    – Output shape (same as the forward FFT shape).
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::{fftn_real, ifftn_real};
///
/// let shape = vec![4, 4];
/// let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
/// let spectrum = fftn_real(&data, &shape).expect("fftn_real");
/// let recovered = ifftn_real(&spectrum, &shape).expect("ifftn_real");
/// for (a, b) in data.iter().zip(recovered.iter()) {
///     assert!((a - b).abs() < 1e-9);
/// }
/// ```
pub fn ifftn_real(spectrum: &[[f64; 2]], shape: &[usize]) -> FFTResult<Vec<f64>> {
    let total: usize = shape.iter().product();
    if total == 0 {
        return Ok(vec![]);
    }
    if spectrum.len() != total {
        return Err(FFTError::InvalidInput(format!(
            "ifftn_real: spectrum.len()={} != product(shape)={}",
            spectrum.len(),
            total
        )));
    }

    let mut buf: Vec<Complex64> = spectrum
        .iter()
        .map(|&[re, im]| Complex64::new(re, im))
        .collect();

    ndim_fft_inplace(&mut buf, shape, true)?;

    Ok(buf.iter().map(|c| c.re).collect())
}

/// Apply N-dimensional FFT (or IFFT) in-place along every axis.
fn ndim_fft_inplace(
    buf: &mut Vec<Complex64>,
    shape: &[usize],
    inverse: bool,
) -> FFTResult<()> {
    let ndim = shape.len();
    let total: usize = shape.iter().product();

    // Process each axis
    let mut stride = total;
    for axis in 0..ndim {
        let n = shape[axis];
        stride /= n;
        let n_transforms = total / n;

        // For each slice along this axis, extract, transform, and put back
        for t in 0..n_transforms {
            // Compute the base index for this transform
            // The transform runs along axis `axis` with stride `stride`
            let outer = t / stride;
            let inner = t % stride;
            let base = outer * stride * n + inner;

            // Extract the 1D slice
            let slice: Vec<Complex64> = (0..n).map(|k| buf[base + k * stride]).collect();

            // Transform
            let transformed = if inverse {
                ifft(&slice, None)?
            } else {
                // For forward: we use real FFT trick — split re/im
                let re: Vec<f64> = slice.iter().map(|c| c.re).collect();
                let im: Vec<f64> = slice.iter().map(|c| c.im).collect();
                let fft_re = fft(&re, Some(n))?;
                let fft_im = fft(&im, Some(n))?;
                (0..n)
                    .map(|k| Complex64::new(fft_re[k].re - fft_im[k].im, fft_re[k].im + fft_im[k].re))
                    .collect()
            };

            // Put back
            for k in 0..n {
                buf[base + k * stride] = transformed[k];
            }
        }
        // stride remains updated for next iteration
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Real FFT (simplified interface)
// ─────────────────────────────────────────────────────────────────────────────

/// Real-to-complex FFT — returns only the positive frequency half-spectrum.
///
/// For a real-valued signal of length N, the output has `N/2 + 1` complex
/// values (frequencies from 0 to the Nyquist frequency).
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::rfft_simple;
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let spectrum = rfft_simple(&signal).expect("rfft_simple");
/// assert_eq!(spectrum.len(), 8 / 2 + 1);
/// ```
pub fn rfft_simple(signal: &[f64]) -> FFTResult<Vec<[f64; 2]>> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::InvalidInput("rfft_simple: empty signal".into()));
    }
    let spectrum = rfft(signal, None)?;
    Ok(spectrum.iter().map(|c| [c.re, c.im]).collect())
}

/// Inverse real FFT — reconstructs the real-valued time-domain signal.
///
/// # Arguments
///
/// * `spectrum` – Half-spectrum of length `n/2 + 1` returned by [`rfft_simple`].
/// * `n`        – Length of the original real-valued signal.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::{rfft_simple, irfft_simple};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let spectrum = rfft_simple(&signal).expect("rfft_simple");
/// let recovered = irfft_simple(&spectrum, signal.len()).expect("irfft_simple");
/// for (a, b) in signal.iter().zip(recovered.iter()) {
///     assert!((a - b).abs() < 1e-9);
/// }
/// ```
pub fn irfft_simple(spectrum: &[[f64; 2]], n: usize) -> FFTResult<Vec<f64>> {
    let complex: Vec<Complex64> = spectrum
        .iter()
        .map(|&[re, im]| Complex64::new(re, im))
        .collect();
    irfft(&complex, Some(n))
}

// ─────────────────────────────────────────────────────────────────────────────
//  FFT-based correlation and convolution
// ─────────────────────────────────────────────────────────────────────────────

/// FFT-based cross-correlation of two real signals.
///
/// Cross-correlation: `(a ★ b)[τ] = Σ_n a[n] · b[n + τ]`
///
/// # Arguments
///
/// * `a`    – First real signal.
/// * `b`    – Second real signal.
/// * `mode` – Output length mode.
///
/// # Returns
///
/// Real-valued cross-correlation vector.
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] for empty inputs.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::{fft_correlate_full, MultidimCorrelationMode};
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![0.0, 1.0, 0.5];
/// let c = fft_correlate_full(&a, &b, MultidimCorrelationMode::Full).expect("fft_correlate_full");
/// assert_eq!(c.len(), a.len() + b.len() - 1);
/// ```
pub fn fft_correlate_full(
    a: &[f64],
    b: &[f64],
    mode: MultidimCorrelationMode,
) -> FFTResult<Vec<f64>> {
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 {
        return Err(FFTError::InvalidInput(
            "fft_correlate_full: inputs must not be empty".into(),
        ));
    }

    let full_len = na + nb - 1;
    let fft_len = next_pow2(full_len);

    // FFT of zero-padded inputs
    let fa = fft(a, Some(fft_len))?;
    let fb = fft(b, Some(fft_len))?;

    // Cross-spectrum: A* · B  (complex conjugate of A times B)
    let cross: Vec<Complex64> = fa
        .iter()
        .zip(fb.iter())
        .map(|(a_c, b_c)| a_c.conj() * b_c)
        .collect();

    // IFFT
    let corr_complex = ifft(&cross, None)?;

    // Extract real part and rearrange to match conventions
    // The output of ifft gives the correlation with zero lag at index 0,
    // positive lags at 1..na-1, and negative lags at fft_len-nb+1..fft_len-1
    let mut full = vec![0.0f64; full_len];
    let neg_lags = nb - 1;
    // Positive lags (0 to na-1): indices 0..na
    // Negative lags (-(nb-1) to -1): indices fft_len-nb+1..fft_len
    for i in 0..neg_lags {
        full[i] = corr_complex[fft_len - neg_lags + i].re;
    }
    for i in 0..na {
        let idx = i + neg_lags;
        if idx < full_len {
            full[idx] = corr_complex[i].re;
        }
    }

    match mode {
        MultidimCorrelationMode::Full => Ok(full),
        MultidimCorrelationMode::Same => {
            let out_len = na.max(nb);
            let start = (full_len - out_len) / 2;
            Ok(full[start..start + out_len].to_vec())
        }
        MultidimCorrelationMode::Valid => {
            let short = na.min(nb);
            let out_len = full_len - 2 * (short - 1);
            let start = short - 1;
            Ok(full[start..start + out_len].to_vec())
        }
    }
}

/// FFT-based 1D convolution of two real signals.
///
/// Computes the linear convolution `(a * b)[n] = Σ_k a[k] · b[n-k]`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::fft_convolve_1d;
///
/// let a = vec![1.0, 1.0, 1.0];
/// let b = vec![1.0, 1.0];
/// let c = fft_convolve_1d(&a, &b).expect("fft_convolve_1d");
/// assert_eq!(c.len(), a.len() + b.len() - 1);
/// ```
pub fn fft_convolve_1d(a: &[f64], b: &[f64]) -> FFTResult<Vec<f64>> {
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 {
        return Err(FFTError::InvalidInput(
            "fft_convolve_1d: inputs must not be empty".into(),
        ));
    }

    let full_len = na + nb - 1;
    let fft_len = next_pow2(full_len);

    let fa = fft(a, Some(fft_len))?;
    let fb = fft(b, Some(fft_len))?;

    let product: Vec<Complex64> = fa.iter().zip(fb.iter()).map(|(ac, bc)| ac * bc).collect();
    let conv_complex = ifft(&product, None)?;

    Ok(conv_complex[..full_len].iter().map(|c| c.re).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Cross-power spectrum and phase correlation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the normalised cross-power spectrum of two 1D signals.
///
/// The cross-power spectrum is:
///   G(f) = A*(f) · B(f) / |A*(f) · B(f)|
///
/// It is used in phase correlation for sub-sample shift estimation.
///
/// # Returns
///
/// Complex cross-power spectrum as `[re, im]` pairs of length `n`.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::cross_power_spectrum;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![0.0, 1.0, 2.0, 3.0];
/// let cps = cross_power_spectrum(&a, &b).expect("cross_power_spectrum");
/// assert_eq!(cps.len(), a.len());
/// ```
pub fn cross_power_spectrum(a: &[f64], b: &[f64]) -> FFTResult<Vec<[f64; 2]>> {
    if a.is_empty() || b.is_empty() {
        return Err(FFTError::InvalidInput(
            "cross_power_spectrum: inputs must not be empty".into(),
        ));
    }
    if a.len() != b.len() {
        return Err(FFTError::InvalidInput(format!(
            "cross_power_spectrum: length mismatch {} vs {}",
            a.len(),
            b.len()
        )));
    }

    let n = a.len();
    let fa = fft(a, Some(n))?;
    let fb = fft(b, Some(n))?;

    let result: Vec<[f64; 2]> = fa
        .iter()
        .zip(fb.iter())
        .map(|(ac, bc)| {
            let cross = ac.conj() * bc;
            let mag = cross.norm();
            if mag < f64::EPSILON {
                [0.0, 0.0]
            } else {
                [cross.re / mag, cross.im / mag]
            }
        })
        .collect();

    Ok(result)
}

/// Phase correlation for sub-pixel 2D image registration.
///
/// Estimates the translation `(dy, dx)` between two images `img1` and `img2`
/// such that `img2 ≈ img1` shifted by `(dy, dx)` pixels.
///
/// # Algorithm
///
/// 1. Compute 2D FFT of both images.
/// 2. Form the normalised cross-power spectrum.
/// 3. Inverse FFT to get the phase correlation surface.
/// 4. Find the peak of the surface (integer shift).
/// 5. Refine to sub-pixel accuracy by fitting a parabola around the peak.
///
/// # Arguments
///
/// * `img1` – Reference image as `Vec<Vec<f64>>` (rows × cols).
/// * `img2` – Template image of the same size.
///
/// # Returns
///
/// `(dy, dx)` estimated sub-pixel shift in pixels.
///
/// # Errors
///
/// Returns [`FFTError::InvalidInput`] for empty or mismatched images.
///
/// # Examples
///
/// ```rust
/// use scirs2_fft::multidim_utils::phase_correlation_2d;
///
/// // Create a simple shifted signal
/// let rows = 16usize;
/// let cols = 16usize;
/// let mut img1 = vec![vec![0.0f64; cols]; rows];
/// let mut img2 = vec![vec![0.0f64; cols]; rows];
/// img1[4][4] = 1.0;
/// img2[5][6] = 1.0; // shifted by (1, 2)
///
/// let (dy, dx) = phase_correlation_2d(&img1, &img2).expect("phase_correlation_2d");
/// // Should be close to (1.0, 2.0)
/// assert!((dy - 1.0).abs() < 1.0, "dy={dy}");
/// assert!((dx - 2.0).abs() < 1.0, "dx={dx}");
/// ```
pub fn phase_correlation_2d(
    img1: &[Vec<f64>],
    img2: &[Vec<f64>],
) -> FFTResult<(f64, f64)> {
    let rows = img1.len();
    if rows == 0 {
        return Err(FFTError::InvalidInput(
            "phase_correlation_2d: empty image".into(),
        ));
    }
    let cols = img1[0].len();
    if cols == 0 {
        return Err(FFTError::InvalidInput(
            "phase_correlation_2d: empty image cols".into(),
        ));
    }
    if img2.len() != rows || img2.iter().any(|r| r.len() != cols) {
        return Err(FFTError::InvalidInput(
            "phase_correlation_2d: image size mismatch".into(),
        ));
    }

    // Flatten images
    let flat1: Vec<f64> = img1.iter().flat_map(|r| r.iter().copied()).collect();
    let flat2: Vec<f64> = img2.iter().flat_map(|r| r.iter().copied()).collect();

    // 2D FFT
    let f1 = fft2d_flat(&flat1, rows, cols)?;
    let f2 = fft2d_flat(&flat2, rows, cols)?;

    // Normalised cross-power spectrum
    let cross: Vec<Complex64> = f1
        .iter()
        .zip(f2.iter())
        .map(|(&[r1, i1], &[r2, i2])| {
            let a = Complex64::new(r1, i1);
            let b = Complex64::new(r2, i2);
            let c = a.conj() * b;
            let mag = c.norm();
            if mag < f64::EPSILON { Complex64::new(0.0, 0.0) } else { c / mag }
        })
        .collect();

    // 2D IFFT of cross-power spectrum
    let cross_flat: Vec<[f64; 2]> = cross.iter().map(|c| [c.re, c.im]).collect();
    let surface = ifft2d_flat(&cross_flat, rows, cols)?;

    // Find peak
    let (peak_idx, _) = surface
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .ok_or_else(|| FFTError::ComputationError("phase_correlation_2d: empty surface".into()))?;

    let peak_row = peak_idx / cols;
    let peak_col = peak_idx % cols;

    // Convert to signed shift (wrap around Nyquist)
    let dy_int = if peak_row > rows / 2 {
        peak_row as f64 - rows as f64
    } else {
        peak_row as f64
    };
    let dx_int = if peak_col > cols / 2 {
        peak_col as f64 - cols as f64
    } else {
        peak_col as f64
    };

    // Sub-pixel refinement via parabolic interpolation
    let dy = if peak_row > 0 && peak_row < rows - 1 {
        let ym1 = surface[(peak_row - 1) * cols + peak_col];
        let y0 = surface[peak_row * cols + peak_col];
        let yp1 = surface[(peak_row + 1) * cols + peak_col];
        let denom = 2.0 * (2.0 * y0 - ym1 - yp1);
        if denom.abs() > f64::EPSILON {
            dy_int + (yp1 - ym1) / denom
        } else {
            dy_int
        }
    } else {
        dy_int
    };

    let dx = if peak_col > 0 && peak_col < cols - 1 {
        let xm1 = surface[peak_row * cols + peak_col - 1];
        let x0 = surface[peak_row * cols + peak_col];
        let xp1 = surface[peak_row * cols + peak_col + 1];
        let denom = 2.0 * (2.0 * x0 - xm1 - xp1);
        if denom.abs() > f64::EPSILON {
            dx_int + (xp1 - xm1) / denom
        } else {
            dx_int
        }
    } else {
        dx_int
    };

    Ok((dy, dx))
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_fft2d_flat_roundtrip() {
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let spectrum = fft2d_flat(&data, 4, 4).expect("fft2d_flat");
        assert_eq!(spectrum.len(), 16);
        let recovered = ifft2d_flat(&spectrum, 4, 4).expect("ifft2d_flat");
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "Roundtrip error: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_fft2d_vec_roundtrip() {
        let data: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let spectrum = fft2d_vec(&data).expect("fft2d_vec");
        let recovered = ifft2d_vec(&spectrum).expect("ifft2d_vec");
        for (ri, row) in recovered.iter().enumerate() {
            for (ci, &v) in row.iter().enumerate() {
                assert!((v - data[ri][ci]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_fft2d_size_mismatch_error() {
        let data = vec![1.0f64; 10];
        let err = fft2d_flat(&data, 4, 4).unwrap_err();
        assert!(matches!(err, FFTError::InvalidInput(_)));
    }

    #[test]
    fn test_fft3d_flat_shape() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let spectrum = fft3d_flat(&data, 2, 3, 4).expect("fft3d_flat");
        assert_eq!(spectrum.len(), 24);
    }

    #[test]
    fn test_fft3d_vec_shape() {
        let data: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            vec![vec![5.0, 6.0], vec![7.0, 8.0]],
        ];
        let spectrum = fft3d_vec(&data).expect("fft3d_vec");
        assert_eq!(spectrum.len(), 2);
        assert_eq!(spectrum[0].len(), 2);
        assert_eq!(spectrum[0][0].len(), 2);
    }

    #[test]
    fn test_fftn_real_roundtrip() {
        let shape = vec![4, 4];
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let spectrum = fftn_real(&data, &shape).expect("fftn_real");
        let recovered = ifftn_real(&spectrum, &shape).expect("ifftn_real");
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-8, "fftn roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_fftn_real_3d_roundtrip() {
        let shape = vec![2, 3, 4];
        let data: Vec<f64> = (0..24).map(|i| i as f64 * 0.5).collect();
        let spectrum = fftn_real(&data, &shape).expect("fftn_real 3d");
        let recovered = ifftn_real(&spectrum, &shape).expect("ifftn_real 3d");
        for (a, b) in data.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-8, "3d roundtrip error");
        }
    }

    #[test]
    fn test_rfft_simple_length() {
        let signal: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let spectrum = rfft_simple(&signal).expect("rfft_simple");
        assert_eq!(spectrum.len(), 8 / 2 + 1);
    }

    #[test]
    fn test_irfft_simple_roundtrip() {
        let signal: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let spectrum = rfft_simple(&signal).expect("rfft_simple");
        let recovered = irfft_simple(&spectrum, signal.len()).expect("irfft_simple");
        assert_eq!(recovered.len(), signal.len());
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-9, "irfft_simple roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_fft_correlate_full_length() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, -1.0];
        let c = fft_correlate_full(&a, &b, MultidimCorrelationMode::Full).expect("correlate full");
        assert_eq!(c.len(), a.len() + b.len() - 1);
    }

    #[test]
    fn test_fft_correlate_same_length() {
        let a = vec![1.0f64; 8];
        let b = vec![1.0f64; 4];
        let c = fft_correlate_full(&a, &b, MultidimCorrelationMode::Same).expect("correlate same");
        assert_eq!(c.len(), a.len().max(b.len()));
    }

    #[test]
    fn test_fft_convolve_1d_length() {
        let a = vec![1.0, 1.0, 1.0];
        let b = vec![1.0, 1.0];
        let c = fft_convolve_1d(&a, &b).expect("fft_convolve_1d");
        assert_eq!(c.len(), a.len() + b.len() - 1);
    }

    #[test]
    fn test_fft_convolve_1d_values() {
        // [1, 1] * [1, 1] = [1, 2, 1]
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        let c = fft_convolve_1d(&a, &b).expect("fft_convolve_1d");
        assert_eq!(c.len(), 3);
        assert!((c[0] - 1.0).abs() < 1e-9);
        assert!((c[1] - 2.0).abs() < 1e-9);
        assert!((c[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cross_power_spectrum_length() {
        let a: Vec<f64> = (0..8).map(|i| (i as f64).sin()).collect();
        let b: Vec<f64> = (0..8).map(|i| (i as f64 + 0.5).sin()).collect();
        let cps = cross_power_spectrum(&a, &b).expect("cross_power_spectrum");
        assert_eq!(cps.len(), 8);
    }

    #[test]
    fn test_cross_power_spectrum_unit_magnitude() {
        let a: Vec<f64> = (0..16).map(|i| (2.0 * PI * 0.1 * i as f64).sin()).collect();
        let b: Vec<f64> = (0..16).map(|i| (2.0 * PI * 0.1 * i as f64 + 0.3).sin()).collect();
        let cps = cross_power_spectrum(&a, &b).expect("cross_power_spectrum");
        for [re, im] in &cps {
            let mag = (re * re + im * im).sqrt();
            // Non-zero entries should have unit magnitude
            if mag > 0.1 {
                assert!((mag - 1.0).abs() < 0.01, "Expected unit magnitude, got {mag}");
            }
        }
    }

    #[test]
    fn test_phase_correlation_2d_shift() {
        let rows = 16usize;
        let cols = 16usize;
        let mut img1 = vec![vec![0.0f64; cols]; rows];
        let mut img2 = vec![vec![0.0f64; cols]; rows];
        img1[4][4] = 1.0;
        img2[5][6] = 1.0; // shifted by (1, 2)

        let (dy, dx) = phase_correlation_2d(&img1, &img2).expect("phase_correlation_2d");
        assert!(
            (dy - 1.0).abs() < 1.5,
            "Expected dy≈1.0, got {dy:.3}"
        );
        assert!(
            (dx - 2.0).abs() < 1.5,
            "Expected dx≈2.0, got {dx:.3}"
        );
    }

    #[test]
    fn test_phase_correlation_2d_zero_shift() {
        let rows = 8usize;
        let cols = 8usize;
        let img: Vec<Vec<f64>> = (0..rows)
            .map(|r| (0..cols).map(|c| (r + c) as f64).collect())
            .collect();
        let (dy, dx) = phase_correlation_2d(&img, &img).expect("phase_correlation_2d");
        assert!(dy.abs() < 0.1, "Zero shift: expected dy≈0, got {dy}");
        assert!(dx.abs() < 0.1, "Zero shift: expected dx≈0, got {dx}");
    }

    #[test]
    fn test_fft_correlate_empty_error() {
        let err = fft_correlate_full(&[], &[1.0], MultidimCorrelationMode::Full).unwrap_err();
        assert!(matches!(err, FFTError::InvalidInput(_)));
    }
}
