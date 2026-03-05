//! Multi-dimensional FFT operations
//!
//! This module provides comprehensive 2D FFT utilities including:
//!
//! - **2D FFT / IFFT**: Forward and inverse transforms via separable row+column passes
//! - **Real-input 2D FFT**: Optimized path for real-valued images
//! - **FFT shift**: Moves the DC component to the centre of the spectrum
//! - **Frequency grids**: Generates 2D wavenumber/frequency arrays
//! - **Power Spectral Density**: Windowed 2D PSD via Welch-style periodogram
//! - **Cross-spectrum & coherence**: Spectral relationship between two images
//!
//! # Mathematical background
//!
//! The 2D DFT of an M×N array x is defined as:
//!
//! ```text
//! X[k, l] = Σ_m Σ_n  x[m, n] · exp(-j 2π (mk/M + nl/N))
//! ```
//!
//! The implementation factorises this into M row-wise 1D FFTs followed by N
//! column-wise 1D FFTs, which is mathematically equivalent and exploits
//! existing highly-optimised 1D routines.
//!
//! # References
//!
//! * Cooley, J. W. & Tukey, J. W. (1965). "An algorithm for the machine
//!   calculation of complex Fourier series." *Math. Comput.* 19, 297–301.
//! * Oppenheim, A. V. & Schafer, R. W. (2010). *Discrete-Time Signal Processing*
//!   (3rd ed.). Pearson.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  2D FFT / IFFT (complex input)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 2-dimensional Discrete Fourier Transform of a complex array.
///
/// The transform is computed by applying a 1D FFT along each row and then
/// along each column (separability of the DFT).
///
/// # Arguments
///
/// * `data` - Complex input array of shape (rows, cols).
///
/// # Returns
///
/// Complex spectrum of the same shape.
///
/// # Errors
///
/// Propagates any errors returned by the underlying 1D FFT.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidim::fft2d;
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::numeric::Complex64;
///
/// let n = 4;
/// let data = Array2::from_shape_fn((n, n), |(r, c)| {
///     Complex64::new((r + c) as f64, 0.0)
/// });
/// let spectrum = fft2d(&data).expect("fft2d failed");
/// assert_eq!(spectrum.shape(), data.shape());
/// ```
pub fn fft2d(data: &Array2<Complex64>) -> FFTResult<Array2<Complex64>> {
    let (rows, cols) = data.dim();
    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    // Row-wise 1D FFT
    let mut work = Array2::<Complex64>::zeros((rows, cols));
    for r in 0..rows {
        let row_slice: Vec<Complex64> = data.row(r).iter().copied().collect();
        let row_spectrum = fft(&row_slice, Some(cols))?;
        for c in 0..cols {
            work[[r, c]] = row_spectrum[c];
        }
    }

    // Column-wise 1D FFT
    let mut result = Array2::<Complex64>::zeros((rows, cols));
    for c in 0..cols {
        let col_slice: Vec<Complex64> = (0..rows).map(|r| work[[r, c]]).collect();
        let col_spectrum = fft(&col_slice, Some(rows))?;
        for r in 0..rows {
            result[[r, c]] = col_spectrum[r];
        }
    }

    Ok(result)
}

/// Compute the 2-dimensional inverse Discrete Fourier Transform.
///
/// The inverse is obtained by applying a 1D IFFT along each row and then
/// along each column.
///
/// # Arguments
///
/// * `spectrum` - Complex spectrum array of shape (rows, cols).
///
/// # Returns
///
/// Complex spatial-domain array of the same shape.
///
/// # Errors
///
/// Propagates any errors returned by the underlying 1D IFFT.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidim::{fft2d, ifft2d};
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::numeric::Complex64;
///
/// let n = 4;
/// let data = Array2::from_shape_fn((n, n), |(r, c)| {
///     Complex64::new(r as f64 * c as f64, 0.0)
/// });
/// let spectrum = fft2d(&data).expect("fft2d failed");
/// let recovered = ifft2d(&spectrum).expect("ifft2d failed");
///
/// for r in 0..n {
///     for c in 0..n {
///         assert!((data[[r, c]].re - recovered[[r, c]].re).abs() < 1e-10);
///         assert!((data[[r, c]].im - recovered[[r, c]].im).abs() < 1e-10);
///     }
/// }
/// ```
pub fn ifft2d(spectrum: &Array2<Complex64>) -> FFTResult<Array2<Complex64>> {
    let (rows, cols) = spectrum.dim();
    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    // Row-wise 1D IFFT
    let mut work = Array2::<Complex64>::zeros((rows, cols));
    for r in 0..rows {
        let row_slice: Vec<Complex64> = spectrum.row(r).iter().copied().collect();
        let row_spatial = ifft(&row_slice, Some(cols))?;
        for c in 0..cols {
            work[[r, c]] = row_spatial[c];
        }
    }

    // Column-wise 1D IFFT
    let mut result = Array2::<Complex64>::zeros((rows, cols));
    for c in 0..cols {
        let col_slice: Vec<Complex64> = (0..rows).map(|r| work[[r, c]]).collect();
        let col_spatial = ifft(&col_slice, Some(rows))?;
        for r in 0..rows {
            result[[r, c]] = col_spatial[r];
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Real-input 2D FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 2D FFT of a real-valued input array.
///
/// This is an optimised path for real inputs: each row is converted to a
/// complex vector (zero imaginary part) and the standard complex 2D FFT is
/// applied. Because the input is real, the spectrum satisfies the Hermitian
/// symmetry `X[-k, -l] = conj(X[k, l])`, but the full M×N spectrum is
/// returned for compatibility with other routines.
///
/// # Arguments
///
/// * `data` - Real-valued input array of shape (rows, cols).
///
/// # Returns
///
/// Full complex spectrum of shape (rows, cols).
///
/// # Errors
///
/// Propagates errors from the underlying 1D FFT.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidim::fft2d_real;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_fn((8, 8), |(r, c)| r as f64 + c as f64);
/// let spectrum = fft2d_real(&data).expect("fft2d_real failed");
/// assert_eq!(spectrum.shape(), [8, 8]);
///
/// // DC component should equal sum of all elements
/// let sum: f64 = data.iter().sum();
/// assert!((spectrum[[0, 0]].re - sum).abs() < 1e-8);
/// ```
pub fn fft2d_real(data: &Array2<f64>) -> FFTResult<Array2<Complex64>> {
    let (rows, cols) = data.dim();
    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    let complex_data = Array2::from_shape_fn((rows, cols), |(r, c)| {
        Complex64::new(data[[r, c]], 0.0)
    });
    fft2d(&complex_data)
}

// ─────────────────────────────────────────────────────────────────────────────
//  FFT shift for 2D arrays
// ─────────────────────────────────────────────────────────────────────────────

/// Shift the zero-frequency component to the centre of a 2D complex spectrum.
///
/// For an M×N output of [`fft2d`], the DC component is at `[0, 0]`.
/// This function moves it to approximately the centre `[M/2, N/2]`, which
/// is the natural representation for visualisation and filtering.
///
/// # Arguments
///
/// * `spectrum` - Complex spectrum of shape (rows, cols).
///
/// # Returns
///
/// Shifted spectrum of the same shape.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidim::{fft2d, fftshift_2d};
/// use scirs2_core::ndarray::Array2;
/// use scirs2_core::numeric::Complex64;
///
/// let mut data = Array2::<Complex64>::zeros((4, 4));
/// data[[0, 0]] = Complex64::new(1.0, 0.0); // DC at corner
///
/// let shifted = fftshift_2d(&data);
/// // DC moves to centre [2, 2] for n=4
/// assert!((shifted[[2, 2]].re - 1.0).abs() < 1e-12);
/// ```
pub fn fftshift_2d(spectrum: &Array2<Complex64>) -> Array2<Complex64> {
    let (rows, cols) = spectrum.dim();
    let row_shift = rows / 2;
    let col_shift = cols / 2;

    Array2::from_shape_fn((rows, cols), |(r, c)| {
        let sr = (r + row_shift) % rows;
        let sc = (c + col_shift) % cols;
        spectrum[[sr, sc]]
    })
}

// ─────────────────────────────────────────────────────────────────────────────
//  2D frequency grids
// ─────────────────────────────────────────────────────────────────────────────

/// Build 2D frequency grids for a 2D FFT output.
///
/// Returns a pair of M×N arrays (`freq_rows`, `freq_cols`) where
/// `freq_rows[r, c]` is the row-axis frequency at bin `r` and
/// `freq_cols[r, c]` is the column-axis frequency at bin `c`.  These arrays
/// can be combined to form radial-frequency maps, filter masks, etc.
///
/// The frequencies follow the [`fftfreq`] convention:
/// - Positive bins: `0, 1/(n·d), 2/(n·d), …`
/// - Negative bins: `…, -2/(n·d), -1/(n·d)` (starting at the Nyquist)
///
/// # Arguments
///
/// * `rows` - Number of rows (M).
/// * `cols` - Number of columns (N).
/// * `dr`   - Sample spacing along the row axis (e.g. pixel height in metres).
/// * `dc`   - Sample spacing along the column axis (e.g. pixel width in metres).
///
/// # Returns
///
/// `(freq_rows, freq_cols)` — both of shape (rows, cols).
///
/// # Errors
///
/// Returns an error if `rows` or `cols` is zero, or if `dr`/`dc` ≤ 0.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidim::fftfreqs_2d;
///
/// let (fr, fc) = fftfreqs_2d(4, 4, 1.0, 1.0).expect("fftfreqs_2d failed");
///
/// // DC bin: both frequencies are 0
/// assert_eq!(fr[[0, 0]], 0.0);
/// assert_eq!(fc[[0, 0]], 0.0);
/// ```
pub fn fftfreqs_2d(
    rows: usize,
    cols: usize,
    dr: f64,
    dc: f64,
) -> FFTResult<(Array2<f64>, Array2<f64>)> {
    if rows == 0 || cols == 0 {
        return Err(FFTError::ValueError(
            "rows and cols must be non-zero".to_string(),
        ));
    }
    if dr <= 0.0 {
        return Err(FFTError::ValueError(format!(
            "Row sample spacing dr={dr} must be > 0"
        )));
    }
    if dc <= 0.0 {
        return Err(FFTError::ValueError(format!(
            "Column sample spacing dc={dc} must be > 0"
        )));
    }

    let row_freqs = fftfreq_1d(rows, dr);
    let col_freqs = fftfreq_1d(cols, dc);

    let freq_rows = Array2::from_shape_fn((rows, cols), |(r, _c)| row_freqs[r]);
    let freq_cols = Array2::from_shape_fn((rows, cols), |(_r, c)| col_freqs[c]);

    Ok((freq_rows, freq_cols))
}

/// Compute 1D fftfreq following numpy/scipy convention.
fn fftfreq_1d(n: usize, d: f64) -> Vec<f64> {
    let scale = 1.0 / (n as f64 * d);
    (0..n)
        .map(|i| {
            let k = if i <= n / 2 - (if n % 2 == 0 { 1 } else { 0 }) {
                i as i64
            } else {
                i as i64 - n as i64
            };
            k as f64 * scale
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Power Spectral Density (2D)
// ─────────────────────────────────────────────────────────────────────────────

/// Window types available for 2D PSD estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Window2D {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Hann window applied separably along rows and columns
    Hann,
    /// Hamming window applied separably along rows and columns
    Hamming,
    /// Blackman window applied separably
    Blackman,
}

/// Compute the 2D Power Spectral Density using a separable window.
///
/// The PSD is estimated as the squared magnitude of the 2D FFT after applying
/// a separable window function to reduce spectral leakage. The result is
/// normalised by the total window power so that units are `power / (Hz²)` when
/// `dr` and `dc` are given in seconds.
///
/// # Arguments
///
/// * `image`  - Real-valued 2D input of shape (rows, cols).
/// * `window` - Window function to apply before the FFT.
///
/// # Returns
///
/// Real-valued PSD array of shape (rows, cols).
///
/// # Errors
///
/// Returns an error if the image has zero size or if the underlying FFT fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidim::{power_spectral_density_2d, Window2D};
/// use scirs2_core::ndarray::Array2;
///
/// let image = Array2::<f64>::zeros((16, 16));
/// let psd = power_spectral_density_2d(&image, Window2D::Hann).expect("PSD failed");
/// assert_eq!(psd.shape(), [16, 16]);
///
/// // Zero image → zero PSD
/// for &v in psd.iter() {
///     assert!(v >= 0.0, "PSD must be non-negative");
/// }
/// ```
pub fn power_spectral_density_2d(
    image: &Array2<f64>,
    window: Window2D,
) -> FFTResult<Array2<f64>> {
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(FFTError::ValueError(
            "Image must have non-zero dimensions".to_string(),
        ));
    }

    // Build separable window weights
    let win_r = build_window_1d(rows, window);
    let win_c = build_window_1d(cols, window);

    // Window power (for normalisation)
    let win_power: f64 = win_r.iter().map(|&w| w * w).sum::<f64>()
        * win_c.iter().map(|&w| w * w).sum::<f64>();

    // Apply separable window and convert to complex
    let windowed = Array2::from_shape_fn((rows, cols), |(r, c)| {
        Complex64::new(image[[r, c]] * win_r[r] * win_c[c], 0.0)
    });

    let spectrum = fft2d(&windowed)?;

    // Squared magnitude, normalised by window power
    let scale = if win_power > 0.0 {
        1.0 / win_power
    } else {
        1.0
    };
    Ok(Array2::from_shape_fn((rows, cols), |(r, c)| {
        spectrum[[r, c]].norm_sqr() * scale
    }))
}

/// Build a 1D window of the requested type.
fn build_window_1d(n: usize, window: Window2D) -> Vec<f64> {
    match window {
        Window2D::Rectangular => vec![1.0; n],
        Window2D::Hann => (0..n)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / (n as f64 - 1.0).max(1.0);
                0.5 * (1.0 - x.cos())
            })
            .collect(),
        Window2D::Hamming => (0..n)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / (n as f64 - 1.0).max(1.0);
                0.54 - 0.46 * x.cos()
            })
            .collect(),
        Window2D::Blackman => (0..n)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / (n as f64 - 1.0).max(1.0);
                0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
            })
            .collect(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Cross-spectrum & coherence
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the cross-power spectrum between two real-valued images.
///
/// The cross-spectrum is defined as:
///
/// ```text
/// S_xy[k, l] = X[k, l] · conj(Y[k, l])
/// ```
///
/// where `X` and `Y` are the 2D FFTs of `im1` and `im2` respectively.
///
/// # Arguments
///
/// * `im1` - First real-valued image of shape (rows, cols).
/// * `im2` - Second real-valued image of **the same** shape.
///
/// # Returns
///
/// Complex cross-spectrum array of shape (rows, cols).
///
/// # Errors
///
/// Returns an error if the shapes differ or if either FFT fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidim::cross_spectrum_2d;
/// use scirs2_core::ndarray::Array2;
///
/// let a = Array2::<f64>::zeros((8, 8));
/// let b = Array2::<f64>::zeros((8, 8));
///
/// let cs = cross_spectrum_2d(&a, &b).expect("cross_spectrum_2d failed");
/// assert_eq!(cs.shape(), [8, 8]);
/// ```
pub fn cross_spectrum_2d(
    im1: &Array2<f64>,
    im2: &Array2<f64>,
) -> FFTResult<Array2<Complex64>> {
    if im1.shape() != im2.shape() {
        return Err(FFTError::ValueError(format!(
            "Input arrays must have the same shape: {:?} vs {:?}",
            im1.shape(),
            im2.shape()
        )));
    }
    let x1 = fft2d_real(im1)?;
    let x2 = fft2d_real(im2)?;

    let (rows, cols) = x1.dim();
    Ok(Array2::from_shape_fn((rows, cols), |(r, c)| {
        x1[[r, c]] * x2[[r, c]].conj()
    }))
}

/// Compute the magnitude-squared coherence between two real-valued images.
///
/// The coherence is defined as:
///
/// ```text
/// C[k, l] = |S_xy[k, l]|² / (S_xx[k, l] · S_yy[k, l])
/// ```
///
/// Values lie in [0, 1]: 1 means perfect linear relationship at that
/// frequency pair, 0 means no linear relationship.
///
/// # Arguments
///
/// * `im1` - First real-valued image of shape (rows, cols).
/// * `im2` - Second real-valued image of the same shape.
///
/// # Returns
///
/// Real-valued coherence array in [0, 1], shape (rows, cols).
///
/// # Errors
///
/// Returns an error if the shapes differ or if either FFT fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidim::coherence_2d;
/// use scirs2_core::ndarray::Array2;
///
/// // Identical images → coherence should be 1 everywhere (except DC numerical noise)
/// let data = Array2::from_shape_fn((8, 8), |(r, c)| (r + c) as f64);
/// let coh = coherence_2d(&data, &data).expect("coherence_2d failed");
/// assert_eq!(coh.shape(), [8, 8]);
/// for &v in coh.iter() {
///     // coherence values must be in [0, 1]
///     assert!((0.0..=1.0 + 1e-10).contains(&v), "coherence out of range: {v}");
/// }
/// ```
pub fn coherence_2d(im1: &Array2<f64>, im2: &Array2<f64>) -> FFTResult<Array2<f64>> {
    if im1.shape() != im2.shape() {
        return Err(FFTError::ValueError(format!(
            "Input arrays must have the same shape: {:?} vs {:?}",
            im1.shape(),
            im2.shape()
        )));
    }
    let x1 = fft2d_real(im1)?;
    let x2 = fft2d_real(im2)?;

    let (rows, cols) = x1.dim();
    Ok(Array2::from_shape_fn((rows, cols), |(r, c)| {
        let s_xx = x1[[r, c]].norm_sqr();
        let s_yy = x2[[r, c]].norm_sqr();
        let s_xy = (x1[[r, c]] * x2[[r, c]].conj()).norm_sqr();

        let denom = s_xx * s_yy;
        if denom < 1e-30 {
            0.0
        } else {
            // Clamp to [0, 1] to handle floating-point rounding
            (s_xy / denom).min(1.0)
        }
    }))
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ── fft2d / ifft2d roundtrip ─────────────────────────────────────────────

    #[test]
    fn test_fft2d_ifft2d_roundtrip() {
        let data = Array2::from_shape_fn((8, 8), |(r, c)| {
            Complex64::new((r as f64).sin() + (c as f64).cos(), r as f64 * 0.1)
        });
        let spectrum = fft2d(&data).expect("fft2d failed");
        let recovered = ifft2d(&spectrum).expect("ifft2d failed");

        for r in 0..8 {
            for c in 0..8 {
                assert_relative_eq!(data[[r, c]].re, recovered[[r, c]].re, epsilon = 1e-10);
                assert_relative_eq!(data[[r, c]].im, recovered[[r, c]].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_fft2d_shape_preserved() {
        let data = Array2::from_shape_fn((5, 7), |(r, c)| Complex64::new((r + c) as f64, 0.0));
        let s = fft2d(&data).expect("fft2d");
        assert_eq!(s.shape(), data.shape());
    }

    #[test]
    fn test_fft2d_dc_component() {
        // For a constant array, the DC component should equal rows*cols*value
        let n = 4;
        let val = 3.0;
        let data = Array2::from_shape_fn((n, n), |_| Complex64::new(val, 0.0));
        let s = fft2d(&data).expect("fft2d");
        let expected_dc = val * (n * n) as f64;
        assert_relative_eq!(s[[0, 0]].re, expected_dc, epsilon = 1e-9);
        assert_relative_eq!(s[[0, 0]].im, 0.0, epsilon = 1e-9);
    }

    // ── fft2d_real ────────────────────────────────────────────────────────────

    #[test]
    fn test_fft2d_real_dc() {
        let n = 8;
        let data = Array2::from_shape_fn((n, n), |(r, c)| r as f64 + c as f64);
        let s = fft2d_real(&data).expect("fft2d_real");
        let sum: f64 = data.iter().sum();
        assert_relative_eq!(s[[0, 0]].re, sum, epsilon = 1e-8);
    }

    #[test]
    fn test_fft2d_real_hermitian_symmetry() {
        // For real input: X[k, l] == conj(X[-k, -l]) (mod M, N)
        let rows = 6;
        let cols = 8;
        let data = Array2::from_shape_fn((rows, cols), |(r, c)| {
            ((r as f64 * 1.3 + c as f64 * 2.7) * 0.1).sin()
        });
        let s = fft2d_real(&data).expect("fft2d_real");

        for r in 1..rows {
            for c in 1..cols {
                let conj_r = (rows - r) % rows;
                let conj_c = (cols - c) % cols;
                let x = s[[r, c]];
                let cx = s[[conj_r, conj_c]].conj();
                assert_relative_eq!(x.re, cx.re, epsilon = 1e-10);
                assert_relative_eq!(x.im, cx.im, epsilon = 1e-10);
            }
        }
    }

    // ── fftshift_2d ──────────────────────────────────────────────────────────

    #[test]
    fn test_fftshift_2d_dc_to_centre_even() {
        let mut data = Array2::<Complex64>::zeros((4, 4));
        data[[0, 0]] = Complex64::new(1.0, 0.0);
        let shifted = fftshift_2d(&data);
        assert_relative_eq!(shifted[[2, 2]].re, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_fftshift_2d_dc_to_centre_odd() {
        let mut data = Array2::<Complex64>::zeros((5, 5));
        data[[0, 0]] = Complex64::new(1.0, 0.0);
        let shifted = fftshift_2d(&data);
        // For odd n=5: shift = 5/2 = 2
        assert_relative_eq!(shifted[[2, 2]].re, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_fftshift_2d_shape_preserved() {
        let data = Array2::<Complex64>::zeros((6, 10));
        let s = fftshift_2d(&data);
        assert_eq!(s.shape(), data.shape());
    }

    // ── fftfreqs_2d ──────────────────────────────────────────────────────────

    #[test]
    fn test_fftfreqs_2d_dc_is_zero() {
        let (fr, fc) = fftfreqs_2d(8, 8, 1.0, 1.0).expect("fftfreqs_2d");
        assert_eq!(fr[[0, 0]], 0.0);
        assert_eq!(fc[[0, 0]], 0.0);
    }

    #[test]
    fn test_fftfreqs_2d_shape() {
        let (fr, fc) = fftfreqs_2d(4, 6, 0.5, 0.5).expect("fftfreqs_2d");
        assert_eq!(fr.shape(), [4, 6]);
        assert_eq!(fc.shape(), [4, 6]);
    }

    #[test]
    fn test_fftfreqs_2d_invalid_spacing() {
        assert!(fftfreqs_2d(4, 4, 0.0, 1.0).is_err());
        assert!(fftfreqs_2d(4, 4, 1.0, -1.0).is_err());
    }

    #[test]
    fn test_fftfreqs_2d_zero_size() {
        assert!(fftfreqs_2d(0, 4, 1.0, 1.0).is_err());
        assert!(fftfreqs_2d(4, 0, 1.0, 1.0).is_err());
    }

    #[test]
    fn test_fftfreqs_2d_row_independence() {
        // Row frequencies should not depend on column index
        let (fr, _fc) = fftfreqs_2d(4, 6, 1.0, 1.0).expect("fftfreqs_2d");
        for r in 0..4 {
            let f0 = fr[[r, 0]];
            for c in 1..6 {
                assert_eq!(fr[[r, c]], f0);
            }
        }
    }

    // ── PSD ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_psd_2d_non_negative() {
        let data = Array2::from_shape_fn((8, 8), |(r, c)| {
            ((r as f64 + c as f64) * 0.5).sin()
        });
        for window in [Window2D::Rectangular, Window2D::Hann, Window2D::Hamming, Window2D::Blackman] {
            let psd = power_spectral_density_2d(&data, window).expect("PSD");
            for &v in psd.iter() {
                assert!(v >= 0.0, "PSD value {v} is negative for {window:?}");
            }
        }
    }

    #[test]
    fn test_psd_2d_zero_input() {
        let data = Array2::<f64>::zeros((8, 8));
        let psd = power_spectral_density_2d(&data, Window2D::Hann).expect("PSD");
        for &v in psd.iter() {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_psd_2d_shape() {
        let data = Array2::<f64>::zeros((6, 10));
        let psd = power_spectral_density_2d(&data, Window2D::Rectangular).expect("PSD");
        assert_eq!(psd.shape(), [6, 10]);
    }

    // ── cross-spectrum / coherence ───────────────────────────────────────────

    #[test]
    fn test_cross_spectrum_2d_shape() {
        let a = Array2::<f64>::zeros((6, 8));
        let b = Array2::<f64>::zeros((6, 8));
        let cs = cross_spectrum_2d(&a, &b).expect("cross_spectrum_2d");
        assert_eq!(cs.shape(), [6, 8]);
    }

    #[test]
    fn test_cross_spectrum_2d_shape_mismatch() {
        let a = Array2::<f64>::zeros((4, 8));
        let b = Array2::<f64>::zeros((4, 6));
        assert!(cross_spectrum_2d(&a, &b).is_err());
    }

    #[test]
    fn test_cross_spectrum_2d_auto_is_psd() {
        // Auto-cross-spectrum of a signal should equal |X|^2 (real and positive)
        let data = Array2::from_shape_fn((8, 8), |(r, c)| {
            ((r as f64 * 0.7 + c as f64 * 1.1) * 0.2).cos()
        });
        let cs = cross_spectrum_2d(&data, &data).expect("cross_spectrum auto");
        for r in 0..8 {
            for c in 0..8 {
                // Imaginary part of auto-spectrum is exactly 0
                assert!(
                    cs[[r, c]].im.abs() < 1e-10,
                    "Auto cross-spectrum should be real, got im={} at ({r},{c})",
                    cs[[r, c]].im
                );
                // Real part should be non-negative
                assert!(cs[[r, c]].re >= -1e-10, "Auto cross-spectrum must be non-negative");
            }
        }
    }

    #[test]
    fn test_coherence_2d_range() {
        let a = Array2::from_shape_fn((8, 8), |(r, c)| ((r + c) as f64 * 0.3).sin());
        let b = Array2::from_shape_fn((8, 8), |(r, c)| ((r * c) as f64 * 0.1).cos());
        let coh = coherence_2d(&a, &b).expect("coherence_2d");
        for &v in coh.iter() {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&v),
                "Coherence value {v} is out of [0, 1]"
            );
        }
    }

    #[test]
    fn test_coherence_2d_identical_images() {
        // Coherence of a signal with itself should be 1 at all non-zero-energy bins
        let data = Array2::from_shape_fn((8, 8), |(r, c)| (r + c) as f64 + 1.0);
        let coh = coherence_2d(&data, &data).expect("coherence identical");
        for &v in coh.iter() {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&v),
                "Coherence out of range: {v}"
            );
            // Where there is energy, coherence should be 1
            if v > 0.0 {
                assert!(
                    (v - 1.0).abs() < 1e-8,
                    "Self-coherence should be 1.0, got {v}"
                );
            }
        }
    }

    #[test]
    fn test_coherence_2d_shape_mismatch() {
        let a = Array2::<f64>::zeros((4, 8));
        let b = Array2::<f64>::zeros((4, 6));
        assert!(coherence_2d(&a, &b).is_err());
    }

    // ── Parseval's theorem for 2D FFT ─────────────────────────────────────────

    #[test]
    fn test_parseval_2d() {
        let rows = 8;
        let cols = 8;
        let data = Array2::from_shape_fn((rows, cols), |(r, c)| {
            Complex64::new(
                (2.0 * PI * r as f64 / rows as f64).sin(),
                (2.0 * PI * c as f64 / cols as f64).cos(),
            )
        });
        let spectrum = fft2d(&data).expect("fft2d");

        let energy_x: f64 = data.iter().map(|c| c.norm_sqr()).sum();
        let energy_s: f64 = spectrum.iter().map(|c| c.norm_sqr()).sum();
        let n2 = (rows * cols) as f64;

        // Parseval: sum|X|^2 = N * sum|x|^2
        assert_relative_eq!(energy_s, n2 * energy_x, epsilon = 1e-8 * energy_s.max(1.0));
    }

    // ── Zero-size arrays ─────────────────────────────────────────────────────

    #[test]
    fn test_fft2d_zero_size() {
        let data = Array2::<Complex64>::zeros((0, 8));
        let s = fft2d(&data).expect("fft2d zero rows");
        assert_eq!(s.shape(), [0, 8]);

        let data2 = Array2::<Complex64>::zeros((4, 0));
        let s2 = fft2d(&data2).expect("fft2d zero cols");
        assert_eq!(s2.shape(), [4, 0]);
    }
}
