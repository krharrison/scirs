//! Multidimensional FFT utilities.
//!
//! This module provides 2D and N-dimensional FFT operations including:
//!
//! - [`fft2d`] / [`ifft2d`] — 2-D complex DFT via row-column decomposition
//! - [`fftn`] — N-dimensional FFT for flat arrays with a given shape
//! - [`fftshift`] / [`ifftshift`] — shift DC to/from centre
//! - [`fft_frequencies`] — frequency bin values for a 1-D transform
//! - [`rfft2d`] / [`irfft2d`] — 2-D RFFT exploiting Hermitian symmetry

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use crate::rfft::{irfft, rfft};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ─────────────────────────────────────────────────────────────────────────────
//  2D FFT helpers
// ─────────────────────────────────────────────────────────────────────────────

fn rows(m: &[Vec<Complex64>]) -> usize {
    m.len()
}

fn cols(m: &[Vec<Complex64>]) -> FFTResult<usize> {
    if m.is_empty() {
        return Ok(0);
    }
    let c = m[0].len();
    for row in m {
        if row.len() != c {
            return Err(FFTError::DimensionError(
                "fft2d: all rows must have the same length".to_string(),
            ));
        }
    }
    Ok(c)
}

// ─────────────────────────────────────────────────────────────────────────────
//  2D complex FFT / IFFT
// ─────────────────────────────────────────────────────────────────────────────

/// 2-D forward FFT of a complex matrix (row-major nested Vec).
///
/// Applies a 1-D FFT along each row, then along each column.  This is
/// equivalent to the separable 2-D DFT.
///
/// # Errors
///
/// Returns [`FFTError::DimensionError`] if rows have unequal length.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidimensional::fft2d;
/// use scirs2_core::numeric::Complex64;
///
/// let input = vec![
///     vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
///     vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
/// ];
/// let output = fft2d(&input).expect("valid input");
/// assert_eq!(output.len(), 2);
/// assert_eq!(output[0].len(), 2);
/// ```
pub fn fft2d(x: &[Vec<Complex64>]) -> FFTResult<Vec<Vec<Complex64>>> {
    let nr = rows(x);
    let nc = cols(x)?;
    if nr == 0 || nc == 0 {
        return Ok(Vec::new());
    }

    // Transform along rows (preserve exact row length, no power-of-2 padding)
    let mut tmp: Vec<Vec<Complex64>> = x
        .iter()
        .map(|row| fft(row, Some(nc)))
        .collect::<FFTResult<Vec<_>>>()?;

    // Transform along columns (preserve exact column count = nr)
    for c in 0..nc {
        let col: Vec<Complex64> = tmp.iter().map(|row| row[c]).collect();
        let col_fft = fft(&col, Some(nr))?;
        for (r, val) in col_fft.into_iter().enumerate() {
            tmp[r][c] = val;
        }
    }
    Ok(tmp)
}

/// 2-D inverse FFT of a complex matrix.
///
/// Applies IFFT along columns, then along rows.
///
/// # Errors
///
/// Returns [`FFTError::DimensionError`] if rows have unequal length.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidimensional::{fft2d, ifft2d};
/// use scirs2_core::numeric::Complex64;
///
/// let input = vec![
///     vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
///     vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
/// ];
/// let freq = fft2d(&input).expect("valid input");
/// let recovered = ifft2d(&freq).expect("valid input");
/// ```
pub fn ifft2d(x: &[Vec<Complex64>]) -> FFTResult<Vec<Vec<Complex64>>> {
    let nr = rows(x);
    let nc = cols(x)?;
    if nr == 0 || nc == 0 {
        return Ok(Vec::new());
    }

    // Inverse transform along columns (preserve exact sizes)
    let mut tmp: Vec<Vec<Complex64>> = x.to_vec();
    for c in 0..nc {
        let col: Vec<Complex64> = tmp.iter().map(|row| row[c]).collect();
        let col_ifft = ifft(&col, Some(nr))?;
        for (r, val) in col_ifft.into_iter().enumerate() {
            tmp[r][c] = val;
        }
    }

    // Inverse transform along rows (preserve exact sizes)
    tmp.iter_mut()
        .map(|row| ifft(row, Some(nc)))
        .collect::<FFTResult<Vec<_>>>()
}

// ─────────────────────────────────────────────────────────────────────────────
//  N-dimensional FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Compute an N-dimensional FFT of a flat real array reshaped to `shape`.
///
/// The transform is applied as sequential 1-D FFTs along each axis.  The
/// output is a flat `Vec<Complex64>` in the same row-major order as the input.
///
/// # Arguments
///
/// * `x` — flat real array; `x.len()` must equal `shape.iter().product()`.
/// * `shape` — sizes along each dimension.
///
/// # Errors
///
/// Returns [`FFTError::DimensionError`] if `x.len() != shape.iter().product()`.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidimensional::fftn;
///
/// let x = vec![1.0, 0.0, 0.0, 0.0]; // 2×2 array
/// let out = fftn(&x, &[2, 2]).expect("valid input");
/// assert_eq!(out.len(), 4);
/// ```
pub fn fftn(x: &[f64], shape: &[usize]) -> FFTResult<Vec<Complex64>> {
    let total: usize = shape.iter().product();
    if total == 0 {
        return Ok(Vec::new());
    }
    if x.len() != total {
        return Err(FFTError::DimensionError(format!(
            "fftn: input length {} does not match shape product {}",
            x.len(),
            total
        )));
    }

    // Start with complex representation
    let mut data: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();

    // Apply 1-D FFT along each axis
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }

    for d in 0..ndim {
        let n = shape[d];
        let stride = strides[d];
        let outer: usize = total / (n * stride);

        for i in 0..outer {
            for s in 0..stride {
                // Extract the n-length slice along dimension d
                let base = i * n * stride + s;
                let slice: Vec<Complex64> = (0..n).map(|k| data[base + k * stride]).collect();
                let result = fft(&slice, None)?;
                for (k, val) in result.into_iter().enumerate() {
                    data[base + k * stride] = val;
                }
            }
        }
    }
    Ok(data)
}

// ─────────────────────────────────────────────────────────────────────────────
//  FFT shift / iFFT shift
// ─────────────────────────────────────────────────────────────────────────────

/// Shift the zero-frequency component to the centre of a 2-D spectrum.
///
/// Equivalent to `numpy.fft.fftshift` in 2-D.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidimensional::fftshift;
/// use scirs2_core::numeric::Complex64;
///
/// let x = vec![
///     vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
///     vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
/// ];
/// let shifted = fftshift(&x).expect("valid input");
/// ```
pub fn fftshift(x: &[Vec<Complex64>]) -> FFTResult<Vec<Vec<Complex64>>> {
    let nr = rows(x);
    let nc = cols(x)?;
    if nr == 0 || nc == 0 {
        return Ok(Vec::new());
    }
    let sr = nr / 2;
    let sc = nc / 2;
    let mut out = vec![vec![Complex64::new(0.0, 0.0); nc]; nr];
    for r in 0..nr {
        for c in 0..nc {
            out[(r + sr) % nr][(c + sc) % nc] = x[r][c];
        }
    }
    Ok(out)
}

/// Inverse of [`fftshift`]: move DC back from centre to corner.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidimensional::{fftshift, ifftshift};
/// use scirs2_core::numeric::Complex64;
///
/// let x = vec![
///     vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
///     vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
/// ];
/// let shifted = fftshift(&x).expect("valid input");
/// let recovered = ifftshift(&shifted).expect("valid input");
/// ```
pub fn ifftshift(x: &[Vec<Complex64>]) -> FFTResult<Vec<Vec<Complex64>>> {
    let nr = rows(x);
    let nc = cols(x)?;
    if nr == 0 || nc == 0 {
        return Ok(Vec::new());
    }
    // ifftshift is fftshift with ceil(n/2) shift
    let sr = (nr + 1) / 2;
    let sc = (nc + 1) / 2;
    let mut out = vec![vec![Complex64::new(0.0, 0.0); nc]; nr];
    for r in 0..nr {
        for c in 0..nc {
            out[(r + sr) % nr][(c + sc) % nc] = x[r][c];
        }
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Frequency grid
// ─────────────────────────────────────────────────────────────────────────────

/// Return the DFT frequency bin values for a transform of length `n`.
///
/// The result is `[0, 1, …, n/2−1, −n/2, …, −1] / (n / sample_rate)`.
/// This matches the convention of `numpy.fft.fftfreq`.
///
/// # Arguments
///
/// * `n` — transform length.
/// * `sample_rate` — number of samples per unit time (Hz, default 1.0).
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidimensional::fft_frequencies;
/// let freqs = fft_frequencies(4, 1.0);
/// assert_eq!(freqs, vec![0.0, 0.25, -0.5, -0.25]);
/// ```
pub fn fft_frequencies(n: usize, sample_rate: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    let d = sample_rate / n as f64;
    // Matches numpy.fft.fftfreq: [0, 1, ..., (n-1)/2, -n/2, ..., -1] / n * sample_rate
    // For even n: positive = [0, 1, ..., n/2-1], negative = [-n/2, ..., -1]
    // For odd n: positive = [0, 1, ..., (n-1)/2], negative = [-(n-1)/2, ..., -1]
    let mut freqs = Vec::with_capacity(n);
    let pos_count = (n - 1) / 2 + 1; // = n/2 for even, (n+1)/2 for odd
    for i in 0..pos_count {
        freqs.push(i as f64 * d);
    }
    // negative frequencies
    let neg_start = -(n as isize / 2);
    for k in neg_start..0 {
        freqs.push(k as f64 * d);
    }
    freqs
}

// ─────────────────────────────────────────────────────────────────────────────
//  2D RFFT / IRFFT
// ─────────────────────────────────────────────────────────────────────────────

/// 2-D RFFT: real-input 2-D FFT exploiting Hermitian symmetry.
///
/// Applies RFFT along each row (so the column count of the output is
/// `ncols/2 + 1`), then a full complex FFT along each column.
///
/// # Errors
///
/// Returns [`FFTError::DimensionError`] if rows have unequal length.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidimensional::rfft2d;
///
/// let input = vec![
///     vec![1.0, 2.0, 3.0, 4.0],
///     vec![5.0, 6.0, 7.0, 8.0],
/// ];
/// let out = rfft2d(&input).expect("valid input");
/// assert_eq!(out.len(), 2);
/// assert_eq!(out[0].len(), 3); // 4/2+1
/// ```
pub fn rfft2d(x: &[Vec<f64>]) -> FFTResult<Vec<Vec<Complex64>>> {
    let nr = x.len();
    if nr == 0 {
        return Ok(Vec::new());
    }
    let nc = x[0].len();
    for row in x {
        if row.len() != nc {
            return Err(FFTError::DimensionError(
                "rfft2d: all rows must have the same length".to_string(),
            ));
        }
    }
    if nc == 0 {
        return Ok(Vec::new());
    }

    // RFFT along each row → output is nr × (nc/2+1)
    let nc_out = nc / 2 + 1;
    let mut tmp: Vec<Vec<Complex64>> = x
        .iter()
        .map(|row| rfft(row, None))
        .collect::<FFTResult<Vec<_>>>()?;

    // Full complex FFT along each column
    for c in 0..nc_out {
        let col: Vec<Complex64> = tmp.iter().map(|row| row[c]).collect();
        let col_fft = fft(&col, None)?;
        for (r, val) in col_fft.into_iter().enumerate() {
            tmp[r][c] = val;
        }
    }
    Ok(tmp)
}

/// 2-D inverse RFFT: inverse of [`rfft2d`].
///
/// # Arguments
///
/// * `x` — complex spectrum of shape rows × (original_cols/2+1).
/// * `original_cols` — the number of columns in the original real array.
///
/// # Errors
///
/// Returns [`FFTError::DimensionError`] if rows have unequal length.
///
/// # Examples
///
/// ```
/// use scirs2_fft::multidimensional::{rfft2d, irfft2d};
///
/// let input = vec![
///     vec![1.0, 2.0, 3.0, 4.0],
///     vec![5.0, 6.0, 7.0, 8.0],
/// ];
/// let freq = rfft2d(&input).expect("valid input");
/// let recovered = irfft2d(&freq, 4).expect("valid input");
/// ```
pub fn irfft2d(x: &[Vec<Complex64>], original_cols: usize) -> FFTResult<Vec<Vec<f64>>> {
    let nr = x.len();
    if nr == 0 {
        return Ok(Vec::new());
    }
    let nc_rfft = x[0].len();
    for row in x {
        if row.len() != nc_rfft {
            return Err(FFTError::DimensionError(
                "irfft2d: all rows must have the same length".to_string(),
            ));
        }
    }

    // Inverse complex FFT along each column first
    let mut tmp: Vec<Vec<Complex64>> = x.to_vec();
    for c in 0..nc_rfft {
        let col: Vec<Complex64> = tmp.iter().map(|row| row[c]).collect();
        let col_ifft = ifft(&col, None)?;
        for (r, val) in col_ifft.into_iter().enumerate() {
            tmp[r][c] = val;
        }
    }

    // IRFFT along each row → real output
    tmp.iter()
        .map(|row| irfft(row, Some(original_cols)))
        .collect::<FFTResult<Vec<_>>>()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn test_fft2d_roundtrip() {
        let input = vec![
            vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)],
            vec![c(5.0, 0.0), c(6.0, 0.0), c(7.0, 0.0), c(8.0, 0.0)],
            vec![c(9.0, 0.0), c(10.0, 0.0), c(11.0, 0.0), c(12.0, 0.0)],
            vec![c(13.0, 0.0), c(14.0, 0.0), c(15.0, 0.0), c(16.0, 0.0)],
        ];
        let freq = fft2d(&input).expect("failed to create freq");
        let recovered = ifft2d(&freq).expect("failed to create recovered");
        for (r, row) in input.iter().enumerate() {
            for (col_idx, &orig) in row.iter().enumerate() {
                assert_relative_eq!(orig.re, recovered[r][col_idx].re, epsilon = 1e-10);
                assert_relative_eq!(orig.im, recovered[r][col_idx].im, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_fft2d_dc_term() {
        // For a constant array all(v), X[0,0] = rows * cols * v
        let v = 2.0;
        let nr = 3usize;
        let nc = 4usize;
        let input: Vec<Vec<Complex64>> = (0..nr)
            .map(|_| (0..nc).map(|_| c(v, 0.0)).collect())
            .collect();
        let freq = fft2d(&input).expect("failed to create freq");
        let expected_dc = nr as f64 * nc as f64 * v;
        assert_relative_eq!(freq[0][0].re, expected_dc, epsilon = 1e-10);
    }

    #[test]
    fn test_fftshift_ifftshift_roundtrip() {
        let input: Vec<Vec<Complex64>> = (0..4)
            .map(|r| (0..4).map(|col| c((r * 4 + col) as f64, 0.0)).collect())
            .collect();
        let shifted = fftshift(&input).expect("failed to create shifted");
        let recovered = ifftshift(&shifted).expect("failed to create recovered");
        for (r, row) in input.iter().enumerate() {
            for (col, &orig) in row.iter().enumerate() {
                assert_relative_eq!(orig.re, recovered[r][col].re, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_fft_frequencies_n4() {
        let freqs = fft_frequencies(4, 1.0);
        let expected = [0.0, 0.25, -0.5, -0.25];
        for (a, b) in freqs.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_fft_frequencies_sample_rate() {
        // With sample_rate=100, freqs for n=4 should be 0, 25, -50, -25
        let freqs = fft_frequencies(4, 100.0);
        let expected = [0.0, 25.0, -50.0, -25.0];
        for (a, b) in freqs.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_rfft2d_irfft2d_roundtrip() {
        let input = vec![
            vec![1.0_f64, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![-1.0, 0.0, 1.0, 2.0],
            vec![3.0, -2.0, 1.0, 0.0],
        ];
        let freq = rfft2d(&input).expect("failed to create freq");
        assert_eq!(freq[0].len(), 3); // 4/2+1
        let recovered = irfft2d(&freq, 4).expect("failed to create recovered");
        for (r, row) in input.iter().enumerate() {
            for (col, &orig) in row.iter().enumerate() {
                assert_relative_eq!(orig, recovered[r][col], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_fftn_2d() {
        // fftn on a 2×2 real array should match 2D DFT
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = fftn(&x, &[2, 2]).expect("failed to create out");
        assert_eq!(out.len(), 4);
        // Compare with 2d fft
        let input2d = vec![
            vec![c(1.0, 0.0), c(2.0, 0.0)],
            vec![c(3.0, 0.0), c(4.0, 0.0)],
        ];
        let out2d = fft2d(&input2d).expect("failed to create out2d");
        assert_relative_eq!(out[0].re, out2d[0][0].re, epsilon = 1e-10);
        assert_relative_eq!(out[1].re, out2d[0][1].re, epsilon = 1e-10);
        assert_relative_eq!(out[2].re, out2d[1][0].re, epsilon = 1e-10);
        assert_relative_eq!(out[3].re, out2d[1][1].re, epsilon = 1e-10);
    }

    #[test]
    fn test_fftn_wrong_size() {
        assert!(fftn(&[1.0, 2.0, 3.0], &[2, 2]).is_err());
    }

    #[test]
    fn test_fft2d_empty() {
        let empty: Vec<Vec<Complex64>> = Vec::new();
        let out = fft2d(&empty).expect("failed to create out");
        assert!(out.is_empty());
    }
}
