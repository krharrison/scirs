//! Optimized Real-valued FFT using the split-radix packing trick.
//!
//! This module exploits the conjugate-symmetry of the DFT of a real signal to
//! reduce the work roughly by half compared to a general complex FFT.  The key
//! insight is:
//!
//! > Given two real signals `a[n]` and `b[n]` of length `N`, we can pack them
//! > into a single complex signal `z[n] = a[n] + i·b[n]`, compute **one**
//! > N-point complex FFT, and then unpack the individual spectra using only
//! > `O(N)` additional arithmetic.
//!
//! For a single real signal `x[n]` of length `N` we split it into its even and
//! odd sub-sequences, which effectively doubles the packing factor:
//!
//! ```text
//! a[k] = x[2k]           (even-indexed samples)
//! b[k] = x[2k+1]         (odd-indexed samples)
//! z[k] = a[k] + i·b[k]
//! Z[k] = FFT(z)[k]
//! A[k] = (Z[k] + conj(Z[N/2-k])) / 2
//! B[k] = (Z[k] - conj(Z[N/2-k])) / (2i)
//! X[k] = A[k] + W_N^{-k} · B[k],   W_N = exp(-2πi/N)
//! ```
//!
//! The result is a length `N/2 + 1` complex spectrum (the positive-frequency
//! half, including DC and Nyquist), consistent with NumPy/SciPy `rfft`.
//!
//! # 2-D Real FFT
//!
//! `rfft2` / `irfft2` apply the 1-D real/inverse-real FFT along each axis of a
//! 2-D array, re-using the 1-D routines with appropriate transposing.
//!
//! # Normalization
//!
//! `rfft_norm` / `irfft_norm` follow the "ortho" convention where both the
//! forward and inverse transforms are scaled by `1/sqrt(N)`, yielding a
//! unitary transform pair.
//!
//! # References
//!
//! * Sorensen, H. V.; Jones, D. L.; Heideman, M. T.; Burrus, C. S.
//!   "Real-valued fast Fourier transform algorithms."
//!   *IEEE Trans. ASSP* 35(6) (1987), pp. 849–863.
//! * Numerical Recipes, 3rd ed., §12.3.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Complex64, NumCast, Zero};
use std::f64::consts::PI;
use std::fmt::Debug;

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a real slice to a `Vec<f64>`, returning an error on failure.
fn cast_to_f64<T: NumCast + Copy + Debug>(x: &[T]) -> FFTResult<Vec<f64>> {
    x.iter()
        .map(|&v| {
            NumCast::from(v).ok_or_else(|| {
                FFTError::ValueError(format!("Cannot cast {v:?} to f64"))
            })
        })
        .collect()
}

/// Build twiddle factor `exp(-2πi·k/n)` for k = 0..half+1.
fn twiddle_forward(n: usize, half: usize) -> Vec<Complex64> {
    (0..=half)
        .map(|k| {
            let phase = -2.0 * PI * k as f64 / n as f64;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect()
}

/// Build twiddle factor `exp(+2πi·k/n)` for k = 0..half+1 (inverse).
fn twiddle_inverse(n: usize, half: usize) -> Vec<Complex64> {
    (0..=half)
        .map(|k| {
            let phase = 2.0 * PI * k as f64 / n as f64;
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  rfft — real-to-complex FFT (packed split-radix algorithm)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 1-D FFT of a real-valued signal.
///
/// Uses the split-radix packing trick: packs the even and odd sub-sequences of
/// the (zero-padded) input into one N/2-point complex FFT, then unpacks with
/// `O(N)` twiddle arithmetic.  This roughly halves the work compared to a
/// straight complex FFT.
///
/// # Arguments
///
/// * `x` - Real input signal.
/// * `n` - Number of points for the transform (zero-pads or truncates `x`
///   if `Some`).  Defaults to `x.len()`.
///
/// # Returns
///
/// A `Vec<Complex64>` of length `n/2 + 1` (the non-redundant, non-negative
/// frequency bins from DC to Nyquist).
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `x` is empty.
///
/// # Examples
///
/// ```
/// use scirs2_fft::real_fft::rfft_optimized;
///
/// let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let spectrum = rfft_optimized(&signal, None).expect("rfft_optimized failed");
/// assert_eq!(spectrum.len(), signal.len() / 2 + 1); // 3 bins
/// ```
pub fn rfft_optimized<T>(x: &[T], n: Option<usize>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    if x.is_empty() {
        return Err(FFTError::ValueError("rfft_optimized: input is empty".into()));
    }

    let input_f64 = cast_to_f64(x)?;
    let n_val = n.unwrap_or(input_f64.len());

    if n_val == 0 {
        return Err(FFTError::ValueError("rfft_optimized: n must be > 0".into()));
    }

    // Zero-pad or truncate to n_val.
    let mut padded = vec![0.0_f64; n_val];
    let copy_len = input_f64.len().min(n_val);
    padded[..copy_len].copy_from_slice(&input_f64[..copy_len]);

    // Trivial case: length 1
    if n_val == 1 {
        return Ok(vec![Complex64::new(padded[0], 0.0)]);
    }

    // Pack even/odd samples into a complex signal of length n_val/2 (rounded down).
    // Odd-length: pad by 0 so the half-length is (n_val+1)/2 - 1 = n_val/2.
    let half = n_val / 2;

    // Build complex z[k] = x[2k] + i·x[2k+1].
    let z: Vec<Complex64> = (0..half)
        .map(|k| {
            let re = padded[2 * k];
            let im = if 2 * k + 1 < n_val { padded[2 * k + 1] } else { 0.0 };
            Complex64::new(re, im)
        })
        .collect();

    // One N/2-point complex FFT.
    let z_fft = fft(&z, None)?;

    // Number of output bins (DC + positive freqs + Nyquist).
    let n_out = n_val / 2 + 1;
    let mut result = Vec::with_capacity(n_out);

    // Pre-compute twiddle factors W_N^{-k} = exp(-2πi·k/N).
    let twiddles = twiddle_forward(n_val, half);

    // Unpack:  X[k] = A[k] + W^{-k} · B[k]
    //   A[k] = (Z[k] + conj(Z[half - k])) / 2
    //   B[k] = (Z[k] - conj(Z[half - k])) / (2i) = -i·(Z[k] - conj(Z[half-k])) / 2
    for k in 0..n_out {
        let zk = if k < half { z_fft[k] } else { z_fft[0] }; // Z[0] repeated for Nyquist
        let zm = if k == 0 {
            z_fft[0]
        } else if k < half {
            z_fft[half - k]
        } else {
            z_fft[0] // k == half (Nyquist)
        };

        let zk_c = Complex64::new(zm.re, -zm.im); // conj(Z[half-k])

        let a_k = (zk + zk_c) * 0.5;
        // B[k] = (zk - zk_c) / (2i) = -i * (zk - zk_c) / 2
        let diff = zk - zk_c;
        let b_k = Complex64::new(diff.im * 0.5, -diff.re * 0.5);

        result.push(a_k + twiddles[k] * b_k);
    }

    // Special correction for the Nyquist bin when n_val is even.
    if n_val % 2 == 0 && n_val > 1 {
        // The formula above gives X[half] = A[0] - i·B[0] with wrong twiddle;
        // use the symmetry X[N/2] = sum x[n]·(-1)^n directly from the packed FFT.
        // Actually, the standard result: X[N/2] = A[0] - B[0] (twiddle = -1 at k=N/2).
        // Our twiddle vector already holds exp(-2πi·(N/2)/N) = exp(-πi) = -1, correct.
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  irfft — complex-to-real inverse FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the inverse real FFT (complex spectrum → real signal).
///
/// Reconstructs the full conjugate-symmetric spectrum from the half-spectrum
/// produced by [`rfft_optimized`], then applies the inverse complex FFT and
/// returns the real part.
///
/// # Arguments
///
/// * `x` - Half-spectrum of length `N/2 + 1`.
/// * `n` - Length of the output signal.  If `None`, uses `2 * (x.len() - 1)`.
///
/// # Returns
///
/// A `Vec<f64>` of length `n`.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `x` is empty.
///
/// # Examples
///
/// ```
/// use scirs2_fft::real_fft::{rfft_optimized, irfft_optimized};
///
/// let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let spectrum = rfft_optimized(&signal, None).expect("rfft failed");
/// let recovered = irfft_optimized(&spectrum, Some(signal.len())).expect("irfft failed");
/// for (a, b) in signal.iter().zip(recovered.iter()) {
///     assert!((a - b).abs() < 1e-9);
/// }
/// ```
pub fn irfft_optimized(x: &[Complex64], n: Option<usize>) -> FFTResult<Vec<f64>> {
    if x.is_empty() {
        return Err(FFTError::ValueError("irfft_optimized: input is empty".into()));
    }

    let n_out = n.unwrap_or(2 * (x.len() - 1));

    if n_out == 0 {
        return Ok(Vec::new());
    }

    // Reconstruct the full N-point complex spectrum from the half-spectrum.
    let mut full = vec![Complex64::zero(); n_out];
    let n_half = x.len();

    // Copy positive-frequency bins (0 .. n_half).
    for (k, &val) in x.iter().enumerate() {
        if k < n_out {
            full[k] = val;
        }
    }

    // Fill negative-frequency bins using conjugate symmetry.
    // X[N-k] = conj(X[k]) for k = 1 .. N/2-1
    let n_conj = if n_out % 2 == 0 { n_half - 1 } else { n_half };
    for k in 1..n_conj {
        let neg_k = n_out - k;
        if neg_k < n_out && k < x.len() {
            full[neg_k] = Complex64::new(x[k].re, -x[k].im);
        }
    }

    // Inverse FFT.
    let complex_result = ifft(&full, None)?;

    // Return real parts, scaled by 1/N (ifft already scales by 1/N).
    Ok(complex_result.iter().map(|c| c.re).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
//  rfft2 / irfft2 — 2-D real FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 2-D FFT of a real-valued matrix.
///
/// Applies `rfft_optimized` along rows, then a full complex FFT along columns.
///
/// # Arguments
///
/// * `x` - 2-D real input as an `Array2<f64>`.
/// * `s` - Optional output shape `(nrows, ncols)`.  Defaults to the shape of `x`.
///
/// # Returns
///
/// An `Array2<Complex64>` of shape `(nrows, ncols/2 + 1)`.
///
/// # Errors
///
/// Returns an error if any row or column FFT fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::real_fft::rfft2_optimized;
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec((4, 4), vec![
///     1.0_f64, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
///     9.0, 10.0, 11.0, 12.0,
///     13.0, 14.0, 15.0, 16.0,
/// ]).expect("shape error");
///
/// let spectrum = rfft2_optimized(&data, None).expect("rfft2 failed");
/// assert_eq!(spectrum.shape(), &[4, 3]); // ncols/2 + 1 = 3
/// ```
pub fn rfft2_optimized(
    x: &Array2<f64>,
    s: Option<(usize, usize)>,
) -> FFTResult<Array2<Complex64>> {
    let (nrows, ncols) = x.dim();
    let (out_rows, out_cols) = s.unwrap_or((nrows, ncols));

    if out_rows == 0 || out_cols == 0 {
        return Err(FFTError::ValueError("rfft2_optimized: output shape must be > 0".into()));
    }

    let col_out = out_cols / 2 + 1;

    // Step 1: rfft along each row.
    let mut row_spectra: Vec<Vec<Complex64>> = Vec::with_capacity(out_rows);
    for i in 0..out_rows {
        let row: Vec<f64> = (0..out_cols)
            .map(|j| if i < nrows && j < ncols { x[[i, j]] } else { 0.0 })
            .collect();
        let spec = rfft_optimized(&row, Some(out_cols))?;
        row_spectra.push(spec);
    }

    // Step 2: complex FFT along each column of the row-spectra matrix.
    // row_spectra is (out_rows × col_out).
    let mut result_data = vec![Complex64::zero(); out_rows * col_out];

    for j in 0..col_out {
        let col: Vec<Complex64> = row_spectra.iter().map(|r| r[j]).collect();
        let col_fft = fft(&col, Some(out_rows))?;
        for i in 0..out_rows {
            result_data[i * col_out + j] = col_fft[i];
        }
    }

    Array2::from_shape_vec((out_rows, col_out), result_data)
        .map_err(|e| FFTError::DimensionError(format!("rfft2_optimized shape error: {e}")))
}

/// Compute the 2-D inverse real FFT.
///
/// # Arguments
///
/// * `x` - Half-spectrum of shape `(nrows, ncols_half)` where `ncols_half = N/2 + 1`.
/// * `s` - Full output shape `(out_rows, out_cols)`.
///
/// # Returns
///
/// An `Array2<f64>` of shape `out_s`.
///
/// # Errors
///
/// Returns an error if any inverse FFT fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::real_fft::{rfft2_optimized, irfft2_optimized};
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec((4, 4), (0..16).map(|x| x as f64).collect())
///     .expect("shape error");
/// let spectrum = rfft2_optimized(&data, None).expect("rfft2 failed");
/// let recovered = irfft2_optimized(&spectrum, Some((4, 4))).expect("irfft2 failed");
/// for i in 0..4 {
///     for j in 0..4 {
///         assert!((data[[i, j]] - recovered[[i, j]]).abs() < 1e-9,
///             "mismatch at [{i},{j}]: {} vs {}", data[[i,j]], recovered[[i,j]]);
///     }
/// }
/// ```
pub fn irfft2_optimized(
    x: &Array2<Complex64>,
    s: Option<(usize, usize)>,
) -> FFTResult<Array2<f64>> {
    let (nrows_in, ncols_half) = x.dim();
    let out_cols = s.map(|(_, c)| c).unwrap_or(2 * (ncols_half - 1));
    let out_rows = s.map(|(r, _)| r).unwrap_or(nrows_in);

    if out_rows == 0 || out_cols == 0 {
        return Err(FFTError::ValueError("irfft2_optimized: output shape must be > 0".into()));
    }

    // Step 1: inverse complex FFT along each column first.
    let mut col_ifft_mat: Vec<Vec<Complex64>> = vec![vec![Complex64::zero(); ncols_half]; out_rows];

    for j in 0..ncols_half {
        let col: Vec<Complex64> = (0..nrows_in).map(|i| x[[i, j]]).collect();
        let col_result = ifft(&col, Some(out_rows))?;
        for i in 0..out_rows {
            col_ifft_mat[i][j] = col_result[i];
        }
    }

    // Step 2: irfft along each row.
    let mut result_data = vec![0.0_f64; out_rows * out_cols];
    for i in 0..out_rows {
        let row_real = irfft_optimized(&col_ifft_mat[i], Some(out_cols))?;
        for j in 0..out_cols {
            result_data[i * out_cols + j] = row_real[j];
        }
    }

    Array2::from_shape_vec((out_rows, out_cols), result_data)
        .map_err(|e| FFTError::DimensionError(format!("irfft2_optimized shape error: {e}")))
}

// ─────────────────────────────────────────────────────────────────────────────
//  Normalized variants (ortho convention)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the normalized (orthonormal) real FFT.
///
/// Each output bin is scaled by `1/sqrt(N)`, making the transform unitary:
/// `rfft_norm` and `irfft_norm` are exact inverses of each other under the same
/// normalization.
///
/// # Examples
///
/// ```
/// use scirs2_fft::real_fft::{rfft_norm, irfft_norm};
///
/// let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let spectrum = rfft_norm(&signal, None).expect("rfft_norm failed");
/// let recovered = irfft_norm(&spectrum, Some(signal.len())).expect("irfft_norm failed");
/// for (a, b) in signal.iter().zip(recovered.iter()) {
///     assert!((a - b).abs() < 1e-9);
/// }
/// ```
pub fn rfft_norm<T>(x: &[T], n: Option<usize>) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let n_val = n.unwrap_or(x.len());
    let mut result = rfft_optimized(x, n)?;
    let scale = 1.0 / (n_val as f64).sqrt();
    for c in &mut result {
        *c = Complex64::new(c.re * scale, c.im * scale);
    }
    Ok(result)
}

/// Compute the normalized (orthonormal) inverse real FFT.
///
/// Counterpart to [`rfft_norm`].  Scales the result of the inverse transform by
/// `sqrt(N)` to undo the forward normalization, yielding unitary round-trip
/// behavior.
pub fn irfft_norm(x: &[Complex64], n: Option<usize>) -> FFTResult<Vec<f64>> {
    let n_val = n.unwrap_or(2 * (x.len() - 1));
    let scale = (n_val as f64).sqrt();
    let result = irfft_optimized(x, n)?;
    Ok(result.iter().map(|&v| v * scale).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
//  N-D real FFT helpers (1-D real + complex FFT along remaining axes)
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a complex twiddle correction along the last axis of a 2-D spectrum.
///
/// Used internally to post-correct the row-RFFT result so that the per-column
/// FFT correctly handles the circular-shift implied by the even/odd packing.
#[allow(dead_code)]
fn apply_twiddle_correction(
    spec: &[Vec<Complex64>],
    n_cols: usize,
) -> Vec<Vec<Complex64>> {
    let n_rows = spec.len();
    let col_half = n_cols / 2 + 1;
    let twiddles = twiddle_forward(n_cols, col_half);

    let mut out = vec![vec![Complex64::zero(); col_half]; n_rows];
    for i in 0..n_rows {
        for j in 0..col_half {
            out[i][j] = spec[i][j] * twiddles[j];
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
//  Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Helper: brute-force DFT
    fn brute_dft(x: &[f64]) -> Vec<Complex64> {
        let n = x.len();
        (0..n / 2 + 1)
            .map(|k| {
                x.iter().enumerate().fold(Complex64::zero(), |acc, (m, &xm)| {
                    let phase = -2.0 * PI * k as f64 * m as f64 / n as f64;
                    acc + Complex64::new(xm * phase.cos(), xm * phase.sin())
                })
            })
            .collect()
    }

    fn assert_complex_slice_eq(a: &[Complex64], b: &[Complex64], tol: f64) {
        assert_eq!(a.len(), b.len(), "length mismatch");
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            assert_relative_eq!(ai.re, bi.re, epsilon = tol,
                max_relative = tol, var_name = format!("bin {i} re"));
            assert_relative_eq!(ai.im, bi.im, epsilon = tol,
                max_relative = tol, var_name = format!("bin {i} im"));
        }
    }

    #[test]
    fn test_rfft_length_4_vs_brute() {
        let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
        let got = rfft_optimized(&signal, None).expect("rfft failed");
        let expected = brute_dft(&signal);
        assert_complex_slice_eq(&got, &expected, 1e-9);
    }

    #[test]
    fn test_rfft_length_8_vs_brute() {
        let signal: Vec<f64> = (0..8).map(|k| (k as f64).cos()).collect();
        let got = rfft_optimized(&signal, None).expect("rfft failed");
        let expected = brute_dft(&signal);
        assert_complex_slice_eq(&got, &expected, 1e-9);
    }

    #[test]
    fn test_rfft_odd_length_5_vs_brute() {
        let signal = vec![1.0_f64, -1.0, 2.0, -2.0, 0.5];
        let got = rfft_optimized(&signal, None).expect("rfft failed");
        let expected = brute_dft(&signal);
        assert_complex_slice_eq(&got, &expected, 1e-9);
    }

    #[test]
    fn test_rfft_output_length() {
        for n in [2, 3, 4, 7, 8, 16, 32, 100] {
            let sig: Vec<f64> = (0..n).map(|k| k as f64).collect();
            let spec = rfft_optimized(&sig, None).expect("rfft");
            assert_eq!(spec.len(), n / 2 + 1, "output length for N={n}");
        }
    }

    #[test]
    fn test_rfft_irfft_roundtrip_even() {
        let signal: Vec<f64> = (0..8).map(|k| k as f64 * 0.5).collect();
        let spectrum = rfft_optimized(&signal, None).expect("rfft");
        let recovered = irfft_optimized(&spectrum, Some(signal.len())).expect("irfft");
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_rfft_irfft_roundtrip_odd() {
        let signal = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0];
        let spectrum = rfft_optimized(&signal, None).expect("rfft");
        let recovered = irfft_optimized(&spectrum, Some(signal.len())).expect("irfft");
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_rfft_norm_roundtrip() {
        let signal = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let spec = rfft_norm(&signal, None).expect("rfft_norm");
        let rec = irfft_norm(&spec, Some(signal.len())).expect("irfft_norm");
        for (a, b) in signal.iter().zip(rec.iter()) {
            assert_relative_eq!(*a, *b, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_rfft2_shape() {
        let data = Array2::from_shape_vec(
            (4, 8),
            (0..32).map(|k| k as f64).collect(),
        )
        .expect("shape");
        let spec = rfft2_optimized(&data, None).expect("rfft2");
        assert_eq!(spec.shape(), &[4, 5]); // 8/2+1 = 5
    }

    #[test]
    fn test_rfft2_irfft2_roundtrip() {
        let n = 4usize;
        let data = Array2::from_shape_vec(
            (n, n),
            (0..n * n).map(|k| k as f64).collect(),
        )
        .expect("shape");
        let spectrum = rfft2_optimized(&data, None).expect("rfft2");
        let recovered = irfft2_optimized(&spectrum, Some((n, n))).expect("irfft2");
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(data[[i, j]], recovered[[i, j]], epsilon = 1e-8,
                    var_name = format!("[{i},{j}]"));
            }
        }
    }

    #[test]
    fn test_rfft_empty_error() {
        let empty: Vec<f64> = vec![];
        assert!(rfft_optimized(&empty, None).is_err());
    }

    #[test]
    fn test_irfft_empty_error() {
        let empty: Vec<Complex64> = vec![];
        assert!(irfft_optimized(&empty, None).is_err());
    }

    #[test]
    fn test_rfft_single_element() {
        let x = vec![42.0_f64];
        let spec = rfft_optimized(&x, None).expect("rfft single");
        assert_eq!(spec.len(), 1);
        assert_relative_eq!(spec[0].re, 42.0, epsilon = 1e-12);
        assert_relative_eq!(spec[0].im, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_rfft_dc_component() {
        // DC bin should equal sum of all samples
        let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
        let spec = rfft_optimized(&signal, None).expect("rfft");
        let dc_expected = signal.iter().sum::<f64>();
        assert_relative_eq!(spec[0].re, dc_expected, epsilon = 1e-9);
        assert_relative_eq!(spec[0].im, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_rfft_with_n_truncate() {
        let signal = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let spec = rfft_optimized(&signal, Some(4)).expect("rfft truncate");
        assert_eq!(spec.len(), 3); // 4/2+1
    }

    #[test]
    fn test_rfft_with_n_zeropad() {
        let signal = vec![1.0_f64, 2.0];
        let spec = rfft_optimized(&signal, Some(8)).expect("rfft pad");
        assert_eq!(spec.len(), 5); // 8/2+1
    }
}
