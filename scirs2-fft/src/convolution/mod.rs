//! FFT-based convolution and correlation.
//!
//! This module provides efficient convolution and correlation routines built on
//! top of the FFT:
//!
//! - [`fft_convolve`] — linear convolution of two real slices
//! - [`fft_convolve_complex`] — linear convolution of complex slices
//! - [`circular_convolve`] — circular (periodic) convolution
//! - [`fft_correlate`] — cross-correlation with Full/Same/Valid modes
//! - [`overlap_add_convolve`] — overlap-add algorithm for long signals
//! - [`overlap_save_convolve`] — overlap-save algorithm for long signals

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use crate::helper::next_fast_len;
use scirs2_core::numeric::Complex64;

// ─────────────────────────────────────────────────────────────────────────────
//  CorrelationMode
// ─────────────────────────────────────────────────────────────────────────────

/// Output size mode for cross-correlation and convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationMode {
    /// Full output of length `a.len() + b.len() - 1`.
    Full,
    /// Output trimmed to `max(a.len(), b.len())` (centred).
    Same,
    /// Only the valid (fully-overlapping) portion, length
    /// `max(a.len(), b.len()) - min(a.len(), b.len()) + 1`.
    Valid,
}

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a real slice to complex, zero-padded to length `n`.
fn real_to_complex_padded(x: &[f64], n: usize) -> Vec<Complex64> {
    let mut v = vec![Complex64::new(0.0, 0.0); n];
    for (i, &xi) in x.iter().enumerate().take(n) {
        v[i] = Complex64::new(xi, 0.0);
    }
    v
}

/// Extract real parts of a complex slice.
fn extract_real(v: &[Complex64]) -> Vec<f64> {
    v.iter().map(|c| c.re).collect()
}

/// Point-wise multiplication in place: a[i] *= b[i].
fn pointwise_mul_inplace(a: &mut Vec<Complex64>, b: &[Complex64]) {
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        *ai = *ai * *bi;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Linear convolution — real
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the linear convolution `c = a * b` using FFT.
///
/// The output has length `a.len() + b.len() - 1`. Both inputs are padded to
/// the next fast FFT length (≥ output length) before the transform.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if either input is empty.
///
/// # Examples
///
/// ```
/// use scirs2_fft::convolution::fft_convolve;
/// // [1,1] * [1,1] = [1,2,1]
/// let c = fft_convolve(&[1.0, 1.0], &[1.0, 1.0]).expect("valid input");
/// assert!((c[0] - 1.0).abs() < 1e-10);
/// assert!((c[1] - 2.0).abs() < 1e-10);
/// assert!((c[2] - 1.0).abs() < 1e-10);
/// ```
pub fn fft_convolve(a: &[f64], b: &[f64]) -> FFTResult<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Err(FFTError::ValueError(
            "fft_convolve: inputs must be non-empty".to_string(),
        ));
    }
    let out_len = a.len() + b.len() - 1;
    let n = next_fast_len(out_len, true);

    let mut fa = real_to_complex_padded(a, n);
    let fb = real_to_complex_padded(b, n);

    let mut fa = fft(&fa, Some(n))?;
    let fb = fft(&fb, Some(n))?;
    pointwise_mul_inplace(&mut fa, &fb);
    let result = ifft(&fa, Some(n))?;
    let mut out = extract_real(&result);
    out.truncate(out_len);
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Linear convolution — complex
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the linear convolution of two complex slices using FFT.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if either input is empty.
///
/// # Examples
///
/// ```
/// use scirs2_fft::convolution::fft_convolve_complex;
/// use scirs2_core::numeric::Complex64;
/// let a = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
/// let b = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
/// let c = fft_convolve_complex(&a, &b).expect("valid input");
/// assert_eq!(c.len(), 3);
/// ```
pub fn fft_convolve_complex(a: &[Complex64], b: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    if a.is_empty() || b.is_empty() {
        return Err(FFTError::ValueError(
            "fft_convolve_complex: inputs must be non-empty".to_string(),
        ));
    }
    let out_len = a.len() + b.len() - 1;
    let n = next_fast_len(out_len, true);

    let mut fa_padded = a.to_vec();
    fa_padded.resize(n, Complex64::new(0.0, 0.0));
    let mut fb_padded = b.to_vec();
    fb_padded.resize(n, Complex64::new(0.0, 0.0));

    let mut fa = fft(&fa_padded, Some(n))?;
    let fb = fft(&fb_padded, Some(n))?;
    pointwise_mul_inplace(&mut fa, &fb);
    let mut result = ifft(&fa, Some(n))?;
    result.truncate(out_len);
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Circular convolution
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the circular convolution of two real slices.
///
/// Both inputs must have the same length N. The result is
/// `c[n] = sum_k a[k] * b[(n-k) mod N]`.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if inputs are empty or have different lengths.
///
/// # Examples
///
/// ```
/// use scirs2_fft::convolution::circular_convolve;
/// let a = [1.0, 2.0, 3.0, 4.0];
/// let b = [1.0, 0.0, 0.0, 0.0];
/// let c = circular_convolve(&a, &b).expect("valid input");
/// // Convolving with a unit impulse returns the original
/// for (ai, ci) in a.iter().zip(c.iter()) {
///     assert!((ai - ci).abs() < 1e-10);
/// }
/// ```
pub fn circular_convolve(a: &[f64], b: &[f64]) -> FFTResult<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Err(FFTError::ValueError(
            "circular_convolve: inputs must be non-empty".to_string(),
        ));
    }
    if a.len() != b.len() {
        return Err(FFTError::ValueError(format!(
            "circular_convolve: inputs must have equal length ({} vs {})",
            a.len(),
            b.len()
        )));
    }
    let n = a.len();
    let fa_c: Vec<Complex64> = a.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let fb_c: Vec<Complex64> = b.iter().map(|&v| Complex64::new(v, 0.0)).collect();

    let mut fa = fft(&fa_c, None)?;
    let fb = fft(&fb_c, None)?;
    pointwise_mul_inplace(&mut fa, &fb);
    let result = ifft(&fa, None)?;
    let mut out = extract_real(&result);
    out.truncate(n);
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Cross-correlation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the cross-correlation of `a` and `b`: `r[n] = sum_k a[k] * b[k+n]`.
///
/// This is equivalent to convolving `a` with the time-reversed version of `b`.
///
/// # Arguments
///
/// * `a` — reference signal.
/// * `b` — signal to correlate against.
/// * `mode` — output size mode.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if either input is empty.
///
/// # Examples
///
/// ```
/// use scirs2_fft::convolution::{fft_correlate, CorrelationMode};
/// let a = [1.0, 2.0, 3.0];
/// let b = [0.0, 1.0, 0.5];
/// let r = fft_correlate(&a, &b, CorrelationMode::Full).expect("valid input");
/// assert_eq!(r.len(), 5);
/// ```
pub fn fft_correlate(a: &[f64], b: &[f64], mode: CorrelationMode) -> FFTResult<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Err(FFTError::ValueError(
            "fft_correlate: inputs must be non-empty".to_string(),
        ));
    }

    let full_len = a.len() + b.len() - 1;
    let n = next_fast_len(full_len, true);

    // Correlation = convolution of a with time-reverse of b
    let b_rev: Vec<f64> = b.iter().rev().copied().collect();

    let fa_c = real_to_complex_padded(a, n);
    let fb_c = real_to_complex_padded(&b_rev, n);

    let mut fa = fft(&fa_c, Some(n))?;
    let fb = fft(&fb_c, Some(n))?;
    pointwise_mul_inplace(&mut fa, &fb);
    let result = ifft(&fa, Some(n))?;
    let full: Vec<f64> = extract_real(&result)[..full_len].to_vec();

    match mode {
        CorrelationMode::Full => Ok(full),
        CorrelationMode::Same => {
            let out_len = a.len().max(b.len());
            let start = (full_len - out_len) / 2;
            Ok(full[start..start + out_len].to_vec())
        }
        CorrelationMode::Valid => {
            let la = a.len();
            let lb = b.len();
            if la < lb {
                return Err(FFTError::ValueError(
                    "fft_correlate Valid mode: a must be at least as long as b".to_string(),
                ));
            }
            let out_len = la - lb + 1;
            // In correlation the valid part starts at offset lb-1
            let start = lb - 1;
            Ok(full[start..start + out_len].to_vec())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Overlap-add convolution
// ─────────────────────────────────────────────────────────────────────────────

/// Convolve `signal` with `kernel` using the overlap-add algorithm.
///
/// The signal is split into blocks of `block_size` samples. Each block is
/// zero-padded and FFT-convolved with the kernel. Successive output blocks are
/// then added (overlapped) to produce the final result.
///
/// This method is efficient when the signal is much longer than the kernel.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `block_size == 0` or either input is empty.
///
/// # Examples
///
/// ```
/// use scirs2_fft::convolution::overlap_add_convolve;
/// let signal: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
/// let kernel = vec![0.5, 0.5];
/// let out = overlap_add_convolve(&signal, &kernel, 16).expect("valid input");
/// assert_eq!(out.len(), signal.len() + kernel.len() - 1);
/// ```
pub fn overlap_add_convolve(signal: &[f64], kernel: &[f64], block_size: usize) -> FFTResult<Vec<f64>> {
    if block_size == 0 {
        return Err(FFTError::ValueError(
            "overlap_add_convolve: block_size must be > 0".to_string(),
        ));
    }
    if signal.is_empty() || kernel.is_empty() {
        return Err(FFTError::ValueError(
            "overlap_add_convolve: inputs must be non-empty".to_string(),
        ));
    }

    let m = kernel.len();
    let out_len = signal.len() + m - 1;
    let fft_size = next_fast_len(block_size + m - 1, true);

    // Pre-compute FFT of zero-padded kernel
    let h_padded = real_to_complex_padded(kernel, fft_size);
    let h_fft = fft(&h_padded, Some(fft_size))?;

    let mut output = vec![0.0_f64; out_len + fft_size]; // extra room for the tail

    let mut pos = 0usize;
    while pos < signal.len() {
        let end = (pos + block_size).min(signal.len());
        let block = &signal[pos..end];

        let x_padded = real_to_complex_padded(block, fft_size);
        let mut x_fft = fft(&x_padded, Some(fft_size))?;
        pointwise_mul_inplace(&mut x_fft, &h_fft);
        let y = ifft(&x_fft, Some(fft_size))?;
        let y_real = extract_real(&y);

        // Add the block result to the output starting at position `pos`
        for (k, &val) in y_real.iter().enumerate().take(fft_size) {
            if pos + k < output.len() {
                output[pos + k] += val;
            }
        }
        pos += block_size;
    }

    output.truncate(out_len);
    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Overlap-save convolution
// ─────────────────────────────────────────────────────────────────────────────

/// Convolve `signal` with `kernel` using the overlap-save algorithm.
///
/// Each FFT block of size `fft_size = block_size + kernel.len() - 1` takes
/// `kernel.len() - 1` samples from the previous block (the saved overlap)
/// plus `block_size` new samples.  Only the last `block_size` output samples
/// of each block are valid and are written to the output.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `block_size == 0` or either input is empty.
///
/// # Examples
///
/// ```
/// use scirs2_fft::convolution::overlap_save_convolve;
/// let signal: Vec<f64> = (0..64).map(|i| (i as f64).sin()).collect();
/// let kernel = vec![0.5, 0.5];
/// let out = overlap_save_convolve(&signal, &kernel, 16).expect("valid input");
/// assert_eq!(out.len(), signal.len() + kernel.len() - 1);
/// ```
pub fn overlap_save_convolve(signal: &[f64], kernel: &[f64], block_size: usize) -> FFTResult<Vec<f64>> {
    if block_size == 0 {
        return Err(FFTError::ValueError(
            "overlap_save_convolve: block_size must be > 0".to_string(),
        ));
    }
    if signal.is_empty() || kernel.is_empty() {
        return Err(FFTError::ValueError(
            "overlap_save_convolve: inputs must be non-empty".to_string(),
        ));
    }

    let m = kernel.len();
    let out_len = signal.len() + m - 1;
    let fft_size = next_fast_len(block_size + m - 1, true);
    let overlap = m - 1;

    // Pre-compute kernel FFT
    let h_padded = real_to_complex_padded(kernel, fft_size);
    let h_fft = fft(&h_padded, Some(fft_size))?;

    let mut output = Vec::with_capacity(out_len);

    // Zero-pad the signal on the left by (m-1) so the first block has a
    // valid "saved" region.
    let mut padded_signal = vec![0.0_f64; m - 1];
    padded_signal.extend_from_slice(signal);

    let mut pos = 0usize;
    while output.len() < out_len {
        let end = (pos + fft_size).min(padded_signal.len());
        let mut block = vec![0.0_f64; fft_size];
        let slice = &padded_signal[pos..end];
        block[..slice.len()].copy_from_slice(slice);

        let x_padded: Vec<Complex64> = block.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let mut x_fft = fft(&x_padded, Some(fft_size))?;
        pointwise_mul_inplace(&mut x_fft, &h_fft);
        let y = ifft(&x_fft, Some(fft_size))?;
        let y_real = extract_real(&y);

        // Discard the first `overlap` samples; keep the rest
        for &val in &y_real[overlap..fft_size] {
            if output.len() < out_len {
                output.push(val);
            }
        }
        pos += block_size;
    }

    output.truncate(out_len);
    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fft_convolve_unit_impulse() {
        // Convolving with [1] = identity
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0];
        let c = fft_convolve(&a, &b).expect("failed to create c");
        assert_eq!(c.len(), 4);
        for (ai, ci) in a.iter().zip(c.iter()) {
            assert_relative_eq!(ai, ci, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_convolve_known() {
        // [1,1] * [1,1] = [1,2,1]
        let c = fft_convolve(&[1.0, 1.0], &[1.0, 1.0]).expect("failed to create c");
        let expected = [1.0, 2.0, 1.0];
        assert_eq!(c.len(), 3);
        for (a, b) in c.iter().zip(expected.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fft_convolve_complex_length() {
        let a: Vec<Complex64> = (0..4).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let b: Vec<Complex64> = vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let c = fft_convolve_complex(&a, &b).expect("failed to create c");
        assert_eq!(c.len(), 5);
    }

    #[test]
    fn test_circular_convolve_identity() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 0.0];
        let c = circular_convolve(&a, &b).expect("failed to create c");
        for (ai, ci) in a.iter().zip(c.iter()) {
            assert_relative_eq!(ai, ci, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_circular_convolve_unequal_error() {
        assert!(circular_convolve(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn test_fft_correlate_full() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 0.0];
        let r = fft_correlate(&a, &b, CorrelationMode::Full).expect("failed to create r");
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_fft_correlate_same() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, -1.0];
        let r = fft_correlate(&a, &b, CorrelationMode::Same).expect("failed to create r");
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_fft_correlate_valid() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, -1.0];
        let r = fft_correlate(&a, &b, CorrelationMode::Valid).expect("failed to create r");
        // la=4, lb=2  → valid_len = la - lb + 1 = 3
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn test_overlap_add_matches_direct() {
        let signal: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).sin()).collect();
        let kernel = vec![0.25, 0.5, 0.25];
        let ola = overlap_add_convolve(&signal, &kernel, 8).expect("failed to create ola");
        let direct = fft_convolve(&signal, &kernel).expect("failed to create direct");
        assert_eq!(ola.len(), direct.len());
        for (a, b) in ola.iter().zip(direct.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_overlap_save_matches_direct() {
        let signal: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).sin()).collect();
        let kernel = vec![0.25, 0.5, 0.25];
        let ols = overlap_save_convolve(&signal, &kernel, 8).expect("failed to create ols");
        let direct = fft_convolve(&signal, &kernel).expect("failed to create direct");
        assert_eq!(ols.len(), direct.len());
        for (a, b) in ols.iter().zip(direct.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_fft_convolve_empty_error() {
        assert!(fft_convolve(&[], &[1.0]).is_err());
        assert!(fft_convolve(&[1.0], &[]).is_err());
    }
}
