//! Enhanced Butterfly Operations for FFT
//!
//! This module provides optimized, in-place butterfly operations for radix-2,
//! radix-4, radix-8, and split-radix FFT algorithms.  All operations work
//! directly on mutable slices / arrays without any heap allocation.
//!
//! # Overview
//!
//! | Function | Radix | Points | Twiddles |
//! |----------|-------|--------|----------|
//! | [`butterfly2`] | 2 | 2 | 1 |
//! | [`butterfly4`] | 4 | 4 | 3 |
//! | [`butterfly8`] | 8 | 8 | 7 |
//! | [`split_radix_butterfly`] | 2/4 | any | computed |
//!
//! Additionally, [`generate_twiddle_table`] pre-computes the unit-root vector
//! `e^{-2 pi i k / N}` for `k = 0 .. N-1`.

use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

use crate::error::{FFTError, FFTResult};

// ─────────────────────────────────────────────────────────────────────────────
//  Twiddle-table generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate twiddle factor table for an N-point DFT.
///
/// Returns `W[k] = e^{-2 pi i k / N}` for `k = 0, 1, ..., N-1`.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `n == 0`.
pub fn generate_twiddle_table(n: usize) -> FFTResult<Vec<Complex64>> {
    if n == 0 {
        return Err(FFTError::ValueError(
            "generate_twiddle_table: n must be > 0".into(),
        ));
    }
    if n == 1 {
        return Ok(vec![Complex64::new(1.0, 0.0)]);
    }
    let inv_n = -2.0 * PI / n as f64;
    Ok((0..n)
        .map(|k| {
            let angle = inv_n * k as f64;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect())
}

/// Generate twiddle factor table for an inverse N-point DFT.
///
/// Returns `W[k] = e^{+2 pi i k / N}` for `k = 0, 1, ..., N-1`.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `n == 0`.
pub fn generate_inverse_twiddle_table(n: usize) -> FFTResult<Vec<Complex64>> {
    if n == 0 {
        return Err(FFTError::ValueError(
            "generate_inverse_twiddle_table: n must be > 0".into(),
        ));
    }
    if n == 1 {
        return Ok(vec![Complex64::new(1.0, 0.0)]);
    }
    let inv_n = 2.0 * PI / n as f64;
    Ok((0..n)
        .map(|k| {
            let angle = inv_n * k as f64;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Radix-2 butterfly
// ─────────────────────────────────────────────────────────────────────────────

/// In-place radix-2 butterfly.
///
/// Computes:
/// ```text
///   a' = a + twiddle * b
///   b' = a - twiddle * b
/// ```
///
/// This is the fundamental building block of a decimation-in-time (DIT)
/// Cooley-Tukey FFT.  It is completely allocation-free.
#[inline(always)]
pub fn butterfly2(a: &mut Complex64, b: &mut Complex64, twiddle: Complex64) {
    let t = twiddle * *b;
    let new_a = *a + t;
    let new_b = *a - t;
    *a = new_a;
    *b = new_b;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Radix-4 butterfly
// ─────────────────────────────────────────────────────────────────────────────

/// In-place radix-4 DFT butterfly.
///
/// Computes a 4-point DFT in-place using twiddle factors.
/// `twiddles[0]` = W_4^1, `twiddles[1]` = W_4^2, `twiddles[2]` = W_4^3.
///
/// For a standalone 4-point DFT: `W_4^1 = e^{-pi i/2} = -j`,
/// `W_4^2 = -1`, `W_4^3 = j`.
///
/// The standard radix-4 DFT decomposition:
/// ```text
///   X[0] = x[0] + x[1] + x[2] + x[3]
///   X[1] = x[0] - j*x[1] - x[2] + j*x[3]
///   X[2] = x[0] - x[1] + x[2] - x[3]
///   X[3] = x[0] + j*x[1] - x[2] - j*x[3]
/// ```
#[inline]
pub fn butterfly4(a: &mut [Complex64; 4], twiddles: &[Complex64; 3]) {
    // Standard 4-point DFT using the DFT matrix approach:
    // X[k] = sum_{n=0}^{3} x[n] * W_4^{nk}
    // where W_4 = e^{-2*pi*i/4} = -j
    //
    // twiddles[0] = W_4^1 = -j
    // twiddles[1] = W_4^2 = -1
    // twiddles[2] = W_4^3 = j

    let x0 = a[0];
    let x1 = a[1];
    let x2 = a[2];
    let x3 = a[3];

    // X[0] = x0 + x1 + x2 + x3
    a[0] = x0 + x1 + x2 + x3;

    // X[1] = x0 + W^1*x1 + W^2*x2 + W^3*x3
    a[1] = x0 + twiddles[0] * x1 + twiddles[1] * x2 + twiddles[2] * x3;

    // X[2] = x0 + W^2*x1 + W^4*x2 + W^6*x3
    //      = x0 + W^2*x1 + (W^2)^2*x2 + (W^2)^3*x3
    let w2 = twiddles[1]; // W^2
    let w4 = w2 * w2; // W^4 = (W^2)^2
    let w6 = w4 * w2; // W^6
    a[2] = x0 + w2 * x1 + w4 * x2 + w6 * x3;

    // X[3] = x0 + W^3*x1 + W^6*x2 + W^9*x3
    let w3 = twiddles[2]; // W^3
    let w9 = w3 * w3 * w3; // W^9
    a[3] = x0 + w3 * x1 + w6 * x2 + w9 * x3;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Radix-8 butterfly
// ─────────────────────────────────────────────────────────────────────────────

/// In-place radix-8 DFT butterfly.
///
/// Computes an 8-point DFT in-place. `twiddles[k]` = `W_8^{k+1}` for k=0..6,
/// i.e., `twiddles` contains `W_8^1` through `W_8^7`.
#[inline]
pub fn butterfly8(a: &mut [Complex64; 8], twiddles: &[Complex64; 7]) {
    // Direct 8-point DFT: X[k] = sum_{n=0}^{7} x[n] * W_8^{n*k}
    // We pre-compute the powers W_8^m for m = 0..7 from the twiddle table.
    let w = [
        Complex64::new(1.0, 0.0), // W^0
        twiddles[0],              // W^1
        twiddles[1],              // W^2
        twiddles[2],              // W^3
        twiddles[3],              // W^4
        twiddles[4],              // W^5
        twiddles[5],              // W^6
        twiddles[6],              // W^7
    ];

    let input = *a;
    for k in 0..8 {
        let mut sum = Complex64::new(0.0, 0.0);
        for n in 0..8 {
            let idx = (n * k) % 8;
            sum += input[n] * w[idx];
        }
        a[k] = sum;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Split-radix butterfly (L-shaped)
// ─────────────────────────────────────────────────────────────────────────────

/// In-place split-radix FFT for a complex array of length N.
///
/// Implements the Cooley-Tukey radix-2 DIT FFT with bit-reversal
/// permutation.  This is the standard iterative butterfly algorithm
/// that achieves O(N log N) complexity.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `data.len()` is not a power of two or is < 4.
pub fn split_radix_butterfly(data: &mut [Complex64]) -> FFTResult<()> {
    let n = data.len();
    if n < 4 {
        return Err(FFTError::ValueError(
            "split_radix_butterfly: length must be >= 4".into(),
        ));
    }
    if !n.is_power_of_two() {
        return Err(FFTError::ValueError(
            "split_radix_butterfly: length must be a power of two".into(),
        ));
    }

    // Bit-reversal permutation
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = reverse_bits(i, bits);
        if i < j {
            data.swap(i, j);
        }
    }

    // Iterative butterfly passes
    let mut size = 2;
    while size <= n {
        let half = size / 2;
        let angle_step = -2.0 * PI / size as f64;

        let mut group_start = 0;
        while group_start < n {
            for k in 0..half {
                let angle = angle_step * k as f64;
                let twiddle = Complex64::new(angle.cos(), angle.sin());

                let i = group_start + k;
                let j = i + half;

                let t = twiddle * data[j];
                data[j] = data[i] - t;
                data[i] = data[i] + t;
            }
            group_start += size;
        }
        size *= 2;
    }

    Ok(())
}

/// Reverse the lower `bits` bits of `x`.
fn reverse_bits(x: usize, bits: u32) -> usize {
    let mut result = 0usize;
    let mut val = x;
    for _ in 0..bits {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
//  Direct DFT for small sizes (base-case for recursive algorithms)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute a direct (naive) DFT for small `N`.
///
/// Complexity is O(N^2) so this should only be used for small base cases
/// (typically N <= 16).
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `data` is empty.
pub fn direct_dft(data: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let n = data.len();
    if n == 0 {
        return Err(FFTError::ValueError("direct_dft: empty input".into()));
    }
    if n == 1 {
        return Ok(data.to_vec());
    }

    let angle_base = -2.0 * PI / n as f64;
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for j in 0..n {
            let angle = angle_base * (k * j) as f64;
            let w = Complex64::new(angle.cos(), angle.sin());
            sum += data[j] * w;
        }
        result[k] = sum;
    }
    Ok(result)
}

/// Compute a direct (naive) inverse DFT for small `N`.
///
/// # Errors
///
/// Returns [`FFTError::ValueError`] if `data` is empty.
pub fn direct_idft(data: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    let n = data.len();
    if n == 0 {
        return Err(FFTError::ValueError("direct_idft: empty input".into()));
    }
    if n == 1 {
        return Ok(data.to_vec());
    }

    let angle_base = 2.0 * PI / n as f64;
    let inv_n = 1.0 / n as f64;
    let mut result = vec![Complex64::new(0.0, 0.0); n];
    for k in 0..n {
        let mut sum = Complex64::new(0.0, 0.0);
        for j in 0..n {
            let angle = angle_base * (k * j) as f64;
            let w = Complex64::new(angle.cos(), angle.sin());
            sum += data[j] * w;
        }
        result[k] = sum * inv_n;
    }
    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Helper: maximum absolute error between two complex slices.
    fn max_abs_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0_f64, f64::max)
    }

    // ── twiddle table ────────────────────────────────────────────────────
    #[test]
    fn test_twiddle_table_size_1() {
        let tw = generate_twiddle_table(1).expect("should succeed");
        assert_eq!(tw.len(), 1);
        assert_relative_eq!(tw[0].re, 1.0, epsilon = 1e-15);
        assert_relative_eq!(tw[0].im, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_twiddle_table_values() {
        let n = 8;
        let tw = generate_twiddle_table(n).expect("should succeed");
        assert_eq!(tw.len(), n);

        // W^0 = 1
        assert_relative_eq!(tw[0].re, 1.0, epsilon = 1e-14);
        assert_relative_eq!(tw[0].im, 0.0, epsilon = 1e-14);

        // W^(N/4) = e^{-pi i / 2} = -j
        assert_relative_eq!(tw[n / 4].re, 0.0, epsilon = 1e-14);
        assert_relative_eq!(tw[n / 4].im, -1.0, epsilon = 1e-14);

        // W^(N/2) = e^{-pi i} = -1
        assert_relative_eq!(tw[n / 2].re, -1.0, epsilon = 1e-14);
        assert_relative_eq!(tw[n / 2].im, 0.0, epsilon = 1e-14);

        // All magnitudes should be 1
        for w in &tw {
            assert_relative_eq!(w.norm(), 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_twiddle_table_error_on_zero() {
        assert!(generate_twiddle_table(0).is_err());
    }

    // ── butterfly2 ──────────────────────────────────────────────────────
    #[test]
    fn test_butterfly2_trivial_twiddle() {
        let mut a = Complex64::new(3.0, 0.0);
        let mut b = Complex64::new(1.0, 0.0);
        butterfly2(&mut a, &mut b, Complex64::new(1.0, 0.0));
        assert_relative_eq!(a.re, 4.0, epsilon = 1e-14);
        assert_relative_eq!(b.re, 2.0, epsilon = 1e-14);
    }

    #[test]
    fn test_butterfly2_with_twiddle() {
        // W = -1  =>  a' = a + (-1)*b = a-b,  b' = a - (-1)*b = a+b
        let mut a = Complex64::new(5.0, 0.0);
        let mut b = Complex64::new(3.0, 0.0);
        butterfly2(&mut a, &mut b, Complex64::new(-1.0, 0.0));
        assert_relative_eq!(a.re, 2.0, epsilon = 1e-14);
        assert_relative_eq!(b.re, 8.0, epsilon = 1e-14);
    }

    // ── butterfly4 ──────────────────────────────────────────────────────
    #[test]
    fn test_butterfly4_matches_direct_dft() {
        let input = [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let expected = direct_dft(&input).expect("direct_dft failed");

        // For a 4-point DFT: W_4 = e^{-2*pi*i/4} = e^{-pi*i/2} = -j
        // twiddles[0] = W_4^1 = -j
        // twiddles[1] = W_4^2 = -1
        // twiddles[2] = W_4^3 = j
        let twiddles = [
            Complex64::new(0.0, -1.0), // W4^1 = -j
            Complex64::new(-1.0, 0.0), // W4^2 = -1
            Complex64::new(0.0, 1.0),  // W4^3 = j
        ];
        let mut data = input;
        butterfly4(&mut data, &twiddles);

        let err = max_abs_err(&data, &expected);
        assert!(err < 1e-12, "butterfly4 error = {err}");
    }

    // ── butterfly8 ──────────────────────────────────────────────────────
    #[test]
    fn test_butterfly8_matches_direct_dft() {
        let input: [Complex64; 8] = [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(0.5, 0.5),
            Complex64::new(3.0, 0.0),
            Complex64::new(-1.0, 1.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(1.5, -0.5),
            Complex64::new(-0.5, 0.0),
        ];
        let expected = direct_dft(&input).expect("direct_dft failed");

        // W_8^k for k=1..7
        let twiddles: [Complex64; 7] = std::array::from_fn(|k| {
            let angle = -2.0 * PI * (k + 1) as f64 / 8.0;
            Complex64::new(angle.cos(), angle.sin())
        });

        let mut data = input;
        butterfly8(&mut data, &twiddles);

        let err = max_abs_err(&data, &expected);
        assert!(err < 1e-10, "butterfly8 error = {err}");
    }

    // ── direct DFT ──────────────────────────────────────────────────────
    #[test]
    fn test_direct_dft_known_result() {
        // DFT of [1, 1, 1, 1] = [4, 0, 0, 0]
        let input = vec![Complex64::new(1.0, 0.0); 4];
        let result = direct_dft(&input).expect("direct_dft failed");
        assert_relative_eq!(result[0].re, 4.0, epsilon = 1e-12);
        for k in 1..4 {
            assert!(result[k].norm() < 1e-12, "non-zero at k={k}");
        }
    }

    #[test]
    fn test_direct_dft_idft_roundtrip() {
        let input = vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(0.5, 0.5),
            Complex64::new(-2.0, 1.5),
        ];
        let spectrum = direct_dft(&input).expect("dft failed");
        let recovered = direct_idft(&spectrum).expect("idft failed");
        let err = max_abs_err(&input, &recovered);
        assert!(err < 1e-12, "roundtrip error = {err}");
    }

    #[test]
    fn test_direct_dft_empty() {
        assert!(direct_dft(&[]).is_err());
    }

    // ── split-radix butterfly ───────────────────────────────────────────
    #[test]
    fn test_split_radix_butterfly_size_4() {
        let input = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(0.0, -1.0),
        ];
        let expected = direct_dft(&input).expect("dft failed");
        let mut data = input;
        split_radix_butterfly(&mut data).expect("split_radix failed");
        let err = max_abs_err(&data, &expected);
        assert!(err < 1e-10, "split_radix error (n=4) = {err}");
    }

    #[test]
    fn test_split_radix_butterfly_size_8() {
        let input: Vec<Complex64> = (0..8)
            .map(|k| Complex64::new(k as f64, -(k as f64) * 0.5))
            .collect();
        let expected = direct_dft(&input).expect("dft failed");
        let mut data = input;
        split_radix_butterfly(&mut data).expect("split_radix failed");
        let err = max_abs_err(&data, &expected);
        assert!(err < 1e-10, "split_radix error (n=8) = {err}");
    }

    #[test]
    fn test_split_radix_butterfly_size_16() {
        let input: Vec<Complex64> = (0..16)
            .map(|k| Complex64::new((k as f64 * 0.5).sin(), (k as f64 * 0.3).cos()))
            .collect();
        let expected = direct_dft(&input).expect("dft failed");
        let mut data = input;
        split_radix_butterfly(&mut data).expect("split_radix failed");
        let err = max_abs_err(&data, &expected);
        assert!(err < 1e-10, "split_radix error (n=16) = {err}");
    }

    #[test]
    fn test_split_radix_butterfly_not_power_of_two() {
        let mut data = vec![Complex64::new(1.0, 0.0); 6];
        assert!(split_radix_butterfly(&mut data).is_err());
    }

    #[test]
    fn test_split_radix_butterfly_too_small() {
        let mut data = vec![Complex64::new(1.0, 0.0); 2];
        assert!(split_radix_butterfly(&mut data).is_err());
    }

    // ── inverse twiddle table ───────────────────────────────────────────
    #[test]
    fn test_inverse_twiddle_table() {
        let n = 8;
        let fw = generate_twiddle_table(n).expect("forward failed");
        let inv = generate_inverse_twiddle_table(n).expect("inverse failed");
        // W[k] * W_inv[k] = |W|^2 = 1, since W_inv[k] = conj(W[k])
        for k in 0..n {
            let product = fw[k] * inv[k];
            assert_relative_eq!(product.re, 1.0, epsilon = 1e-14);
            assert_relative_eq!(product.im, 0.0, epsilon = 1e-14);
        }
    }
}
