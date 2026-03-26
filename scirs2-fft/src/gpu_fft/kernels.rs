//! Simulated GPU FFT kernel implementations.
//!
//! All functions operate entirely in Rust and emulate the tiling, butterfly,
//! and permutation passes that would run on actual GPU hardware.  The API
//! surface is designed so that a future CUDA/ROCm back-end can be dropped in
//! without altering the `pipeline` layer.

use std::f64::consts::PI;

use scirs2_core::numeric::Complex64;

use super::types::{FftDirection, GpuFftError, GpuFftResult, NormalizationMode};

// ─────────────────────────────────────────────────────────────────────────────
// Twiddle factor computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute twiddle factors `W_N^k = exp(−2πi·k/N)` for `k = 0..n/2`.
///
/// Uses a recurrence relation to avoid repeated `sin`/`cos` evaluations:
/// `W^(k+1) = W^k * W^1`.
///
/// # Errors
///
/// Returns [`GpuFftError::SizeTooSmall`] if `n < 2`.
pub fn compute_twiddles_gpu(n: usize) -> GpuFftResult<Vec<Complex64>> {
    if n < 2 {
        return Err(GpuFftError::SizeTooSmall(n));
    }
    let half = n / 2;
    let mut twiddles = Vec::with_capacity(half);

    // W^0 = 1
    let angle_step = -2.0 * PI / n as f64;
    let w1 = Complex64::new(angle_step.cos(), angle_step.sin());
    let mut wk = Complex64::new(1.0, 0.0);
    for _ in 0..half {
        twiddles.push(wk);
        wk = wk * w1;
    }
    Ok(twiddles)
}

/// Compute *inverse* twiddle factors `W_N^{-k} = exp(+2πi·k/N)` for `k = 0..n/2`.
///
/// Used by the Bluestein kernel for the IFFT direction.
///
/// # Errors
///
/// Returns [`GpuFftError::SizeTooSmall`] if `n < 2`.
pub fn compute_inverse_twiddles_gpu(n: usize) -> GpuFftResult<Vec<Complex64>> {
    if n < 2 {
        return Err(GpuFftError::SizeTooSmall(n));
    }
    let half = n / 2;
    let mut twiddles = Vec::with_capacity(half);

    let angle_step = 2.0 * PI / n as f64;
    let w1 = Complex64::new(angle_step.cos(), angle_step.sin());
    let mut wk = Complex64::new(1.0, 0.0);
    for _ in 0..half {
        twiddles.push(wk);
        wk = wk * w1;
    }
    Ok(twiddles)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bit-reversal permutation
// ─────────────────────────────────────────────────────────────────────────────

/// Iterative bit-reversal permutation in place.
///
/// After this pass the elements of `data` are in bit-reversed order, which is
/// the initial condition required by the iterative Cooley-Tukey algorithm.
///
/// # Panics
///
/// Panics in debug mode if `data.len()` is not a power of two.
pub fn bit_reverse_permute_gpu(data: &mut [Complex64]) {
    let n = data.len();
    debug_assert!(
        n.is_power_of_two(),
        "bit_reverse_permute_gpu: n must be a power of two"
    );
    if n <= 1 {
        return;
    }
    let log2n = n.trailing_zeros() as usize;
    for i in 0..n {
        let rev = bit_reverse(i, log2n);
        if i < rev {
            data.swap(i, rev);
        }
    }
}

/// Reverse the `bits` least significant bits of `x`.
#[inline]
fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Butterfly pass
// ─────────────────────────────────────────────────────────────────────────────

/// Execute one butterfly stage of the Cooley-Tukey DIT algorithm.
///
/// For a butterfly group of size `2·stride`, each butterfly reads a pair
/// `(u, v)` at distance `stride` apart and produces:
/// ```text
///   out[i]        = u + W · v
///   out[i+stride] = u − W · v
/// ```
/// where `W` is the twiddle factor `twiddles[k * (n / (2*stride))]`.
///
/// This function modifies `data` in-place.
pub fn butterfly_pass_gpu(data: &mut [Complex64], stride: usize, twiddles: &[Complex64]) {
    let n = data.len();
    let step = 2 * stride;
    let twiddle_step = if !twiddles.is_empty() {
        twiddles.len() / stride
    } else {
        1
    };

    let mut pos = 0;
    while pos < n {
        for k in 0..stride {
            let twiddle_idx = k * twiddle_step;
            let w = if twiddle_idx < twiddles.len() {
                twiddles[twiddle_idx]
            } else {
                Complex64::new(1.0, 0.0)
            };
            let u = data[pos + k];
            let v = w * data[pos + k + stride];
            data[pos + k] = u + v;
            data[pos + k + stride] = u - v;
        }
        pos += step;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cooley-Tukey (power-of-two)
// ─────────────────────────────────────────────────────────────────────────────

/// Full iterative Cooley-Tukey FFT for power-of-two sizes.
///
/// Steps:
/// 1. Bit-reversal permutation.
/// 2. log₂(N) butterfly stages.
///
/// # Errors
///
/// * [`GpuFftError::SizeTooSmall`] – if `data.len() < 2`.
/// * [`GpuFftError::NonPowerOfTwo`] – if `data.len()` is not a power of two.
pub fn cooley_tukey_gpu(
    data: &mut [Complex64],
    direction: FftDirection,
    twiddles: &[Complex64],
) -> GpuFftResult<()> {
    let n = data.len();
    if n < 2 {
        return Err(GpuFftError::SizeTooSmall(n));
    }
    if !n.is_power_of_two() {
        return Err(GpuFftError::NonPowerOfTwo(n));
    }

    // For inverse direction we conjugate the twiddles: W_N^{-k} = conj(W_N^k).
    let effective_twiddles: Vec<Complex64> = match direction {
        FftDirection::Forward => twiddles.to_vec(),
        FftDirection::Inverse => twiddles.iter().map(|w| w.conj()).collect(),
    };

    bit_reverse_permute_gpu(data);

    let mut stride = 1usize;
    while stride < n {
        butterfly_pass_gpu(data, stride, &effective_twiddles);
        stride <<= 1;
    }

    // The inverse DFT needs an overall 1/N scale.
    if direction == FftDirection::Inverse {
        let scale = 1.0 / n as f64;
        for x in data.iter_mut() {
            *x = *x * scale;
        }
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Bluestein (arbitrary size)
// ─────────────────────────────────────────────────────────────────────────────

/// Bluestein's chirp-z algorithm for arbitrary (non-power-of-two) sizes.
///
/// Expresses an N-point DFT as a convolution, then pads to the next power of
/// two and uses [`cooley_tukey_gpu`] internally.
///
/// # Errors
///
/// * [`GpuFftError::SizeTooSmall`] – if `data.len() < 2`.
/// * [`GpuFftError::KernelLaunchFailed`] – if an internal sub-kernel fails.
pub fn bluestein_gpu(data: &mut [Complex64], direction: FftDirection) -> GpuFftResult<()> {
    let n = data.len();
    if n < 2 {
        return Err(GpuFftError::SizeTooSmall(n));
    }
    if n.is_power_of_two() {
        // Fast path: no Bluestein needed.
        let twiddles = compute_twiddles_gpu(n)?;
        return cooley_tukey_gpu(data, direction, &twiddles);
    }

    // Sign of the Bluestein exponent matches DFT direction.
    let sign: f64 = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };

    // Chirp sequence: c[k] = exp(i·π·sign·k²/N)
    let chirp: Vec<Complex64> = (0..n)
        .map(|k| {
            let angle = sign * PI * (k * k) as f64 / n as f64;
            Complex64::new(angle.cos(), angle.sin())
        })
        .collect();

    // Pre-multiply input by chirp.
    let mut a: Vec<Complex64> = data
        .iter()
        .zip(chirp.iter())
        .map(|(&x, &c)| x * c)
        .collect();

    // Choose convolution length M ≥ 2N−1 that is a power of two.
    let m = next_pow2(2 * n - 1);

    // Zero-pad a to length M.
    a.resize(m, Complex64::new(0.0, 0.0));

    // Build the convolution kernel b of length M:
    //   b[0..n]       = conj(chirp[0..n])
    //   b[(M-n+1)..M] = conj(chirp[1..n])  (wrap-around)
    let mut b = vec![Complex64::new(0.0, 0.0); m];
    for k in 0..n {
        b[k] = chirp[k].conj();
    }
    for k in 1..n {
        b[m - k] = chirp[k].conj();
    }

    // FFT of both sequences.
    let tw = compute_twiddles_gpu(m)?;
    cooley_tukey_gpu(&mut a, FftDirection::Forward, &tw)
        .map_err(|e| GpuFftError::KernelLaunchFailed(format!("bluestein sub-fft a: {e}")))?;
    cooley_tukey_gpu(&mut b, FftDirection::Forward, &tw)
        .map_err(|e| GpuFftError::KernelLaunchFailed(format!("bluestein sub-fft b: {e}")))?;

    // Pointwise multiply.
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        *ai = *ai * *bi;
    }

    // IFFT (using Cooley-Tukey, which includes 1/M scaling).
    cooley_tukey_gpu(&mut a, FftDirection::Inverse, &tw)
        .map_err(|e| GpuFftError::KernelLaunchFailed(format!("bluestein ifft: {e}")))?;

    // Multiply output by chirp and scale (the Cooley-Tukey IFFT already divided
    // by M; the 1/N factor for the inverse DFT is applied here when needed).
    let inv_scale = if direction == FftDirection::Inverse {
        1.0 / n as f64
    } else {
        1.0
    };

    for (k, out) in data.iter_mut().enumerate() {
        // The forward DFT Bluestein result has an implicit *M factor from the
        // convolution; since cooley_tukey_gpu's IFFT divided by M, we just
        // multiply by chirp[k].
        *out = a[k] * chirp[k] * inv_scale;
    }

    Ok(())
}

/// Return the smallest power of two ≥ `n`.
fn next_pow2(n: usize) -> usize {
    if n.is_power_of_two() {
        return n;
    }
    1usize << (usize::BITS - n.leading_zeros()) as usize
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiled 1-D FFT
// ─────────────────────────────────────────────────────────────────────────────

/// Tile-based 1-D FFT that processes `data` in chunks of `tile_size`.
///
/// Each tile that is a power of two is handled by Cooley-Tukey; non-power-of-two
/// tiles fall back to Bluestein.  This mirrors the cache-tiling strategy used
/// in GPU shared-memory implementations.
///
/// # Errors
///
/// Propagates any error from the per-tile kernel.
pub fn tiled_fft_1d(
    data: &mut [Complex64],
    tile_size: usize,
    twiddles: &[Complex64],
    direction: FftDirection,
) -> GpuFftResult<()> {
    let n = data.len();
    if n < 2 {
        return Err(GpuFftError::SizeTooSmall(n));
    }

    // If the whole signal fits within a single tile (or is power-of-two),
    // just run the single-pass kernel.
    if n <= tile_size && n.is_power_of_two() {
        return cooley_tukey_gpu(data, direction, twiddles);
    }

    // Process tile by tile.
    let effective_tile = tile_size.max(2);
    let mut offset = 0;
    while offset < n {
        let end = (offset + effective_tile).min(n);
        let chunk = &mut data[offset..end];
        let chunk_n = chunk.len();

        if chunk_n < 2 {
            // A single-element "tile" is already in DFT form.
            offset += effective_tile;
            continue;
        }

        if chunk_n.is_power_of_two() && chunk_n <= twiddles.len() * 2 {
            cooley_tukey_gpu(chunk, direction, twiddles)
                .map_err(|e| GpuFftError::KernelLaunchFailed(format!("tiled chunk: {e}")))?;
        } else {
            bluestein_gpu(chunk, direction)
                .map_err(|e| GpuFftError::KernelLaunchFailed(format!("tiled bluestein: {e}")))?;
        }
        offset += effective_tile;
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Normalisation
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the requested normalisation to `data` (in-place).
pub fn apply_normalization(data: &mut [Complex64], mode: NormalizationMode) {
    let n = data.len();
    if n == 0 {
        return;
    }
    match mode {
        NormalizationMode::None => {}
        NormalizationMode::Forward | NormalizationMode::Backward => {
            let scale = 1.0 / n as f64;
            for x in data.iter_mut() {
                *x = *x * scale;
            }
        }
        NormalizationMode::Ortho => {
            let scale = 1.0 / (n as f64).sqrt();
            for x in data.iter_mut() {
                *x = *x * scale;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn nearly_equal(a: Complex64, b: Complex64) -> bool {
        (a.re - b.re).abs() < EPS && (a.im - b.im).abs() < EPS
    }

    #[test]
    fn test_twiddle_size_zero_fails() {
        assert!(compute_twiddles_gpu(0).is_err());
        assert!(compute_twiddles_gpu(1).is_err());
    }

    #[test]
    fn test_twiddle_size_two() {
        let tw = compute_twiddles_gpu(2).expect("twiddles for n=2");
        assert_eq!(tw.len(), 1);
        // W_2^0 = 1
        assert!(nearly_equal(tw[0], Complex64::new(1.0, 0.0)));
    }

    #[test]
    fn test_bit_reverse_size8() {
        let mut data: Vec<Complex64> = (0..8_u64).map(|i| Complex64::new(i as f64, 0.0)).collect();
        bit_reverse_permute_gpu(&mut data);
        // Expected bit-reversed order for indices 0..8:
        // 0(000→000=0), 1(001→100=4), 2(010→010=2), 3(011→110=6),
        // 4(100→001=1), 5(101→101=5), 6(110→011=3), 7(111→111=7)
        let expected = [0.0, 4.0, 2.0, 6.0, 1.0, 5.0, 3.0, 7.0];
        for (i, &e) in expected.iter().enumerate() {
            assert!(
                (data[i].re - e).abs() < EPS,
                "index {i}: got {}",
                data[i].re
            );
        }
    }

    #[test]
    fn test_butterfly_pass_size2() {
        // Two-point butterfly: [a, b] → [a+b, a−b]
        let mut data = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let twiddles = vec![Complex64::new(1.0, 0.0)];
        butterfly_pass_gpu(&mut data, 1, &twiddles);
        assert!(nearly_equal(data[0], Complex64::new(3.0, 0.0)));
        assert!(nearly_equal(data[1], Complex64::new(-1.0, 0.0)));
    }

    #[test]
    fn test_cooley_tukey_identity() {
        // DFT then IDFT should recover the original signal.
        let original: Vec<Complex64> = (0..8).map(|i| Complex64::new(i as f64, 0.0)).collect();
        let mut data = original.clone();
        let tw = compute_twiddles_gpu(8).expect("twiddles");
        cooley_tukey_gpu(&mut data, FftDirection::Forward, &tw).expect("fft");
        cooley_tukey_gpu(&mut data, FftDirection::Inverse, &tw).expect("ifft");
        for (i, (got, exp)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                (got.re - exp.re).abs() < 1e-10,
                "index {i}: re mismatch got {} exp {}",
                got.re,
                exp.re
            );
        }
    }
}
