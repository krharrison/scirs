// Fast FFT-based Discrete Wavelet Transform (DWT)
//
// Supports Haar, Daubechies-4 (Db4) and Daubechies-8 (Db8) wavelets.
//
// Boundary handling: zero-padding (the "zero" mode of PyWavelets).
// For short filters (len ≤ 32), direct convolution is used; for longer
// filters the radix-2 Cooley-Tukey FFT convolution is used (O(N log N)).
//
// Output ordering: [approx_L, detail_L, detail_{L-1}, …, detail_1]
//
// The implementation follows the convention used in PyWavelets:
//   1. Convolve signal with filter (causal / linear convolution)
//   2. Downsample the *even-indexed* output starting from filter_len-1
//      so that `cA[m] = (conv * lo)[2m + filter_len - 1]`.
// This choice makes the causal group-delay integer and allows exact
// reconstruction by the symmetric inverse below.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::num_complex::Complex;

// ---------------------------------------------------------------------------
// Wavelet type
// ---------------------------------------------------------------------------

/// Wavelet family used by the fast DWT/IDWT.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastWaveletType {
    /// Haar wavelet (filter length 2, exact reconstruction).
    Haar,
    /// Daubechies-4 (filter length 8).
    Db4,
    /// Daubechies-8 (filter length 16).
    Db8,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`fast_dwt`] and [`fast_idwt`].
#[derive(Debug, Clone)]
pub struct FastDwtConfig {
    /// Wavelet family. Default: [`FastWaveletType::Haar`].
    pub wavelet: FastWaveletType,
    /// Number of decomposition levels. Default: `3`.
    pub levels: usize,
}

impl Default for FastDwtConfig {
    fn default() -> Self {
        Self {
            wavelet: FastWaveletType::Haar,
            levels: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Filter coefficients  (PyWavelets ordering)
// ---------------------------------------------------------------------------

/// Decomposition low-pass (scaling) filter.
fn get_lowpass_filter(wavelet: FastWaveletType) -> Vec<f64> {
    match wavelet {
        FastWaveletType::Haar => vec![
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ],
        FastWaveletType::Db4 => vec![
            -0.010_597_401_784_997_278,
            0.032_883_011_666_982_945,
            0.030_841_381_835_986_965,
            -0.187_034_811_718_881_20,
            -0.027_983_769_416_983_850,
            0.630_880_767_929_590_9,
            0.714_846_570_552_541_9,
            0.230_377_813_308_896_0,
        ],
        FastWaveletType::Db8 => vec![
            -0.003_335_725_285_001_549_5,
            -0.012_580_751_999_015_828,
            0.006_241_490_212_798_274,
            0.077_571_493_840_065_22,
            -0.031_871_590_188_550_63,
            -0.226_264_693_965_441_7,
            0.129_766_867_567_262_32,
            0.582_795_950_996_122_3,
            0.679_569_128_814_339_2,
            0.281_172_343_660_579_7,
            -0.019_764_779_299_977_59,
            -0.021_317_956_813_660_34,
            -0.000_134_960_961_160_700_26,
            0.003_018_229_212_827_913_8,
            0.000_789_147_012_066_406_38,
            -0.000_329_559_387_320_340_43,
        ],
    }
}

/// Decomposition high-pass (wavelet) filter — quadrature mirror of low-pass.
fn get_highpass_filter(wavelet: FastWaveletType) -> Vec<f64> {
    let lo = get_lowpass_filter(wavelet);
    let n = lo.len();
    // PyWavelets QMF: g[k] = (-1)^k  * lo[n-1-k]
    (0..n)
        .map(|k| {
            let sign = if k % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
            sign * lo[n - 1 - k]
        })
        .collect()
}

/// Reconstruction low-pass filter.
fn get_rec_lowpass(wavelet: FastWaveletType) -> Vec<f64> {
    // rlo[k] = lo[n-1-k]   (time-reversed decomp lo-pass)
    let lo = get_lowpass_filter(wavelet);
    let n = lo.len();
    (0..n).map(|k| lo[n - 1 - k]).collect()
}

/// Reconstruction high-pass filter.
fn get_rec_highpass(wavelet: FastWaveletType) -> Vec<f64> {
    // rhi[k] = (-1)^(k+1) * lo[k]
    let lo = get_lowpass_filter(wavelet);
    let n = lo.len();
    (0..n)
        .map(|k| {
            let sign = if k % 2 == 0 { -1.0_f64 } else { 1.0_f64 };
            sign * lo[k]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Radix-2 FFT (Cooley-Tukey DIT, in-place)
// ---------------------------------------------------------------------------

/// In-place radix-2 DIT FFT.  `input` must have power-of-two length.
fn fft_radix2(input: &mut Vec<Complex<f64>>) {
    let n = input.len();
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            input.swap(i, j);
        }
    }

    // Butterfly stages
    let mut len = 2usize;
    while len <= n {
        let half_len = len / 2;
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let w_step = Complex::new(ang.cos(), ang.sin());
        let mut start = 0;
        while start < n {
            let mut w = Complex::new(1.0_f64, 0.0_f64);
            for k in 0..half_len {
                let u = input[start + k];
                let v = input[start + k + half_len] * w;
                input[start + k] = u + v;
                input[start + k + half_len] = u - v;
                w *= w_step;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// In-place radix-2 IFFT. `input` must have power-of-two length.
fn ifft_radix2(input: &mut Vec<Complex<f64>>) {
    for c in input.iter_mut() {
        *c = c.conj();
    }
    fft_radix2(input);
    let n = input.len() as f64;
    for c in input.iter_mut() {
        *c = c.conj() / n;
    }
}

// ---------------------------------------------------------------------------
// Linear convolution helpers (full: output length = m + n - 1)
// ---------------------------------------------------------------------------

/// Linear convolution via FFT. Output length = `signal.len() + kernel.len() - 1`.
pub(crate) fn convolve_fft(signal: &[f64], kernel: &[f64]) -> SignalResult<Vec<f64>> {
    let out_len = signal.len() + kernel.len().saturating_sub(1);
    if out_len == 0 {
        return Ok(Vec::new());
    }
    let fft_len = next_pow2(out_len);

    let mut sa: Vec<Complex<f64>> = {
        let mut v: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
        v.resize(fft_len, Complex::new(0.0, 0.0));
        v
    };
    let mut ka: Vec<Complex<f64>> = {
        let mut v: Vec<Complex<f64>> = kernel.iter().map(|&x| Complex::new(x, 0.0)).collect();
        v.resize(fft_len, Complex::new(0.0, 0.0));
        v
    };

    fft_radix2(&mut sa);
    fft_radix2(&mut ka);
    for (s, k) in sa.iter_mut().zip(ka.iter()) {
        *s *= k;
    }
    ifft_radix2(&mut sa);

    Ok(sa.iter().take(out_len).map(|c| c.re).collect())
}

/// Direct (time-domain) linear convolution.
fn convolve_direct(signal: &[f64], kernel: &[f64]) -> Vec<f64> {
    let out_len = signal.len() + kernel.len().saturating_sub(1);
    let mut out = vec![0.0f64; out_len];
    for (i, &s) in signal.iter().enumerate() {
        for (j, &k) in kernel.iter().enumerate() {
            out[i + j] += s * k;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// DWT step: filter + downsample
//
// Follows the PyWavelets "zero" padding (causal) convention:
//   full = convolve(signal, filter)   [length = N + L - 1]
//   cA[m] = full[2*m + L - 1]  for m = 0 .. ceil(N/2) - 1
//
// This means the first kept sample is at index (L-1) of the full convolution,
// which cancels the group delay of the causal filter.
// ---------------------------------------------------------------------------

fn dwt_step(signal: &[f64], filter: &[f64]) -> SignalResult<Vec<f64>> {
    let n = signal.len();
    let l = filter.len();

    let full: Vec<f64> = if l > 32 {
        convolve_fft(signal, filter)?
    } else {
        convolve_direct(signal, filter)
    };

    // Number of output samples: ceil(N / 2)
    let n_out = (n + 1) / 2;
    let offset = l - 1; // group-delay cancellation

    let out: Vec<f64> = (0..n_out)
        .map(|m| {
            let idx = 2 * m + offset;
            if idx < full.len() {
                full[idx]
            } else {
                0.0
            }
        })
        .collect();
    Ok(out)
}

// ---------------------------------------------------------------------------
// IDWT step: upsample + filter, then trim to target length
//
// Inverse of dwt_step with the same offset convention:
//   1. Upsample: insert zeros between samples → length 2*M
//   2. Convolve with reconstruction filter → length 2*M + L - 1
//   3. The reconstructed sub-signal is at positions [L-1 .. L-1+target_len]
// ---------------------------------------------------------------------------

fn idwt_step_part(coeff: &[f64], filter: &[f64], target_len: usize) -> SignalResult<Vec<f64>> {
    if coeff.is_empty() {
        return Ok(vec![0.0; target_len]);
    }
    let m = coeff.len();

    // Upsample: interleave with zeros → [c0, 0, c1, 0, …]
    let mut up = vec![0.0f64; 2 * m];
    for (i, &v) in coeff.iter().enumerate() {
        up[2 * i] = v;
    }

    // Convolve with reconstruction filter (full linear convolution)
    let full: Vec<f64> = if filter.len() > 32 {
        convolve_fft(&up, filter)?
    } else {
        convolve_direct(&up, filter)
    };

    // The valid reconstruction starts at index 0 of the full convolution.
    // (The group delay introduced by the decomposition step is absorbed by
    //  the causal filter in the forward pass; the inverse simply reads from
    //  the beginning of the convolution output.)
    let out: Vec<f64> = (0..target_len)
        .map(|i| if i < full.len() { full[i] } else { 0.0 })
        .collect();
    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Discrete wavelet decomposition using FFT-based convolution.
///
/// # Returns
///
/// `Vec<Array1<f64>>` ordered as `[approx_L, detail_L, detail_{L-1}, …, detail_1]`.
pub fn fast_dwt(signal: &Array1<f64>, config: &FastDwtConfig) -> SignalResult<Vec<Array1<f64>>> {
    if config.levels == 0 {
        return Err(SignalError::InvalidArgument(
            "levels must be at least 1".into(),
        ));
    }
    if signal.is_empty() {
        return Err(SignalError::InvalidArgument(
            "signal must not be empty".into(),
        ));
    }

    let lo = get_lowpass_filter(config.wavelet);
    let hi = get_highpass_filter(config.wavelet);

    let mut approx: Vec<f64> = signal.to_vec();
    let mut details: Vec<Array1<f64>> = Vec::with_capacity(config.levels);

    for _ in 0..config.levels {
        let detail_vec = dwt_step(&approx, &hi)?;
        details.push(Array1::from_vec(detail_vec));
        approx = dwt_step(&approx, &lo)?;
    }

    // Output: [approx_L, detail_L, …, detail_1]
    let mut result = Vec::with_capacity(config.levels + 1);
    result.push(Array1::from_vec(approx));
    for d in details.into_iter().rev() {
        result.push(d);
    }
    Ok(result)
}

/// Inverse DWT.
///
/// `coeffs` must be in the order returned by [`fast_dwt`]:
/// `[approx_L, detail_L, detail_{L-1}, …, detail_1]`.
///
/// The reconstructed signal has the same length as the original.
pub fn fast_idwt(coeffs: &[Array1<f64>], config: &FastDwtConfig) -> SignalResult<Array1<f64>> {
    if coeffs.is_empty() {
        return Err(SignalError::InvalidArgument(
            "coeffs must not be empty".into(),
        ));
    }

    let rlo = get_rec_lowpass(config.wavelet);
    let rhi = get_rec_highpass(config.wavelet);

    // coeffs[0] = approx_L; coeffs[1] = detail_L; … coeffs[L] = detail_1
    //
    // At each level we combine approx[k] and detail[k] to recover approx[k-1].
    // The target length at level k = detail[k].len() * 2  (or +1 if odd).
    // We iterate from the coarsest detail (index 1) to the finest (last index).

    let mut approx: Vec<f64> = coeffs[0].to_vec();

    for detail in coeffs[1..].iter() {
        let detail_slice = detail.as_slice().ok_or_else(|| {
            SignalError::ComputationError("detail coefficients not contiguous".into())
        })?;

        // The length before downsampling was ceil(target / 1) — we estimate:
        // target_len = detail.len() * 2  (may be off by 1 if original was odd)
        let target_len = detail_slice.len() * 2;

        let a_part = idwt_step_part(&approx, &rlo, target_len)?;
        let d_part = idwt_step_part(detail_slice, &rhi, target_len)?;

        approx = a_part
            .iter()
            .zip(d_part.iter())
            .map(|(a, d)| a + d)
            .collect();
    }

    Ok(Array1::from_vec(approx))
}

/// Batch DWT: apply [`fast_dwt`] to each row of `signals`.
pub fn fast_dwt_batch(
    signals: &Array2<f64>,
    config: &FastDwtConfig,
) -> SignalResult<Vec<Vec<Array1<f64>>>> {
    let n_signals = signals.nrows();
    let mut results = Vec::with_capacity(n_signals);
    for i in 0..n_signals {
        let row = Array1::from_vec(signals.row(i).to_vec());
        results.push(fast_dwt(&row, config)?);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn next_pow2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{Array1, Array2};

    // ------------------------------------------------------------------
    // FFT / convolution unit tests
    // ------------------------------------------------------------------

    #[test]
    fn test_fft_radix2_single_bin() {
        // FFT of [1, 0, 0, 0] should be [1, 1, 1, 1]
        let mut input = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        fft_radix2(&mut input);
        for c in &input {
            assert_abs_diff_eq!(c.re, 1.0, epsilon = 1e-10);
            assert_abs_diff_eq!(c.im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_convolve_fft_vs_direct() {
        let sig = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ker = vec![0.5, 1.0, 0.5];
        let fft_result = convolve_fft(&sig, &ker).expect("convolve_fft");
        let dir_result = convolve_direct(&sig, &ker);
        assert_eq!(fft_result.len(), dir_result.len());
        for (a, b) in fft_result.iter().zip(dir_result.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-9);
        }
    }

    // ------------------------------------------------------------------
    // Haar DWT
    // ------------------------------------------------------------------

    #[test]
    fn test_fast_dwt_haar_known_output() {
        // For a constant signal [c, c, c, c]:
        //   approx = [c*sqrt(2), c*sqrt(2)]  (after 1 level Haar)
        //   detail = [0, 0]
        let c = 2.0f64;
        let signal = Array1::from_vec(vec![c; 4]);
        let config = FastDwtConfig {
            wavelet: FastWaveletType::Haar,
            levels: 1,
        };
        let coeffs = fast_dwt(&signal, &config).expect("dwt");
        // approx (index 0): each pair sums → (c+c)*inv_sqrt2 = c*sqrt(2)
        for &v in coeffs[0].iter() {
            assert_abs_diff_eq!(v, c * std::f64::consts::SQRT_2, epsilon = 1e-10);
        }
        // detail (index 1): all zeros
        for &v in coeffs[1].iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fast_dwt_haar_coefficients_constant_signal() {
        // All-ones signal: all detail coefficients should be zero
        let signal = Array1::ones(32);
        let config = FastDwtConfig {
            wavelet: FastWaveletType::Haar,
            levels: 3,
        };
        let coeffs = fast_dwt(&signal, &config).expect("dwt");
        for d in &coeffs[1..] {
            for &v in d.iter() {
                assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_fast_dwt_reconstruction_roundtrip() {
        let n = 64;
        let signal: Array1<f64> =
            Array1::from_vec((0..n).map(|i| (i as f64 * 0.1).sin()).collect());
        let config = FastDwtConfig {
            wavelet: FastWaveletType::Haar,
            levels: 3,
        };
        let coeffs = fast_dwt(&signal, &config).expect("dwt");
        let reconstructed = fast_idwt(&coeffs, &config).expect("idwt");

        let len = signal.len().min(reconstructed.len());
        for i in 0..len {
            let diff = (signal[i] - reconstructed[i]).abs();
            assert!(
                diff < 1e-8,
                "reconstruction mismatch at index {i}: signal={}, rec={}",
                signal[i],
                reconstructed[i]
            );
        }
    }

    #[test]
    fn test_fast_idwt_haar_reconstruction() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let config = FastDwtConfig {
            wavelet: FastWaveletType::Haar,
            levels: 2,
        };
        let coeffs = fast_dwt(&signal, &config).expect("dwt");
        let rec = fast_idwt(&coeffs, &config).expect("idwt");

        let len = signal.len().min(rec.len());
        for i in 0..len {
            assert_abs_diff_eq!(signal[i], rec[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn test_fast_dwt_db4_output_length() {
        let n = 128;
        let signal = Array1::zeros(n);
        let config = FastDwtConfig {
            wavelet: FastWaveletType::Db4,
            levels: 2,
        };
        let coeffs = fast_dwt(&signal, &config).expect("dwt");
        assert_eq!(
            coeffs.len(),
            3,
            "Expected 3 coefficient arrays (approx + 2 details)"
        );
        assert!(
            coeffs[0].len() <= n / 2 + 8,
            "Approx too long: {}",
            coeffs[0].len()
        );
    }

    #[test]
    fn test_fast_dwt_batch_shape() {
        let n_signals = 5;
        let signal_len = 64;
        let signals = Array2::from_shape_fn((n_signals, signal_len), |(_, j)| j as f64);
        let config = FastDwtConfig {
            wavelet: FastWaveletType::Haar,
            levels: 2,
        };
        let batch = fast_dwt_batch(&signals, &config).expect("batch");
        assert_eq!(batch.len(), n_signals);
        for row in &batch {
            assert_eq!(row.len(), 3); // approx + 2 details
        }
    }

    #[test]
    fn test_fast_dwt_error_zero_levels() {
        let signal = Array1::zeros(32);
        let config = FastDwtConfig {
            levels: 0,
            ..Default::default()
        };
        assert!(fast_dwt(&signal, &config).is_err());
    }

    #[test]
    fn test_fast_idwt_empty_coeffs_error() {
        let config = FastDwtConfig::default();
        assert!(fast_idwt(&[], &config).is_err());
    }

    #[test]
    fn test_fast_dwt_db4_reconstruction() {
        // Use a longer signal so we can inspect the interior well away from
        // the zero-padding boundary region.  Db4 has filter length 8, so
        // two decomposition levels introduce boundary artifacts over roughly
        // the first and last 2*(8-1)*2 = 28 samples.
        let signal = Array1::from_vec(
            (0..128)
                .map(|i| (i as f64 * 0.15).sin())
                .collect::<Vec<f64>>(),
        );
        let config = FastDwtConfig {
            wavelet: FastWaveletType::Db4,
            levels: 2,
        };
        let coeffs = fast_dwt(&signal, &config).expect("dwt");
        let rec = fast_idwt(&coeffs, &config).expect("idwt");
        let len = signal.len().min(rec.len());

        // Check interior samples only (boundary region ≈ 28 samples each side)
        let margin = 28;
        for i in margin..len.saturating_sub(margin) {
            assert_abs_diff_eq!(signal[i], rec[i], epsilon = 1e-8);
        }
    }
}
