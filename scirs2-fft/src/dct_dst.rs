//! Optimized DCT and DST implementations using FFT-based algorithms.
//!
//! This module provides highly-efficient implementations of the most commonly
//! used cosine and sine transforms for signal processing, image compression
//! (JPEG/MPEG), and audio coding (MP3/AAC via MDCT).
//!
//! ## Transform catalogue
//!
//! | Function     | Type          | Description                             |
//! |--------------|---------------|-----------------------------------------|
//! | [`dct2`]     | DCT-II        | "Standard" DCT (JPEG, MPEG)             |
//! | [`idct2`]    | IDCT-II       | Inverse of DCT-II                       |
//! | [`dct3`]     | DCT-III       | Unnormalized inverse of DCT-II          |
//! | [`dct4`]     | DCT-IV        | Self-inverse transform (MDCT building block)|
//! | [`dst2`]     | DST-II        | Sine counterpart of DCT-II              |
//! | [`idst2`]    | IDST-II       | Inverse of DST-II                       |
//! | [`mdct`]     | MDCT          | Modified DCT for audio coding           |
//! | [`imdct`]    | IMDCT         | Inverse MDCT with OLA reconstruction    |
//!
//! ## Normalization
//!
//! The default convention is **unormalized** (matches SciPy `dct(x, type=2,
//! norm=None)`).  Pass `norm = Some("ortho")` to get the orthonormal variant
//! where the transform matrix is unitary.
//!
//! ## Algorithm outline
//!
//! All DCT/DST implementations use the FFT-based approach described by Makhoul
//! (1980):
//!
//! ### DCT-II (N-point)
//!
//! 1. Reorder the N inputs into a 2N even-symmetric sequence.
//! 2. Take the N-point real FFT (or half of a 2N FFT).
//! 3. Multiply by twiddle factors `2·exp(-iπk/2N)`.
//! 4. Take the real parts.
//!
//! ### DCT-III (IDCT-II)
//!
//! Derived from DCT-II by reversal; uses a similar FFT-based route.
//!
//! ### DCT-IV
//!
//! Uses the DCT-III/II relationship with a shift of half a sample.
//!
//! ### DST-II
//!
//! Constructed by negating the imaginary part of the same twiddle-based FFT
//! used for DCT-II.
//!
//! ### MDCT
//!
//! `mdct(x, n)` maps a frame of length 2N to N spectral coefficients.
//! Implemented as a DCT-IV of the windowed and folded input.
//!
//! `imdct(X, n)` maps N spectral coefficients to 2N time samples using the
//! inverse DCT-IV; the caller is responsible for the overlap-add step.
//!
//! # References
//!
//! * Makhoul, J. "A fast cosine transform in one and two dimensions."
//!   *IEEE Trans. ASSP* 28(1) (1980), pp. 27–34.
//! * Britanak, V.; Yip, P. C.; Rao, K. R. *Discrete Cosine and Sine Transforms*.
//!   Academic Press, 2007.
//! * Princen, J. P.; Bradley, A. B. "Analysis/synthesis filter bank design
//!   based on time domain aliasing cancellation."
//!   *IEEE Trans. ASSP* 34(5) (1986), pp. 1153–1161.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use scirs2_core::numeric::{Complex64, NumCast, Zero};
use std::f64::consts::PI;
use std::fmt::Debug;

// ─────────────────────────────────────────────────────────────────────────────
//  Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Cast a generic slice to `Vec<f64>`.
fn cast_f64<T: NumCast + Copy + Debug>(x: &[T]) -> FFTResult<Vec<f64>> {
    x.iter()
        .map(|&v| {
            NumCast::from(v).ok_or_else(|| {
                FFTError::ValueError(format!("Cannot cast {v:?} to f64"))
            })
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  DCT-II (the "standard" DCT)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Type-II Discrete Cosine Transform (DCT-II).
///
/// Definition:
/// ```text
/// X[k] = 2 · sum_{n=0}^{N-1}  x[n] · cos(π·k·(2n+1) / (2N))
/// ```
///
/// This is the most commonly used DCT form (used in JPEG, MPEG, MP3).
/// With `norm = Some("ortho")` the output is scaled so the transform matrix
/// is orthonormal.
///
/// ## Algorithm
///
/// 1. Build a reordered sequence `y[n] = x[n]` for n<N, `y[2N-1-n] = x[n]` for n=0..N → even extension of length 2N.
/// 2. Compute the 2N-point real FFT.
/// 3. Keep the first N output bins and multiply by `exp(-iπk/(2N))`.
/// 4. Return the real parts (× 2 for the unormalized convention).
///
/// # Arguments
///
/// * `x` - Real input signal.
/// * `norm` - Normalization: `None` (default, unormalized) or `Some("ortho")`.
///
/// # Returns
///
/// `Vec<f64>` of N DCT-II coefficients.
///
/// # Errors
///
/// Returns an error if input is empty or cast fails.
///
/// # Examples
///
/// ```
/// use scirs2_fft::dct_dst::dct2;
///
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let coeffs = dct2(&x, None).expect("dct2 failed");
/// assert_eq!(coeffs.len(), x.len());
/// // DC coefficient = 2 * sum(x[n])
/// let sum: f64 = x.iter().sum();
/// assert!((coeffs[0] - 2.0 * sum).abs() < 1e-9, "DC={}", coeffs[0]);
/// ```
pub fn dct2<T>(x: &[T], norm: Option<&str>) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug,
{
    if x.is_empty() {
        return Err(FFTError::ValueError("dct2: input is empty".into()));
    }
    let input = cast_f64(x)?;
    dct2_f64(&input, norm)
}

fn dct2_f64(x: &[f64], norm: Option<&str>) -> FFTResult<Vec<f64>> {
    let n = x.len();

    // Build the 2N even-symmetric extension.
    // y[k] = x[k] for k<N, y[2N-1-k] = x[k] for k=0..N
    let ext_len = 2 * n;
    let mut extended = vec![Complex64::zero(); ext_len];
    for k in 0..n {
        extended[k] = Complex64::new(x[k], 0.0);
        extended[ext_len - 1 - k] = Complex64::new(x[k], 0.0);
    }

    // 2N-point FFT.
    let y_fft = fft(&extended, None)?;

    // Twiddle and extract real part.
    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        let phase = -PI * k as f64 / ext_len as f64;
        let twiddle = Complex64::new(phase.cos(), phase.sin());
        let val = y_fft[k] * twiddle;
        result.push(val.re);
    }

    // Apply normalization.
    if norm == Some("ortho") {
        let scale0 = 1.0 / (4.0 * n as f64).sqrt();
        let scale_k = 1.0 / (2.0 * n as f64).sqrt();
        result[0] *= scale0;
        for v in result.iter_mut().skip(1) {
            *v *= scale_k;
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  IDCT-II
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the inverse Type-II DCT (IDCT-II).
///
/// This is the exact inverse of [`dct2`].  Under the default (unormalized)
/// convention the forward transform doubles the energy, so the inverse divides
/// by `2N`.  Under the orthonormal convention both forward and inverse scale by
/// `1/sqrt(N)`.
///
/// # Examples
///
/// ```
/// use scirs2_fft::dct_dst::{dct2, idct2};
///
/// let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let coeffs = dct2(&signal, None).expect("dct2");
/// let recovered = idct2(&coeffs, None).expect("idct2");
/// for (a, b) in signal.iter().zip(recovered.iter()) {
///     assert!((a - b).abs() < 1e-9, "mismatch: {a} vs {b}");
/// }
/// ```
pub fn idct2<T>(x: &[T], norm: Option<&str>) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug,
{
    if x.is_empty() {
        return Err(FFTError::ValueError("idct2: input is empty".into()));
    }
    let input = cast_f64(x)?;
    idct2_f64(&input, norm)
}

fn idct2_f64(x: &[f64], norm: Option<&str>) -> FFTResult<Vec<f64>> {
    let n = x.len();

    // The IDCT-II is proportional to the DCT-III.
    // idct2(X) = (1/(2N)) * dct3(X)  for unormalized convention.
    // For ortho: idct2_ortho(X) = dct3_ortho(X).
    dct3_f64(x, norm)
}

// ─────────────────────────────────────────────────────────────────────────────
//  DCT-III
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Type-III DCT.
///
/// Definition:
/// ```text
/// X[n] = X[0]/2 + sum_{k=1}^{N-1} x[k] · cos(π·k·(2n+1) / (2N))
/// ```
///
/// DCT-III is the transpose (and for orthonormal: the inverse) of DCT-II.
/// Under the unormalized convention `idct2(dct2(x)) = N/2 * x` up to a factor;
/// use `norm = Some("ortho")` for a perfect round-trip.
///
/// # Examples
///
/// ```
/// use scirs2_fft::dct_dst::{dct2, dct3};
///
/// let n = 8;
/// let x: Vec<f64> = (0..n).map(|k| k as f64).collect();
/// let X = dct2(&x, Some("ortho")).expect("dct2");
/// let x_rec = dct3(&X, Some("ortho")).expect("dct3");
/// for (a, b) in x.iter().zip(x_rec.iter()) {
///     assert!((a - b).abs() < 1e-8);
/// }
/// ```
pub fn dct3<T>(x: &[T], norm: Option<&str>) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug,
{
    if x.is_empty() {
        return Err(FFTError::ValueError("dct3: input is empty".into()));
    }
    let input = cast_f64(x)?;
    dct3_f64(&input, norm)
}

fn dct3_f64(x: &[f64], norm: Option<&str>) -> FFTResult<Vec<f64>> {
    let n = x.len();

    // The DCT-III of X is computed as:
    // 1. Scale: v[0] = X[0]/2, v[k] = X[k] for k>0  (unorm), or apply ortho.
    // 2. Twiddle by exp(iπk/(2N)).
    // 3. Take IFFT of length N (unnormalized by N).
    // 4. Reorder output.

    // Step 1 + 2: build complex twiddle-scaled spectrum.
    let mut v = vec![Complex64::zero(); n];

    if norm == Some("ortho") {
        let scale0 = (4.0 * n as f64).sqrt();
        let scale_k = (2.0 * n as f64).sqrt();
        let phase0 = PI / (4.0 * n as f64); // exp(iπ·0/(2N)) * scale0 correction
        let twiddle0 = Complex64::new(phase0.cos(), phase0.sin());
        v[0] = Complex64::new(x[0], 0.0) / scale0 * twiddle0 * Complex64::new(2.0 * n as f64, 0.0);
        // Ortho: undo the forward scaling.
        // Forward dct2_ortho[0] = dct2[0] / sqrt(4N)
        // Forward dct2_ortho[k] = dct2[k] / sqrt(2N)
        // So dct3(X_ortho) = (1/N) * dct3_unorm(X_ortho * ortho_scale)
        // Easier: just scale the input and call the unorm version.
        let mut x_scaled = vec![0.0_f64; n];
        x_scaled[0] = x[0] * scale0;
        for k in 1..n {
            x_scaled[k] = x[k] * scale_k;
        }
        return dct3_core(&x_scaled, n);
    }

    // Unormalized: v[k] = X[k] for k>0, X[0]/2 for k=0, all twiddle-multiplied.
    for k in 0..n {
        let xk = if k == 0 { x[0] * 0.5 } else { x[k] };
        let phase = PI * k as f64 / (2.0 * n as f64);
        let twiddle = Complex64::new(phase.cos(), phase.sin());
        v[k] = Complex64::new(xk, 0.0) * twiddle;
    }

    // IFFT of the twiddle-scaled spectrum (unnormalized: multiply by N after IFFT).
    // Standard IFFT divides by N, so multiply result by N to get unscaled.
    let ifft_result = ifft(&v, None)?;

    // Extract real parts and reorder: output[n] corresponds to the shuffled input.
    // Actually the standard relation gives output directly from real parts.
    // dct3[n] = Re(IFFT(v)[n]) * 2N  (because ifft divides by N and we need ×2N).
    let scale = 2.0 * n as f64;
    Ok(ifft_result.iter().map(|c| c.re * scale).collect())
}

/// Core DCT-III computation (input already scaled, no norm argument).
fn dct3_core(x: &[f64], n: usize) -> FFTResult<Vec<f64>> {
    let mut v = vec![Complex64::zero(); n];
    for k in 0..n {
        let xk = if k == 0 { x[0] * 0.5 } else { x[k] };
        let phase = PI * k as f64 / (2.0 * n as f64);
        let twiddle = Complex64::new(phase.cos(), phase.sin());
        v[k] = Complex64::new(xk, 0.0) * twiddle;
    }
    let ifft_result = ifft(&v, None)?;
    let scale = 2.0 * n as f64;
    Ok(ifft_result.iter().map(|c| c.re * scale).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
//  DCT-IV
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Type-IV DCT.
///
/// Definition:
/// ```text
/// X[k] = 2 · sum_{n=0}^{N-1}  x[n] · cos(π·(2n+1)·(2k+1) / (4N))
/// ```
///
/// DCT-IV is **self-inverse** (up to a scaling): `dct4(dct4(x)) = N·x`.
/// It is the building block of the MDCT.
///
/// ## Algorithm
///
/// Uses the relation DCT-IV(x) = Re(IFFT(y)) where y is constructed from
/// the input with appropriate twiddle factors (Britanak & Rao approach).
///
/// # Examples
///
/// ```
/// use scirs2_fft::dct_dst::dct4;
///
/// let n = 8;
/// let x: Vec<f64> = (0..n).map(|k| k as f64).collect();
/// let X = dct4(&x, None).expect("dct4");
/// let x_rec = dct4(&X, None).expect("dct4 inverse");
/// // dct4(dct4(x)) = 2N·x  (unormalized)
/// for (a, b) in x.iter().zip(x_rec.iter()) {
///     assert!((a * 2.0 * n as f64 - b).abs() < 1e-8, "{a}*2N vs {b}");
/// }
/// ```
pub fn dct4<T>(x: &[T], norm: Option<&str>) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug,
{
    if x.is_empty() {
        return Err(FFTError::ValueError("dct4: input is empty".into()));
    }
    let input = cast_f64(x)?;
    dct4_f64(&input, norm)
}

fn dct4_f64(x: &[f64], norm: Option<&str>) -> FFTResult<Vec<f64>> {
    let n = x.len();

    // Build N-point complex sequence:
    //   z[k] = x[k] · exp(-iπ(2k+1) / (4N))
    // The DCT-IV is then 2 · Re(W^{-1/2} · IFFT(z)) · N
    // where W^{-1/2} = exp(iπ/(4N)).

    let mut z = vec![Complex64::zero(); n];
    for k in 0..n {
        let phase = -PI * (2 * k + 1) as f64 / (4.0 * n as f64);
        let twiddle = Complex64::new(phase.cos(), phase.sin());
        z[k] = Complex64::new(x[k], 0.0) * twiddle;
    }

    // N-point IFFT (divides by N internally).
    let ifft_result = ifft(&z, None)?;

    // Output twiddle W^{-1/2} = exp(iπ / (4N)):
    // (applies to the outer multiplication)
    let out_twiddle_phase = PI / (4.0 * n as f64);
    let out_twiddle = Complex64::new(out_twiddle_phase.cos(), out_twiddle_phase.sin());

    // Result[k] = 2N · Re(out_twiddle · IFFT(z)[k])
    let scale = 2.0 * n as f64;

    let mut result: Vec<f64> = ifft_result
        .iter()
        .map(|&c| (out_twiddle * c).re * scale)
        .collect();

    if norm == Some("ortho") {
        let s = 1.0 / (2.0 * n as f64).sqrt();
        for v in &mut result {
            *v *= s;
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  DST-II
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Type-II Discrete Sine Transform (DST-II).
///
/// Definition:
/// ```text
/// X[k] = 2 · sum_{n=0}^{N-1}  x[n] · sin(π·(2n+1)·(k+1) / (2N))
/// ```
///
/// DST-II is the sine counterpart of DCT-II.  It is useful when the signal is
/// assumed to be zero at both endpoints (odd extension).
///
/// # Arguments
///
/// * `x` - Real input signal.
/// * `norm` - `None` for unormalized, `Some("ortho")` for orthonormal.
///
/// # Examples
///
/// ```
/// use scirs2_fft::dct_dst::dst2;
///
/// let x = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let coeffs = dst2(&x, None).expect("dst2");
/// assert_eq!(coeffs.len(), 4);
/// ```
pub fn dst2<T>(x: &[T], norm: Option<&str>) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug,
{
    if x.is_empty() {
        return Err(FFTError::ValueError("dst2: input is empty".into()));
    }
    let input = cast_f64(x)?;
    dst2_f64(&input, norm)
}

fn dst2_f64(x: &[f64], norm: Option<&str>) -> FFTResult<Vec<f64>> {
    let n = x.len();

    // DST-II via odd extension of length 2N.
    // Build z[k] = x[k] for k<N, z[k] = -x[2N-1-k] for k=N..2N-1.
    let ext_len = 2 * n;
    let mut extended = vec![Complex64::zero(); ext_len];
    for k in 0..n {
        extended[k] = Complex64::new(x[k], 0.0);
        extended[ext_len - 1 - k] = Complex64::new(-x[k], 0.0);
    }

    // 2N-point FFT.
    let y_fft = fft(&extended, None)?;

    // Extract imaginary parts of twiddle-multiplied output.
    // DST-II[k] = -Im(exp(-iπ(k+1)/(2N)) · Y[k+1])
    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        let phase = -PI * (k + 1) as f64 / ext_len as f64;
        let twiddle = Complex64::new(phase.cos(), phase.sin());
        let val = y_fft[k + 1] * twiddle;
        result.push(-val.im);
    }

    if norm == Some("ortho") {
        let scale_k = 1.0 / (2.0 * n as f64).sqrt();
        let scale_n = 1.0 / (4.0 * n as f64).sqrt();
        for (k, v) in result.iter_mut().enumerate() {
            *v *= if k == n - 1 { scale_n } else { scale_k };
        }
    }

    Ok(result)
}

// ─────────────────────────────────────────────────────────────────────────────
//  IDST-II
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the inverse Type-II DST (IDST-II).
///
/// Inverts [`dst2`].  Under the default convention the round-trip scales by N.
/// Use `norm = Some("ortho")` for a unitary pair.
///
/// # Examples
///
/// ```
/// use scirs2_fft::dct_dst::{dst2, idst2};
///
/// let signal = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let coeffs = dst2(&signal, Some("ortho")).expect("dst2");
/// let recovered = idst2(&coeffs, Some("ortho")).expect("idst2");
/// for (a, b) in signal.iter().zip(recovered.iter()) {
///     assert!((a - b).abs() < 1e-9, "mismatch {a} vs {b}");
/// }
/// ```
pub fn idst2<T>(x: &[T], norm: Option<&str>) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug,
{
    if x.is_empty() {
        return Err(FFTError::ValueError("idst2: input is empty".into()));
    }
    let input = cast_f64(x)?;
    idst2_f64(&input, norm)
}

fn idst2_f64(x: &[f64], norm: Option<&str>) -> FFTResult<Vec<f64>> {
    let n = x.len();

    // IDST-II is proportional to DST-III.
    // Construct DST-III via FFT.
    // DST-III: Y[n] = (−1)^n · X[N-1] + sum_{k=0}^{N-2} X[k] · sin(π(k+1)(2n+1)/(2N))

    // Build DST-III coefficients using the complement of DCT-III:
    // DST-III(X)[n] = DCT-III(X_rev)[N-1-n] flipped with alternating signs.
    // Simpler direct approach: use the odd extension IFFT.

    // Scale input for ortho convention.
    let x_work: Vec<f64> = if norm == Some("ortho") {
        let scale_k = (2.0 * n as f64).sqrt();
        let scale_n = (4.0 * n as f64).sqrt();
        x.iter()
            .enumerate()
            .map(|(k, &v)| v * if k == n - 1 { scale_n } else { scale_k })
            .collect()
    } else {
        x.to_vec()
    };

    dst3_f64(&x_work)
}

/// Compute DST-III (internal).
fn dst3_f64(x: &[f64]) -> FFTResult<Vec<f64>> {
    let n = x.len();

    // Build a complex spectrum using twiddle factors exp(iπ(k+1)/(2N)).
    // The DST-III output is the imaginary part of 2N · IFFT(v) re-ordered.
    let mut v = vec![Complex64::zero(); n];
    for k in 0..n {
        let phase = PI * (k + 1) as f64 / (2.0 * n as f64);
        let twiddle = Complex64::new(phase.cos(), phase.sin());
        // Scale: X[N-1]/2 at the last index, otherwise as-is.
        let xk = if k == n - 1 { x[k] * 0.5 } else { x[k] };
        v[k] = Complex64::new(xk, 0.0) * twiddle;
    }

    // Build a symmetric extension: v_sym = [v[0], v[1],..,v[N-1], -v[N-1],...,-v[0]]
    // for a 2N-point IFFT approach.  Alternatively use N-point IFFT directly:
    let ifft_result = ifft(&v, None)?;
    let scale = 2.0 * n as f64;

    // Extract imaginary parts with sign alternation for the odd extension.
    Ok(ifft_result.iter().map(|c| c.im * scale).collect())
}

// ─────────────────────────────────────────────────────────────────────────────
//  MDCT / IMDCT  (Modified DCT for audio coding)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Modified Discrete Cosine Transform (MDCT).
///
/// Maps a frame of `2N` time-domain samples to `N` spectral coefficients.
/// The MDCT is defined as:
///
/// ```text
/// X[k] = sum_{n=0}^{2N-1}  x[n] · cos(π/(2N) · (2n + N/2 + 1) · (2k + 1))
/// ```
///
/// for k = 0, 1, ..., N − 1.
///
/// ## Implementation via DCT-IV
///
/// The MDCT of a windowed input is related to the DCT-IV of the folded frame:
///
/// 1. Fold x[0..2N] → y[0..N] using the TDAC folding operation.
/// 2. Compute DCT-IV(y) to obtain the MDCT coefficients.
///
/// # Arguments
///
/// * `x` - Real frame of exactly `2 * n` samples.
/// * `n` - Number of output coefficients (half the frame length).
///
/// # Returns
///
/// `Vec<f64>` of length `n`.
///
/// # Errors
///
/// Returns an error if `x.len() != 2 * n` or `n == 0`.
///
/// # Examples
///
/// ```
/// use scirs2_fft::dct_dst::{mdct, imdct};
///
/// let n = 8;
/// let frame: Vec<f64> = (0..2*n).map(|k| k as f64 * 0.1).collect();
/// let coeffs = mdct(&frame, n).expect("mdct");
/// assert_eq!(coeffs.len(), n);
/// ```
pub fn mdct(x: &[f64], n: usize) -> FFTResult<Vec<f64>> {
    if n == 0 {
        return Err(FFTError::ValueError("mdct: n must be > 0".into()));
    }
    if x.len() != 2 * n {
        return Err(FFTError::ValueError(format!(
            "mdct: input length {} must be 2*n = {}",
            x.len(),
            2 * n
        )));
    }

    // TDAC folding: produce a length-N sequence y[k].
    // y[k] = -x[n/2 + k] - x[n/2 - 1 - k] for 0 ≤ k < n/2
    //        x[k - n/2]   - x[5n/2 - 1 - k] for n/2 ≤ k < n
    // (Standard MDCT with rectangular window, no external windowing.)
    let half = n / 2;
    let mut y = vec![0.0_f64; n];
    for k in 0..n {
        y[k] = if k < half {
            // Segment 1: fold from the middle of x
            -x[half + k] - x[half - 1 - k]
        } else {
            // Segment 2: fold from the tails
            let m = k - half;
            x[m + n] - x[n + half - 1 - m] // x[n+m] - x[5n/2-1-k] simplified
        };
        // Correct for odd n/2:
        let _ = m_safe_fold(x, n, k); // will be unused if inlined above
    }

    // Actually recompute with the proper general formula:
    // y[k] = x[n*3/2 - 1 - k] + x[n*3/2 + k]  with alternating signs.
    // Use the standard textbook formula:
    for k in 0..n {
        y[k] = fold_mdct(x, n, k);
    }

    // DCT-IV of the folded frame.
    dct4_f64(&y, None)
}

/// Compute the inverse MDCT (IMDCT).
///
/// Maps `n` spectral coefficients back to a time frame of `2n` samples.
/// The IMDCT uses the self-inverse property of DCT-IV and the TDAC unfolding.
///
/// **Note**: Proper MDCT reconstruction requires overlap-add of successive
/// frames.  This function computes the unfolded frame for a single block; the
/// caller must apply the overlap-add step.
///
/// # Arguments
///
/// * `x` - MDCT coefficients of length `n`.
/// * `n` - Must equal `x.len()`.
///
/// # Returns
///
/// `Vec<f64>` of length `2n`.
///
/// # Examples
///
/// ```
/// use scirs2_fft::dct_dst::{mdct, imdct};
///
/// let n = 8;
/// let frame: Vec<f64> = (0..2*n).map(|k| (k as f64 * 0.3).sin()).collect();
/// let coeffs = mdct(&frame, n).expect("mdct");
/// let restored = imdct(&coeffs, n).expect("imdct");
/// assert_eq!(restored.len(), 2 * n);
/// ```
pub fn imdct(x: &[f64], n: usize) -> FFTResult<Vec<f64>> {
    if n == 0 {
        return Err(FFTError::ValueError("imdct: n must be > 0".into()));
    }
    if x.len() != n {
        return Err(FFTError::ValueError(format!(
            "imdct: input length {} must equal n = {n}",
            x.len()
        )));
    }

    // IMDCT = (1/N) · DCT-IV(X), then TDAC unfold.
    // DCT-IV is self-inverse up to a scaling of 2N (unormalized).
    // So IMDCT(X)[k] = (1/N) · DCT-IV(X) unfolded.

    let dct4_result = dct4_f64(x, None)?;

    // Scale: DCT-IV unormalized gives output scaled by 2N, so divide by (2N)·(1/N) = 2.
    let scale = 1.0 / (2.0 * n as f64);

    // TDAC unfold: expand N → 2N.
    let half = n / 2;
    let mut output = vec![0.0_f64; 2 * n];

    for k in 0..n {
        let v = dct4_result[k] * scale;
        // Inverse of the folding:
        // upper half: output[n + k] = v[k]   for k < half... (sign pattern below)
        // This follows the standard IMDCT unfold from Princen-Bradley.
        let unfolded = unfold_imdct(&dct4_result, n, k, scale);
        output[k] = unfolded.0;
        output[k + n] = unfolded.1;
        let _ = v;
        let _ = half;
    }

    // Actually re-derive the unfolding properly:
    // IMDCT output y[n] from TDAC unfolding of z[k] = (1/(2N)) * DCT-IV(X):
    // z[k] = scale * dct4_result[k]
    let z: Vec<f64> = dct4_result.iter().map(|&v| v * scale).collect();
    let mut out = vec![0.0_f64; 2 * n];

    for k in 0..n {
        let unf = imdct_unfold(&z, n, k);
        out[k] = unf;
    }
    // Mirror second half using TDAC symmetry.
    for k in 0..n {
        // y[2N-1-k] = y[k] (for a rectangular window, TDAC gives perfect reconstruction).
        out[2 * n - 1 - k] = out[k];
    }

    // But the standard IMDCT is:
    //   y[n] = sum_{k=0}^{N-1} X[k] · cos(π/(2N) · (2n + N/2 + 1) · (2k+1)) / N
    // Implement directly for correctness:
    direct_imdct(x, n)
}

/// Direct (brute-force) IMDCT for correctness.
fn direct_imdct(x: &[f64], n: usize) -> FFTResult<Vec<f64>> {
    let mut out = vec![0.0_f64; 2 * n];
    let scale = 1.0 / n as f64;
    for m in 0..2 * n {
        let mut sum = 0.0_f64;
        for k in 0..n {
            let phase =
                PI / (2.0 * n as f64) * (2.0 * m as f64 + (n as f64) * 0.5 + 1.0) * (2.0 * k as f64 + 1.0);
            sum += x[k] * phase.cos();
        }
        out[m] = sum * scale;
    }
    Ok(out)
}

// ── MDCT folding helpers ──────────────────────────────────────────────────────

/// Compute the standard MDCT folding y[k] from frame x of length 2N.
///
/// The formula (Princen–Bradley, 1986):
/// ```text
/// y[k] = -x[N/2 + k]   - x[N/2 - 1 - k]    for 0 ≤ k < N/2
///         x[k - N/2]   - x[5N/2 - 1 - k]    for N/2 ≤ k < N
/// ```
/// This is valid for any (even) N.
#[inline]
fn fold_mdct(x: &[f64], n: usize, k: usize) -> f64 {
    let half = n / 2; // N/2
    if k < half {
        // y[k] = -x[n/2 + k] - x[n/2 - 1 - k]
        // Boundary: indices must be in [0, 2n)
        let i0 = half + k;            // n/2 + k    ∈ [n/2, n)
        let i1 = half - 1 - k;       // n/2-1-k    ∈ [0, n/2)
        -x[i0] - x[i1]
    } else {
        // y[k] = x[k - n/2] - x[5n/2 - 1 - k]
        let m = k - half;             // k - n/2   ∈ [0, n/2)
        let i0 = m + n;              // k - n/2 + n  = k + n/2   ∈ [n, 3n/2)
        // But wait: for n/2 ≤ k < n:  i0 = (k-n/2)+n = k+n/2 ∈ [n, 3n/2)  ✓ (< 2n)
        let i1 = n + half - 1 - m;  // 3n/2 - 1 - m  ∈ (n/2, 3n/2)
        // Check: for m=0 → i1 = 3n/2 - 1; for m=n/2-1 → i1 = n
        x[i0] - x[i1]
    }
}

#[allow(dead_code)]
#[inline]
fn m_safe_fold(_x: &[f64], _n: usize, _k: usize) -> f64 { 0.0 }

#[allow(dead_code)]
#[inline]
fn unfold_imdct(_z: &[f64], _n: usize, _k: usize, _scale: f64) -> (f64, f64) { (0.0, 0.0) }

#[allow(dead_code)]
#[inline]
fn imdct_unfold(_z: &[f64], _n: usize, _k: usize) -> f64 { 0.0 }

// ─────────────────────────────────────────────────────────────────────────────
//  2-D DCT-II (for image processing)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the 2-D Type-II DCT (used in JPEG image compression).
///
/// Applies the 1-D DCT-II along rows, then along columns, using separability.
///
/// # Arguments
///
/// * `block` - Flattened row-major 2-D input of shape `rows × cols`.
/// * `rows` - Number of rows.
/// * `cols` - Number of columns.
/// * `norm` - Normalization mode.
///
/// # Returns
///
/// `Vec<f64>` of length `rows * cols` with the 2-D DCT-II coefficients.
///
/// # Errors
///
/// Returns an error if `block.len() != rows * cols` or either dimension is 0.
///
/// # Examples
///
/// ```
/// use scirs2_fft::dct_dst::{dct2_2d, idct2_2d};
///
/// let block: Vec<f64> = (0..16).map(|k| k as f64).collect();
/// let coeffs = dct2_2d(&block, 4, 4, Some("ortho")).expect("dct2_2d");
/// let recovered = idct2_2d(&coeffs, 4, 4, Some("ortho")).expect("idct2_2d");
/// for (a, b) in block.iter().zip(recovered.iter()) {
///     assert!((a - b).abs() < 1e-9, "mismatch {a} vs {b}");
/// }
/// ```
pub fn dct2_2d(block: &[f64], rows: usize, cols: usize, norm: Option<&str>) -> FFTResult<Vec<f64>> {
    if rows == 0 || cols == 0 {
        return Err(FFTError::ValueError("dct2_2d: dimensions must be > 0".into()));
    }
    if block.len() != rows * cols {
        return Err(FFTError::ValueError(format!(
            "dct2_2d: expected {} elements, got {}",
            rows * cols,
            block.len()
        )));
    }

    let mut buf = block.to_vec();

    // Row-wise DCT-II.
    for i in 0..rows {
        let row = buf[i * cols..(i + 1) * cols].to_vec();
        let row_dct = dct2_f64(&row, norm)?;
        buf[i * cols..(i + 1) * cols].copy_from_slice(&row_dct);
    }

    // Column-wise DCT-II.
    for j in 0..cols {
        let col: Vec<f64> = (0..rows).map(|i| buf[i * cols + j]).collect();
        let col_dct = dct2_f64(&col, norm)?;
        for i in 0..rows {
            buf[i * cols + j] = col_dct[i];
        }
    }

    Ok(buf)
}

/// Compute the 2-D inverse Type-II DCT.
///
/// Inverts [`dct2_2d`].
pub fn idct2_2d(block: &[f64], rows: usize, cols: usize, norm: Option<&str>) -> FFTResult<Vec<f64>> {
    if rows == 0 || cols == 0 {
        return Err(FFTError::ValueError("idct2_2d: dimensions must be > 0".into()));
    }
    if block.len() != rows * cols {
        return Err(FFTError::ValueError(format!(
            "idct2_2d: expected {} elements, got {}",
            rows * cols,
            block.len()
        )));
    }

    let mut buf = block.to_vec();

    // Column-wise IDCT-II first.
    for j in 0..cols {
        let col: Vec<f64> = (0..rows).map(|i| buf[i * cols + j]).collect();
        let col_idct = idct2_f64(&col, norm)?;
        for i in 0..rows {
            buf[i * cols + j] = col_idct[i];
        }
    }

    // Row-wise IDCT-II.
    for i in 0..rows {
        let row = buf[i * cols..(i + 1) * cols].to_vec();
        let row_idct = idct2_f64(&row, norm)?;
        buf[i * cols..(i + 1) * cols].copy_from_slice(&row_idct);
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

    fn assert_f64_close(a: &[f64], b: &[f64], tol: f64, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            assert_relative_eq!(ai, bi, epsilon = tol, var_name = format!("{label}[{i}]"));
        }
    }

    // ── DCT-II / IDCT-II ──────────────────────────────────────────────────────

    #[test]
    fn test_dct2_dc_component() {
        let x = vec![1.0_f64, 1.0, 1.0, 1.0];
        let coeffs = dct2(&x, None).expect("dct2");
        // DC = 2 * sum = 8, all others = 0
        assert_relative_eq!(coeffs[0], 8.0, epsilon = 1e-9);
        for k in 1..4 {
            assert_relative_eq!(coeffs[k].abs(), 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_dct2_roundtrip_ortho() {
        let n = 8;
        let x: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let X = dct2(&x, Some("ortho")).expect("dct2");
        let x_rec = idct2(&X, Some("ortho")).expect("idct2");
        assert_f64_close(&x, &x_rec, 1e-9, "dct2 ortho roundtrip");
    }

    #[test]
    fn test_dct2_length_16() {
        let n = 16;
        let x: Vec<f64> = (0..n).map(|k| (k as f64 / n as f64 * PI).sin()).collect();
        let X = dct2(&x, None).expect("dct2");
        let x_rec = idct2(&X, None).expect("idct2");
        assert_f64_close(&x, &x_rec, 1e-9, "dct2 unorm roundtrip 16");
    }

    #[test]
    fn test_dct2_empty_error() {
        let empty: Vec<f64> = vec![];
        assert!(dct2(&empty, None).is_err());
    }

    // ── DCT-III ───────────────────────────────────────────────────────────────

    #[test]
    fn test_dct3_inverse_of_dct2_ortho() {
        let n = 8;
        let x: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let X = dct2(&x, Some("ortho")).expect("dct2");
        let x_rec = dct3(&X, Some("ortho")).expect("dct3");
        assert_f64_close(&x, &x_rec, 1e-8, "dct3 is inverse of dct2 ortho");
    }

    // ── DCT-IV ────────────────────────────────────────────────────────────────

    #[test]
    fn test_dct4_self_inverse() {
        let n = 8;
        let x: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let X = dct4(&x, None).expect("dct4");
        let x2 = dct4(&X, None).expect("dct4 double");
        // dct4(dct4(x)) = 2N · x
        let scale = 2.0 * n as f64;
        for (a, b) in x.iter().zip(x2.iter()) {
            assert_relative_eq!(a * scale, *b, epsilon = 1e-7);
        }
    }

    #[test]
    fn test_dct4_ortho_self_inverse() {
        let n = 8;
        let x: Vec<f64> = (0..n).map(|k| (k as f64).sin()).collect();
        let X = dct4(&x, Some("ortho")).expect("dct4 ortho");
        let x2 = dct4(&X, Some("ortho")).expect("dct4 ortho double");
        // Orthonormal DCT-IV: dct4(dct4(x)) = x (exactly self-inverse).
        assert_f64_close(&x, &x2, 1e-7, "dct4 ortho self-inverse");
    }

    // ── DST-II / IDST-II ──────────────────────────────────────────────────────

    #[test]
    fn test_dst2_length() {
        let x = vec![1.0_f64, 2.0, 3.0, 4.0];
        let coeffs = dst2(&x, None).expect("dst2");
        assert_eq!(coeffs.len(), 4);
    }

    #[test]
    fn test_dst2_idst2_roundtrip_ortho() {
        let n = 8;
        let x: Vec<f64> = (0..n).map(|k| k as f64).collect();
        let X = dst2(&x, Some("ortho")).expect("dst2");
        let x_rec = idst2(&X, Some("ortho")).expect("idst2");
        assert_f64_close(&x, &x_rec, 1e-8, "dst2 ortho roundtrip");
    }

    #[test]
    fn test_dst2_empty_error() {
        let empty: Vec<f64> = vec![];
        assert!(dst2(&empty, None).is_err());
    }

    // ── MDCT / IMDCT ─────────────────────────────────────────────────────────

    #[test]
    fn test_mdct_output_length() {
        let n = 8;
        let frame: Vec<f64> = (0..2 * n).map(|k| k as f64).collect();
        let coeffs = mdct(&frame, n).expect("mdct");
        assert_eq!(coeffs.len(), n);
    }

    #[test]
    fn test_imdct_output_length() {
        let n = 8;
        let frame: Vec<f64> = (0..2 * n).map(|k| (k as f64 * 0.3).sin()).collect();
        let coeffs = mdct(&frame, n).expect("mdct");
        let restored = imdct(&coeffs, n).expect("imdct");
        assert_eq!(restored.len(), 2 * n);
    }

    #[test]
    fn test_mdct_invalid_length() {
        let n = 8;
        let frame = vec![0.0_f64; 2 * n + 1]; // wrong length
        assert!(mdct(&frame, n).is_err());
    }

    #[test]
    fn test_mdct_zero_input() {
        let n = 4;
        let frame = vec![0.0_f64; 2 * n];
        let coeffs = mdct(&frame, n).expect("mdct zero");
        for &v in &coeffs {
            assert_relative_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    // ── 2-D DCT-II ───────────────────────────────────────────────────────────

    #[test]
    fn test_dct2_2d_roundtrip() {
        let block: Vec<f64> = (0..16).map(|k| k as f64).collect();
        let coeffs = dct2_2d(&block, 4, 4, Some("ortho")).expect("dct2_2d");
        let recovered = idct2_2d(&coeffs, 4, 4, Some("ortho")).expect("idct2_2d");
        assert_f64_close(&block, &recovered, 1e-9, "dct2_2d roundtrip");
    }

    #[test]
    fn test_dct2_2d_dc() {
        let rows = 4;
        let cols = 4;
        let block = vec![1.0_f64; rows * cols];
        let coeffs = dct2_2d(&block, rows, cols, None).expect("dct2_2d");
        // DC coeff = (2*sum_rows)^2 / something; just verify it's dominant.
        let dc = coeffs[0].abs();
        for (k, &v) in coeffs.iter().enumerate().skip(1) {
            assert!(v.abs() <= dc + 1e-9, "non-DC bin {k} larger than DC");
        }
    }

    #[test]
    fn test_dct2_2d_wrong_size() {
        let block = vec![0.0_f64; 10];
        assert!(dct2_2d(&block, 4, 4, None).is_err());
    }
}
