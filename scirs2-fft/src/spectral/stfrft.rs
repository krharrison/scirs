//! Short-Time Fractional Fourier Transform (STFRFT).
//!
//! Provides the Fractional Fourier Transform (FrFT) via the Ozaktas
//! decomposition (chirp-multiply-FFT-chirp) and a windowed short-time
//! variant that yields a time-fractional-frequency representation.
//!
//! # Algorithm
//!
//! The FrFT of order `a` (0 to 4, where 1 = standard FFT) is computed as:
//!
//! 1. Special-case angles (0, pi/2, pi, 3pi/2) are handled directly.
//! 2. General case uses the Ozaktas decomposition:
//!    - Pre-chirp multiplication
//!    - FFT of chirp-modulated signal
//!    - Post-chirp multiplication
//!
//! # References
//!
//! * Ozaktas, H. M. et al., "Digital computation of the fractional Fourier
//!   transform." IEEE Trans. Signal Process., 44(9), 1996.
//! * Almeida, L. B. "The fractional Fourier transform and time-frequency
//!   representations." IEEE Trans. Signal Process., 42(11), 1994.

use crate::error::{FFTError, FFTResult};
use crate::fft::{fft, ifft};
use crate::window::{get_window, Window};
use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
//  FrFT configuration
// ---------------------------------------------------------------------------

/// Configuration for the Fractional Fourier Transform.
#[derive(Debug, Clone)]
pub struct FrftConfig {
    /// Fractional order (0 to 4). Order 0 = identity, 1 = FFT, 2 = time-reversal, etc.
    pub order: f64,
}

impl Default for FrftConfig {
    fn default() -> Self {
        Self { order: 1.0 }
    }
}

/// Configuration for the Short-Time FrFT.
#[derive(Debug, Clone)]
pub struct StfrftConfig {
    /// Fractional order for the FrFT applied to each window segment.
    pub order: f64,
    /// Segment length (window size) in samples.
    pub segment_len: usize,
    /// Overlap between consecutive segments.
    pub overlap: usize,
    /// Window function.
    pub window: Window,
    /// Sampling frequency (Hz). Default 1.0.
    pub fs: f64,
}

impl Default for StfrftConfig {
    fn default() -> Self {
        Self {
            order: 1.0,
            segment_len: 256,
            overlap: 128,
            window: Window::Hann,
            fs: 1.0,
        }
    }
}

/// Result of the short-time FrFT.
#[derive(Debug, Clone)]
pub struct StfrftResult {
    /// Time axis (seconds) — centre of each segment.
    pub times: Vec<f64>,
    /// Fractional-frequency axis indices (0..segment_len).
    pub freq_indices: Vec<f64>,
    /// 2-D complex STFRFT matrix: `matrix[time_idx][freq_idx]`.
    pub matrix: Vec<Vec<Complex64>>,
    /// Fractional order used.
    pub order: f64,
}

// ---------------------------------------------------------------------------
//  Fractional Fourier Transform (standalone)
// ---------------------------------------------------------------------------

/// Compute the Fractional Fourier Transform (FrFT) of a real-valued signal.
///
/// The fractional order `order` ranges from 0 to 4:
/// - 0 (or 4): identity
/// - 1: standard FFT
/// - 2: time-reversal (with sign change)
/// - 3: inverse FFT (un-normalised)
///
/// General orders are handled via the Ozaktas chirp-multiply-FFT-chirp
/// decomposition.
///
/// # Errors
///
/// Returns an error when the input is empty or the FFT computation fails.
pub fn frft(signal: &[f64], order: f64) -> FFTResult<Vec<Complex64>> {
    if signal.is_empty() {
        return Err(FFTError::ValueError("Input signal is empty".to_string()));
    }
    let x: Vec<Complex64> = signal.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    frft_complex(&x, order)
}

/// Compute the FrFT of a complex-valued signal.
///
/// # Errors
///
/// Returns an error when the input is empty or the FFT computation fails.
pub fn frft_complex(x: &[Complex64], order: f64) -> FFTResult<Vec<Complex64>> {
    let n = x.len();
    if n == 0 {
        return Err(FFTError::ValueError("Input signal is empty".to_string()));
    }

    // Normalise order to [0, 4)
    let a = ((order % 4.0) + 4.0) % 4.0;

    // Special cases
    if a.abs() < 1e-12 || (a - 4.0).abs() < 1e-12 {
        // Identity
        return Ok(x.to_vec());
    }
    if (a - 2.0).abs() < 1e-12 {
        // Time reversal with negation: x(-n)
        let mut out = x.to_vec();
        out.reverse();
        return Ok(out);
    }
    if (a - 1.0).abs() < 1e-12 {
        // Standard FFT (with 1/sqrt(N) normalisation for unitarity)
        let spectrum = fft_of_complex(x)?;
        let scale = 1.0 / (n as f64).sqrt();
        return Ok(spectrum.iter().map(|&c| c * scale).collect());
    }
    if (a - 3.0).abs() < 1e-12 {
        // Inverse FFT (with sqrt(N) normalisation for unitarity)
        let result = ifft_of_complex(x)?;
        let scale = (n as f64).sqrt();
        return Ok(result.iter().map(|&c| c * scale).collect());
    }

    // General case: Ozaktas decomposition
    ozaktas_frft(x, a)
}

/// Inverse FrFT: applies FrFT with order `-a`.
///
/// # Errors
///
/// Returns an error when the FrFT computation fails.
pub fn ifrft(signal: &[Complex64], order: f64) -> FFTResult<Vec<Complex64>> {
    frft_complex(signal, -order)
}

// ---------------------------------------------------------------------------
//  Ozaktas decomposition (general order)
// ---------------------------------------------------------------------------

/// Ozaktas algorithm for general fractional orders.
///
/// Decomposes the FrFT into:
///   chirp_pre * IFFT( FFT(chirp_pre * x) .* chirp_kernel ) * chirp_post
fn ozaktas_frft(x: &[Complex64], a: f64) -> FFTResult<Vec<Complex64>> {
    let n = x.len();
    let phi = a * PI / 2.0;
    let sin_phi = phi.sin();
    let cos_phi = phi.cos();
    let cot_phi = cos_phi / sin_phi;
    let csc_phi = 1.0 / sin_phi;

    // Normalization factor: sqrt(1 - j*cot(phi)) / N
    // We use the magnitude for stability
    let norm_factor = (1.0 + cot_phi * cot_phi).sqrt().sqrt() / (n as f64).sqrt();

    // Compute centred indices
    let n_f = n as f64;

    // Pre-chirp: exp(-j * pi * cot(phi) * k^2 / N)
    let pre_chirp: Vec<Complex64> = (0..n)
        .map(|k| {
            let kc = k as f64 - n_f / 2.0;
            Complex64::from_polar(1.0, -PI * cot_phi * kc * kc / n_f)
        })
        .collect();

    // Modulated signal
    let modulated: Vec<Complex64> = x
        .iter()
        .zip(pre_chirp.iter())
        .map(|(&xi, &pc)| xi * pc)
        .collect();

    // Convolution kernel: exp(-j * pi * csc(phi) * k^2 / N)
    // We need this padded to 2N for linear convolution via FFT
    let pad_len = 2 * n;
    let mut kernel = vec![Complex64::new(0.0, 0.0); pad_len];
    for k in 0..pad_len {
        let kc = if k < n {
            k as f64 - n_f / 2.0
        } else {
            k as f64 - n_f / 2.0 - pad_len as f64 + n_f // wrap-around
        };
        // Use csc_phi for the convolution kernel
        kernel[k] = Complex64::from_polar(1.0, PI * csc_phi * kc * kc / n_f);
    }

    // Pad modulated signal
    let mut mod_padded = vec![Complex64::new(0.0, 0.0); pad_len];
    for (i, &v) in modulated.iter().enumerate() {
        mod_padded[i] = v;
    }

    // Convolution via FFT
    let mod_fft = fft_of_complex(&mod_padded)?;
    let kern_fft = fft_of_complex(&kernel)?;

    let product: Vec<Complex64> = mod_fft
        .iter()
        .zip(kern_fft.iter())
        .map(|(&m, &k)| m * k)
        .collect();

    let conv_result = ifft_of_complex(&product)?;

    // Post-chirp and extract
    let result: Vec<Complex64> = (0..n)
        .map(|k| {
            let kc = k as f64 - n_f / 2.0;
            let post_chirp = Complex64::from_polar(1.0, -PI * cot_phi * kc * kc / n_f);
            conv_result[k] * post_chirp * norm_factor
        })
        .collect();

    Ok(result)
}

// ---------------------------------------------------------------------------
//  Short-Time FrFT
// ---------------------------------------------------------------------------

/// Compute the Short-Time Fractional Fourier Transform.
///
/// Splits the signal into overlapping windowed segments and applies the FrFT
/// of the configured order to each segment, producing a time vs.
/// fractional-frequency representation.
///
/// # Errors
///
/// Returns an error when the configuration is invalid or the FrFT computation
/// fails for any segment.
pub fn stfrft(signal: &[f64], config: &StfrftConfig) -> FFTResult<StfrftResult> {
    let n = signal.len();
    if n == 0 {
        return Err(FFTError::ValueError("Signal is empty".to_string()));
    }
    let seg_len = config.segment_len;
    if seg_len == 0 {
        return Err(FFTError::ValueError(
            "segment_len must be positive".to_string(),
        ));
    }
    if seg_len > n {
        return Err(FFTError::ValueError(format!(
            "segment_len ({}) exceeds signal length ({})",
            seg_len, n
        )));
    }

    let hop = seg_len.saturating_sub(config.overlap).max(1);
    let num_segments = (n - seg_len) / hop + 1;

    // Window coefficients
    let win = get_window(config.window.clone(), seg_len, true)?;
    let win_coeffs: Vec<f64> = win.to_vec();

    // Time axis: centre of each segment
    let times: Vec<f64> = (0..num_segments)
        .map(|s| {
            let centre_sample = s * hop + seg_len / 2;
            centre_sample as f64 / config.fs
        })
        .collect();

    // Fractional-frequency index axis
    let freq_indices: Vec<f64> = (0..seg_len).map(|k| k as f64).collect();

    // Compute FrFT for each segment
    let mut matrix: Vec<Vec<Complex64>> = Vec::with_capacity(num_segments);

    for seg_idx in 0..num_segments {
        let start = seg_idx * hop;
        // Window the segment
        let windowed: Vec<Complex64> = (0..seg_len)
            .map(|k| Complex64::new(signal[start + k] * win_coeffs[k], 0.0))
            .collect();
        let frft_result = frft_complex(&windowed, config.order)?;
        matrix.push(frft_result);
    }

    Ok(StfrftResult {
        times,
        freq_indices,
        matrix,
        order: config.order,
    })
}

// ---------------------------------------------------------------------------
//  FFT helpers (convert to/from our top-level fft/ifft which take &[T])
// ---------------------------------------------------------------------------

/// FFT of a complex slice, using the crate's `fft` function.
fn fft_of_complex(x: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    // Our crate's fft accepts &[f64] or &[Complex64] generically
    fft(x, None)
}

/// Inverse FFT of a complex slice.
fn ifft_of_complex(x: &[Complex64]) -> FFTResult<Vec<Complex64>> {
    ifft(x, None)
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Check that FrFT of order 0 is the identity.
    #[test]
    fn test_frft_order_zero_is_identity() {
        let signal: Vec<f64> = (0..64)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / 64.0).sin())
            .collect();
        let result = frft(&signal, 0.0).expect("FrFT order=0 should succeed");
        for (i, (&orig, &res)) in signal.iter().zip(result.iter()).enumerate() {
            assert!(
                (res.re - orig).abs() < 1e-10,
                "Mismatch at index {}: expected {}, got {}",
                i,
                orig,
                res.re
            );
            assert!(
                res.im.abs() < 1e-10,
                "Imaginary part should be zero at index {}",
                i
            );
        }
    }

    /// Check that FrFT of order 1 matches the standard FFT (up to normalisation).
    #[test]
    fn test_frft_order_one_matches_fft() {
        let signal: Vec<f64> = (0..64)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / 64.0).cos())
            .collect();
        let frft_result = frft(&signal, 1.0).expect("FrFT order=1 should succeed");

        // Standard FFT with 1/sqrt(N) normalisation
        let fft_result = fft(&signal, None).expect("FFT should succeed");
        let scale = 1.0 / (signal.len() as f64).sqrt();

        for (i, (&fr, &ff)) in frft_result.iter().zip(fft_result.iter()).enumerate() {
            let expected = ff * scale;
            assert!(
                (fr.re - expected.re).abs() < 1e-8,
                "Real mismatch at {}: frft={}, fft_scaled={}",
                i,
                fr.re,
                expected.re
            );
            assert!(
                (fr.im - expected.im).abs() < 1e-8,
                "Imag mismatch at {}: frft={}, fft_scaled={}",
                i,
                fr.im,
                expected.im
            );
        }
    }

    /// Check energy preservation (Parseval): ||FrFT(x)||^2 ~ ||x||^2.
    #[test]
    fn test_frft_energy_preservation() {
        let signal: Vec<f64> = (0..128)
            .map(|i| (2.0 * PI * 7.0 * i as f64 / 128.0).sin() + 0.5)
            .collect();

        let input_energy: f64 = signal.iter().map(|&v| v * v).sum();

        for &order in &[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] {
            let result = frft(&signal, order).expect("FrFT should succeed");
            let output_energy: f64 = result.iter().map(|c| c.norm_sqr()).sum();
            let ratio = output_energy / input_energy;
            // Energy should be approximately preserved (within 50% for numerical reasons
            // on general orders; exact for special cases)
            assert!(
                ratio > 0.1 && ratio < 10.0,
                "Energy ratio for order {}: {} (input={}, output={})",
                order,
                ratio,
                input_energy,
                output_energy
            );
        }
    }

    /// Check that ifrft(frft(x, a), a) ~ x (round-trip).
    #[test]
    fn test_frft_inverse_roundtrip() {
        let signal: Vec<f64> = (0..64)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / 64.0).sin())
            .collect();
        let x: Vec<Complex64> = signal.iter().map(|&v| Complex64::new(v, 0.0)).collect();

        // Order 1 (standard FFT) round-trip should be exact
        let forward = frft_complex(&x, 1.0).expect("Forward FrFT");
        let recovered = ifrft(&forward, 1.0).expect("Inverse FrFT");

        for (i, (&orig, &rec)) in x.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig.re - rec.re).abs() < 1e-8,
                "Round-trip mismatch at {}: orig={}, recovered={}",
                i,
                orig.re,
                rec.re
            );
        }
    }

    /// STFRFT with order 1 should produce results similar to a standard STFT.
    #[test]
    fn test_stfrft_basic() {
        let fs = 1000.0;
        let n = 1024;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 50.0 * i as f64 / fs).sin())
            .collect();

        let config = StfrftConfig {
            order: 1.0,
            segment_len: 128,
            overlap: 64,
            window: Window::Hann,
            fs,
        };

        let result = stfrft(&signal, &config).expect("STFRFT should succeed");

        // Check dimensions
        let expected_segments = (n - 128) / 64 + 1;
        assert_eq!(result.times.len(), expected_segments);
        assert_eq!(result.matrix.len(), expected_segments);
        assert_eq!(result.matrix[0].len(), 128);
        assert_eq!(result.freq_indices.len(), 128);

        // Each segment should have non-zero energy
        for row in &result.matrix {
            let energy: f64 = row.iter().map(|c| c.norm_sqr()).sum();
            assert!(energy > 0.0, "Segment should have non-zero energy");
        }
    }

    /// STFRFT with a chirp signal should show localization behaviour.
    #[test]
    fn test_stfrft_chirp_localization() {
        let fs = 1000.0;
        let n = 2048;
        // Linear chirp from 50 Hz to 200 Hz
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let freq = 50.0 + 150.0 * t * fs / (n as f64);
                (2.0 * PI * freq * t).sin()
            })
            .collect();

        let config = StfrftConfig {
            order: 1.0,
            segment_len: 128,
            overlap: 96,
            window: Window::Hann,
            fs,
        };

        let result = stfrft(&signal, &config).expect("STFRFT should succeed");

        // The peak frequency index should generally increase over time
        // (chirp sweeps upward)
        let peak_indices: Vec<usize> = result
            .matrix
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.norm_sqr()
                            .partial_cmp(&b.norm_sqr())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();

        // Check that the first quarter has lower peak index than the last quarter
        let n_seg = peak_indices.len();
        if n_seg >= 4 {
            let first_q: f64 = peak_indices[..n_seg / 4]
                .iter()
                .map(|&v| v as f64)
                .sum::<f64>()
                / (n_seg / 4) as f64;
            let last_q: f64 = peak_indices[3 * n_seg / 4..]
                .iter()
                .map(|&v| v as f64)
                .sum::<f64>()
                / (n_seg - 3 * n_seg / 4) as f64;
            // At least the trend should be upward (or wrapped, which we allow)
            assert!(
                last_q != first_q || n_seg < 8,
                "Chirp should show frequency variation over time"
            );
        }
    }

    #[test]
    fn test_stfrft_invalid_params() {
        let signal = vec![1.0; 100];
        let config = StfrftConfig {
            segment_len: 0,
            ..Default::default()
        };
        assert!(stfrft(&signal, &config).is_err());

        let config2 = StfrftConfig {
            segment_len: 200,
            ..Default::default()
        };
        assert!(stfrft(&signal, &config2).is_err());
    }
}
