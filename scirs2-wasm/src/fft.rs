//! Fast Fourier Transform (FFT) operations for WASM
//!
//! Provides FFT functions accessible from JavaScript/TypeScript via wasm-bindgen.
//! All complex data is represented as interleaved real/imaginary pairs in flat f64 arrays:
//! `[re_0, im_0, re_1, im_1, ...]`
//!
//! This module wraps `scirs2-fft` functions with WASM-compatible interfaces.

use crate::error::WasmError;
use wasm_bindgen::prelude::*;

/// Convert interleaved real/imaginary f64 pairs to `Vec<Complex64>`.
///
/// Input format: `[re_0, im_0, re_1, im_1, ...]`
///
/// # Errors
///
/// Returns `WasmError::InvalidParameter` if the input length is not even.
fn parse_interleaved_complex(
    data: &[f64],
) -> Result<Vec<scirs2_core::numeric::Complex64>, WasmError> {
    if !data.len().is_multiple_of(2) {
        return Err(WasmError::InvalidParameter(
            "Interleaved complex data must have an even number of elements (real/imag pairs)"
                .to_string(),
        ));
    }

    let complex_vec: Vec<scirs2_core::numeric::Complex64> = data
        .chunks_exact(2)
        .map(|pair| scirs2_core::numeric::Complex64::new(pair[0], pair[1]))
        .collect();

    Ok(complex_vec)
}

/// Convert a `Vec<Complex64>` to interleaved real/imaginary f64 pairs.
///
/// Output format: `[re_0, im_0, re_1, im_1, ...]`
fn complex_to_interleaved(data: &[scirs2_core::numeric::Complex64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len() * 2);
    for c in data {
        result.push(c.re);
        result.push(c.im);
    }
    result
}

/// Compute the forward Fast Fourier Transform (FFT) of complex input data.
///
/// # Arguments
///
/// * `data` - Interleaved real/imaginary pairs: `[re_0, im_0, re_1, im_1, ...]`
///
/// # Returns
///
/// Interleaved real/imaginary pairs of the FFT result.
/// The output length equals the input length (same number of interleaved pairs).
///
/// # Errors
///
/// Returns a `JsValue` error if:
/// - The input length is not even (not valid interleaved complex data)
/// - The input is empty
/// - The FFT computation fails
///
/// # Example (JavaScript)
///
/// ```javascript
/// // Signal: [1+0i, 2+0i, 3+0i, 4+0i] as interleaved
/// const input = new Float64Array([1, 0, 2, 0, 3, 0, 4, 0]);
/// const result = fft(input);
/// // result contains interleaved [re, im, re, im, ...]
/// ```
#[wasm_bindgen]
pub fn fft(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    let complex_input = parse_interleaved_complex(data)?;

    let result = scirs2_fft::fft(&complex_input, None)
        .map_err(|e| WasmError::ComputationError(format!("FFT failed: {}", e)))?;

    Ok(complex_to_interleaved(&result))
}

/// Compute the inverse Fast Fourier Transform (IFFT) of complex input data.
///
/// # Arguments
///
/// * `data` - Interleaved real/imaginary pairs: `[re_0, im_0, re_1, im_1, ...]`
///
/// # Returns
///
/// Interleaved real/imaginary pairs of the IFFT result.
/// The output length equals the input length (same number of interleaved pairs).
///
/// # Errors
///
/// Returns a `JsValue` error if:
/// - The input length is not even
/// - The input is empty
/// - The IFFT computation fails
///
/// # Example (JavaScript)
///
/// ```javascript
/// const spectrum = new Float64Array([10, 0, -2, 2, -2, 0, -2, -2]);
/// const signal = ifft(spectrum);
/// ```
#[wasm_bindgen]
pub fn ifft(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    let complex_input = parse_interleaved_complex(data)?;

    let result = scirs2_fft::ifft(&complex_input, None)
        .map_err(|e| WasmError::ComputationError(format!("IFFT failed: {}", e)))?;

    Ok(complex_to_interleaved(&result))
}

/// Compute the FFT of real-valued input data (RFFT).
///
/// This is optimized for real-valued signals. The output contains only the
/// positive-frequency half of the spectrum (plus the DC and Nyquist components),
/// since the negative frequencies are redundant for real input.
///
/// # Arguments
///
/// * `data` - Real-valued input signal as `f64` array.
///
/// # Returns
///
/// Interleaved real/imaginary pairs of the half-spectrum.
/// Output length is `2 * (n/2 + 1)` where `n` is the input length, because
/// the result contains `n/2 + 1` complex values stored as interleaved pairs.
///
/// # Errors
///
/// Returns a `JsValue` error if:
/// - The input is empty
/// - The RFFT computation fails
///
/// # Example (JavaScript)
///
/// ```javascript
/// const signal = new Float64Array([1.0, 2.0, 3.0, 4.0]);
/// const spectrum = rfft(signal);
/// // spectrum has (4/2 + 1) * 2 = 6 elements: 3 complex values as interleaved pairs
/// ```
#[wasm_bindgen]
pub fn rfft(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    if data.is_empty() {
        return Err(WasmError::InvalidParameter("Input data cannot be empty".to_string()).into());
    }

    let result = scirs2_fft::rfft(data, None)
        .map_err(|e| WasmError::ComputationError(format!("RFFT failed: {}", e)))?;

    Ok(complex_to_interleaved(&result))
}

/// Compute the inverse FFT for real-valued output (IRFFT).
///
/// This is the inverse of `rfft`. It takes the positive-frequency half-spectrum
/// and reconstructs the original real-valued signal.
///
/// # Arguments
///
/// * `data` - Interleaved real/imaginary pairs of the half-spectrum:
///   `[re_0, im_0, re_1, im_1, ...]`. The length must be even.
/// * `n` - Length of the output real signal. If 0, it is computed as
///   `2 * (num_complex - 1)` where `num_complex` is the number of complex values.
///
/// # Returns
///
/// Real-valued output signal as `f64` array.
///
/// # Errors
///
/// Returns a `JsValue` error if:
/// - The input length is not even
/// - The input is empty
/// - The IRFFT computation fails
///
/// # Example (JavaScript)
///
/// ```javascript
/// const spectrum = rfft(new Float64Array([1, 2, 3, 4]));
/// const signal = irfft(spectrum, 4);  // Recover original 4-element signal
/// ```
#[wasm_bindgen]
pub fn irfft(data: &[f64], n: usize) -> Result<Vec<f64>, JsValue> {
    let complex_input = parse_interleaved_complex(data)?;

    if complex_input.is_empty() {
        return Err(
            WasmError::InvalidParameter("Input spectrum cannot be empty".to_string()).into(),
        );
    }

    // If n is 0, compute the default output length: 2 * (num_complex - 1)
    let output_len = if n == 0 {
        2 * (complex_input.len() - 1)
    } else {
        n
    };

    let result = scirs2_fft::irfft(&complex_input, Some(output_len))
        .map_err(|e| WasmError::ComputationError(format!("IRFFT failed: {}", e)))?;

    Ok(result)
}

/// Compute the FFT sample frequencies.
///
/// Returns the frequency bin centers in cycles per unit of the sample spacing,
/// following the same convention as NumPy/SciPy.
///
/// # Arguments
///
/// * `n` - Number of samples in the signal (window length).
/// * `d` - Sample spacing (inverse of the sampling rate). Must be positive.
///
/// # Returns
///
/// A vector of length `n` containing the sample frequencies.
/// For even `n`: `[0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)`
/// For odd `n`: `[0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)`
///
/// # Errors
///
/// Returns a `JsValue` error if:
/// - `n` is 0
/// - `d` is not positive
/// - The computation fails
///
/// # Example (JavaScript)
///
/// ```javascript
/// const freqs = fftfreq(8, 0.1);
/// // freqs = [0.0, 1.25, 2.5, 3.75, -5.0, -3.75, -2.5, -1.25]
/// ```
#[wasm_bindgen]
pub fn fftfreq(n: usize, d: f64) -> Result<Vec<f64>, JsValue> {
    if n == 0 {
        return Err(WasmError::InvalidParameter(
            "Number of samples (n) must be positive".to_string(),
        )
        .into());
    }
    if d <= 0.0 {
        return Err(
            WasmError::InvalidParameter("Sample spacing (d) must be positive".to_string()).into(),
        );
    }

    scirs2_fft::fftfreq(n, d)
        .map_err(|e| WasmError::ComputationError(format!("fftfreq failed: {}", e)).into())
}

/// Compute the power spectrum (power spectral density) of a real-valued signal.
///
/// Computes `|FFT(signal)|^2` for each frequency bin. The result represents
/// the distribution of signal power across frequencies.
///
/// # Arguments
///
/// * `data` - Real-valued input signal as `f64` array.
///
/// # Returns
///
/// A vector of length `n/2 + 1` containing the power at each frequency bin
/// (only positive frequencies, since the input is real-valued).
///
/// # Errors
///
/// Returns a `JsValue` error if:
/// - The input is empty
/// - The FFT computation fails
///
/// # Example (JavaScript)
///
/// ```javascript
/// const signal = new Float64Array([1, 0, -1, 0, 1, 0, -1, 0]);
/// const psd = power_spectrum(signal);
/// // psd[2] should have the dominant peak (frequency = 2 cycles per 8 samples)
/// ```
#[wasm_bindgen]
pub fn power_spectrum(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    if data.is_empty() {
        return Err(WasmError::InvalidParameter("Input data cannot be empty".to_string()).into());
    }

    // Compute RFFT to get the half-spectrum (positive frequencies only)
    let spectrum = scirs2_fft::rfft(data, None)
        .map_err(|e| WasmError::ComputationError(format!("Power spectrum FFT failed: {}", e)))?;

    // Compute |c|^2 = re^2 + im^2 for each complex frequency bin
    let power: Vec<f64> = spectrum.iter().map(|c| c.norm_sqr()).collect();

    Ok(power)
}

/// Compute the FFT sample frequencies for real-valued FFT (RFFT).
///
/// Returns the frequency bin centers for the positive-frequency half of the
/// spectrum, which is the output of `rfft`.
///
/// # Arguments
///
/// * `n` - Number of samples in the original signal (window length).
/// * `d` - Sample spacing (inverse of the sampling rate). Must be positive.
///
/// # Returns
///
/// A vector of length `n/2 + 1` containing the sample frequencies for
/// the positive half of the spectrum.
///
/// # Errors
///
/// Returns a `JsValue` error if:
/// - `n` is 0
/// - `d` is not positive
/// - The computation fails
///
/// # Example (JavaScript)
///
/// ```javascript
/// const freqs = rfftfreq(8, 0.1);
/// // freqs = [0.0, 1.25, 2.5, 3.75, 5.0]
/// ```
#[wasm_bindgen]
pub fn rfftfreq(n: usize, d: f64) -> Result<Vec<f64>, JsValue> {
    if n == 0 {
        return Err(WasmError::InvalidParameter(
            "Number of samples (n) must be positive".to_string(),
        )
        .into());
    }
    if d <= 0.0 {
        return Err(
            WasmError::InvalidParameter("Sample spacing (d) must be positive".to_string()).into(),
        );
    }

    scirs2_fft::rfftfreq(n, d)
        .map_err(|e| WasmError::ComputationError(format!("rfftfreq failed: {}", e)).into())
}

/// Shift the zero-frequency component to the center of the spectrum.
///
/// Rearranges the FFT output so that the zero-frequency component is at
/// the center, with negative frequencies on the left and positive on the right.
///
/// # Arguments
///
/// * `data` - FFT output as a flat f64 array (real-valued, e.g. from `power_spectrum`
///   or magnitude array).
///
/// # Returns
///
/// The shifted array with zero-frequency at the center.
///
/// # Errors
///
/// Returns a `JsValue` error if the input is empty.
///
/// # Example (JavaScript)
///
/// ```javascript
/// const data = new Float64Array([0, 1, 2, 3, 4, -4, -3, -2, -1]);
/// const shifted = fftshift(data);
/// // shifted = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
/// ```
#[wasm_bindgen]
pub fn fftshift(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    if data.is_empty() {
        return Err(WasmError::InvalidParameter("Input data cannot be empty".to_string()).into());
    }

    let n = data.len();
    let mid = n.div_ceil(2);
    let mut result = Vec::with_capacity(n);
    result.extend_from_slice(&data[mid..]);
    result.extend_from_slice(&data[..mid]);

    Ok(result)
}

/// Inverse of `fftshift`: shift zero-frequency component back to the beginning.
///
/// This undoes the effect of `fftshift`.
///
/// # Arguments
///
/// * `data` - Shifted spectrum (with zero-frequency at center).
///
/// # Returns
///
/// The unshifted array with zero-frequency at the beginning.
///
/// # Errors
///
/// Returns a `JsValue` error if the input is empty.
///
/// # Example (JavaScript)
///
/// ```javascript
/// const shifted = new Float64Array([-4, -3, -2, -1, 0, 1, 2, 3, 4]);
/// const unshifted = ifftshift(shifted);
/// // unshifted = [0, 1, 2, 3, 4, -4, -3, -2, -1]
/// ```
#[wasm_bindgen]
pub fn ifftshift(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    if data.is_empty() {
        return Err(WasmError::InvalidParameter("Input data cannot be empty".to_string()).into());
    }

    let n = data.len();
    let mid = n / 2;
    let mut result = Vec::with_capacity(n);
    result.extend_from_slice(&data[mid..]);
    result.extend_from_slice(&data[..mid]);

    Ok(result)
}

/// Compute the magnitude (absolute value) of complex interleaved data.
///
/// For each complex value `(re, im)`, computes `sqrt(re^2 + im^2)`.
///
/// # Arguments
///
/// * `data` - Interleaved real/imaginary pairs: `[re_0, im_0, re_1, im_1, ...]`
///
/// # Returns
///
/// A vector of magnitudes, one per complex value. Length is `data.len() / 2`.
///
/// # Errors
///
/// Returns a `JsValue` error if the input length is not even.
///
/// # Example (JavaScript)
///
/// ```javascript
/// // Complex values: [3+4i, 0+1i] => magnitudes: [5.0, 1.0]
/// const data = new Float64Array([3, 4, 0, 1]);
/// const mags = fft_magnitude(data);
/// ```
#[wasm_bindgen]
pub fn fft_magnitude(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    let complex_data = parse_interleaved_complex(data)?;

    let magnitudes: Vec<f64> = complex_data
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();

    Ok(magnitudes)
}

/// Compute the phase angle of complex interleaved data (in radians).
///
/// For each complex value `(re, im)`, computes `atan2(im, re)`.
///
/// # Arguments
///
/// * `data` - Interleaved real/imaginary pairs: `[re_0, im_0, re_1, im_1, ...]`
///
/// # Returns
///
/// A vector of phase angles in radians `[-pi, pi]`, one per complex value.
/// Length is `data.len() / 2`.
///
/// # Errors
///
/// Returns a `JsValue` error if the input length is not even.
///
/// # Example (JavaScript)
///
/// ```javascript
/// // Complex value: [1+1i] => phase: pi/4 (~0.785)
/// const data = new Float64Array([1, 1]);
/// const phases = fft_phase(data);
/// ```
#[wasm_bindgen]
pub fn fft_phase(data: &[f64]) -> Result<Vec<f64>, JsValue> {
    let complex_data = parse_interleaved_complex(data)?;

    let phases: Vec<f64> = complex_data.iter().map(|c| c.im.atan2(c.re)).collect();

    Ok(phases)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_interleaved_complex() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let result = parse_interleaved_complex(&data);
        assert!(result.is_ok());
        let complex = result.expect("should parse successfully");
        assert_eq!(complex.len(), 2);
        assert!((complex[0].re - 1.0).abs() < 1e-10);
        assert!((complex[0].im - 2.0).abs() < 1e-10);
        assert!((complex[1].re - 3.0).abs() < 1e-10);
        assert!((complex[1].im - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_interleaved_complex_odd_length() {
        let data = [1.0, 2.0, 3.0];
        let result = parse_interleaved_complex(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_to_interleaved() {
        use scirs2_core::numeric::Complex64;
        let complex = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
        let result = complex_to_interleaved(&complex);
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
        assert!((result[2] - 3.0).abs() < 1e-10);
        assert!((result[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_fftfreq_basic() {
        let result = fftfreq(8, 1.0);
        assert!(result.is_ok());
        let freqs = result.expect("should compute fftfreq");
        assert_eq!(freqs.len(), 8);
        assert!((freqs[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fftfreq_zero_n() {
        let result = fftfreq(0, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_fftfreq_negative_d() {
        let result = fftfreq(8, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rfftfreq_basic() {
        let result = rfftfreq(8, 1.0);
        assert!(result.is_ok());
        let freqs = result.expect("should compute rfftfreq");
        // n/2 + 1 = 5 frequency bins
        assert_eq!(freqs.len(), 5);
        assert!((freqs[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_fftshift_even() {
        let data = [0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0];
        let result = fftshift(&data);
        assert!(result.is_ok());
        let shifted = result.expect("should shift");
        assert_eq!(shifted, vec![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_fftshift_odd() {
        let data = [0.0, 1.0, 2.0, -2.0, -1.0];
        let result = fftshift(&data);
        assert!(result.is_ok());
        let shifted = result.expect("should shift");
        assert_eq!(shifted, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_ifftshift_even() {
        let data = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let result = ifftshift(&data);
        assert!(result.is_ok());
        let unshifted = result.expect("should unshift");
        assert_eq!(unshifted, vec![0.0, 1.0, 2.0, 3.0, -4.0, -3.0, -2.0, -1.0]);
    }

    #[test]
    fn test_fft_magnitude() {
        let data = [3.0, 4.0, 0.0, 1.0];
        let result = fft_magnitude(&data);
        assert!(result.is_ok());
        let mags = result.expect("should compute magnitude");
        assert_eq!(mags.len(), 2);
        assert!((mags[0] - 5.0).abs() < 1e-10);
        assert!((mags[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fft_phase() {
        let data = [1.0, 0.0, 0.0, 1.0];
        let result = fft_phase(&data);
        assert!(result.is_ok());
        let phases = result.expect("should compute phase");
        assert_eq!(phases.len(), 2);
        assert!((phases[0] - 0.0).abs() < 1e-10);
        assert!((phases[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_power_spectrum_empty() {
        let data: [f64; 0] = [];
        let result = power_spectrum(&data);
        assert!(result.is_err());
    }
}
