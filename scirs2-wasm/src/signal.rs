//! Signal processing operations for WASM
//!
//! Provides wasm_bindgen bindings for digital signal processing functions
//! including filter design, convolution, correlation, window functions,
//! FIR filter design, and digital filter application.

use crate::error::WasmError;
use wasm_bindgen::prelude::*;

/// Design a Butterworth digital filter.
///
/// Returns a JSON object with `b` (numerator) and `a` (denominator) coefficients.
///
/// # Arguments
///
/// * `order` - Filter order (positive integer)
/// * `cutoff` - Cutoff frequency, normalized 0..1 where 1 is Nyquist
/// * `btype` - Filter type: "lowpass", "highpass", "bandpass", or "bandstop"
/// * `fs` - Sampling frequency in Hz (cutoff will be normalized by fs/2)
///
/// # Returns
///
/// A JsValue containing a JSON object `{ b: number[], a: number[] }`.
#[wasm_bindgen]
pub fn butter(order: u32, cutoff: f64, btype: &str, fs: f64) -> Result<JsValue, JsValue> {
    if order == 0 {
        return Err(
            WasmError::InvalidParameter("Filter order must be positive".to_string()).into(),
        );
    }
    if fs <= 0.0 {
        return Err(
            WasmError::InvalidParameter("Sampling frequency must be positive".to_string()).into(),
        );
    }

    // Normalize cutoff frequency: cutoff_hz / (fs/2)
    let nyquist = fs / 2.0;
    let normalized_cutoff = cutoff / nyquist;

    if normalized_cutoff <= 0.0 || normalized_cutoff >= 1.0 {
        return Err(WasmError::InvalidParameter(
            "Normalized cutoff frequency must be in the range (0, 1) exclusive".to_string(),
        )
        .into());
    }

    let filter_type_str = match btype {
        "lowpass" | "low" | "lp" => "lowpass",
        "highpass" | "high" | "hp" => "highpass",
        "bandpass" | "band" | "bp" => "bandpass",
        "bandstop" | "bs" | "notch" => "bandstop",
        other => {
            return Err(WasmError::InvalidParameter(format!(
                "Unknown filter type '{}'. Use 'lowpass', 'highpass', 'bandpass', or 'bandstop'",
                other
            ))
            .into());
        }
    };

    let (b_coeffs, a_coeffs) =
        scirs2_signal::filter::butter(order as usize, normalized_cutoff, filter_type_str).map_err(
            |e| WasmError::ComputationError(format!("Butterworth design failed: {}", e)),
        )?;

    let result = serde_json::json!({
        "b": b_coeffs,
        "a": a_coeffs,
    });

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| WasmError::SerializationError(e.to_string()).into())
}

/// Perform 1D convolution of two arrays.
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
/// * `mode` - Output mode: "full", "same", or "valid"
///
/// # Returns
///
/// A `Vec<f64>` containing the convolution result.
#[wasm_bindgen]
pub fn convolve(a: &[f64], b: &[f64], mode: &str) -> Result<Vec<f64>, JsValue> {
    if a.is_empty() || b.is_empty() {
        return Err(
            WasmError::InvalidParameter("Input arrays must not be empty".to_string()).into(),
        );
    }

    validate_conv_mode(mode)?;

    scirs2_signal::convolve(a, b, mode)
        .map_err(|e| WasmError::ComputationError(format!("Convolution failed: {}", e)).into())
}

/// Perform 1D cross-correlation of two arrays.
///
/// Cross-correlation is equivalent to convolution with the second array reversed.
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array
/// * `mode` - Output mode: "full", "same", or "valid"
///
/// # Returns
///
/// A `Vec<f64>` containing the correlation result.
#[wasm_bindgen]
pub fn correlate(a: &[f64], b: &[f64], mode: &str) -> Result<Vec<f64>, JsValue> {
    if a.is_empty() || b.is_empty() {
        return Err(
            WasmError::InvalidParameter("Input arrays must not be empty".to_string()).into(),
        );
    }

    validate_conv_mode(mode)?;

    scirs2_signal::correlate(a, b, mode)
        .map_err(|e| WasmError::ComputationError(format!("Correlation failed: {}", e)).into())
}

/// Generate a Hanning (Hann) window of length `n`.
///
/// The Hann window is a raised cosine window with zero values at both endpoints,
/// providing good frequency resolution and moderate sidelobe suppression.
///
/// # Arguments
///
/// * `n` - Number of points in the window
///
/// # Returns
///
/// A `Vec<f64>` containing the window coefficients.
#[wasm_bindgen]
pub fn hanning(n: usize) -> Result<Vec<f64>, JsValue> {
    if n == 0 {
        return Err(
            WasmError::InvalidParameter("Window length must be positive".to_string()).into(),
        );
    }

    scirs2_signal::window::hann(n, true)
        .map_err(|e| WasmError::ComputationError(format!("Hanning window failed: {}", e)).into())
}

/// Generate a Hamming window of length `n`.
///
/// The Hamming window is a raised cosine window with non-zero endpoints,
/// providing better sidelobe suppression than the Hann window.
///
/// # Arguments
///
/// * `n` - Number of points in the window
///
/// # Returns
///
/// A `Vec<f64>` containing the window coefficients.
#[wasm_bindgen]
pub fn hamming(n: usize) -> Result<Vec<f64>, JsValue> {
    if n == 0 {
        return Err(
            WasmError::InvalidParameter("Window length must be positive".to_string()).into(),
        );
    }

    scirs2_signal::window::hamming(n, true)
        .map_err(|e| WasmError::ComputationError(format!("Hamming window failed: {}", e)).into())
}

/// Generate a Blackman window of length `n`.
///
/// The Blackman window is a three-term cosine series window providing
/// excellent sidelobe suppression at the cost of wider main lobe.
///
/// # Arguments
///
/// * `n` - Number of points in the window
///
/// # Returns
///
/// A `Vec<f64>` containing the window coefficients.
#[wasm_bindgen]
pub fn blackman(n: usize) -> Result<Vec<f64>, JsValue> {
    if n == 0 {
        return Err(
            WasmError::InvalidParameter("Window length must be positive".to_string()).into(),
        );
    }

    scirs2_signal::window::blackman(n, true)
        .map_err(|e| WasmError::ComputationError(format!("Blackman window failed: {}", e)).into())
}

/// Design an FIR (Finite Impulse Response) lowpass filter using the window method.
///
/// Designs a linear-phase FIR lowpass filter with the specified number of taps
/// and cutoff frequency, using a Hamming window.
///
/// # Arguments
///
/// * `n` - Number of filter taps (must be >= 3)
/// * `cutoff` - Cutoff frequency in Hz
/// * `fs` - Sampling frequency in Hz
///
/// # Returns
///
/// A `Vec<f64>` containing the FIR filter coefficients.
#[wasm_bindgen]
pub fn firwin(n: usize, cutoff: f64, fs: f64) -> Result<Vec<f64>, JsValue> {
    if n < 3 {
        return Err(
            WasmError::InvalidParameter("Number of taps must be at least 3".to_string()).into(),
        );
    }
    if fs <= 0.0 {
        return Err(
            WasmError::InvalidParameter("Sampling frequency must be positive".to_string()).into(),
        );
    }

    let nyquist = fs / 2.0;
    let normalized_cutoff = cutoff / nyquist;

    if normalized_cutoff <= 0.0 || normalized_cutoff >= 1.0 {
        return Err(WasmError::InvalidParameter(
            "Normalized cutoff frequency must be in the range (0, 1) exclusive".to_string(),
        )
        .into());
    }

    // Use Hamming window and lowpass (pass_zero=true) by default
    scirs2_signal::filter::firwin(n, normalized_cutoff, "hamming", true)
        .map_err(|e| WasmError::ComputationError(format!("FIR filter design failed: {}", e)).into())
}

/// Apply a digital filter to a signal using direct form II transposed structure.
///
/// Implements causal filtering with the inherent group delay of the filter.
/// Both IIR and FIR filters are supported.
///
/// # Arguments
///
/// * `b` - Numerator (feedforward) coefficients
/// * `a` - Denominator (feedback) coefficients. For FIR filters, use `[1.0]`.
/// * `x` - Input signal
///
/// # Returns
///
/// A `Vec<f64>` containing the filtered signal (same length as input).
#[wasm_bindgen]
pub fn lfilter(b: &[f64], a: &[f64], x: &[f64]) -> Result<Vec<f64>, JsValue> {
    if b.is_empty() {
        return Err(WasmError::InvalidParameter(
            "Numerator coefficients must not be empty".to_string(),
        )
        .into());
    }
    if a.is_empty() {
        return Err(WasmError::InvalidParameter(
            "Denominator coefficients must not be empty".to_string(),
        )
        .into());
    }
    if x.is_empty() {
        return Err(
            WasmError::InvalidParameter("Input signal must not be empty".to_string()).into(),
        );
    }

    scirs2_signal::filter::lfilter(b, a, x)
        .map_err(|e| WasmError::ComputationError(format!("lfilter failed: {}", e)).into())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate convolution/correlation mode string.
fn validate_conv_mode(mode: &str) -> Result<(), JsValue> {
    match mode {
        "full" | "same" | "valid" => Ok(()),
        other => Err(WasmError::InvalidParameter(format!(
            "Unknown mode '{}'. Use 'full', 'same', or 'valid'",
            other
        ))
        .into()),
    }
}
