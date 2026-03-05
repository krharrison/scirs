//! Cosine Window Family
//!
//! This module implements the cosine family of window functions including:
//! - Hann window (raised cosine)
//! - Hamming window (raised cosine with non-zero endpoints)
//! - Blackman window (three-term cosine series)
//! - Blackman-Harris window (four-term cosine series)
//! - Nuttall window (four-term cosine series optimized for minimum sidelobe)
//! - Flat-top window (five-term cosine series for amplitude accuracy)

use super::super::{_extend, _len_guards, _truncate};
use crate::error::{SignalError, SignalResult};
use std::f64::consts::PI;

/// Hann window (raised cosine)
///
/// The Hann window is a taper formed by using a raised cosine.
/// It has zero values at both endpoints and good frequency resolution.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
///
/// # Examples
/// ```
/// use scirs2_signal::window::families::cosine::hann;
/// let window = hann(10, true).expect("operation should succeed");
/// assert_eq!(window.len(), 10);
/// assert!((window[0] - 0.0).abs() < 1e-10); // Zero at endpoints
/// ```
pub fn hann(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos());
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Hamming window (raised cosine with non-zero endpoints)
///
/// The Hamming window is a taper formed by using a raised cosine with
/// non-zero endpoints. It provides good sidelobe suppression.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
///
/// # Examples
/// ```
/// use scirs2_signal::window::families::cosine::hamming;
/// let window = hamming(10, true).expect("operation should succeed");
/// assert_eq!(window.len(), 10);
/// assert!(window[0] > 0.0); // Non-zero at endpoints
/// ```
pub fn hamming(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Blackman window (three-term cosine series)
///
/// The Blackman window is a taper formed by using the first three terms of
/// a summation of cosines. It provides excellent sidelobe suppression.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
///
/// # Examples
/// ```
/// use scirs2_signal::window::families::cosine::blackman;
/// let window = blackman(10, true).expect("operation should succeed");
/// assert_eq!(window.len(), 10);
/// ```
pub fn blackman(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * PI * i as f64 / (n - 1) as f64;
        let w_val = 0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Blackman-Harris window (four-term cosine series)
///
/// The Blackman-Harris window is a generalization of the Blackman window
/// with four cosine terms for improved sidelobe suppression.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn blackmanharris(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    // Blackman-Harris coefficients
    let a0 = 0.35875;
    let a1 = 0.48829;
    let a2 = 0.14128;
    let a3 = 0.01168;

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * PI * i as f64 / (n - 1) as f64;
        let w_val = a0 - a1 * angle.cos() + a2 * (2.0 * angle).cos() - a3 * (3.0 * angle).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Nuttall window (four-term cosine series optimized for minimum sidelobe)
///
/// The Nuttall window is designed to minimize the maximum sidelobe level.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn nuttall(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    // Nuttall coefficients for minimum maximum sidelobe
    let a0 = 0.355768;
    let a1 = 0.487396;
    let a2 = 0.144232;
    let a3 = 0.012604;

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * PI * i as f64 / (n - 1) as f64;
        let w_val = a0 - a1 * angle.cos() + a2 * (2.0 * angle).cos() - a3 * (3.0 * angle).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Flat-top window (five-term cosine series for amplitude accuracy)
///
/// The flat-top window is designed for accurate amplitude measurements
/// in spectrum analysis applications. It has a very flat passband.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn flattop(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    // Flat-top coefficients for amplitude accuracy
    let a0 = 0.21557895;
    let a1 = 0.41663158;
    let a2 = 0.277263158;
    let a3 = 0.083578947;
    let a4 = 0.006947368;

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * PI * i as f64 / (n - 1) as f64;
        let w_val = a0 - a1 * angle.cos() + a2 * (2.0 * angle).cos() - a3 * (3.0 * angle).cos()
            + a4 * (4.0 * angle).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Simple cosine window (single cosine term)
///
/// A single period cosine window, also known as the cosine window.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn cosine(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let w_val = (PI * i as f64 / (n - 1) as f64).sin();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Generalized cosine window with customizable coefficients
///
/// Creates a generalized cosine window with user-specified coefficients.
/// This allows for creating custom windows or exact replicas of standard windows.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `coeffs` - Cosine coefficients [a0, a1, a2, ...]
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
///
/// # Examples
/// ```
/// // Create a Hann window using generalized cosine
/// use scirs2_signal::window::families::cosine::generalized_cosine;
/// let coeffs = vec![0.5, -0.5]; // Hann coefficients
/// let window = generalized_cosine(10, &coeffs, true).expect("operation should succeed");
/// ```
pub fn generalized_cosine(m: usize, coeffs: &[f64], sym: bool) -> SignalResult<Vec<f64>> {
    if coeffs.is_empty() {
        return Err(SignalError::ValueError(
            "At least one coefficient must be provided".to_string(),
        ));
    }

    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let mut w_val = 0.0;
        let base_angle = 2.0 * PI * i as f64 / (n - 1) as f64;

        for (k, &coeff) in coeffs.iter().enumerate() {
            if k == 0 {
                w_val += coeff; // DC component
            } else {
                w_val += coeff * (k as f64 * base_angle).cos();
            }
        }
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Modified Bartlett-Hann window (cosine-based modification of Bartlett)
///
/// A modified version of the Bartlett window using cosine terms.
///
/// # Arguments
/// * `m` - Number of points in the output window
/// * `sym` - If true, generates a symmetric window, otherwise a periodic window
///
/// # Returns
/// A `Vec<f64>` of window values
pub fn barthann(m: usize, sym: bool) -> SignalResult<Vec<f64>> {
    if _len_guards(m) {
        return Ok(vec![1.0; m]);
    }

    let (n, needs_trunc) = _extend(m, sym);

    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let frac = i as f64 / (n - 1) as f64;
        let w_val = 0.62 - 0.48 * (frac - 0.5).abs() + 0.38 * (2.0 * PI * frac).cos();
        w.push(w_val);
    }

    Ok(_truncate(w, needs_trunc))
}

/// Analyze cosine window properties
///
/// Computes important properties of cosine windows for analysis
pub fn analyze_cosine_window(window: &[f64]) -> CosineWindowAnalysis {
    let n = window.len();

    // Coherent gain (DC response)
    let coherent_gain = window.iter().sum::<f64>() / n as f64;

    // Processing gain (equivalent noise bandwidth)
    let sum_squares = window.iter().map(|&w| w * w).sum::<f64>();
    let processing_gain = window.iter().sum::<f64>().powi(2) / (n as f64 * sum_squares);

    // Scalloping loss (worst-case loss between bins)
    let scalloping_loss = estimate_scalloping_loss(window);

    // Main lobe width (approximate)
    let main_lobe_width = estimate_main_lobe_width(window);

    // Maximum sidelobe level (approximate)
    let max_sidelobe_level = estimate_max_sidelobe_level(window);

    CosineWindowAnalysis {
        coherent_gain,
        processing_gain,
        scalloping_loss,
        main_lobe_width,
        max_sidelobe_level,
    }
}

/// Cosine window analysis results
#[derive(Debug, Clone)]
pub struct CosineWindowAnalysis {
    /// Coherent gain (DC response normalized)
    pub coherent_gain: f64,
    /// Processing gain (ratio of coherent to incoherent gain)
    pub processing_gain: f64,
    /// Scalloping loss in dB
    pub scalloping_loss: f64,
    /// Main lobe width in bins
    pub main_lobe_width: f64,
    /// Maximum sidelobe level in dB
    pub max_sidelobe_level: f64,
}

// Helper functions for window analysis

fn estimate_scalloping_loss(window: &[f64]) -> f64 {
    // Simplified scalloping loss estimation
    // In practice would compute FFT and analyze frequency response
    let coherent_gain = window.iter().sum::<f64>() / window.len() as f64;

    // Approximate worst-case loss for cosine windows
    match coherent_gain {
        g if g > 0.5 => 3.92, // Hann-like windows
        g if g > 0.4 => 1.42, // Hamming-like windows
        _ => 0.83,            // Blackman-like windows
    }
}

fn estimate_main_lobe_width(window: &[f64]) -> f64 {
    // Simplified main lobe width estimation based on window characteristics
    let coherent_gain = window.iter().sum::<f64>() / window.len() as f64;

    // Approximate main lobe width for different cosine windows
    match coherent_gain {
        g if g > 0.5 => 4.0, // Hann: ~4 bins
        g if g > 0.4 => 4.0, // Hamming: ~4 bins
        _ => 6.0,            // Blackman and variants: ~6 bins
    }
}

fn estimate_max_sidelobe_level(_window: &[f64]) -> f64 {
    // Would compute actual FFT to get precise sidelobe levels
    // For now, return typical values for common cosine windows
    -40.0 // Approximate for most cosine windows
}
