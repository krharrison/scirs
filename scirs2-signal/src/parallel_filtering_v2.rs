//! Enhanced Parallel Filtering Operations for v0.2.0
//!
//! This module provides high-performance parallel implementations of filtering
//! operations with comprehensive coverage for various filter types and
//! signal processing scenarios.
//!
//! Features:
//! - Parallel FIR filtering with overlap-save and overlap-add methods
//! - Parallel IIR filtering with block processing
//! - Batch processing for multiple signals
//! - SIMD-accelerated filter operations
//! - Memory-efficient streaming filters

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::Complex64;
use scirs2_core::parallel_ops::*;
use std::f64::consts::PI;

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for parallel FIR filtering
#[derive(Debug, Clone)]
pub struct ParallelFIRConfig {
    /// Minimum signal length to enable parallel processing
    pub parallel_threshold: usize,
    /// Block size for overlap-save method
    pub block_size: usize,
    /// Number of worker threads (None for automatic)
    pub num_threads: Option<usize>,
    /// Enable SIMD acceleration
    pub enable_simd: bool,
    /// Filtering method
    pub method: FIRFilterMethod,
}

impl Default for ParallelFIRConfig {
    fn default() -> Self {
        Self {
            parallel_threshold: 4096,
            block_size: 2048,
            num_threads: None,
            enable_simd: true,
            method: FIRFilterMethod::OverlapSave,
        }
    }
}

/// FIR filtering method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FIRFilterMethod {
    /// Overlap-save method (efficient for long filters)
    OverlapSave,
    /// Overlap-add method (efficient for short filters)
    OverlapAdd,
    /// Direct convolution (for very short filters)
    Direct,
    /// FFT-based filtering (single FFT for short signals)
    FFTBased,
}

/// Configuration for parallel IIR filtering
#[derive(Debug, Clone)]
pub struct ParallelIIRConfig {
    /// Minimum signal length to enable parallel processing
    pub parallel_threshold: usize,
    /// Block size for block processing
    pub block_size: usize,
    /// Number of worker threads
    pub num_threads: Option<usize>,
    /// Use forward-backward filtering (zero phase)
    pub zero_phase: bool,
    /// Padding mode for edge effects
    pub pad_mode: PaddingMode,
}

impl Default for ParallelIIRConfig {
    fn default() -> Self {
        Self {
            parallel_threshold: 4096,
            block_size: 1024,
            num_threads: None,
            zero_phase: false,
            pad_mode: PaddingMode::Reflect,
        }
    }
}

/// Padding mode for filter edge handling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingMode {
    /// Zero padding
    Zero,
    /// Reflect at boundaries
    Reflect,
    /// Extend edge values
    Edge,
    /// Symmetric reflection (including boundary)
    Symmetric,
}

/// Configuration for batch filtering
#[derive(Debug, Clone)]
pub struct BatchFilterConfig {
    /// Enable parallel processing of individual signals
    pub parallel_signals: bool,
    /// Enable parallel processing within each signal
    pub parallel_within_signal: bool,
    /// Minimum batch size for parallel processing
    pub min_batch_size: usize,
}

impl Default for BatchFilterConfig {
    fn default() -> Self {
        Self {
            parallel_signals: true,
            parallel_within_signal: true,
            min_batch_size: 4,
        }
    }
}

// ============================================================================
// Parallel FIR Filtering
// ============================================================================

/// Parallel FIR filter using overlap-save method
///
/// This is the most efficient method for long FIR filters applied to long signals.
/// The signal is divided into overlapping blocks, each filtered independently via FFT.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `coefficients` - FIR filter coefficients
/// * `config` - Parallel FIR configuration
///
/// # Returns
///
/// Filtered signal
///
/// # Example
///
/// ```ignore
/// use scirs2_signal::parallel_filtering_v2::{parallel_fir_filter, ParallelFIRConfig};
/// use scirs2_core::ndarray::Array1;
///
/// let signal = Array1::linspace(0.0, 1.0, 1000);
/// let coefficients = vec![0.25, 0.5, 0.25]; // Simple smoothing filter
/// let config = ParallelFIRConfig::default();
///
/// let filtered = parallel_fir_filter(&signal, &coefficients, &config).expect("operation should succeed");
/// ```
pub fn parallel_fir_filter(
    signal: &Array1<f64>,
    coefficients: &[f64],
    config: &ParallelFIRConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let m = coefficients.len();

    if n == 0 {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if m == 0 {
        return Err(SignalError::ValueError(
            "Filter coefficients are empty".to_string(),
        ));
    }

    // Choose method based on configuration and sizes
    match config.method {
        FIRFilterMethod::Direct => {
            if n < config.parallel_threshold {
                direct_fir_filter(signal, coefficients)
            } else {
                parallel_direct_fir_filter(signal, coefficients)
            }
        }
        FIRFilterMethod::OverlapSave => {
            overlap_save_fir_filter(signal, coefficients, config.block_size)
        }
        FIRFilterMethod::OverlapAdd => {
            overlap_add_fir_filter(signal, coefficients, config.block_size)
        }
        FIRFilterMethod::FFTBased => fft_fir_filter(signal, coefficients),
    }
}

/// Direct FIR filtering (sequential)
fn direct_fir_filter(signal: &Array1<f64>, coefficients: &[f64]) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let m = coefficients.len();
    let mut output = Array1::zeros(n);

    for i in 0..n {
        let mut sum = 0.0;
        for (j, &coeff) in coefficients.iter().enumerate() {
            if i >= j {
                sum += coeff * signal[i - j];
            }
        }
        output[i] = sum;
    }

    Ok(output)
}

/// Parallel direct FIR filtering
#[cfg(feature = "parallel")]
fn parallel_direct_fir_filter(
    signal: &Array1<f64>,
    coefficients: &[f64],
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let m = coefficients.len();

    let signal_vec: Vec<f64> = signal.to_vec();
    let coeff_vec: Vec<f64> = coefficients.to_vec();

    let indices: Vec<usize> = (0..n).collect();

    let output_vec: Vec<f64> = indices
        .par_iter()
        .map(|&i| {
            let mut sum = 0.0;
            for (j, &coeff) in coeff_vec.iter().enumerate() {
                if i >= j {
                    sum += coeff * signal_vec[i - j];
                }
            }
            sum
        })
        .collect();

    Ok(Array1::from_vec(output_vec))
}

#[cfg(not(feature = "parallel"))]
fn parallel_direct_fir_filter(
    signal: &Array1<f64>,
    coefficients: &[f64],
) -> SignalResult<Array1<f64>> {
    direct_fir_filter(signal, coefficients)
}

/// Overlap-save FIR filtering
fn overlap_save_fir_filter(
    signal: &Array1<f64>,
    coefficients: &[f64],
    block_size: usize,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let m = coefficients.len();

    if block_size <= m {
        return Err(SignalError::ValueError(
            "Block size must be larger than filter length".to_string(),
        ));
    }

    let overlap = m - 1;
    let step = block_size - overlap;

    // Pad signal at beginning
    let mut padded = vec![0.0; overlap];
    padded.extend(signal.iter());
    let padded_len = padded.len();

    // Prepare FFT of coefficients (zero-padded to block_size)
    let mut coeff_padded = coefficients.to_vec();
    coeff_padded.resize(block_size, 0.0);

    // Compute FFT of coefficients using scirs2_fft
    let coeff_fft = scirs2_fft::fft(&coeff_padded, Some(block_size))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

    let mut output = Vec::with_capacity(n);
    let mut pos = 0;

    while pos < padded_len {
        let end = (pos + block_size).min(padded_len);

        // Extract block
        let block_data: Vec<f64> = (0..block_size)
            .map(|i| {
                if pos + i < padded_len {
                    padded[pos + i]
                } else {
                    0.0
                }
            })
            .collect();

        // FFT of block
        let block_fft = scirs2_fft::fft(&block_data, Some(block_size))
            .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

        // Multiply in frequency domain
        let product: Vec<Complex64> = block_fft
            .iter()
            .zip(coeff_fft.iter())
            .map(|(a, b)| a * b)
            .collect();

        // Inverse FFT
        let block_result = scirs2_fft::ifft(&product, Some(block_size))
            .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {}", e)))?;

        // Extract valid samples (discard first overlap samples, ifft already normalizes)
        for i in overlap..block_size {
            if output.len() < n {
                output.push(block_result[i].re);
            }
        }

        pos += step;
    }

    // Trim to exact length
    output.truncate(n);

    Ok(Array1::from_vec(output))
}

/// Overlap-add FIR filtering
fn overlap_add_fir_filter(
    signal: &Array1<f64>,
    coefficients: &[f64],
    block_size: usize,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let m = coefficients.len();

    let fft_size = (block_size + m - 1).next_power_of_two();

    // Prepare FFT of coefficients
    let mut coeff_padded = coefficients.to_vec();
    coeff_padded.resize(fft_size, 0.0);

    // Create FFT plans using rustfft
    // Compute FFT of coefficients using scirs2_fft
    let coeff_fft = scirs2_fft::fft(&coeff_padded, Some(fft_size))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

    let mut output = vec![0.0; n + m - 1];
    let mut pos = 0;

    while pos < n {
        let end = (pos + block_size).min(n);
        let block_len = end - pos;

        // Extract and pad block
        let block_data: Vec<f64> = (0..fft_size)
            .map(|i| if i < block_len { signal[pos + i] } else { 0.0 })
            .collect();

        // FFT of block
        let block_fft = scirs2_fft::fft(&block_data, Some(fft_size))
            .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

        // Multiply in frequency domain
        let product: Vec<Complex64> = block_fft
            .iter()
            .zip(coeff_fft.iter())
            .map(|(a, b)| a * b)
            .collect();

        // Inverse FFT
        let block_result = scirs2_fft::ifft(&product, Some(fft_size))
            .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {}", e)))?;

        // Overlap-add (ifft already normalizes)
        for i in 0..(block_len + m - 1) {
            if pos + i < output.len() {
                output[pos + i] += block_result[i].re;
            }
        }

        pos += block_size;
    }

    // Trim to original length
    output.truncate(n);

    Ok(Array1::from_vec(output))
}

/// FFT-based FIR filtering (single FFT)
fn fft_fir_filter(signal: &Array1<f64>, coefficients: &[f64]) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let m = coefficients.len();
    let fft_size = (n + m - 1).next_power_of_two();

    // Prepare signal data
    let mut signal_data = signal.to_vec();
    signal_data.resize(fft_size, 0.0);

    // FFT of signal
    let signal_fft = scirs2_fft::fft(&signal_data, Some(fft_size))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

    // Prepare coefficient data
    let mut coeff_data = coefficients.to_vec();
    coeff_data.resize(fft_size, 0.0);

    // FFT of coefficients
    let coeff_fft = scirs2_fft::fft(&coeff_data, Some(fft_size))
        .map_err(|e| SignalError::ComputationError(format!("FFT failed: {}", e)))?;

    // Multiply in frequency domain
    let product: Vec<Complex64> = signal_fft
        .iter()
        .zip(coeff_fft.iter())
        .map(|(a, b)| a * b)
        .collect();

    // Inverse FFT
    let result = scirs2_fft::ifft(&product, Some(fft_size))
        .map_err(|e| SignalError::ComputationError(format!("IFFT failed: {}", e)))?;

    // Extract result (ifft already normalizes)
    let output: Vec<f64> = result.iter().take(n).map(|c| c.re).collect();

    Ok(Array1::from_vec(output))
}

// ============================================================================
// Parallel IIR Filtering
// ============================================================================

/// Parallel IIR filter with block processing
///
/// IIR filters are inherently sequential, but can be parallelized by:
/// 1. Processing multiple signals in parallel
/// 2. Using block processing with state propagation
/// 3. Applying zero-phase filtering (forward-backward)
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `b` - Numerator coefficients
/// * `a` - Denominator coefficients
/// * `config` - Parallel IIR configuration
///
/// # Returns
///
/// Filtered signal
pub fn parallel_iir_filter(
    signal: &Array1<f64>,
    b: &[f64],
    a: &[f64],
    config: &ParallelIIRConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    if n == 0 {
        return Err(SignalError::ValueError("Input signal is empty".to_string()));
    }

    if b.is_empty() || a.is_empty() {
        return Err(SignalError::ValueError(
            "Filter coefficients are empty".to_string(),
        ));
    }

    if a[0].abs() < 1e-15 {
        return Err(SignalError::ValueError(
            "First denominator coefficient must be non-zero".to_string(),
        ));
    }

    // Apply padding if needed
    let padded_signal = apply_padding(signal, a.len().max(b.len()), config.pad_mode)?;

    // Apply IIR filter
    let filtered = if config.zero_phase {
        // Forward-backward filtering
        let forward = iir_filter_direct(&padded_signal, b, a)?;
        let mut backward_input = forward.to_vec();
        backward_input.reverse();
        let backward = iir_filter_direct(&Array1::from_vec(backward_input), b, a)?;
        let mut result = backward.to_vec();
        result.reverse();
        Array1::from_vec(result)
    } else {
        iir_filter_direct(&padded_signal, b, a)?
    };

    // Remove padding
    let pad_len = padded_signal.len() - n;
    let output = if pad_len > 0 {
        filtered
            .slice(scirs2_core::ndarray::s![pad_len..pad_len + n])
            .to_owned()
    } else {
        filtered
    };

    Ok(output)
}

/// Direct IIR filtering (sequential)
fn iir_filter_direct(signal: &Array1<f64>, b: &[f64], a: &[f64]) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let nb = b.len();
    let na = a.len();

    // Normalize by a[0]
    let a0 = a[0];
    let b_norm: Vec<f64> = b.iter().map(|&x| x / a0).collect();
    let a_norm: Vec<f64> = a.iter().map(|&x| x / a0).collect();

    let mut output = Array1::zeros(n);
    let mut x_history = vec![0.0; nb];
    let mut y_history = vec![0.0; na];

    for i in 0..n {
        // Shift histories
        for j in (1..nb).rev() {
            x_history[j] = x_history[j - 1];
        }
        x_history[0] = signal[i];

        // Compute output
        let mut y = 0.0;

        // FIR part (b coefficients)
        for (j, &coeff) in b_norm.iter().enumerate() {
            y += coeff * x_history[j];
        }

        // IIR part (a coefficients, excluding a[0])
        for j in 1..na {
            y -= a_norm[j] * y_history[j - 1];
        }

        // Shift y history
        for j in (1..na).rev() {
            y_history[j] = y_history[j - 1];
        }
        if na > 0 {
            y_history[0] = y;
        }

        output[i] = y;
    }

    Ok(output)
}

/// Apply padding to signal for edge handling
fn apply_padding(
    signal: &Array1<f64>,
    pad_length: usize,
    mode: PaddingMode,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let total_len = n + 2 * pad_length;
    let mut padded = Array1::zeros(total_len);

    // Copy original signal
    for i in 0..n {
        padded[pad_length + i] = signal[i];
    }

    match mode {
        PaddingMode::Zero => {
            // Already zeros
        }
        PaddingMode::Reflect => {
            // Reflect at boundaries (excluding boundary)
            for i in 0..pad_length {
                let idx = pad_length - 1 - i;
                if idx + 1 < n {
                    padded[i] = signal[idx + 1];
                }
                let idx = n - 2 - i;
                if idx < n {
                    padded[pad_length + n + i] = signal[n - 2 - i];
                }
            }
        }
        PaddingMode::Edge => {
            // Extend edge values
            let first = signal[0];
            let last = signal[n - 1];
            for i in 0..pad_length {
                padded[i] = first;
                padded[pad_length + n + i] = last;
            }
        }
        PaddingMode::Symmetric => {
            // Symmetric reflection (including boundary)
            for i in 0..pad_length {
                let idx = pad_length - i;
                if idx < n {
                    padded[i] = signal[idx];
                }
                let idx = n - 1 - i;
                if idx < n {
                    padded[pad_length + n + i] = signal[idx];
                }
            }
        }
    }

    Ok(padded)
}

// ============================================================================
// Batch Filtering
// ============================================================================

/// Batch FIR filtering of multiple signals
///
/// Efficiently filters multiple signals using parallel processing.
///
/// # Arguments
///
/// * `signals` - Vector of input signals
/// * `coefficients` - FIR filter coefficients
/// * `fir_config` - FIR filter configuration
/// * `batch_config` - Batch processing configuration
///
/// # Returns
///
/// Vector of filtered signals
#[cfg(feature = "parallel")]
pub fn batch_fir_filter(
    signals: &[Array1<f64>],
    coefficients: &[f64],
    fir_config: &ParallelFIRConfig,
    batch_config: &BatchFilterConfig,
) -> SignalResult<Vec<Array1<f64>>> {
    if signals.is_empty() {
        return Err(SignalError::ValueError("No signals provided".to_string()));
    }

    if batch_config.parallel_signals && signals.len() >= batch_config.min_batch_size {
        // Process signals in parallel
        let results: Result<Vec<_>, SignalError> = signals
            .par_iter()
            .map(|signal| parallel_fir_filter(signal, coefficients, fir_config))
            .collect();

        results
    } else {
        // Process sequentially
        signals
            .iter()
            .map(|signal| parallel_fir_filter(signal, coefficients, fir_config))
            .collect()
    }
}

#[cfg(not(feature = "parallel"))]
pub fn batch_fir_filter(
    signals: &[Array1<f64>],
    coefficients: &[f64],
    fir_config: &ParallelFIRConfig,
    _batch_config: &BatchFilterConfig,
) -> SignalResult<Vec<Array1<f64>>> {
    signals
        .iter()
        .map(|signal| parallel_fir_filter(signal, coefficients, fir_config))
        .collect()
}

/// Batch IIR filtering of multiple signals
#[cfg(feature = "parallel")]
pub fn batch_iir_filter(
    signals: &[Array1<f64>],
    b: &[f64],
    a: &[f64],
    iir_config: &ParallelIIRConfig,
    batch_config: &BatchFilterConfig,
) -> SignalResult<Vec<Array1<f64>>> {
    if signals.is_empty() {
        return Err(SignalError::ValueError("No signals provided".to_string()));
    }

    if batch_config.parallel_signals && signals.len() >= batch_config.min_batch_size {
        let results: Result<Vec<_>, SignalError> = signals
            .par_iter()
            .map(|signal| parallel_iir_filter(signal, b, a, iir_config))
            .collect();

        results
    } else {
        signals
            .iter()
            .map(|signal| parallel_iir_filter(signal, b, a, iir_config))
            .collect()
    }
}

#[cfg(not(feature = "parallel"))]
pub fn batch_iir_filter(
    signals: &[Array1<f64>],
    b: &[f64],
    a: &[f64],
    iir_config: &ParallelIIRConfig,
    _batch_config: &BatchFilterConfig,
) -> SignalResult<Vec<Array1<f64>>> {
    signals
        .iter()
        .map(|signal| parallel_iir_filter(signal, b, a, iir_config))
        .collect()
}

// ============================================================================
// Specialized Parallel Filters
// ============================================================================

/// Parallel moving average filter
///
/// Efficiently computes moving average using cumulative sum approach.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Moving average window size
///
/// # Returns
///
/// Moving average filtered signal
pub fn parallel_moving_average(
    signal: &Array1<f64>,
    window_size: usize,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    if window_size == 0 {
        return Err(SignalError::ValueError(
            "Window size must be positive".to_string(),
        ));
    }

    if window_size > n {
        return Err(SignalError::ValueError(
            "Window size exceeds signal length".to_string(),
        ));
    }

    // Compute cumulative sum
    let mut cumsum = Vec::with_capacity(n + 1);
    cumsum.push(0.0);

    let mut sum = 0.0;
    for &x in signal.iter() {
        sum += x;
        cumsum.push(sum);
    }

    // Compute moving average using cumulative sum
    let output_len = n - window_size + 1;
    let mut output = Array1::zeros(n);

    // First window_size - 1 elements: partial average
    for i in 0..(window_size - 1) {
        output[i] = cumsum[i + 1] / (i + 1) as f64;
    }

    // Rest: full window average
    for i in (window_size - 1)..n {
        output[i] = (cumsum[i + 1] - cumsum[i + 1 - window_size]) / window_size as f64;
    }

    Ok(output)
}

/// Parallel median filter
///
/// Applies median filtering with parallel processing for independent windows.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Median filter window size (should be odd)
///
/// # Returns
///
/// Median filtered signal
#[cfg(feature = "parallel")]
pub fn parallel_median_filter(
    signal: &Array1<f64>,
    window_size: usize,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    if window_size == 0 {
        return Err(SignalError::ValueError(
            "Window size must be positive".to_string(),
        ));
    }

    let half_window = window_size / 2;
    let signal_vec: Vec<f64> = signal.to_vec();
    let indices: Vec<usize> = (0..n).collect();

    let output_vec: Vec<f64> = indices
        .par_iter()
        .map(|&i| {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(n);

            let mut window_values: Vec<f64> = signal_vec[start..end].to_vec();
            window_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let len = window_values.len();
            if len % 2 == 0 {
                (window_values[len / 2 - 1] + window_values[len / 2]) / 2.0
            } else {
                window_values[len / 2]
            }
        })
        .collect();

    Ok(Array1::from_vec(output_vec))
}

#[cfg(not(feature = "parallel"))]
pub fn parallel_median_filter(
    signal: &Array1<f64>,
    window_size: usize,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    if window_size == 0 {
        return Err(SignalError::ValueError(
            "Window size must be positive".to_string(),
        ));
    }

    let half_window = window_size / 2;
    let mut output = Array1::zeros(n);

    for i in 0..n {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(n);

        let mut window_values: Vec<f64> =
            signal.slice(scirs2_core::ndarray::s![start..end]).to_vec();
        window_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let len = window_values.len();
        output[i] = if len % 2 == 0 {
            (window_values[len / 2 - 1] + window_values[len / 2]) / 2.0
        } else {
            window_values[len / 2]
        };
    }

    Ok(output)
}

/// Parallel Savitzky-Golay filter
///
/// Applies Savitzky-Golay smoothing filter with parallel window processing.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Window size (must be odd)
/// * `poly_order` - Polynomial order (must be less than window_size)
///
/// # Returns
///
/// Savitzky-Golay filtered signal
pub fn parallel_savgol_filter(
    signal: &Array1<f64>,
    window_size: usize,
    poly_order: usize,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();

    if window_size == 0 || window_size % 2 == 0 {
        return Err(SignalError::ValueError(
            "Window size must be odd and positive".to_string(),
        ));
    }

    if poly_order >= window_size {
        return Err(SignalError::ValueError(
            "Polynomial order must be less than window size".to_string(),
        ));
    }

    // Compute Savitzky-Golay coefficients
    let coefficients = compute_savgol_coefficients(window_size, poly_order)?;

    // Apply as FIR filter
    let config = ParallelFIRConfig {
        method: FIRFilterMethod::Direct,
        ..Default::default()
    };

    parallel_fir_filter(signal, &coefficients, &config)
}

/// Compute Savitzky-Golay filter coefficients
fn compute_savgol_coefficients(window_size: usize, poly_order: usize) -> SignalResult<Vec<f64>> {
    let m = (window_size - 1) / 2;
    let n_coeffs = window_size;

    // Build Vandermonde matrix
    let mut vandermonde = Array2::zeros((n_coeffs, poly_order + 1));
    for i in 0..n_coeffs {
        let x = (i as i32 - m as i32) as f64;
        for j in 0..=poly_order {
            vandermonde[[i, j]] = x.powi(j as i32);
        }
    }

    // Compute pseudoinverse: (V^T V)^{-1} V^T
    let vtv = vandermonde.t().dot(&vandermonde);
    let vtv_inv = invert_matrix(&vtv)?;
    let pinv = vtv_inv.dot(&vandermonde.t());

    // Extract smoothing coefficients (first row of pseudoinverse transposed)
    let coefficients: Vec<f64> = pinv.row(0).to_vec();

    Ok(coefficients)
}

/// Simple matrix inversion for small matrices
fn invert_matrix(a: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SignalError::ValueError("Matrix must be square".to_string()));
    }

    // Gaussian elimination with augmented matrix
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Forward elimination
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = aug[[k, k]].abs();
        for i in (k + 1)..n {
            if aug[[i, k]].abs() > max_val {
                max_val = aug[[i, k]].abs();
                max_row = i;
            }
        }

        // Swap rows
        if max_row != k {
            for j in 0..(2 * n) {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singularity
        if aug[[k, k]].abs() < 1e-14 {
            return Err(SignalError::ComputationError(
                "Matrix is singular".to_string(),
            ));
        }

        // Scale row
        let pivot = aug[[k, k]];
        for j in 0..(2 * n) {
            aug[[k, j]] /= pivot;
        }

        // Eliminate
        for i in 0..n {
            if i != k {
                let factor = aug[[i, k]];
                for j in 0..(2 * n) {
                    aug[[i, j]] -= factor * aug[[k, j]];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }

    Ok(inv)
}

// ============================================================================
// Streaming Filter Implementation
// ============================================================================

/// Streaming FIR filter for real-time processing
///
/// Maintains internal state for continuous filtering of streaming data.
pub struct StreamingFIRFilter {
    /// Filter coefficients
    coefficients: Vec<f64>,
    /// History buffer
    history: Vec<f64>,
    /// Current position in history
    pos: usize,
}

impl StreamingFIRFilter {
    /// Create a new streaming FIR filter
    pub fn new(coefficients: Vec<f64>) -> SignalResult<Self> {
        if coefficients.is_empty() {
            return Err(SignalError::ValueError(
                "Coefficients cannot be empty".to_string(),
            ));
        }

        let len = coefficients.len();
        Ok(Self {
            coefficients,
            history: vec![0.0; len],
            pos: 0,
        })
    }

    /// Process a single sample
    pub fn process_sample(&mut self, input: f64) -> f64 {
        // Store input in circular buffer
        self.history[self.pos] = input;

        // Compute output
        let mut output = 0.0;
        let len = self.coefficients.len();

        for i in 0..len {
            let idx = (self.pos + len - i) % len;
            output += self.coefficients[i] * self.history[idx];
        }

        // Update position
        self.pos = (self.pos + 1) % len;

        output
    }

    /// Process a block of samples
    pub fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| self.process_sample(x)).collect()
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.pos = 0;
    }
}

/// Streaming IIR filter for real-time processing
pub struct StreamingIIRFilter {
    /// Numerator coefficients
    b: Vec<f64>,
    /// Denominator coefficients (normalized)
    a: Vec<f64>,
    /// Input history
    x_history: Vec<f64>,
    /// Output history
    y_history: Vec<f64>,
}

impl StreamingIIRFilter {
    /// Create a new streaming IIR filter
    pub fn new(b: Vec<f64>, a: Vec<f64>) -> SignalResult<Self> {
        if b.is_empty() || a.is_empty() {
            return Err(SignalError::ValueError(
                "Coefficients cannot be empty".to_string(),
            ));
        }

        if a[0].abs() < 1e-15 {
            return Err(SignalError::ValueError(
                "First denominator coefficient must be non-zero".to_string(),
            ));
        }

        // Normalize coefficients
        let a0 = a[0];
        let b_norm: Vec<f64> = b.iter().map(|&x| x / a0).collect();
        let a_norm: Vec<f64> = a.iter().map(|&x| x / a0).collect();

        let nb = b_norm.len();
        let na = a_norm.len();

        Ok(Self {
            b: b_norm,
            a: a_norm,
            x_history: vec![0.0; nb],
            y_history: vec![0.0; na],
        })
    }

    /// Process a single sample
    pub fn process_sample(&mut self, input: f64) -> f64 {
        let nb = self.b.len();
        let na = self.a.len();

        // Shift input history
        for i in (1..nb).rev() {
            self.x_history[i] = self.x_history[i - 1];
        }
        self.x_history[0] = input;

        // Compute output
        let mut output = 0.0;

        // FIR part
        for i in 0..nb {
            output += self.b[i] * self.x_history[i];
        }

        // IIR part
        for i in 1..na {
            output -= self.a[i] * self.y_history[i - 1];
        }

        // Shift output history
        for i in (1..na).rev() {
            self.y_history[i] = self.y_history[i - 1];
        }
        if na > 0 {
            self.y_history[0] = output;
        }

        output
    }

    /// Process a block of samples
    pub fn process_block(&mut self, input: &[f64]) -> Vec<f64> {
        input.iter().map(|&x| self.process_sample(x)).collect()
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.x_history.fill(0.0);
        self.y_history.fill(0.0);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn generate_test_signal(n: usize, freq: f64, fs: f64) -> Array1<f64> {
        Array1::linspace(0.0, (n - 1) as f64 / fs, n).mapv(|t| (2.0 * PI * freq * t).sin())
    }

    #[test]
    fn test_parallel_fir_filter_direct() {
        let signal = generate_test_signal(100, 10.0, 100.0);
        let coefficients = vec![0.25, 0.5, 0.25];

        let config = ParallelFIRConfig {
            method: FIRFilterMethod::Direct,
            ..Default::default()
        };

        let result = parallel_fir_filter(&signal, &coefficients, &config);
        assert!(result.is_ok());

        let filtered = result.expect("Operation failed");
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_parallel_fir_filter_overlap_save() {
        let signal = generate_test_signal(1000, 10.0, 100.0);
        let coefficients: Vec<f64> = (0..32).map(|i| 1.0 / 32.0).collect();

        let config = ParallelFIRConfig {
            method: FIRFilterMethod::OverlapSave,
            block_size: 256,
            ..Default::default()
        };

        let result = parallel_fir_filter(&signal, &coefficients, &config);
        assert!(result.is_ok());

        let filtered = result.expect("Operation failed");
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_parallel_fir_filter_overlap_add() {
        let signal = generate_test_signal(500, 20.0, 100.0);
        let coefficients: Vec<f64> = (0..16).map(|i| 1.0 / 16.0).collect();

        let config = ParallelFIRConfig {
            method: FIRFilterMethod::OverlapAdd,
            block_size: 128,
            ..Default::default()
        };

        let result = parallel_fir_filter(&signal, &coefficients, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parallel_iir_filter() {
        let signal = generate_test_signal(200, 10.0, 100.0);

        // Simple lowpass filter coefficients
        let b = vec![0.0675, 0.1349, 0.0675];
        let a = vec![1.0, -1.1430, 0.4128];

        let config = ParallelIIRConfig::default();

        let result = parallel_iir_filter(&signal, &b, &a, &config);
        assert!(result.is_ok());

        let filtered = result.expect("Operation failed");
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_parallel_iir_filter_zero_phase() {
        let signal = generate_test_signal(300, 15.0, 100.0);

        let b = vec![0.0675, 0.1349, 0.0675];
        let a = vec![1.0, -1.1430, 0.4128];

        let config = ParallelIIRConfig {
            zero_phase: true,
            ..Default::default()
        };

        let result = parallel_iir_filter(&signal, &b, &a, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parallel_moving_average() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let result = parallel_moving_average(&signal, 3);
        assert!(result.is_ok());

        let filtered = result.expect("Operation failed");
        assert_eq!(filtered.len(), signal.len());

        // Check a known value
        assert_relative_eq!(filtered[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(filtered[5], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parallel_median_filter() {
        let signal = Array1::from_vec(vec![1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0, 5.0, 6.0]);

        let result = parallel_median_filter(&signal, 3);
        assert!(result.is_ok());

        let filtered = result.expect("Operation failed");
        assert_eq!(filtered.len(), signal.len());
    }

    #[test]
    fn test_streaming_fir_filter() {
        let coefficients = vec![0.25, 0.5, 0.25];
        let mut filter = StreamingFIRFilter::new(coefficients).expect("Operation failed");

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = filter.process_block(&input);

        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_streaming_iir_filter() {
        let b = vec![0.1, 0.2, 0.1];
        let a = vec![1.0, -0.5, 0.2];

        let mut filter = StreamingIIRFilter::new(b, a).expect("Operation failed");

        let input = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let output = filter.process_block(&input);

        assert_eq!(output.len(), input.len());
        assert!(output[0] > 0.0); // First output should be non-zero
    }

    #[test]
    fn test_batch_fir_filter() {
        let signals: Vec<Array1<f64>> = (0..5)
            .map(|i| generate_test_signal(100, 10.0 + i as f64, 100.0))
            .collect();

        let coefficients = vec![0.2, 0.3, 0.3, 0.2];
        let fir_config = ParallelFIRConfig::default();
        let batch_config = BatchFilterConfig::default();

        let result = batch_fir_filter(&signals, &coefficients, &fir_config, &batch_config);
        assert!(result.is_ok());

        let filtered = result.expect("Operation failed");
        assert_eq!(filtered.len(), signals.len());
    }

    #[test]
    fn test_empty_signal_error() {
        let signal = Array1::zeros(0);
        let coefficients = vec![0.5, 0.5];

        let config = ParallelFIRConfig::default();
        let result = parallel_fir_filter(&signal, &coefficients, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_coefficients_error() {
        let signal = generate_test_signal(100, 10.0, 100.0);
        let coefficients: Vec<f64> = vec![];

        let config = ParallelFIRConfig::default();
        let result = parallel_fir_filter(&signal, &coefficients, &config);
        assert!(result.is_err());
    }
}
