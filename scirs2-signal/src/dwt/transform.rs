// Core DWT decomposition and reconstruction functions
//
// This module provides the core functions for single-level discrete wavelet transform
// decomposition and reconstruction.

use super::boundary::extend_signal;
use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

#[allow(unused_imports)]
/// Perform single-level discrete wavelet transform (DWT) decomposition
///
/// # Arguments
///
/// * `data` - Input signal
/// * `wavelet` - Wavelet to use for transform
/// * `mode` - Signal extension mode (default: "symmetric")
///
/// # Returns
///
/// A tuple containing (approximation coefficients, detail coefficients)
///
/// # Examples
///
/// ```
/// use scirs2_signal::dwt::{dwt_decompose, Wavelet};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let (approx, detail) = dwt_decompose(&signal, Wavelet::DB(4), None).expect("operation should succeed");
///
/// // Approximation and detail coefficients
/// println!("Approximation: {:?}", approx);
/// println!("Detail: {:?}", detail);
/// ```
#[allow(dead_code)]
pub fn dwt_decompose<T>(
    data: &[T],
    wavelet: Wavelet,
    mode: Option<&str>,
) -> SignalResult<(Vec<f64>, Vec<f64>)>
where
    T: Float + NumCast + Debug,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Get wavelet filters
    let filters = wavelet.filters()?;
    let filter_len = filters.dec_lo.len();

    // Convert data to f64
    let data_f64: Vec<f64> = data
        .iter()
        .map(|&v| {
            NumCast::from(v).ok_or_else(|| {
                SignalError::ValueError(format!("Failed to convert value {:?} to f64", v))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    // Extend signal according to the specified mode
    let mode_str = mode.unwrap_or("symmetric");
    let extended = extend_signal(&data_f64, filter_len, mode_str)?;

    // Calculate output length
    let input_len = data_f64.len();
    let output_len = (input_len + filter_len - 1) / 2;

    // Allocate output arrays
    let mut approx = vec![0.0; output_len];
    let mut detail = vec![0.0; output_len];

    // Perform the convolution and downsample
    // Start at offset 1 in the extended signal so the first filter window
    // is aligned with the beginning of the original signal data.
    // Use true convolution (reversed filter indices) for correct phase.
    let start_offset = 1;
    for i in 0..output_len {
        let idx = start_offset + 2 * i;

        // Convolve with low-pass filter for approximation coefficients
        let mut approx_sum = 0.0;
        for j in 0..filter_len {
            if idx + j < extended.len() {
                approx_sum += extended[idx + j] * filters.dec_lo[filter_len - 1 - j];
            }
        }
        approx[i] = approx_sum;

        // Convolve with high-pass filter for detail coefficients
        let mut detail_sum = 0.0;
        for j in 0..filter_len {
            if idx + j < extended.len() {
                detail_sum += extended[idx + j] * filters.dec_hi[filter_len - 1 - j];
            }
        }
        detail[i] = detail_sum;
    }

    Ok((approx, detail))
}

/// Perform single-level inverse discrete wavelet transform (IDWT) reconstruction
///
/// # Arguments
///
/// * `approx` - Approximation coefficients
/// * `detail` - Detail coefficients
/// * `wavelet` - Wavelet to use for reconstruction
///
/// # Returns
///
/// The reconstructed signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::dwt::{dwt_decompose, dwt_reconstruct, Wavelet};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let (approx, detail) = dwt_decompose(&signal, Wavelet::DB(4), None).expect("operation should succeed");
///
/// // Reconstruct the signal
/// let reconstructed = dwt_reconstruct(&approx, &detail, Wavelet::DB(4)).expect("operation should succeed");
///
/// // Basic test - reconstruction should succeed
/// assert!(reconstructed.len() > 0);
/// ```
#[allow(dead_code)]
pub fn dwt_reconstruct(
    _approx: &[f64],
    detail: &[f64],
    wavelet: Wavelet,
) -> SignalResult<Vec<f64>> {
    if _approx.is_empty() || detail.is_empty() {
        return Err(SignalError::ValueError(
            "Input arrays are empty".to_string(),
        ));
    }

    if _approx.len() != detail.len() {
        return Err(SignalError::ValueError(
            "Approximation and detail coefficients must have the same length".to_string(),
        ));
    }

    // Get wavelet filters
    let filters = wavelet.filters()?;
    let filter_len = filters.rec_lo.len();
    let pad = filter_len - 1;

    // Calculate output length
    let input_len = _approx.len();
    // The full transposed convolution produces 2*input_len + filter_len - 2 samples
    let full_len = 2 * input_len + pad;

    // Allocate output array
    let mut result = vec![0.0; full_len];

    // Upsample and convolve (transpose of decimated convolution)
    // This mirrors the analysis step: analysis did extended[2i+j] * dec[j]
    // Synthesis does: result[2i+j] += coeff[i] * rec[j]
    for i in 0..input_len {
        for j in 0..filter_len {
            let idx = 2 * i + j;
            if idx < full_len {
                result[idx] += _approx[i] * filters.rec_lo[j] + detail[i] * filters.rec_hi[j];
            }
        }
    }

    // The analysis used start_offset=1 and true convolution (reversed filter).
    // The reconstruction group delay is (filter_len - 2), so we skip that many
    // samples from the beginning to align with the original signal.
    let skip = if pad > 0 { pad - 1 } else { 0 };
    let max_output = 2 * input_len;
    let available = if skip < full_len { full_len - skip } else { 0 };
    let take = max_output.min(available);

    if take > 0 && skip < full_len {
        Ok(result[skip..skip + take].to_vec())
    } else {
        // Fallback: return full result
        Ok(result)
    }
}
