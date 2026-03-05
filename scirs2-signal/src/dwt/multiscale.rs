// Multi-level wavelet transform functions
//
// This module provides functions for multi-level/multi-resolution wavelet analysis,
// including decomposition and reconstruction of signals.

use super::transform::{dwt_decompose, dwt_reconstruct};
use crate::dwt::Wavelet;
use crate::error::{SignalError, SignalResult};
use scirs2_core::numeric::{Float, NumCast};
use std::fmt::Debug;

#[allow(unused_imports)]
/// Perform multi-level wavelet decomposition
///
/// # Arguments
///
/// * `data` - Input signal
/// * `wavelet` - Wavelet to use for decomposition
/// * `level` - Number of decomposition levels (default: maximum possible)
/// * `mode` - Signal extension mode (default: "symmetric")
///
/// # Returns
///
/// A vector of coefficient arrays: [approximation_n, detail_n, detail_n-1, ..., detail_1]
/// where n is the decomposition level.
///
/// # Examples
///
/// ```
/// use scirs2_signal::dwt::{wavedec, Wavelet};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let coeffs = wavedec(&signal, Wavelet::DB(4), Some(2), None).expect("operation should succeed");
///
/// // coeffs[0] contains level-2 approximation
/// // coeffs[1] contains level-2 detail
/// // coeffs[2] contains level-1 detail
/// ```
#[allow(dead_code)]
pub fn wavedec<T>(
    data: &[T],
    wavelet: Wavelet,
    level: Option<usize>,
    mode: Option<&str>,
) -> SignalResult<Vec<Vec<f64>>>
where
    T: Float + NumCast + Debug,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }

    // Convert data to f64
    let data_f64: Vec<f64> = data
        .iter()
        .map(|&v| {
            NumCast::from(v).ok_or_else(|| {
                SignalError::ValueError(format!("Failed to convert value {:?} to f64", v))
            })
        })
        .collect::<SignalResult<Vec<f64>>>()?;

    // Calculate maximum possible decomposition level
    let data_len = data_f64.len();
    let filters = wavelet.filters()?;
    let filter_len = filters.dec_lo.len();

    // Each level of decomposition approximately halves the signal length
    // and requires at least filter_len samples
    let min_length = if let Wavelet::Haar = wavelet {
        2
    } else {
        filter_len
    };
    let max_level = (data_len as f64 / min_length as f64).log2().floor() as usize;
    let decomp_level = level.unwrap_or(max_level).min(max_level);

    if decomp_level == 0 {
        // No decomposition, just return the original signal
        return Ok(vec![data_f64]);
    }

    // Initialize coefficient arrays
    let mut coeffs = Vec::with_capacity(decomp_level + 1);

    // Start with the original signal
    let mut approx = data_f64;

    // Perform decomposition for each level
    for _ in 0..decomp_level {
        // Decompose current approximation
        let (next_approx, detail) = dwt_decompose(&approx, wavelet, mode)?;

        // Store detail coefficients
        coeffs.push(detail);

        // Update approximation for next level
        approx = next_approx;
    }

    // Add final approximation (level 'n')
    coeffs.push(approx);

    // Reverse to get [a_n, d_n, d_n-1, ..., d_1]
    coeffs.reverse();

    Ok(coeffs)
}

/// Perform multi-level inverse wavelet reconstruction
///
/// # Arguments
///
/// * `coeffs` - Wavelet coefficients from wavedec [a_n, d_n, d_n-1, ..., d_1]
/// * `wavelet` - Wavelet to use for reconstruction
///
/// # Returns
///
/// The reconstructed signal
///
/// # Examples
///
/// ```
/// use scirs2_signal::dwt::{wavedec, waverec, Wavelet};
///
/// let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let coeffs = wavedec(&signal, Wavelet::DB(4), Some(2), None).expect("operation should succeed");
///
/// // Reconstruct the signal
/// let reconstructed = waverec(&coeffs, Wavelet::DB(4)).expect("operation should succeed");
///
/// // Check that reconstructed signal is close to original
/// for i in 0..signal.len() {
///     assert!((signal[i] - reconstructed[i]).abs() < 1e-10);
/// }
/// ```
#[allow(dead_code)]
pub fn waverec(coeffs: &[Vec<f64>], wavelet: Wavelet) -> SignalResult<Vec<f64>> {
    if coeffs.is_empty() {
        return Err(SignalError::ValueError(
            "Coefficients array is empty".to_string(),
        ));
    }

    // Case of no transform (just the signal)
    if coeffs.len() == 1 {
        return Ok(coeffs[0].clone());
    }

    // Start with the coarsest approximation
    let mut approx = coeffs[0].clone();

    // Number of reconstruction levels
    let n_levels = coeffs.len() - 1;

    // Reconstruct each level
    for i in 0..n_levels {
        let detail = &coeffs[i + 1];

        // Adjust approximation/detail lengths if they differ slightly
        // due to rounding in the output_len calculation
        if approx.len() != detail.len() {
            let min_len = approx.len().min(detail.len());
            if approx.len() > min_len {
                approx.truncate(min_len);
            }
            let detail = if detail.len() > min_len {
                detail[0..min_len].to_vec()
            } else {
                detail.clone()
            };

            approx = dwt_reconstruct(&approx, &detail, wavelet)?;
        } else {
            approx = dwt_reconstruct(&approx, detail, wavelet)?;
        }

        // After reconstruction, trim the output to match the expected length
        // at the next level. The next level's detail coefficients tell us
        // what length the signal was before that level's decomposition.
        if i + 2 < coeffs.len() {
            let next_detail_len = coeffs[i + 2].len();
            // The original signal length at the next level can be inferred:
            // next_detail_len = (original_len + filter_len - 1) / 2
            // So original_len is approximately 2 * next_detail_len
            // We use the next detail length to compute the expected signal length
            let filters = wavelet.filters()?;
            let filter_len = filters.dec_lo.len();
            // We know that: next_detail_len = (approx_expected_len + filter_len - 1) / 2
            // So: approx_expected_len = 2 * next_detail_len - filter_len + 1
            //   or approx_expected_len = 2 * next_detail_len - filter_len + 2 (if odd)
            // Since we can't know exactly, just truncate to the value that
            // gives the right coefficient count at the next level.
            // The expected length L satisfies: (L + filter_len - 1) / 2 = next_detail_len
            // => L = 2 * next_detail_len - filter_len + 1 (min)
            //    L = 2 * next_detail_len - filter_len + 2 (max)
            // We take the maximum plausible length that doesn't exceed our output
            let expected_len_min = 2 * next_detail_len - filter_len + 1;
            let expected_len_max = 2 * next_detail_len - filter_len + 2;
            if approx.len() > expected_len_max {
                approx.truncate(expected_len_max);
            } else if approx.len() > expected_len_min && approx.len() <= expected_len_max {
                // Already in the right range, keep as is
            }
        }
    }

    Ok(approx)
}

// Compatibility wrapper functions for old API style

/// Compatibility wrapper for wavedec with 3 parameters (old API)
pub fn wavedec_compat<T>(data: &[T], wavelet: Wavelet, level: usize) -> SignalResult<Vec<Vec<f64>>>
where
    T: Float + NumCast + Debug,
{
    wavedec(data, wavelet, Some(level), None)
}

/// Compatibility wrapper for waverec with DecompositionResult input
pub fn waverec_compat(coeffs: &[Vec<f64>], wavelet: Wavelet) -> SignalResult<Vec<f64>> {
    waverec(coeffs, wavelet)
}
