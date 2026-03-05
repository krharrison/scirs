//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{Dwt2dConfig, Dwt2dResult, MemoryPool, ThresholdMethod};
use crate::dwt::{self, Wavelet};
use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::parallel_ops::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::fmt::Debug;

use super::types::{Dwt2dConfig, Dwt2dResult};
use super::functions::{div_ceil, get_temp_buffer, return_temp_buffer};

/// Memory-optimized version of 2D DWT decomposition with configuration options
///
/// This function provides the same functionality as `dwt2d_decompose` but with
/// additional memory optimizations and configuration options for better performance
/// on large arrays.
///
/// # Arguments
///
/// * `data` - The input 2D array (image)
/// * `wavelet` - The wavelet to use for the transform
/// * `mode` - The signal extension mode (default: "symmetric")
/// * `config` - Configuration for memory optimization
///
/// # Returns
///
/// * A `Dwt2dResult` containing the four subbands of the decomposition
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose_optimized, Dwt2dConfig};
///
/// // Create a sample image
/// let data = Array2::from_shape_vec((8, 8), (0..64).map(|x| x as f64).collect()).expect("operation should succeed");
///
/// // Use optimized decomposition with default configuration
/// let config = Dwt2dConfig::default();
/// let result = dwt2d_decompose_optimized(&data, Wavelet::Haar, None, &config).expect("operation should succeed");
/// ```
#[allow(dead_code)]
pub fn dwt2d_decompose_optimized<T>(
    data: &Array2<T>,
    wavelet: Wavelet,
    mode: Option<&str>,
    config: &Dwt2dConfig,
) -> SignalResult<Dwt2dResult>
where
    T: Float + NumCast + Debug,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }
    if let Some(data_slice) = data.as_slice() {}
    let (rows, cols) = data.dim();
    let output_rows = div_ceil(rows, 2);
    let output_cols = div_ceil(cols, 2);
    let mut ll = Array2::zeros((output_rows, output_cols));
    let mut lh = Array2::zeros((output_rows, output_cols));
    let mut hl = Array2::zeros((output_rows, output_cols));
    let mut hh = Array2::zeros((output_rows, output_cols));
    let data_buffer_size = rows * cols;
    let mut data_buffer = if config.preallocate_memory {
        get_temp_buffer(data_buffer_size)
    } else {
        vec![0.0; data_buffer_size]
    };
    for ((i, j), &val) in data.indexed_iter() {
        match NumCast::from(val) {
            Some(converted) => data_buffer[i * cols + j] = converted,
            None => {
                return Err(
                    SignalError::ValueError(
                        "Failed to convert input data to f64".to_string(),
                    ),
                );
            }
        }
    }
    let row_buffer_size = rows * output_cols;
    let mut rows_lo_buffer = if config.preallocate_memory {
        get_temp_buffer(row_buffer_size)
    } else {
        vec![0.0; row_buffer_size]
    };
    let mut rows_hi_buffer = if config.preallocate_memory {
        get_temp_buffer(row_buffer_size)
    } else {
        vec![0.0; row_buffer_size]
    };
    let chunk_size = config.chunk_size.unwrap_or(rows);
    for chunk_start in (0..rows).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(rows);
        #[cfg(feature = "parallel")]
        {
            let chunk_results: Result<Vec<(usize, Vec<f64>, Vec<f64>)>, SignalError> = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|i| {
                    let row_start = i * cols;
                    let row_end = row_start + cols;
                    let row = &data_buffer[row_start..row_end];
                    let (approx, detail) = dwt::dwt_decompose(row, wavelet, mode)
                        .map_err(|e| {
                            SignalError::ComputationError(
                                format!("Row transform failed: {}", e),
                            )
                        })?;
                    Ok((i, approx, detail))
                })
                .collect();
            let chunk_results = chunk_results?;
            for (i, approx, detail) in chunk_results {
                let output_start = i * output_cols;
                for j in 0..approx.len().min(output_cols) {
                    rows_lo_buffer[output_start + j] = approx[j];
                    rows_hi_buffer[output_start + j] = detail[j];
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in chunk_start..chunk_end {
                let row_start = i * cols;
                let row_end = row_start + cols;
                let row = &data_buffer[row_start..row_end];
                let (approx, detail) = dwt::dwt_decompose(row, wavelet, mode)?;
                let output_start = i * output_cols;
                for j in 0..approx.len().min(output_cols) {
                    rows_lo_buffer[output_start + j] = approx[j];
                    rows_hi_buffer[output_start + j] = detail[j];
                }
            }
        }
    }
    let col_chunk_size = config.chunk_size.unwrap_or(output_cols);
    for chunk_start in (0..output_cols).step_by(col_chunk_size) {
        let chunk_end = (chunk_start + col_chunk_size).min(output_cols);
        #[cfg(feature = "parallel")]
        {
            let column_results: Result<
                Vec<(usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)>,
                SignalError,
            > = (chunk_start..chunk_end)
                .into_par_iter()
                .map(|j| {
                    let mut col_lo = vec![0.0; rows];
                    for i in 0..rows {
                        col_lo[i] = rows_lo_buffer[i * output_cols + j];
                    }
                    let (approx_lo, detail_lo) = dwt::dwt_decompose(
                            &col_lo,
                            wavelet,
                            mode,
                        )
                        .map_err(|e| {
                            SignalError::ComputationError(
                                format!("Column transform failed (low-pass): {}", e),
                            )
                        })?;
                    let mut col_hi = vec![0.0; rows];
                    for i in 0..rows {
                        col_hi[i] = rows_hi_buffer[i * output_cols + j];
                    }
                    let (approx_hi, detail_hi) = dwt::dwt_decompose(
                            &col_hi,
                            wavelet,
                            mode,
                        )
                        .map_err(|e| {
                            SignalError::ComputationError(
                                format!("Column transform failed (high-pass): {}", e),
                            )
                        })?;
                    Ok((j, approx_lo, detail_lo, approx_hi, detail_hi))
                })
                .collect();
            let column_results = column_results?;
            for (j, approx_lo, detail_lo, approx_hi, detail_hi) in column_results {
                for i in 0..approx_lo.len().min(output_rows) {
                    ll[[i, j]] = approx_lo[i];
                    hl[[i, j]] = detail_lo[i];
                }
                for i in 0..approx_hi.len().min(output_rows) {
                    lh[[i, j]] = approx_hi[i];
                    hh[[i, j]] = detail_hi[i];
                }
            }
        }
        #[cfg(not(feature = "parallel"))]
        {
            for j in chunk_start..chunk_end {
                let mut col_lo = vec![0.0; rows];
                for i in 0..rows {
                    col_lo[i] = rows_lo_buffer[i * output_cols + j];
                }
                let (approx_lo, detail_lo) = dwt::dwt_decompose(&col_lo, wavelet, mode)?;
                let mut col_hi = vec![0.0; rows];
                for i in 0..rows {
                    col_hi[i] = rows_hi_buffer[i * output_cols + j];
                }
                let (approx_hi, detail_hi) = dwt::dwt_decompose(&col_hi, wavelet, mode)?;
                for i in 0..approx_lo.len().min(output_rows) {
                    ll[[i, j]] = approx_lo[i];
                    hl[[i, j]] = detail_lo[i];
                }
                for i in 0..approx_hi.len().min(output_rows) {
                    lh[[i, j]] = approx_hi[i];
                    hh[[i, j]] = detail_hi[i];
                }
            }
        }
    }
    if config.preallocate_memory {
        return_temp_buffer(data_buffer);
        return_temp_buffer(rows_lo_buffer);
        return_temp_buffer(rows_hi_buffer);
    }
    Ok(Dwt2dResult {
        approx: ll,
        detail_h: lh,
        detail_v: hl,
        detail_d: hh,
    })
}
/// Performs a single-level 2D inverse discrete wavelet transform.
///
/// This function reconstructs a 2D array (such as an image) from its wavelet decomposition.
/// It is the inverse operation of `dwt2d_decompose` and combines the four subbands
/// (approximation and detail coefficients) back into the original signal.
///
/// # Algorithm
///
/// The inverse 2D DWT is computed using separable filtering:
/// 1. First, reconstruct each row by combining the corresponding rows from low-pass and high-pass parts
/// 2. Then, reconstruct each column of the resulting array
/// 3. The process reverses the decomposition steps, using inverse wavelet filters
///
/// # Arguments
///
/// * `decomposition` - The wavelet decomposition to reconstruct from, containing the four subbands
/// * `wavelet` - The wavelet used for the original transform (must match the decomposition wavelet)
/// * `mode` - The signal extension mode (default: "symmetric")
///   - Note: This should match the mode used for decomposition for best results
///
/// # Returns
///
/// * The reconstructed 2D array with dimensions twice the size of each subband
///
/// # Errors
///
/// Returns an error if:
/// * The subbands in the decomposition have different shapes
/// * There are issues with the wavelet filters
/// * Numerical problems occur during reconstruction
///
/// # Performance
///
/// This operation is computationally efficient, with O(N) complexity where N is the
/// total number of elements in the reconstructed array. The actual performance depends
/// on the wavelet filter lengths.
///
/// # Examples
///
/// Basic decomposition and reconstruction:
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple "image"
/// let data = Array2::from_shape_vec((4, 4), vec![
///     1.0, 2.0, 3.0, 4.0,
///     5.0, 6.0, 7.0, 8.0,
///     9.0, 10.0, 11.0, 12.0,
///     13.0, 14.0, 15.0, 16.0
/// ]).expect("operation should succeed");
///
/// // Decompose
/// let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).expect("operation should succeed");
///
/// // Reconstruct
/// let reconstructed = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None).expect("operation should succeed");
///
/// // The reconstructed image should have the same shape as the original
/// assert_eq!(reconstructed.shape(), data.shape());
/// ```
///
/// Modifying coefficients before reconstruction (simple denoising):
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct, Dwt2dResult};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a sample image with gradient pattern
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Decompose using Haar wavelet
/// let mut decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).expect("operation should succeed");
///
/// // Simple denoising by zeroing out small detail coefficients
/// let threshold = 2.0;
/// for h in decomposition.detail_h.iter_mut() {
///     if h.abs() < threshold {
///         *h = 0.0;
///     }
/// }
/// for v in decomposition.detail_v.iter_mut() {
///     if v.abs() < threshold {
///         *v = 0.0;
///     }
/// }
/// for d in decomposition.detail_d.iter_mut() {
///     if d.abs() < threshold {
///         *d = 0.0;
///     }
/// }
///
/// // Reconstruct from modified coefficients
/// let denoised = dwt2d_reconstruct(&decomposition, Wavelet::Haar, None).expect("operation should succeed");
/// ```
#[allow(dead_code)]
pub fn dwt2d_reconstruct(
    decomposition: &Dwt2dResult,
    wavelet: Wavelet,
    mode: Option<&str>,
) -> SignalResult<Array2<f64>> {
    let ll = &decomposition.approx;
    let lh = &decomposition.detail_h;
    let hl = &decomposition.detail_v;
    let hh = &decomposition.detail_d;
    let shape = ll.shape();
    if lh.shape() != shape || hl.shape() != shape || hh.shape() != shape {
        return Err(
            SignalError::ValueError(
                "All decomposition components must have the same shape".to_string(),
            ),
        );
    }
    let (rows, cols) = (shape[0], shape[1]);
    let out_rows = rows * 2;
    let out_cols = cols * 2;
    let mut row_lo = Array2::zeros((out_rows, cols));
    let mut row_hi = Array2::zeros((out_rows, cols));
    #[cfg(feature = "parallel")]
    {
        let col_results: Result<Vec<(usize, Vec<f64>, Vec<f64>)>, SignalError> = (0..cols)
            .into_par_iter()
            .map(|j| {
                let ll_col = ll.slice(scirs2_core::ndarray::s![.., j]).to_vec();
                let hl_col = hl.slice(scirs2_core::ndarray::s![.., j]).to_vec();
                let col_lo = dwt::dwt_reconstruct(&ll_col, &hl_col, wavelet)
                    .map_err(|e| {
                        SignalError::ComputationError(
                            format!("Low-pass column reconstruction failed: {}", e),
                        )
                    })?;
                let lh_col = lh.slice(scirs2_core::ndarray::s![.., j]).to_vec();
                let hh_col = hh.slice(scirs2_core::ndarray::s![.., j]).to_vec();
                let col_hi = dwt::dwt_reconstruct(&lh_col, &hh_col, wavelet)
                    .map_err(|e| {
                        SignalError::ComputationError(
                            format!("High-pass column reconstruction failed: {}", e),
                        )
                    })?;
                Ok((j, col_lo, col_hi))
            })
            .collect();
        let col_results = col_results?;
        for (j, col_lo, col_hi) in col_results {
            for i in 0..col_lo.len() {
                if i < out_rows {
                    row_lo[[i, j]] = col_lo[i];
                    row_hi[[i, j]] = col_hi[i];
                }
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        for j in 0..cols {
            let ll_col = ll.slice(scirs2_core::ndarray::s![.., j]).to_vec();
            let hl_col = hl.slice(scirs2_core::ndarray::s![.., j]).to_vec();
            let col_lo = dwt::dwt_reconstruct(&ll_col, &hl_col, wavelet)?;
            let lh_col = lh.slice(scirs2_core::ndarray::s![.., j]).to_vec();
            let hh_col = hh.slice(scirs2_core::ndarray::s![.., j]).to_vec();
            let col_hi = dwt::dwt_reconstruct(&lh_col, &hh_col, wavelet)?;
            for i in 0..col_lo.len() {
                if i < out_rows {
                    row_lo[[i, j]] = col_lo[i];
                    row_hi[[i, j]] = col_hi[i];
                }
            }
        }
    }
    let mut result = Array2::zeros((out_rows, out_cols));
    #[cfg(feature = "parallel")]
    {
        let row_results: Result<Vec<(usize, Vec<f64>)>, SignalError> = (0..out_rows)
            .into_par_iter()
            .map(|i| {
                let lo_row = row_lo.slice(scirs2_core::ndarray::s![i, ..]).to_vec();
                let hi_row = row_hi.slice(scirs2_core::ndarray::s![i, ..]).to_vec();
                let full_row = dwt::dwt_reconstruct(&lo_row, &hi_row, wavelet)
                    .map_err(|e| {
                        SignalError::ComputationError(
                            format!("Row reconstruction failed: {}", e),
                        )
                    })?;
                Ok((i, full_row))
            })
            .collect();
        let row_results = row_results?;
        for (i, full_row) in row_results {
            for j in 0..full_row.len() {
                if j < out_cols {
                    result[[i, j]] = full_row[j];
                }
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    {
        for i in 0..out_rows {
            let lo_row = row_lo.slice(scirs2_core::ndarray::s![i, ..]).to_vec();
            let hi_row = row_hi.slice(scirs2_core::ndarray::s![i, ..]).to_vec();
            let full_row = dwt::dwt_reconstruct(&lo_row, &hi_row, wavelet)?;
            for j in 0..full_row.len() {
                if j < out_cols {
                    result[[i, j]] = full_row[j];
                }
            }
        }
    }
    Ok(result)
}
