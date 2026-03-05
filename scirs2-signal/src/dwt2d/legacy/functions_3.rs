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

use super::types::{Dwt2dResult, ThresholdMethod, WaveletEnergy};
use super::functions::simd_calculate_energy;

/// Performs a multi-level 2D discrete wavelet transform.
///
/// This function computes the wavelet transform recursively, applying
/// successive decompositions to the approximation coefficients from each level.
/// This creates a multi-resolution analysis with a pyramid structure, where each
/// level captures details at different scales.
///
/// # Algorithm
///
/// The multi-level 2D DWT is computed as follows:
/// 1. Apply a single-level 2D DWT to the input data, generating four subbands (LL, LH, HL, HH)
/// 2. Apply a single-level 2D DWT to the LL (approximation) subband from step 1
/// 3. Repeat until reaching the desired number of levels
/// 4. Return the coefficients from all levels, with the deepest level first
///
/// # Arguments
///
/// * `data` - The input 2D array (image)
/// * `wavelet` - The wavelet to use for the transform
/// * `levels` - The number of decomposition levels to compute
/// * `mode` - The signal extension mode (default: "symmetric")
///
/// # Returns
///
/// * A vector of `Dwt2dResult` objects, where:
///   - index 0 contains coefficients from the deepest level (smallest scale)
///   - each subsequent index contains coefficients from a larger scale
///   - the last index contains the first level of decomposition (largest scale)
///
/// # Errors
///
/// Returns an error if:
/// * The input array is empty
/// * The requested number of levels is 0
/// * The input array is too small for the requested number of levels
/// * Other errors from the underlying `dwt2d_decompose` function
///
/// # Memory Usage
///
/// This function stores coefficients from all levels separately, so memory usage
/// is approximately 4/3 times the original image size (for sufficiently large images).
/// For example, an 8×8 image decomposes into:
/// - Level 1: Four 4×4 subbands
/// - Level 2: Three 2×2 subbands plus the Level 3 approximation
/// - Level 3: Three 1×1 subbands plus a 1×1 approximation
///
/// # Examples
///
/// Basic multi-level decomposition:
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt2d::wavedec2;
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple 8x8 "image"
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * 8 + j + 1)  as f64;
///     }
/// }
///
/// // Perform 3-level 2D DWT
/// let coeffs = wavedec2(&data, Wavelet::Haar, 3, None).expect("operation should succeed");
///
/// // Check the number of decomposition levels
/// assert_eq!(coeffs.len(), 3);
///
/// // Examine the coefficient shapes (each level is half the size of the previous)
/// assert_eq!(coeffs[0].approx.shape(), &[1, 1]);  // Deepest level (smallest)
/// assert_eq!(coeffs[1].approx.shape(), &[2, 2]);
/// assert_eq!(coeffs[2].approx.shape(), &[4, 4]);  // First level (largest)
/// ```
///
/// Using a different wavelet family:
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt2d::wavedec2;
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a larger image to accommodate longer filters
/// let mut data = Array2::zeros((32, 32));
/// for i in 0..32 {
///     for j in 0..32 {
///         data[[i, j]] = ((i+j) % 8)  as f64;  // Create a pattern
///     }
/// }
///
/// // Decompose with Daubechies 4 wavelet
/// let coeffs = wavedec2(&data, Wavelet::DB(4), 2, None).expect("operation should succeed");
/// assert_eq!(coeffs.len(), 2);
/// ```
#[allow(dead_code)]
pub fn wavedec2<T>(
    data: &Array2<T>,
    wavelet: Wavelet,
    levels: usize,
    mode: Option<&str>,
) -> SignalResult<Vec<Dwt2dResult>>
where
    T: Float + NumCast + Debug,
{
    if data.is_empty() {
        return Err(SignalError::ValueError("Input array is empty".to_string()));
    }
    if levels == 0 {
        return Err(SignalError::ValueError("Levels must be greater than 0".to_string()));
    }
    let (rows, cols) = data.dim();
    let min_size = 2usize.pow(levels as u32);
    if rows < min_size || cols < min_size {
        return Err(
            SignalError::ValueError(
                format!(
                    "Data size ({}, {}) is too small for {} levels of decomposition",
                    rows, cols, levels
                ),
            ),
        );
    }
    let mut result = Vec::with_capacity(levels);
    let mut decomposition = dwt2d_decompose(data, wavelet, mode)?;
    result.push(decomposition.clone());
    for _level in 1..levels {
        decomposition = dwt2d_decompose(&decomposition.approx, wavelet, mode)?;
        result.push(decomposition.clone());
    }
    result.reverse();
    Ok(result)
}
/// Reconstructs a 2D signal from its multi-level wavelet decomposition.
///
/// This function is the inverse of `wavedec2` and reconstructs a 2D array (such as an image)
/// from its multi-level wavelet decomposition. It processes the coefficients from deepest
/// to shallowest level, gradually building up the full-resolution image.
///
/// # Algorithm
///
/// The multi-level reconstruction works by:
/// 1. Starting with the approximation coefficients at the deepest level
/// 2. Combining these with the detail coefficients at that level to get a higher-resolution approximation
/// 3. Repeating this process level by level until the full-resolution image is reconstructed
///
/// # Arguments
///
/// * `coeffs` - The wavelet coefficients from `wavedec2`, with deepest level first
/// * `wavelet` - The wavelet used for the original transform (must match)
/// * `mode` - The signal extension mode (default: "symmetric")
///   - Should match the mode used for decomposition
///
/// # Returns
///
/// * The reconstructed 2D array with the same dimensions as the original input to `wavedec2`
///
/// # Errors
///
/// Returns an error if:
/// * The coefficient list is empty
/// * The detail coefficients at any level do not match the approximation shape
/// * Other errors from the underlying `dwt2d_reconstruct` function
///
/// # Applications
///
/// This function is particularly useful for:
/// * Image compression (after coefficient thresholding)
/// * Denoising (after removing noise from detail coefficients)
/// * Feature extraction at multiple scales
/// * Image fusion
///
/// # Examples
///
/// Basic multi-level decomposition and reconstruction:
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt2d::{wavedec2, waverec2};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple 8x8 "image"
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * 8 + j + 1)  as f64;
///     }
/// }
///
/// // Decompose
/// let coeffs = wavedec2(&data, Wavelet::Haar, 3, None).expect("operation should succeed");
///
/// // Reconstruct
/// let reconstructed = waverec2(&coeffs, Wavelet::Haar, None).expect("operation should succeed");
///
/// // Check that reconstruction has the correct shape
/// assert_eq!(reconstructed.shape(), data.shape());
/// ```
///
/// Simple image compression by coefficient thresholding:
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt2d::{wavedec2, waverec2, Dwt2dResult};
/// use scirs2_signal::dwt::Wavelet;
///
/// // Create a simple 16x16 "image" with a pattern
/// let mut data = Array2::zeros((16, 16));
/// for i in 0..16 {
///     for j in 0..16 {
///         data[[i, j]] = ((i as f64 - 8.0).powi(2) + (j as f64 - 8.0).powi(2)).sqrt();
///     }
/// }
///
/// // Multi-level decomposition
/// let mut coeffs = wavedec2(&data, Wavelet::DB(4), 2, None).expect("operation should succeed");
///
/// // Threshold small detail coefficients to achieve compression
/// let threshold = 0.5;
/// for level in &mut coeffs {
///     // Only threshold detail coefficients, not approximation
///     for h in level.detail_h.iter_mut() {
///         if h.abs() < threshold { *h = 0.0; }
///     }
///     for v in level.detail_v.iter_mut() {
///         if v.abs() < threshold { *v = 0.0; }
///     }
///     for d in level.detail_d.iter_mut() {
///         if d.abs() < threshold { *d = 0.0; }
///     }
/// }
///
/// // Reconstruct from thresholded coefficients
/// let compressed = waverec2(&coeffs, Wavelet::DB(4), None).expect("operation should succeed");
/// ```
#[allow(dead_code)]
pub fn waverec2(
    coeffs: &[Dwt2dResult],
    wavelet: Wavelet,
    mode: Option<&str>,
) -> SignalResult<Array2<f64>> {
    if coeffs.is_empty() {
        return Err(SignalError::ValueError("Coefficient list is empty".to_string()));
    }
    let mut approx = coeffs[0].approx.clone();
    for decomp in coeffs {
        let synthetic_decomp = Dwt2dResult {
            approx,
            detail_h: decomp.detail_h.clone(),
            detail_v: decomp.detail_v.clone(),
            detail_d: decomp.detail_d.clone(),
        };
        approx = dwt2d_reconstruct(&synthetic_decomp, wavelet, mode)?;
    }
    Ok(approx)
}
/// Apply thresholding to wavelet coefficients for denoising or compression.
///
/// This function applies a threshold to the detail coefficients of a wavelet decomposition.
/// It is commonly used for denoising (removing low-amplitude noise) and compression
/// (removing less significant coefficients). Only detail coefficients are thresholded;
/// approximation coefficients are left unchanged.
///
/// # Arguments
///
/// * `decomposition` - The wavelet decomposition to threshold (will be modified in-place)
/// * `threshold` - The threshold value (coefficients with absolute value below this will be modified)
/// * `method` - The thresholding method to apply (Hard, Soft, or Garrote)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, dwt2d_reconstruct, threshold_dwt2d, ThresholdMethod};
///
/// // Create a sample "image"
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Decompose with wavelet transform
/// let mut decomposition = dwt2d_decompose(&data, Wavelet::DB(4), None).expect("operation should succeed");
///
/// // Apply hard thresholding to detail coefficients
/// threshold_dwt2d(&mut decomposition, 1.0, ThresholdMethod::Hard);
///
/// // Reconstruct the denoised image
/// let denoised = dwt2d_reconstruct(&decomposition, Wavelet::DB(4), None).expect("operation should succeed");
/// ```
#[allow(dead_code)]
pub fn threshold_dwt2d(
    decomposition: &mut Dwt2dResult,
    threshold: f64,
    method: ThresholdMethod,
) {
    if let Some(h_slice) = decomposition.detail_h.as_slice_mut() {
        simd_threshold_coefficients(h_slice, threshold, method);
    } else {
        for h in decomposition.detail_h.iter_mut() {
            *h = apply_threshold(*h, threshold, method);
        }
    }
    if let Some(v_slice) = decomposition.detail_v.as_slice_mut() {
        simd_threshold_coefficients(v_slice, threshold, method);
    } else {
        for v in decomposition.detail_v.iter_mut() {
            *v = apply_threshold(*v, threshold, method);
        }
    }
    if let Some(d_slice) = decomposition.detail_d.as_slice_mut() {
        simd_threshold_coefficients(d_slice, threshold, method);
    } else {
        for d in decomposition.detail_d.iter_mut() {
            *d = apply_threshold(*d, threshold, method);
        }
    }
}
/// Apply thresholding to multi-level wavelet coefficients.
///
/// Similar to `threshold_dwt2d`, but operates on a multi-level decomposition from `wavedec2`.
/// This allows for level-dependent thresholding, which can be more effective for certain
/// applications.
///
/// # Arguments
///
/// * `coeffs` - The multi-level wavelet decomposition to threshold (modified in-place)
/// * `threshold` - The threshold value, or vector of threshold values (one per level)
/// * `method` - The thresholding method to apply
///
/// # Examples
///
/// Using a different threshold for each level:
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{wavedec2, waverec2, threshold_wavedec2, ThresholdMethod};
///
/// // Create a sample image
/// let mut data = Array2::zeros((16, 16));
/// for i in 0..16 {
///     for j in 0..16 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Multi-level decomposition (3 levels)
/// let mut coeffs = wavedec2(&data, Wavelet::Haar, 3, None).expect("operation should succeed");
///
/// // Apply different thresholds for each level (higher thresholds for finer details)
/// let thresholds = vec![5.0, 10.0, 15.0];
/// threshold_wavedec2(&mut coeffs, &thresholds, ThresholdMethod::Soft);
///
/// // Reconstruct from thresholded coefficients
/// let result = waverec2(&coeffs, Wavelet::Haar, None).expect("operation should succeed");
/// ```
#[allow(dead_code)]
pub fn threshold_wavedec2(
    coeffs: &mut [Dwt2dResult],
    threshold: &[f64],
    method: ThresholdMethod,
) {
    for (i, level) in coeffs.iter_mut().enumerate() {
        let level_threshold = if i < threshold.len() {
            threshold[i]
        } else {
            *threshold.last().unwrap_or(&0.0)
        };
        threshold_dwt2d(level, level_threshold, method);
    }
}
/// Helper function to apply a threshold to a single coefficient.
#[allow(dead_code)]
pub(super) fn apply_threshold(x: f64, threshold: f64, method: ThresholdMethod) -> f64 {
    let abs_x = x.abs();
    if abs_x <= threshold {
        return 0.0;
    }
    match method {
        ThresholdMethod::Hard => x,
        ThresholdMethod::Soft => x.signum() * (abs_x - threshold),
        ThresholdMethod::Garrote => x * (1.0 - (threshold * threshold) / (x * x)),
    }
}
/// Calculate the energy of wavelet coefficients in a decomposition.
///
/// Energy is the sum of squared coefficients. This function is useful for analyzing
/// the distribution of energy across different subbands and for determining appropriate
/// threshold values.
///
/// # Arguments
///
/// * `decomposition` - The wavelet decomposition to analyze
/// * `include_approx` - Whether to include approximation coefficients in the calculation
///
/// # Returns
///
/// * A tuple containing the total energy and a struct with energy by subband:
///   - approx: Energy in the approximation coefficients (LL band)
///   - detail_h: Energy in the horizontal detail coefficients (LH band)
///   - detail_v: Energy in the vertical detail coefficients (HL band)
///   - detail_d: Energy in the diagonal detail coefficients (HH band)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::dwt::Wavelet;
/// use scirs2_signal::dwt2d::{dwt2d_decompose, calculate_energy};
///
/// // Create a sample image
/// let mut data = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         data[[i, j]] = (i * j)  as f64;
///     }
/// }
///
/// // Decompose the image
/// let decomposition = dwt2d_decompose(&data, Wavelet::Haar, None).expect("operation should succeed");
///
/// // Calculate energy, including approximation coefficients
/// let (total_energy, energy_by_subband) = calculate_energy(&decomposition, true);
///
/// // Typically, approximation coefficients contain most of the energy
/// assert!(energy_by_subband.approx > energy_by_subband.detail_h);
/// assert!(energy_by_subband.approx > energy_by_subband.detail_v);
/// assert!(energy_by_subband.approx > energy_by_subband.detail_d);
/// ```
#[allow(dead_code)]
pub fn calculate_energy(
    _decomposition: &Dwt2dResult,
    include_approx: bool,
) -> (f64, WaveletEnergy) {
    let approx_energy = if include_approx {
        if let Some(approx_slice) = decomposition.approx.as_slice() {
            simd_calculate_energy(approx_slice)
        } else {
            decomposition.approx.iter().map(|&x| x * x).sum()
        }
    } else {
        0.0
    };
    let detail_h_energy = if let Some(h_slice) = decomposition.detail_h.as_slice() {
        simd_calculate_energy(h_slice)
    } else {
        decomposition.detail_h.iter().map(|&x| x * x).sum()
    };
    let detail_v_energy = if let Some(v_slice) = decomposition.detail_v.as_slice() {
        simd_calculate_energy(v_slice)
    } else {
        decomposition.detail_v.iter().map(|&x| x * x).sum()
    };
    let detail_d_energy = if let Some(d_slice) = decomposition.detail_d.as_slice() {
        simd_calculate_energy(d_slice)
    } else {
        decomposition.detail_d.iter().map(|&x| x * x).sum()
    };
    let total = approx_energy + detail_h_energy + detail_v_energy + detail_d_energy;
    let energy_by_subband = WaveletEnergy {
        approx: approx_energy,
        detail_h: detail_h_energy,
        detail_v: detail_v_energy,
        detail_d: detail_d_energy,
    };
    (total, energy_by_subband)
}
