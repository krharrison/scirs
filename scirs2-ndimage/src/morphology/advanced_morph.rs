//! Advanced morphological operations
//!
//! This module provides advanced morphological operations:
//! - `morphological_reconstruction_by_dilation` - geodesic reconstruction by dilation
//! - `morphological_reconstruction_by_erosion` - geodesic reconstruction by erosion
//! - `fill_holes_2d` - fill holes in binary 2D images
//! - `morphological_gradient_2d_fast` - dilation minus erosion
//! - `white_tophat_2d_fast` - image minus opening
//! - `black_tophat_2d_fast` - closing minus image

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::VecDeque;
use std::fmt::Debug;

/// Morphological reconstruction by dilation using fast hybrid algorithm
///
/// Reconstructs the mask image by iteratively dilating the marker
/// subject to the constraint that the result never exceeds the mask.
/// Uses a two-pass raster scan followed by a queue-based propagation
/// for O(n) average-case performance.
///
/// # Arguments
///
/// * `marker` - Marker image (must be <= mask everywhere)
/// * `mask` - Mask image (constraining upper bound)
/// * `connectivity` - 4 or 8 connectivity (default: 8)
///
/// # Returns
///
/// * `Result<Array2<T>>` - Reconstructed image
pub fn morphological_reconstruction_by_dilation<T>(
    marker: &Array2<T>,
    mask: &Array2<T>,
    connectivity: Option<usize>,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Copy + 'static,
{
    if marker.shape() != mask.shape() {
        return Err(NdimageError::DimensionError(
            "Marker and mask must have the same shape".into(),
        ));
    }

    let rows = marker.nrows();
    let cols = marker.ncols();

    if rows == 0 || cols == 0 {
        return Ok(marker.clone());
    }

    let conn = connectivity.unwrap_or(8);
    if conn != 4 && conn != 8 {
        return Err(NdimageError::InvalidInput(
            "Connectivity must be 4 or 8".into(),
        ));
    }

    // Validate marker <= mask
    for r in 0..rows {
        for c in 0..cols {
            if marker[[r, c]] > mask[[r, c]] {
                return Err(NdimageError::InvalidInput(
                    "Marker must be pointwise <= mask".into(),
                ));
            }
        }
    }

    // Initialize result with marker clamped to mask
    let mut result = marker.clone();

    let offsets_4: &[(isize, isize)] = &[(-1, 0), (0, -1), (0, 1), (1, 0)];
    let offsets_8: &[(isize, isize)] = &[
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    let offsets = if conn == 4 { offsets_4 } else { offsets_8 };

    // Forward raster scan (top-left to bottom-right)
    // Check forward neighbors: those already visited (above and left)
    let forward_offsets_4: &[(isize, isize)] = &[(-1, 0), (0, -1)];
    let forward_offsets_8: &[(isize, isize)] = &[(-1, -1), (-1, 0), (-1, 1), (0, -1)];
    let fwd = if conn == 4 {
        forward_offsets_4
    } else {
        forward_offsets_8
    };

    for r in 0..rows {
        for c in 0..cols {
            let mut max_val = result[[r, c]];
            for &(dr, dc) in fwd {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                    let nv = result[[nr as usize, nc as usize]];
                    if nv > max_val {
                        max_val = nv;
                    }
                }
            }
            // Clamp to mask
            result[[r, c]] = if max_val < mask[[r, c]] {
                max_val
            } else {
                mask[[r, c]]
            };
        }
    }

    // Backward raster scan (bottom-right to top-left)
    let backward_offsets_4: &[(isize, isize)] = &[(1, 0), (0, 1)];
    let backward_offsets_8: &[(isize, isize)] = &[(0, 1), (1, -1), (1, 0), (1, 1)];
    let bwd = if conn == 4 {
        backward_offsets_4
    } else {
        backward_offsets_8
    };

    // Queue for propagation phase
    let mut queue = VecDeque::new();

    for r in (0..rows).rev() {
        for c in (0..cols).rev() {
            let mut max_val = result[[r, c]];
            for &(dr, dc) in bwd {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                    let nv = result[[nr as usize, nc as usize]];
                    if nv > max_val {
                        max_val = nv;
                    }
                }
            }
            result[[r, c]] = if max_val < mask[[r, c]] {
                max_val
            } else {
                mask[[r, c]]
            };

            // Check if any backward neighbor needs propagation
            for &(dr, dc) in bwd {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                    let nu = nr as usize;
                    let ncu = nc as usize;
                    if result[[nu, ncu]] < result[[r, c]] && result[[nu, ncu]] < mask[[nu, ncu]] {
                        queue.push_back((r, c));
                        break;
                    }
                }
            }
        }
    }

    // Queue-based propagation
    while let Some((r, c)) = queue.pop_front() {
        for &(dr, dc) in offsets {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                let nu = nr as usize;
                let ncu = nc as usize;
                if result[[nu, ncu]] < result[[r, c]] && result[[nu, ncu]] != mask[[nu, ncu]] {
                    let new_val = if result[[r, c]] < mask[[nu, ncu]] {
                        result[[r, c]]
                    } else {
                        mask[[nu, ncu]]
                    };
                    if new_val > result[[nu, ncu]] {
                        result[[nu, ncu]] = new_val;
                        queue.push_back((nu, ncu));
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Morphological reconstruction by erosion using fast hybrid algorithm
///
/// Dual of reconstruction by dilation. Reconstructs by iteratively
/// eroding the marker, constrained to be >= mask everywhere.
///
/// # Arguments
///
/// * `marker` - Marker image (must be >= mask everywhere)
/// * `mask` - Mask image (constraining lower bound)
/// * `connectivity` - 4 or 8 connectivity (default: 8)
///
/// # Returns
///
/// * `Result<Array2<T>>` - Reconstructed image
pub fn morphological_reconstruction_by_erosion<T>(
    marker: &Array2<T>,
    mask: &Array2<T>,
    connectivity: Option<usize>,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Copy + 'static,
{
    if marker.shape() != mask.shape() {
        return Err(NdimageError::DimensionError(
            "Marker and mask must have the same shape".into(),
        ));
    }

    let rows = marker.nrows();
    let cols = marker.ncols();

    if rows == 0 || cols == 0 {
        return Ok(marker.clone());
    }

    let conn = connectivity.unwrap_or(8);
    if conn != 4 && conn != 8 {
        return Err(NdimageError::InvalidInput(
            "Connectivity must be 4 or 8".into(),
        ));
    }

    // Negate both images, apply reconstruction by dilation, negate result
    let neg_marker = marker.mapv(|v| T::zero() - v);
    let neg_mask = mask.mapv(|v| T::zero() - v);

    let neg_result = morphological_reconstruction_by_dilation(&neg_marker, &neg_mask, Some(conn))?;
    Ok(neg_result.mapv(|v| T::zero() - v))
}

/// Fill holes in a binary 2D image
///
/// A hole is a set of background pixels that cannot be reached from the
/// image border by traveling through background pixels only. This function
/// fills all such holes, producing an image where the only background
/// pixels are those connected to the border.
///
/// Uses morphological reconstruction by dilation: the marker is a frame
/// of True pixels on the border and False everywhere else; the mask is
/// the complement of the input. The reconstruction of the complement
/// gives the background; re-complementing gives the filled image.
///
/// # Arguments
///
/// * `input` - Input binary 2D array (true = foreground)
///
/// # Returns
///
/// * `Result<Array2<bool>>` - Image with holes filled
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::morphology::fill_holes_2d;
///
/// let input = array![
///     [false, false, false, false, false],
///     [false, true,  true,  true,  false],
///     [false, true,  false, true,  false],
///     [false, true,  true,  true,  false],
///     [false, false, false, false, false],
/// ];
///
/// let filled = fill_holes_2d(&input).expect("fill_holes_2d should succeed");
/// // The hole at (2,2) should now be filled
/// assert_eq!(filled[[2, 2]], true);
/// ```
pub fn fill_holes_2d(input: &Array2<bool>) -> NdimageResult<Array2<bool>> {
    let rows = input.nrows();
    let cols = input.ncols();

    if rows == 0 || cols == 0 {
        return Ok(input.clone());
    }

    // Complement of input: holes become foreground
    let complement = input.mapv(|v| !v);

    // Create marker: True on the border of complement, False elsewhere
    let mut marker = Array2::from_elem((rows, cols), false);

    // Top and bottom rows
    for c in 0..cols {
        marker[[0, c]] = complement[[0, c]];
        marker[[rows - 1, c]] = complement[[rows - 1, c]];
    }
    // Left and right columns
    for r in 0..rows {
        marker[[r, 0]] = complement[[r, 0]];
        marker[[r, cols - 1]] = complement[[r, cols - 1]];
    }

    // Convert to f64 for reconstruction
    let marker_f64 = marker.mapv(|v| if v { 1.0f64 } else { 0.0f64 });
    let mask_f64 = complement.mapv(|v| if v { 1.0f64 } else { 0.0f64 });

    // Reconstruct: this gives us the background connected to the border
    let reconstructed = morphological_reconstruction_by_dilation(&marker_f64, &mask_f64, Some(8))?;

    // The filled result is the complement of the reconstructed background
    let result = reconstructed.mapv(|v| v < 0.5);

    Ok(result)
}

/// Fast 2D morphological gradient (dilation - erosion)
///
/// The morphological gradient highlights edges by computing the
/// difference between dilation and erosion with the given structuring element.
///
/// # Arguments
///
/// * `input` - Input 2D grayscale image
/// * `size` - Size of the square structuring element (default: 3)
///
/// # Returns
///
/// * `Result<Array2<T>>` - Gradient image
pub fn morphological_gradient_2d_fast<T>(
    input: &Array2<T>,
    size: Option<usize>,
) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Copy + 'static,
{
    let sz = size.unwrap_or(3);
    if sz == 0 || sz % 2 == 0 {
        return Err(NdimageError::InvalidInput(
            "Structuring element size must be odd and > 0".into(),
        ));
    }

    let rows = input.nrows();
    let cols = input.ncols();
    let half = (sz / 2) as isize;

    let mut dilated = Array2::from_elem((rows, cols), T::neg_infinity());
    let mut eroded = Array2::from_elem((rows, cols), T::infinity());

    for r in 0..rows {
        for c in 0..cols {
            for dr in -half..=half {
                for dc in -half..=half {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                        let val = input[[nr as usize, nc as usize]];
                        if val > dilated[[r, c]] {
                            dilated[[r, c]] = val;
                        }
                        if val < eroded[[r, c]] {
                            eroded[[r, c]] = val;
                        }
                    }
                }
            }
        }
    }

    // gradient = dilation - erosion
    let mut result = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            result[[r, c]] = dilated[[r, c]] - eroded[[r, c]];
        }
    }

    Ok(result)
}

/// Fast 2D white top-hat transform (image - opening)
///
/// Extracts bright features smaller than the structuring element.
/// Opening removes small bright regions; subtracting from original
/// isolates those removed features.
///
/// # Arguments
///
/// * `input` - Input 2D grayscale image
/// * `size` - Size of the square structuring element (default: 3)
///
/// # Returns
///
/// * `Result<Array2<T>>` - White top-hat image
pub fn white_tophat_2d_fast<T>(input: &Array2<T>, size: Option<usize>) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Copy + 'static,
{
    let sz = size.unwrap_or(3);
    if sz == 0 || sz % 2 == 0 {
        return Err(NdimageError::InvalidInput(
            "Structuring element size must be odd and > 0".into(),
        ));
    }

    let rows = input.nrows();
    let cols = input.ncols();
    let half = (sz / 2) as isize;

    // Erosion
    let mut eroded = Array2::from_elem((rows, cols), T::infinity());
    for r in 0..rows {
        for c in 0..cols {
            for dr in -half..=half {
                for dc in -half..=half {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                        let val = input[[nr as usize, nc as usize]];
                        if val < eroded[[r, c]] {
                            eroded[[r, c]] = val;
                        }
                    }
                }
            }
        }
    }

    // Dilation of eroded (= opening)
    let mut opened = Array2::from_elem((rows, cols), T::neg_infinity());
    for r in 0..rows {
        for c in 0..cols {
            for dr in -half..=half {
                for dc in -half..=half {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                        let val = eroded[[nr as usize, nc as usize]];
                        if val > opened[[r, c]] {
                            opened[[r, c]] = val;
                        }
                    }
                }
            }
        }
    }

    // white top-hat = input - opening
    let mut result = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let diff = input[[r, c]] - opened[[r, c]];
            result[[r, c]] = if diff > T::zero() { diff } else { T::zero() };
        }
    }

    Ok(result)
}

/// Fast 2D black top-hat transform (closing - image)
///
/// Extracts dark features smaller than the structuring element.
/// Closing fills small dark regions; subtracting original isolates
/// those filled features.
///
/// # Arguments
///
/// * `input` - Input 2D grayscale image
/// * `size` - Size of the square structuring element (default: 3)
///
/// # Returns
///
/// * `Result<Array2<T>>` - Black top-hat image
pub fn black_tophat_2d_fast<T>(input: &Array2<T>, size: Option<usize>) -> NdimageResult<Array2<T>>
where
    T: Float + FromPrimitive + Debug + Copy + 'static,
{
    let sz = size.unwrap_or(3);
    if sz == 0 || sz % 2 == 0 {
        return Err(NdimageError::InvalidInput(
            "Structuring element size must be odd and > 0".into(),
        ));
    }

    let rows = input.nrows();
    let cols = input.ncols();
    let half = (sz / 2) as isize;

    // Dilation
    let mut dilated = Array2::from_elem((rows, cols), T::neg_infinity());
    for r in 0..rows {
        for c in 0..cols {
            for dr in -half..=half {
                for dc in -half..=half {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                        let val = input[[nr as usize, nc as usize]];
                        if val > dilated[[r, c]] {
                            dilated[[r, c]] = val;
                        }
                    }
                }
            }
        }
    }

    // Erosion of dilated (= closing)
    let mut closed = Array2::from_elem((rows, cols), T::infinity());
    for r in 0..rows {
        for c in 0..cols {
            for dr in -half..=half {
                for dc in -half..=half {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                        let val = dilated[[nr as usize, nc as usize]];
                        if val < closed[[r, c]] {
                            closed[[r, c]] = val;
                        }
                    }
                }
            }
        }
    }

    // black top-hat = closing - input
    let mut result = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let diff = closed[[r, c]] - input[[r, c]];
            result[[r, c]] = if diff > T::zero() { diff } else { T::zero() };
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_reconstruction_by_dilation_basic() {
        // Marker is all zeros except one pixel, mask is the original image
        let mask: Array2<f64> = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.5, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let mut marker: Array2<f64> = Array2::zeros((5, 5));
        marker[[1, 1]] = 1.0; // seed at top-left of the bright region

        let result = morphological_reconstruction_by_dilation(&marker, &mask, Some(8))
            .expect("reconstruction should succeed");

        // The bright region (1.0) should be fully reconstructed
        assert_abs_diff_eq!(result[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 2]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 3]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 1]], 1.0, epsilon = 1e-10);
        // Interior lower pixel should be reconstructed to its mask value
        assert_abs_diff_eq!(result[[2, 2]], 0.5, epsilon = 1e-10);
        // Background should remain 0
        assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reconstruction_by_dilation_shape_mismatch() {
        let marker: Array2<f64> = Array2::zeros((3, 3));
        let mask: Array2<f64> = Array2::zeros((4, 4));
        let result = morphological_reconstruction_by_dilation(&marker, &mask, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_reconstruction_by_dilation_marker_exceeds_mask() {
        let marker: Array2<f64> = Array2::from_elem((3, 3), 2.0);
        let mask: Array2<f64> = Array2::from_elem((3, 3), 1.0);
        let result = morphological_reconstruction_by_dilation(&marker, &mask, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_reconstruction_by_erosion_basic() {
        // Reconstruction by erosion iteratively erodes the marker, constrained >= mask.
        // A uniform (flat) marker cannot be reduced by erosion (min of constant = constant),
        // so reconstruction by erosion with a uniform marker always returns the marker value.
        let mask: Array2<f64> = array![[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0],];
        let marker: Array2<f64> = Array2::from_elem((3, 3), 1.0);

        let result = morphological_reconstruction_by_erosion(&marker, &mask, Some(8))
            .expect("erosion reconstruction should succeed");

        // Uniform marker => erosion converges immediately, result = marker = 1.0 everywhere
        assert_abs_diff_eq!(result[[1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);

        // Test with a non-uniform marker: marker has a peak at center, mask is lower.
        // Erosion should reduce the peak to the level of its neighbors (constrained by mask).
        let mask2: Array2<f64> = array![[0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2],];
        let marker2: Array2<f64> = array![[0.2, 0.2, 0.2], [0.2, 1.0, 0.2], [0.2, 0.2, 0.2],];

        let result2 = morphological_reconstruction_by_erosion(&marker2, &mask2, Some(8))
            .expect("erosion reconstruction should succeed");

        // The peak at (1,1) should be eroded down to the surrounding level (0.2),
        // constrained to be >= mask (0.2), so result = 0.2 everywhere.
        assert_abs_diff_eq!(result2[[1, 1]], 0.2, epsilon = 1e-10);
        assert_abs_diff_eq!(result2[[0, 0]], 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_fill_holes_2d_basic() {
        let input = array![
            [false, false, false, false, false],
            [false, true, true, true, false],
            [false, true, false, true, false],
            [false, true, true, true, false],
            [false, false, false, false, false],
        ];

        let filled = fill_holes_2d(&input).expect("fill_holes_2d should succeed");

        // The hole at (2,2) should now be filled
        assert!(filled[[2, 2]]);
        // Border background should remain background
        assert!(!filled[[0, 0]]);
        assert!(!filled[[0, 4]]);
        assert!(!filled[[4, 0]]);
        assert!(!filled[[4, 4]]);
        // Foreground should remain foreground
        assert!(filled[[1, 1]]);
        assert!(filled[[1, 2]]);
    }

    #[test]
    fn test_fill_holes_2d_no_holes() {
        // Solid object with no holes
        let input = array![
            [false, false, false],
            [false, true, false],
            [false, false, false],
        ];

        let filled = fill_holes_2d(&input).expect("no holes should succeed");
        assert_eq!(input, filled);
    }

    #[test]
    fn test_fill_holes_2d_all_background() {
        let input = Array2::from_elem((4, 4), false);
        let filled = fill_holes_2d(&input).expect("all background should succeed");
        assert_eq!(input, filled);
    }

    #[test]
    fn test_fill_holes_2d_empty() {
        let input = Array2::<bool>::from_elem((0, 0), false);
        let filled = fill_holes_2d(&input).expect("empty should succeed");
        assert_eq!(filled.len(), 0);
    }

    #[test]
    fn test_fill_holes_2d_multiple_holes() {
        let input = array![
            [false, false, false, false, false, false, false],
            [false, true, true, true, true, true, false],
            [false, true, false, true, false, true, false],
            [false, true, true, true, true, true, false],
            [false, false, false, false, false, false, false],
        ];

        let filled = fill_holes_2d(&input).expect("multiple holes should succeed");

        // Both holes should be filled
        assert!(filled[[2, 2]]);
        assert!(filled[[2, 4]]);
        // Background exterior should remain
        assert!(!filled[[0, 0]]);
    }

    #[test]
    fn test_morphological_gradient_2d_fast_basic() {
        let input: Array2<f64> = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        let grad =
            morphological_gradient_2d_fast(&input, Some(3)).expect("gradient should succeed");

        // Center pixel: dilation = 1.0, erosion = 1.0, gradient = 0.0
        assert_abs_diff_eq!(grad[[2, 2]], 0.0, epsilon = 1e-10);
        // Edge pixel: dilation = 1.0, erosion = 0.0, gradient = 1.0
        assert_abs_diff_eq!(grad[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_morphological_gradient_invalid_size() {
        let input: Array2<f64> = Array2::zeros((3, 3));
        let result = morphological_gradient_2d_fast(&input, Some(2));
        assert!(result.is_err());
    }

    #[test]
    fn test_white_tophat_2d_fast_basic() {
        // Small bright feature on a flat background
        let mut input: Array2<f64> = Array2::zeros((7, 7));
        input[[3, 3]] = 1.0; // single bright pixel

        let wth = white_tophat_2d_fast(&input, Some(5)).expect("white tophat should succeed");

        // The single bright pixel should appear in the white top-hat
        assert!(wth[[3, 3]] > 0.0);
    }

    #[test]
    fn test_white_tophat_flat_image() {
        // Flat image: opening = image, so top-hat should be zero
        let input: Array2<f64> = Array2::from_elem((5, 5), 5.0);
        let wth = white_tophat_2d_fast(&input, Some(3)).expect("flat tophat should succeed");
        for &v in wth.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_black_tophat_2d_fast_basic() {
        // Small dark feature on bright background
        let mut input: Array2<f64> = Array2::from_elem((7, 7), 1.0);
        input[[3, 3]] = 0.0; // single dark pixel

        let bth = black_tophat_2d_fast(&input, Some(5)).expect("black tophat should succeed");

        // The single dark pixel should appear in the black top-hat
        assert!(bth[[3, 3]] > 0.0);
    }

    #[test]
    fn test_black_tophat_flat_image() {
        // Flat image: closing = image, so black top-hat should be zero
        let input: Array2<f64> = Array2::from_elem((5, 5), 3.0);
        let bth = black_tophat_2d_fast(&input, Some(3)).expect("flat btophat should succeed");
        for &v in bth.iter() {
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_reconstruction_4_connectivity() {
        // Test 4-connectivity: diagonal shouldn't propagate
        let mask: Array2<f64> = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],];
        let mut marker: Array2<f64> = Array2::zeros((3, 3));
        marker[[0, 0]] = 1.0;

        let result = morphological_reconstruction_by_dilation(&marker, &mask, Some(4))
            .expect("4-conn reconstruction should succeed");

        // Only (0,0) should be 1.0; diagonals are not 4-connected
        assert_abs_diff_eq!(result[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[2, 2]], 0.0, epsilon = 1e-10);
    }
}
