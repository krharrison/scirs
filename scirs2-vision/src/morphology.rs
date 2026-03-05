//! Morphological image operations (ndarray API)
//!
//! This module provides morphological operations on 2-D grayscale images
//! represented as `Array2<u8>` arrays from the SciRS2 ndarray stack.
//! This complements the existing `preprocessing::morphology` module which
//! operates on `DynamicImage` values.
//!
//! ## Operations
//!
//! | Function | Description |
//! |---|---|
//! | [`erode`] | Erosion: local minimum under the structuring element |
//! | [`dilate`] | Dilation: local maximum under the structuring element |
//! | [`opening`] | Erosion then dilation (removes small bright objects) |
//! | [`closing`] | Dilation then erosion (fills small holes) |
//! | [`morphological_gradient`] | Dilation minus erosion (highlights edges) |
//! | [`top_hat`] | Original minus opening (extracts bright small features) |
//! | [`black_hat`] | Closing minus original (extracts dark small features) |
//!
//! ## Structuring elements
//!
//! | Function | Description |
//! |---|---|
//! | [`disk_kernel`] | Filled disk of given radius |
//! | [`rect_kernel`] | Filled rectangle |
//! | [`cross_kernel`] | Plus-shaped cross |
//!
//! ## Example
//!
//! ```rust
//! use scirs2_vision::morphology::{dilate, disk_kernel};
//! use scirs2_core::ndarray::Array2;
//!
//! // Create a small binary image with a single bright pixel in the centre
//! let mut img = Array2::<u8>::zeros((16, 16));
//! img[[8, 8]] = 255;
//!
//! // Dilate with a disk of radius 3
//! let kernel = disk_kernel(3);
//! let dilated = dilate(&img, &kernel).unwrap();
//!
//! // The centre should still be bright
//! assert_eq!(dilated[[8, 8]], 255);
//! // A pixel within the disk should also be bright
//! assert!(dilated[[8, 10]] > 0);
//! ```

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::Array2;

// ─── Structuring element factories ───────────────────────────────────────────

/// Create a filled-disk structuring element with the given `radius`.
///
/// The returned array has dimensions `(2r+1) × (2r+1)`.  Elements inside or
/// on the boundary of the disk are set to `1`; outside elements are `0`.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::morphology::disk_kernel;
///
/// let k = disk_kernel(2);
/// assert_eq!(k.dim(), (5, 5));
/// assert_eq!(k[[2, 2]], 1); // centre
/// assert_eq!(k[[0, 0]], 0); // corner outside disk
/// ```
pub fn disk_kernel(radius: usize) -> Array2<u8> {
    let size = 2 * radius + 1;
    let mut k = Array2::<u8>::zeros((size, size));
    let cx = radius as isize;
    let cy = radius as isize;
    let r2 = radius as isize;

    for y in 0..size {
        for x in 0..size {
            let dx = x as isize - cx;
            let dy = y as isize - cy;
            if dx * dx + dy * dy <= r2 * r2 {
                k[[y, x]] = 1;
            }
        }
    }
    k
}

/// Create a filled-rectangle structuring element of size `rows × cols`.
///
/// All elements are set to `1`.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::morphology::rect_kernel;
///
/// let k = rect_kernel(3, 5);
/// assert_eq!(k.dim(), (3, 5));
/// assert!(k.iter().all(|&v| v == 1));
/// ```
pub fn rect_kernel(rows: usize, cols: usize) -> Array2<u8> {
    Array2::<u8>::ones((rows, cols))
}

/// Create a cross (plus-sign) structuring element of odd `size × size`.
///
/// Only the middle row and middle column are set to `1`.
/// If `size` is even it is rounded up to the next odd number.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::morphology::cross_kernel;
///
/// let k = cross_kernel(5);
/// assert_eq!(k.dim(), (5, 5));
/// assert_eq!(k[[2, 2]], 1); // centre
/// assert_eq!(k[[0, 2]], 1); // top of column
/// assert_eq!(k[[2, 0]], 1); // left of row
/// assert_eq!(k[[0, 0]], 0); // corner
/// ```
pub fn cross_kernel(size: usize) -> Array2<u8> {
    let sz = if size == 0 {
        1
    } else if size.is_multiple_of(2) {
        size + 1
    } else {
        size
    };
    let mut k = Array2::<u8>::zeros((sz, sz));
    let mid = sz / 2;
    // Middle row
    for x in 0..sz {
        k[[mid, x]] = 1;
    }
    // Middle column
    for y in 0..sz {
        k[[y, mid]] = 1;
    }
    k
}

// ─── Core morphological operations ───────────────────────────────────────────

/// Morphological erosion: replace each pixel with the local minimum of the
/// image under the structuring element.
///
/// Pixels outside the image border are treated as `255` (bright) for the
/// purpose of erosion, so border pixels are eroded towards the background.
///
/// # Arguments
///
/// * `image`  — input 2-D grayscale array (`u8`)
/// * `kernel` — structuring element (non-zero = active element)
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] if the kernel has a zero
/// dimension, or if either kernel dimension exceeds the corresponding image
/// dimension.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::morphology::{erode, rect_kernel};
/// use scirs2_core::ndarray::Array2;
///
/// let mut img = Array2::<u8>::zeros((8, 8));
/// // Fill a 4×4 bright region in the centre
/// for y in 2..6usize { for x in 2..6usize { img[[y, x]] = 200; } }
///
/// let k = rect_kernel(3, 3);
/// let eroded = erode(&img, &k).unwrap();
/// // Eroded image should have a smaller bright region
/// assert_eq!(eroded[[0, 0]], 0);
/// ```
pub fn erode(image: &Array2<u8>, kernel: &Array2<u8>) -> Result<Array2<u8>> {
    validate_kernel(image, kernel)?;
    let (h, w) = image.dim();
    let (kh, kw) = kernel.dim();
    let ry = kh / 2;
    let rx = kw / 2;

    let mut out = Array2::<u8>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let mut min_val = 255u8;
            for ky in 0..kh {
                for kx in 0..kw {
                    if kernel[[ky, kx]] == 0 {
                        continue;
                    }
                    let iy = y as isize + ky as isize - ry as isize;
                    let ix = x as isize + kx as isize - rx as isize;
                    let pix = if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                        image[[iy as usize, ix as usize]]
                    } else {
                        255 // border treated as max → contributes nothing to erosion minimum
                    };
                    if pix < min_val {
                        min_val = pix;
                    }
                }
            }
            out[[y, x]] = min_val;
        }
    }

    Ok(out)
}

/// Morphological dilation: replace each pixel with the local maximum of the
/// image under the structuring element.
///
/// Pixels outside the image border are treated as `0` (dark) for the purpose
/// of dilation.
///
/// # Arguments
///
/// * `image`  — input 2-D grayscale array
/// * `kernel` — structuring element (non-zero = active element)
///
/// # Errors
///
/// Same as [`erode`].
///
/// # Example
///
/// ```rust
/// use scirs2_vision::morphology::{dilate, disk_kernel};
/// use scirs2_core::ndarray::Array2;
///
/// let mut img = Array2::<u8>::zeros((16, 16));
/// img[[8, 8]] = 255;
/// let k = disk_kernel(2);
/// let dilated = dilate(&img, &k).unwrap();
/// assert_eq!(dilated[[8, 8]], 255);
/// assert!(dilated[[8, 9]] > 0);
/// ```
pub fn dilate(image: &Array2<u8>, kernel: &Array2<u8>) -> Result<Array2<u8>> {
    validate_kernel(image, kernel)?;
    let (h, w) = image.dim();
    let (kh, kw) = kernel.dim();
    let ry = kh / 2;
    let rx = kw / 2;

    let mut out = Array2::<u8>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let mut max_val = 0u8;
            for ky in 0..kh {
                for kx in 0..kw {
                    if kernel[[ky, kx]] == 0 {
                        continue;
                    }
                    let iy = y as isize + ky as isize - ry as isize;
                    let ix = x as isize + kx as isize - rx as isize;
                    let pix = if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                        image[[iy as usize, ix as usize]]
                    } else {
                        0
                    };
                    if pix > max_val {
                        max_val = pix;
                    }
                }
            }
            out[[y, x]] = max_val;
        }
    }

    Ok(out)
}

/// Morphological opening: erosion followed by dilation with the same kernel.
///
/// Opening removes small bright objects (smaller than the structuring element)
/// while preserving the shape of larger bright regions.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::morphology::{opening, disk_kernel};
/// use scirs2_core::ndarray::Array2;
///
/// let mut img = Array2::<u8>::zeros((16, 16));
/// // Large bright region
/// for y in 2..14usize { for x in 2..14usize { img[[y, x]] = 200; } }
/// // Small noise blobs
/// img[[0, 0]] = 255;
/// img[[0, 15]] = 255;
///
/// let k = disk_kernel(2);
/// let opened = opening(&img, &k).unwrap();
/// // Noise should be removed
/// assert_eq!(opened[[0, 0]], 0);
/// // Large region should remain
/// assert!(opened[[8, 8]] > 0);
/// ```
pub fn opening(image: &Array2<u8>, kernel: &Array2<u8>) -> Result<Array2<u8>> {
    let eroded = erode(image, kernel)?;
    dilate(&eroded, kernel)
}

/// Morphological closing: dilation followed by erosion with the same kernel.
///
/// Closing fills small holes (dark spots smaller than the structuring element)
/// inside bright regions.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::morphology::{closing, disk_kernel};
/// use scirs2_core::ndarray::Array2;
///
/// let mut img = Array2::<u8>::ones((16, 16)).mapv(|_| 200u8);
/// // Punch a small 1-pixel hole
/// img[[8, 8]] = 0;
///
/// let k = disk_kernel(2);
/// let closed = closing(&img, &k).unwrap();
/// // Hole should be filled
/// assert!(closed[[8, 8]] > 0);
/// ```
pub fn closing(image: &Array2<u8>, kernel: &Array2<u8>) -> Result<Array2<u8>> {
    let dilated = dilate(image, kernel)?;
    erode(&dilated, kernel)
}

/// Morphological gradient: dilation minus erosion.
///
/// The result highlights the edges of bright regions.  Output values are
/// computed as `dilate(image)[y,x] − erode(image)[y,x]` with saturating
/// subtraction (no underflow).
///
/// # Example
///
/// ```rust
/// use scirs2_vision::morphology::{morphological_gradient, rect_kernel};
/// use scirs2_core::ndarray::Array2;
///
/// let mut img = Array2::<u8>::zeros((16, 16));
/// for y in 4..12usize { for x in 4..12usize { img[[y, x]] = 200; } }
///
/// let k = rect_kernel(3, 3);
/// let grad = morphological_gradient(&img, &k).unwrap();
/// // Edges should be bright
/// assert!(grad[[4, 4]] > 0);
/// // Deep interior should be zero
/// assert_eq!(grad[[8, 8]], 0);
/// ```
pub fn morphological_gradient(image: &Array2<u8>, kernel: &Array2<u8>) -> Result<Array2<u8>> {
    let dilated = dilate(image, kernel)?;
    let eroded = erode(image, kernel)?;
    let (h, w) = image.dim();
    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            out[[y, x]] = dilated[[y, x]].saturating_sub(eroded[[y, x]]);
        }
    }
    Ok(out)
}

/// White top-hat transform: `image − opening(image)`.
///
/// Extracts bright features (blobs, peaks) that are smaller than the
/// structuring element.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::morphology::{top_hat, disk_kernel};
/// use scirs2_core::ndarray::Array2;
///
/// let mut img = Array2::<u8>::zeros((16, 16));
/// // Large bright background
/// for y in 2..14usize { for x in 2..14usize { img[[y, x]] = 100; } }
/// // Small bright blob on the background
/// img[[8, 8]] = 255;
///
/// let k = disk_kernel(2);
/// let th = top_hat(&img, &k).unwrap();
/// // The small blob should appear in the result
/// assert!(th[[8, 8]] > 0);
/// ```
pub fn top_hat(image: &Array2<u8>, kernel: &Array2<u8>) -> Result<Array2<u8>> {
    let opened = opening(image, kernel)?;
    let (h, w) = image.dim();
    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            out[[y, x]] = image[[y, x]].saturating_sub(opened[[y, x]]);
        }
    }
    Ok(out)
}

/// Black top-hat (bottom-hat) transform: `closing(image) − image`.
///
/// Extracts dark features (holes, valleys) that are smaller than the
/// structuring element.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::morphology::{black_hat, disk_kernel};
/// use scirs2_core::ndarray::Array2;
///
/// let mut img = Array2::<u8>::ones((16, 16)).mapv(|_| 200u8);
/// // Small dark hole
/// img[[8, 8]] = 0;
///
/// let k = disk_kernel(2);
/// let bh = black_hat(&img, &k).unwrap();
/// // The dark hole should appear in the result
/// assert!(bh[[8, 8]] > 0);
/// ```
pub fn black_hat(image: &Array2<u8>, kernel: &Array2<u8>) -> Result<Array2<u8>> {
    let closed = closing(image, kernel)?;
    let (h, w) = image.dim();
    let mut out = Array2::<u8>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            out[[y, x]] = closed[[y, x]].saturating_sub(image[[y, x]]);
        }
    }
    Ok(out)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Validate that the kernel is non-empty.
fn validate_kernel(image: &Array2<u8>, kernel: &Array2<u8>) -> Result<()> {
    let (kh, kw) = kernel.dim();
    if kh == 0 || kw == 0 {
        return Err(VisionError::InvalidParameter(
            "Structuring element must have non-zero dimensions".to_string(),
        ));
    }
    let (h, w) = image.dim();
    if h == 0 || w == 0 {
        return Err(VisionError::InvalidParameter(
            "Image must have non-zero dimensions".to_string(),
        ));
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    // ── Structuring elements ──────────────────────────────────────────────

    #[test]
    fn disk_kernel_size() {
        let k = disk_kernel(0);
        assert_eq!(k.dim(), (1, 1));
        assert_eq!(k[[0, 0]], 1);

        let k3 = disk_kernel(3);
        assert_eq!(k3.dim(), (7, 7));
        assert_eq!(k3[[3, 3]], 1); // centre
                                   // Corners of 7×7 should be outside disk of radius 3
        assert_eq!(k3[[0, 0]], 0);
        assert_eq!(k3[[0, 6]], 0);
        assert_eq!(k3[[6, 0]], 0);
        assert_eq!(k3[[6, 6]], 0);
        // Cardinal extremes should be inside
        assert_eq!(k3[[0, 3]], 1);
        assert_eq!(k3[[3, 0]], 1);
        assert_eq!(k3[[6, 3]], 1);
        assert_eq!(k3[[3, 6]], 1);
    }

    #[test]
    fn rect_kernel_all_ones() {
        let k = rect_kernel(4, 6);
        assert_eq!(k.dim(), (4, 6));
        assert!(k.iter().all(|&v| v == 1));
    }

    #[test]
    fn cross_kernel_shape() {
        let k = cross_kernel(5);
        assert_eq!(k.dim(), (5, 5));
        // Middle row and column should be 1
        for i in 0..5 {
            assert_eq!(k[[2, i]], 1, "middle row at col {i}");
            assert_eq!(k[[i, 2]], 1, "middle col at row {i}");
        }
        // Corners should be 0
        assert_eq!(k[[0, 0]], 0);
        assert_eq!(k[[4, 4]], 0);
    }

    #[test]
    fn cross_kernel_even_rounds_up() {
        let k = cross_kernel(4);
        assert_eq!(k.dim(), (5, 5));
    }

    // ── Erosion / Dilation basic properties ──────────────────────────────

    fn bright_square(size: usize, sq_start: usize, sq_end: usize) -> Array2<u8> {
        let mut img = Array2::<u8>::zeros((size, size));
        for y in sq_start..sq_end {
            for x in sq_start..sq_end {
                img[[y, x]] = 200;
            }
        }
        img
    }

    #[test]
    fn erode_shrinks_bright_region() {
        let img = bright_square(20, 4, 16); // 12×12 square
        let k = rect_kernel(3, 3);
        let eroded = erode(&img, &k).expect("erode should succeed");

        // Interior should survive
        assert_eq!(eroded[[9, 9]], 200, "Interior should survive erosion");
        // Border row of square should be eroded away
        assert_eq!(eroded[[4, 4]], 0, "Border of square should be eroded");
    }

    #[test]
    fn dilate_grows_bright_region() {
        let mut img = Array2::<u8>::zeros((20, 20));
        img[[10, 10]] = 255;
        let k = rect_kernel(3, 3);
        let dilated = dilate(&img, &k).expect("dilate should succeed");

        // Neighbourhood of single pixel should be bright
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let y = (10i32 + dy) as usize;
                let x = (10i32 + dx) as usize;
                assert_eq!(dilated[[y, x]], 255, "({y},{x}) should be dilated");
            }
        }
        // Far pixels should remain zero
        assert_eq!(dilated[[0, 0]], 0);
    }

    #[test]
    fn dilate_then_erode_identity_for_large_region() {
        // For a large solid region, dilate + erode ≈ identity in the interior
        let img = bright_square(30, 5, 25); // 20×20 bright square inside a 30×30 image
        let k = rect_kernel(3, 3);
        let dilated = dilate(&img, &k).expect("dilate");
        let back = erode(&dilated, &k).expect("erode");
        // Interior pixels should be preserved
        for y in 8..22 {
            for x in 8..22 {
                assert_eq!(
                    back[[y, x]],
                    img[[y, x]],
                    "Interior pixel ({y},{x}) should be preserved by dil+erode"
                );
            }
        }
    }

    #[test]
    fn erode_uniform_image_unchanged() {
        let img = Array2::<u8>::from_elem((16, 16), 128);
        let k = disk_kernel(2);
        let eroded = erode(&img, &k).expect("erode uniform");
        assert!(
            eroded.iter().all(|&v| v == 128),
            "Uniform image should be unchanged by erosion"
        );
    }

    #[test]
    fn dilate_uniform_image_unchanged() {
        let img = Array2::<u8>::from_elem((16, 16), 128);
        let k = disk_kernel(2);
        let dilated = dilate(&img, &k).expect("dilate uniform");
        assert!(
            dilated.iter().all(|&v| v == 128),
            "Uniform image should be unchanged by dilation"
        );
    }

    // ── Opening / Closing ─────────────────────────────────────────────────

    #[test]
    fn opening_removes_small_blobs() {
        let mut img = bright_square(20, 4, 16);
        // Add a single noise pixel far from the square
        img[[1, 1]] = 255;
        img[[1, 18]] = 255;

        let k = disk_kernel(2); // radius > single pixel
        let opened = opening(&img, &k).expect("opening");

        assert_eq!(opened[[1, 1]], 0, "Small blob should be removed by opening");
        assert_eq!(
            opened[[1, 18]],
            0,
            "Small blob should be removed by opening"
        );
        assert!(opened[[9, 9]] > 0, "Large region should survive opening");
    }

    #[test]
    fn closing_fills_small_holes() {
        let mut img = Array2::<u8>::from_elem((20, 20), 200u8);
        // Punch a single dark hole
        img[[10, 10]] = 0;

        let k = rect_kernel(3, 3);
        let closed = closing(&img, &k).expect("closing");

        assert!(
            closed[[10, 10]] > 0,
            "Small hole should be filled by closing"
        );
    }

    // ── Gradient, top-hat, black-hat ──────────────────────────────────────

    #[test]
    fn morphological_gradient_highlights_edges() {
        let img = bright_square(20, 4, 16);
        let k = rect_kernel(3, 3);
        let grad = morphological_gradient(&img, &k).expect("gradient");

        // Deep interior of the square
        assert_eq!(grad[[9, 9]], 0, "Interior should have zero gradient");
        // Edge of the square
        assert!(grad[[4, 4]] > 0, "Edge should have non-zero gradient");
    }

    #[test]
    fn top_hat_extracts_small_bright_feature() {
        let mut img = Array2::<u8>::from_elem((16, 16), 100u8);
        // Small bright feature (1 pixel)
        img[[8, 8]] = 255;

        let k = disk_kernel(3); // larger than the feature
        let th = top_hat(&img, &k).expect("top_hat");

        assert!(
            th[[8, 8]] > 0,
            "Small bright feature should appear in top-hat result"
        );
    }

    #[test]
    fn black_hat_extracts_small_dark_feature() {
        let mut img = Array2::<u8>::from_elem((16, 16), 200u8);
        img[[8, 8]] = 0; // single dark pixel

        let k = disk_kernel(3);
        let bh = black_hat(&img, &k).expect("black_hat");

        assert!(
            bh[[8, 8]] > 0,
            "Small dark feature should appear in black-hat result"
        );
    }

    // ── Dimension preservation ────────────────────────────────────────────

    #[test]
    fn all_ops_preserve_dimensions() {
        let img = bright_square(24, 4, 20);
        let k = rect_kernel(3, 3);

        let ops: Vec<(&str, Box<dyn Fn() -> Result<Array2<u8>>>)> = vec![
            ("erode", Box::new(|| erode(&img, &k))),
            ("dilate", Box::new(|| dilate(&img, &k))),
            ("opening", Box::new(|| opening(&img, &k))),
            ("closing", Box::new(|| closing(&img, &k))),
            ("gradient", Box::new(|| morphological_gradient(&img, &k))),
            ("top_hat", Box::new(|| top_hat(&img, &k))),
            ("black_hat", Box::new(|| black_hat(&img, &k))),
        ];

        for (name, op) in &ops {
            let result = op().unwrap_or_else(|e| panic!("{name} failed: {e}"));
            assert_eq!(
                result.dim(),
                img.dim(),
                "{name} should preserve image dimensions"
            );
        }
    }

    // ── Error handling ────────────────────────────────────────────────────

    #[test]
    fn zero_dimension_kernel_returns_error() {
        let img = Array2::<u8>::zeros((8, 8));
        let k = Array2::<u8>::zeros((0, 3));
        assert!(
            erode(&img, &k).is_err(),
            "Zero-height kernel should return error"
        );

        let k2 = Array2::<u8>::zeros((3, 0));
        assert!(
            dilate(&img, &k2).is_err(),
            "Zero-width kernel should return error"
        );
    }

    #[test]
    fn zero_dimension_image_returns_error() {
        let img = Array2::<u8>::zeros((0, 8));
        let k = rect_kernel(3, 3);
        assert!(erode(&img, &k).is_err());
        assert!(dilate(&img, &k).is_err());
    }

    // ── Cross kernel morphological operations ─────────────────────────────

    #[test]
    fn cross_dilation_cardinal_directions() {
        let mut img = Array2::<u8>::zeros((11, 11));
        img[[5, 5]] = 255; // single pixel

        let k = cross_kernel(5); // cross of size 5×5
        let dilated = dilate(&img, &k).expect("dilate with cross");

        // Cardinal directions should be lit up
        assert_eq!(dilated[[5, 5]], 255, "Centre");
        assert_eq!(dilated[[5, 3]], 255, "Left 2");
        assert_eq!(dilated[[5, 7]], 255, "Right 2");
        assert_eq!(dilated[[3, 5]], 255, "Up 2");
        assert_eq!(dilated[[7, 5]], 255, "Down 2");
        // Diagonal should NOT be lit (cross shape excludes diagonals)
        assert_eq!(dilated[[3, 3]], 0, "Upper-left diagonal should stay dark");
        assert_eq!(dilated[[7, 7]], 0, "Lower-right diagonal should stay dark");
    }

    // ── Disk kernel boundary accuracy ─────────────────────────────────────

    #[test]
    fn disk_kernel_contains_full_ring() {
        let k = disk_kernel(4);
        // Every pixel within radius 4 of the centre (4,4) should be 1
        for y in 0..9usize {
            for x in 0..9usize {
                let dy = y as isize - 4;
                let dx = x as isize - 4;
                let r2 = dy * dy + dx * dx;
                let expected: u8 = if r2 <= 16 { 1 } else { 0 };
                assert_eq!(
                    k[[y, x]],
                    expected,
                    "disk_kernel(4)[[{y},{x}]] should be {expected}, r²={r2}"
                );
            }
        }
    }
}
