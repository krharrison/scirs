//! Geometric image transformations
//!
//! High-level convenience APIs for common geometric operations:
//! - Image rotation (with bilinear interpolation)
//! - Image scaling (nearest-neighbour, bilinear, bicubic)
//! - Affine transformation
//! - Perspective (homography) transformation
//! - Cropping and padding
//! - Horizontal / vertical flip
//!
//! For lower-level primitives see [`transform`](crate::transform) (AffineTransform,
//! PerspectiveTransform, interpolation utilities).

use crate::error::{Result, VisionError};
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};

// ---------------------------------------------------------------------------
// Interpolation method selection
// ---------------------------------------------------------------------------

/// Interpolation method for geometric transforms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Nearest-neighbour (fastest, blocky)
    Nearest,
    /// Bilinear (good balance of speed and quality)
    Bilinear,
    /// Bicubic (highest quality, slowest)
    Bicubic,
}

// ---------------------------------------------------------------------------
// Image rotation
// ---------------------------------------------------------------------------

/// Rotate an image around its centre by `angle` radians
///
/// The output canvas is enlarged to fit the entire rotated image.
/// Background pixels are filled with `bg` (grayscale) or black for colour.
///
/// # Arguments
///
/// * `img` - Input image
/// * `angle` - Rotation angle in radians (positive = counter-clockwise)
/// * `bg` - Background fill value for pixels outside the source image
///
/// # Returns
///
/// Rotated image
pub fn rotate(img: &DynamicImage, angle: f64, bg: u8) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let wf = w as f64;
    let hf = h as f64;

    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // Compute bounding box of the rotated image
    let cx = wf / 2.0;
    let cy = hf / 2.0;
    let corners = [(0.0, 0.0), (wf, 0.0), (wf, hf), (0.0, hf)];

    let rotated_corners: Vec<(f64, f64)> = corners
        .iter()
        .map(|&(px, py)| {
            let dx = px - cx;
            let dy = py - cy;
            (cos_a * dx - sin_a * dy, sin_a * dx + cos_a * dy)
        })
        .collect();

    let min_x = rotated_corners
        .iter()
        .map(|c| c.0)
        .fold(f64::INFINITY, f64::min);
    let max_x = rotated_corners
        .iter()
        .map(|c| c.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = rotated_corners
        .iter()
        .map(|c| c.1)
        .fold(f64::INFINITY, f64::min);
    let max_y = rotated_corners
        .iter()
        .map(|c| c.1)
        .fold(f64::NEG_INFINITY, f64::max);

    let new_w = (max_x - min_x).ceil() as u32;
    let new_h = (max_y - min_y).ceil() as u32;

    if new_w == 0 || new_h == 0 {
        return Err(VisionError::InvalidParameter(
            "Resulting image has zero size".to_string(),
        ));
    }

    let new_cx = new_w as f64 / 2.0;
    let new_cy = new_h as f64 / 2.0;

    let mut result = RgbImage::new(new_w, new_h);

    for out_y in 0..new_h {
        for out_x in 0..new_w {
            // Map back to source coordinates (inverse rotation)
            let dx = out_x as f64 - new_cx;
            let dy = out_y as f64 - new_cy;
            let src_x = cos_a * dx + sin_a * dy + cx;
            let src_y = -sin_a * dx + cos_a * dy + cy;

            let pixel = bilinear_sample_rgb(&rgb, src_x, src_y, bg);
            result.put_pixel(out_x, out_y, Rgb(pixel));
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

// ---------------------------------------------------------------------------
// Image scaling / resizing
// ---------------------------------------------------------------------------

/// Resize an image using the specified interpolation method
///
/// # Arguments
///
/// * `img` - Input image
/// * `new_width` - Target width in pixels
/// * `new_height` - Target height in pixels
/// * `method` - Interpolation method
///
/// # Returns
///
/// Resized image
pub fn resize(
    img: &DynamicImage,
    new_width: u32,
    new_height: u32,
    method: Interpolation,
) -> Result<DynamicImage> {
    if new_width == 0 || new_height == 0 {
        return Err(VisionError::InvalidParameter(
            "Target dimensions must be positive".to_string(),
        ));
    }

    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();

    if w == 0 || h == 0 {
        return Err(VisionError::InvalidParameter(
            "Source image has zero size".to_string(),
        ));
    }

    let x_scale = w as f64 / new_width as f64;
    let y_scale = h as f64 / new_height as f64;

    let mut result = RgbImage::new(new_width, new_height);

    for out_y in 0..new_height {
        for out_x in 0..new_width {
            let src_x = (out_x as f64 + 0.5) * x_scale - 0.5;
            let src_y = (out_y as f64 + 0.5) * y_scale - 0.5;

            let pixel = match method {
                Interpolation::Nearest => {
                    let sx = src_x.round().clamp(0.0, (w - 1) as f64) as u32;
                    let sy = src_y.round().clamp(0.0, (h - 1) as f64) as u32;
                    rgb.get_pixel(sx, sy).0
                }
                Interpolation::Bilinear => bilinear_sample_rgb(&rgb, src_x, src_y, 0),
                Interpolation::Bicubic => bicubic_sample_rgb(&rgb, src_x, src_y),
            };

            result.put_pixel(out_x, out_y, Rgb(pixel));
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

// ---------------------------------------------------------------------------
// Affine transformation wrapper
// ---------------------------------------------------------------------------

/// Apply an affine transformation specified by a 2x3 matrix
///
/// The matrix maps output coordinates to source coordinates (inverse mapping).
///
/// `[[a, b, tx], [c, d, ty]]`
/// `src_x = a * dst_x + b * dst_y + tx`
/// `src_y = c * dst_x + d * dst_y + ty`
///
/// # Arguments
///
/// * `img` - Input image
/// * `matrix` - 2x3 affine matrix (row-major: `[a, b, tx, c, d, ty]`)
/// * `out_width` - Output width
/// * `out_height` - Output height
/// * `bg` - Background fill value
///
/// # Returns
///
/// Transformed image
pub fn affine_transform(
    img: &DynamicImage,
    matrix: &[f64; 6],
    out_width: u32,
    out_height: u32,
    bg: u8,
) -> Result<DynamicImage> {
    if out_width == 0 || out_height == 0 {
        return Err(VisionError::InvalidParameter(
            "Output dimensions must be positive".to_string(),
        ));
    }

    let rgb = img.to_rgb8();
    let [a, b, tx, c, d, ty] = *matrix;

    let mut result = RgbImage::new(out_width, out_height);

    for out_y in 0..out_height {
        for out_x in 0..out_width {
            let src_x = a * out_x as f64 + b * out_y as f64 + tx;
            let src_y = c * out_x as f64 + d * out_y as f64 + ty;
            let pixel = bilinear_sample_rgb(&rgb, src_x, src_y, bg);
            result.put_pixel(out_x, out_y, Rgb(pixel));
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

// ---------------------------------------------------------------------------
// Perspective (homography) transformation
// ---------------------------------------------------------------------------

/// Apply a perspective (homography) transformation
///
/// The `matrix` is a 3x3 projective matrix in row-major order.
/// It maps output coordinates to source coordinates (inverse mapping).
///
/// # Arguments
///
/// * `img` - Input image
/// * `matrix` - 3x3 homography matrix (row-major, 9 elements)
/// * `out_width` - Output width
/// * `out_height` - Output height
/// * `bg` - Background fill value
///
/// # Returns
///
/// Transformed image
pub fn perspective_transform(
    img: &DynamicImage,
    matrix: &[f64; 9],
    out_width: u32,
    out_height: u32,
    bg: u8,
) -> Result<DynamicImage> {
    if out_width == 0 || out_height == 0 {
        return Err(VisionError::InvalidParameter(
            "Output dimensions must be positive".to_string(),
        ));
    }

    let rgb = img.to_rgb8();

    let mut result = RgbImage::new(out_width, out_height);

    for out_y in 0..out_height {
        for out_x in 0..out_width {
            let xf = out_x as f64;
            let yf = out_y as f64;

            let w = matrix[6] * xf + matrix[7] * yf + matrix[8];
            if w.abs() < 1e-12 {
                result.put_pixel(out_x, out_y, Rgb([bg, bg, bg]));
                continue;
            }

            let src_x = (matrix[0] * xf + matrix[1] * yf + matrix[2]) / w;
            let src_y = (matrix[3] * xf + matrix[4] * yf + matrix[5]) / w;

            let pixel = bilinear_sample_rgb(&rgb, src_x, src_y, bg);
            result.put_pixel(out_x, out_y, Rgb(pixel));
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

// ---------------------------------------------------------------------------
// Cropping
// ---------------------------------------------------------------------------

/// Crop a rectangular region from an image
///
/// # Arguments
///
/// * `img` - Input image
/// * `x` - Left edge of crop region
/// * `y` - Top edge of crop region
/// * `width` - Width of crop region
/// * `height` - Height of crop region
///
/// # Returns
///
/// Cropped image
pub fn crop(img: &DynamicImage, x: u32, y: u32, width: u32, height: u32) -> Result<DynamicImage> {
    let (img_w, img_h) = (img.width(), img.height());

    if x >= img_w || y >= img_h {
        return Err(VisionError::InvalidParameter(format!(
            "Crop origin ({x}, {y}) outside image ({img_w}x{img_h})"
        )));
    }
    if width == 0 || height == 0 {
        return Err(VisionError::InvalidParameter(
            "Crop dimensions must be positive".to_string(),
        ));
    }

    // Clamp to image bounds
    let actual_w = width.min(img_w - x);
    let actual_h = height.min(img_h - y);

    let rgb = img.to_rgb8();
    let mut result = RgbImage::new(actual_w, actual_h);

    for dy in 0..actual_h {
        for dx in 0..actual_w {
            let pixel = rgb.get_pixel(x + dx, y + dy);
            result.put_pixel(dx, dy, *pixel);
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

// ---------------------------------------------------------------------------
// Padding
// ---------------------------------------------------------------------------

/// Padding mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadMode {
    /// Fill with a constant value
    Constant(u8),
    /// Replicate edge pixels
    Replicate,
    /// Reflect pixels at the edge (mirror)
    Reflect,
}

/// Add padding around an image
///
/// # Arguments
///
/// * `img` - Input image
/// * `top` - Padding above the image
/// * `bottom` - Padding below the image
/// * `left` - Padding to the left
/// * `right` - Padding to the right
/// * `mode` - Padding strategy
///
/// # Returns
///
/// Padded image
pub fn pad(
    img: &DynamicImage,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
    mode: PadMode,
) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();

    let new_w = w + left + right;
    let new_h = h + top + bottom;
    if new_w == 0 || new_h == 0 {
        return Err(VisionError::InvalidParameter(
            "Padded image has zero size".to_string(),
        ));
    }

    let mut result = RgbImage::new(new_w, new_h);

    for out_y in 0..new_h {
        for out_x in 0..new_w {
            let sx = out_x as i64 - left as i64;
            let sy = out_y as i64 - top as i64;

            let pixel = if sx >= 0 && sx < w as i64 && sy >= 0 && sy < h as i64 {
                *rgb.get_pixel(sx as u32, sy as u32)
            } else {
                match mode {
                    PadMode::Constant(v) => Rgb([v, v, v]),
                    PadMode::Replicate => {
                        let cx = sx.clamp(0, w as i64 - 1) as u32;
                        let cy = sy.clamp(0, h as i64 - 1) as u32;
                        *rgb.get_pixel(cx, cy)
                    }
                    PadMode::Reflect => {
                        let cx = reflect_index(sx, w as i64);
                        let cy = reflect_index(sy, h as i64);
                        *rgb.get_pixel(cx, cy)
                    }
                }
            };

            result.put_pixel(out_x, out_y, pixel);
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

/// Reflect an index into [0, size)
fn reflect_index(idx: i64, size: i64) -> u32 {
    if size <= 1 {
        return 0;
    }
    let mut i = idx;
    // Bring into [-size, 2*size)
    while i < 0 {
        i += 2 * size;
    }
    i %= 2 * size;
    if i >= size {
        i = 2 * size - 1 - i;
    }
    i.clamp(0, size - 1) as u32
}

// ---------------------------------------------------------------------------
// Flip
// ---------------------------------------------------------------------------

/// Flip an image horizontally (left-right mirror)
pub fn flip_horizontal(img: &DynamicImage) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut result = RgbImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(w - 1 - x, y);
            result.put_pixel(x, y, *pixel);
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

/// Flip an image vertically (top-bottom mirror)
pub fn flip_vertical(img: &DynamicImage) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let mut result = RgbImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x, h - 1 - y);
            result.put_pixel(x, y, *pixel);
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

// ---------------------------------------------------------------------------
// Interpolation helpers
// ---------------------------------------------------------------------------

/// Bilinear sampling for RGB images
fn bilinear_sample_rgb(img: &RgbImage, x: f64, y: f64, bg: u8) -> [u8; 3] {
    let (w, h) = img.dimensions();

    if x < -0.5 || y < -0.5 || x >= w as f64 - 0.5 || y >= h as f64 - 0.5 {
        // Check if significantly out of bounds
        if x < -1.0 || y < -1.0 || x >= w as f64 || y >= h as f64 {
            return [bg, bg, bg];
        }
    }

    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;

    let get_px = |ix: i64, iy: i64| -> [f32; 3] {
        if ix >= 0 && ix < w as i64 && iy >= 0 && iy < h as i64 {
            let p = img.get_pixel(ix as u32, iy as u32);
            [p[0] as f32, p[1] as f32, p[2] as f32]
        } else {
            [bg as f32, bg as f32, bg as f32]
        }
    };

    let p00 = get_px(x0, y0);
    let p10 = get_px(x1, y0);
    let p01 = get_px(x0, y1);
    let p11 = get_px(x1, y1);

    let mut out = [0u8; 3];
    for ch in 0..3 {
        let top = p00[ch] * (1.0 - fx) + p10[ch] * fx;
        let bot = p01[ch] * (1.0 - fx) + p11[ch] * fx;
        let val = top * (1.0 - fy) + bot * fy;
        out[ch] = val.clamp(0.0, 255.0) as u8;
    }
    out
}

/// Cubic interpolation kernel (Catmull-Rom)
fn cubic_weight(t: f64) -> f64 {
    let t = t.abs();
    if t < 1.0 {
        (1.5 * t - 2.5) * t * t + 1.0
    } else if t < 2.0 {
        ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
    } else {
        0.0
    }
}

/// Bicubic sampling for RGB images
fn bicubic_sample_rgb(img: &RgbImage, x: f64, y: f64) -> [u8; 3] {
    let (w, h) = img.dimensions();
    let ix = x.floor() as i64;
    let iy = y.floor() as i64;
    let fx = x - ix as f64;
    let fy = y - iy as f64;

    let mut result = [0.0f64; 3];

    for dy in -1i64..=2 {
        let wy = cubic_weight(fy - dy as f64);
        let sy = (iy + dy).clamp(0, h as i64 - 1) as u32;

        for dx in -1i64..=2 {
            let wx = cubic_weight(fx - dx as f64);
            let sx = (ix + dx).clamp(0, w as i64 - 1) as u32;
            let p = img.get_pixel(sx, sy);
            let weight = wx * wy;

            for ch in 0..3 {
                result[ch] += p[ch] as f64 * weight;
            }
        }
    }

    [
        result[0].clamp(0.0, 255.0) as u8,
        result[1].clamp(0.0, 255.0) as u8,
        result[2].clamp(0.0, 255.0) as u8,
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb(width: u32, height: u32, r: u8, g: u8, b: u8) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([r, g, b]);
        }
        DynamicImage::ImageRgb8(img)
    }

    fn make_checker(size: u32) -> DynamicImage {
        let mut img = RgbImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let val = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
                img.put_pixel(x, y, Rgb([val, val, val]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    // --- Rotation ---
    #[test]
    fn test_rotate_zero() {
        let img = make_rgb(10, 10, 100, 150, 200);
        let rotated = rotate(&img, 0.0, 0).expect("rotation should succeed");
        // With zero rotation, the centre should be preserved
        let orig = img.to_rgb8();
        let rot = rotated.to_rgb8();
        let cx = rot.width() / 2;
        let cy = rot.height() / 2;
        let px = rot.get_pixel(cx, cy).0;
        assert!(
            (px[0] as i16 - 100).unsigned_abs() <= 2,
            "Centre pixel should be near 100, got {}",
            px[0]
        );
    }

    #[test]
    fn test_rotate_90() {
        let img = make_rgb(20, 20, 100, 150, 200);
        let rotated =
            rotate(&img, std::f64::consts::FRAC_PI_2, 0).expect("rotation should succeed");
        // Rotated 90 degrees, dimensions may change slightly due to ceil
        assert!(rotated.width() > 0 && rotated.height() > 0);
    }

    #[test]
    fn test_rotate_360() {
        let img = make_rgb(20, 20, 100, 150, 200);
        let rotated = rotate(&img, 2.0 * std::f64::consts::PI, 0).expect("rotation should succeed");
        let orig = img.to_rgb8();
        let rot = rotated.to_rgb8();
        // Centre pixel should be same as original centre
        let cx_o = orig.width() / 2;
        let cy_o = orig.height() / 2;
        let cx_r = rot.width() / 2;
        let cy_r = rot.height() / 2;
        let px_o = orig.get_pixel(cx_o, cy_o).0;
        let px_r = rot.get_pixel(cx_r, cy_r).0;
        for ch in 0..3 {
            assert!(
                (px_o[ch] as i16 - px_r[ch] as i16).unsigned_abs() <= 2,
                "360-rotation should preserve centre pixel"
            );
        }
    }

    // --- Resize ---
    #[test]
    fn test_resize_nearest() {
        let img = make_rgb(20, 20, 100, 150, 200);
        let resized = resize(&img, 40, 40, Interpolation::Nearest).expect("resize should succeed");
        assert_eq!(resized.width(), 40);
        assert_eq!(resized.height(), 40);
    }

    #[test]
    fn test_resize_bilinear() {
        let img = make_rgb(20, 20, 100, 150, 200);
        let resized = resize(&img, 10, 10, Interpolation::Bilinear).expect("resize should succeed");
        assert_eq!(resized.width(), 10);
        assert_eq!(resized.height(), 10);
        let px = resized.to_rgb8().get_pixel(5, 5).0;
        assert!((px[0] as i16 - 100).unsigned_abs() <= 2);
    }

    #[test]
    fn test_resize_bicubic() {
        let img = make_rgb(20, 20, 100, 150, 200);
        let resized = resize(&img, 30, 30, Interpolation::Bicubic).expect("resize should succeed");
        assert_eq!(resized.width(), 30);
        assert_eq!(resized.height(), 30);
    }

    #[test]
    fn test_resize_invalid() {
        let img = make_rgb(10, 10, 100, 100, 100);
        assert!(resize(&img, 0, 10, Interpolation::Nearest).is_err());
        assert!(resize(&img, 10, 0, Interpolation::Nearest).is_err());
    }

    // --- Affine ---
    #[test]
    fn test_affine_identity() {
        let img = make_rgb(10, 10, 100, 150, 200);
        let matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // identity
        let result = affine_transform(&img, &matrix, 10, 10, 0).expect("affine should succeed");
        let px = result.to_rgb8().get_pixel(5, 5).0;
        assert_eq!(px, [100, 150, 200]);
    }

    #[test]
    fn test_affine_translation() {
        let mut img = RgbImage::new(20, 20);
        img.put_pixel(5, 5, Rgb([255, 0, 0]));
        let dyn_img = DynamicImage::ImageRgb8(img);

        // Identity + translate by (3, 2)
        // src = dst -> for this we need inverse: src_x = out_x - 3
        let matrix = [1.0, 0.0, -3.0, 0.0, 1.0, -2.0];
        let result = affine_transform(&dyn_img, &matrix, 20, 20, 0).expect("affine should succeed");

        // Pixel that was at (5,5) should now appear at (8,7)
        let px = result.to_rgb8().get_pixel(8, 7).0;
        assert_eq!(px[0], 255, "Translated pixel should be at (8,7)");
    }

    #[test]
    fn test_affine_invalid() {
        let img = make_rgb(10, 10, 100, 100, 100);
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert!(affine_transform(&img, &m, 0, 10, 0).is_err());
    }

    // --- Perspective ---
    #[test]
    fn test_perspective_identity() {
        let img = make_rgb(10, 10, 100, 150, 200);
        let matrix = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result =
            perspective_transform(&img, &matrix, 10, 10, 0).expect("perspective should succeed");
        let px = result.to_rgb8().get_pixel(5, 5).0;
        assert_eq!(px, [100, 150, 200]);
    }

    #[test]
    fn test_perspective_invalid() {
        let img = make_rgb(10, 10, 100, 100, 100);
        let m = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!(perspective_transform(&img, &m, 0, 10, 0).is_err());
    }

    // --- Crop ---
    #[test]
    fn test_crop_basic() {
        let mut img = RgbImage::new(20, 20);
        img.put_pixel(5, 5, Rgb([255, 0, 0]));
        let dyn_img = DynamicImage::ImageRgb8(img);

        let cropped = crop(&dyn_img, 3, 3, 10, 10).expect("crop should succeed");
        assert_eq!(cropped.width(), 10);
        assert_eq!(cropped.height(), 10);

        // Pixel at (5,5) in original is at (2,2) in cropped
        let px = cropped.to_rgb8().get_pixel(2, 2).0;
        assert_eq!(px[0], 255);
    }

    #[test]
    fn test_crop_clamped() {
        let img = make_rgb(10, 10, 100, 100, 100);
        // Crop extends beyond image boundary - should be clamped
        let cropped = crop(&img, 5, 5, 20, 20).expect("crop should succeed");
        assert_eq!(cropped.width(), 5);
        assert_eq!(cropped.height(), 5);
    }

    #[test]
    fn test_crop_invalid() {
        let img = make_rgb(10, 10, 100, 100, 100);
        assert!(crop(&img, 15, 5, 5, 5).is_err()); // x out of bounds
        assert!(crop(&img, 0, 0, 0, 5).is_err()); // zero width
    }

    // --- Padding ---
    #[test]
    fn test_pad_constant() {
        let img = make_rgb(10, 10, 100, 150, 200);
        let padded = pad(&img, 5, 5, 5, 5, PadMode::Constant(0)).expect("pad should succeed");
        assert_eq!(padded.width(), 20);
        assert_eq!(padded.height(), 20);

        // Corner should be black
        let corner = padded.to_rgb8().get_pixel(0, 0).0;
        assert_eq!(corner, [0, 0, 0]);

        // Original pixel at (0,0) now at (5,5)
        let px = padded.to_rgb8().get_pixel(5, 5).0;
        assert_eq!(px, [100, 150, 200]);
    }

    #[test]
    fn test_pad_replicate() {
        let img = make_rgb(10, 10, 100, 150, 200);
        let padded = pad(&img, 2, 2, 2, 2, PadMode::Replicate).expect("pad should succeed");
        // Edge pixels should be replicated
        let corner = padded.to_rgb8().get_pixel(0, 0).0;
        assert_eq!(corner, [100, 150, 200]);
    }

    #[test]
    fn test_pad_reflect() {
        // Create a simple 3x3 image with gradient
        let mut img = RgbImage::new(3, 1);
        img.put_pixel(0, 0, Rgb([10, 10, 10]));
        img.put_pixel(1, 0, Rgb([20, 20, 20]));
        img.put_pixel(2, 0, Rgb([30, 30, 30]));
        let dyn_img = DynamicImage::ImageRgb8(img);

        let padded = pad(&dyn_img, 0, 0, 2, 2, PadMode::Reflect).expect("pad should succeed");
        assert_eq!(padded.width(), 7);
    }

    // --- Flip ---
    #[test]
    fn test_flip_horizontal() {
        let mut img = RgbImage::new(10, 1);
        for x in 0..10 {
            img.put_pixel(x, 0, Rgb([x as u8 * 25, 0, 0]));
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let flipped = flip_horizontal(&dyn_img).expect("flip should succeed");
        let result = flipped.to_rgb8();

        // First pixel of flipped should be last pixel of original
        assert_eq!(result.get_pixel(0, 0)[0], 225); // 9 * 25
        assert_eq!(result.get_pixel(9, 0)[0], 0); // 0 * 25
    }

    #[test]
    fn test_flip_vertical() {
        let mut img = RgbImage::new(1, 10);
        for y in 0..10 {
            img.put_pixel(0, y, Rgb([y as u8 * 25, 0, 0]));
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let flipped = flip_vertical(&dyn_img).expect("flip should succeed");
        let result = flipped.to_rgb8();

        assert_eq!(result.get_pixel(0, 0)[0], 225); // 9 * 25
        assert_eq!(result.get_pixel(0, 9)[0], 0); // 0 * 25
    }

    #[test]
    fn test_flip_horizontal_twice_is_identity() {
        let img = make_checker(8);
        let f1 = flip_horizontal(&img).expect("flip should succeed");
        let f2 = flip_horizontal(&f1).expect("flip should succeed");

        let orig = img.to_rgb8();
        let rec = f2.to_rgb8();
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(orig.get_pixel(x, y), rec.get_pixel(x, y));
            }
        }
    }

    #[test]
    fn test_flip_vertical_twice_is_identity() {
        let img = make_checker(8);
        let f1 = flip_vertical(&img).expect("flip should succeed");
        let f2 = flip_vertical(&f1).expect("flip should succeed");

        let orig = img.to_rgb8();
        let rec = f2.to_rgb8();
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(orig.get_pixel(x, y), rec.get_pixel(x, y));
            }
        }
    }

    // --- Interpolation ---
    #[test]
    fn test_bicubic_weight() {
        assert!((cubic_weight(0.0) - 1.0).abs() < 1e-10);
        assert!(cubic_weight(2.0).abs() < 1e-10);
        assert!(cubic_weight(3.0).abs() < 1e-10);
    }

    #[test]
    fn test_reflect_index() {
        assert_eq!(reflect_index(0, 10), 0);
        assert_eq!(reflect_index(5, 10), 5);
        assert_eq!(reflect_index(9, 10), 9);
        // reflect(-1, 10) = 0 (reflect at boundary, dcba|abcd convention)
        assert_eq!(reflect_index(-1, 10), 0);
        assert_eq!(reflect_index(-2, 10), 1);
        assert_eq!(reflect_index(10, 10), 9);
        assert_eq!(reflect_index(11, 10), 8);
    }
}
