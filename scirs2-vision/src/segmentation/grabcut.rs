//! GrabCut-like foreground extraction
//!
//! A simplified version of the GrabCut algorithm for interactive foreground
//! extraction. Uses iterative graph-cut segmentation with Gaussian Mixture Models
//! to separate foreground from background.
//!
//! This implementation uses a simplified energy minimization approach inspired
//! by the original GrabCut paper, using K-means initialization and iterative
//! refinement rather than the full graph-cut optimization.
//!
//! # References
//!
//! - Rother, C., Kolmogorov, V. and Blake, A., 2004. "GrabCut": interactive
//!   foreground extraction using iterated graph cuts. ACM TOG, 23(3), pp.309-314.

use crate::error::{Result, VisionError};
use image::{DynamicImage, GenericImageView, GrayImage, Luma, RgbImage};
use scirs2_core::ndarray::Array2;

/// Mask values for GrabCut
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrabCutMask {
    /// Definitely background
    Background = 0,
    /// Definitely foreground
    Foreground = 1,
    /// Probably background
    ProbableBackground = 2,
    /// Probably foreground
    ProbableForeground = 3,
}

impl GrabCutMask {
    /// Check if this mask value is foreground (definite or probable)
    pub fn is_foreground(self) -> bool {
        matches!(
            self,
            GrabCutMask::Foreground | GrabCutMask::ProbableForeground
        )
    }

    /// Check if this mask value is background (definite or probable)
    pub fn is_background(self) -> bool {
        matches!(
            self,
            GrabCutMask::Background | GrabCutMask::ProbableBackground
        )
    }
}

/// Parameters for GrabCut foreground extraction
#[derive(Debug, Clone)]
pub struct GrabCutParams {
    /// Number of Gaussian components per GMM
    pub n_components: usize,
    /// Number of iterations for the segmentation
    pub max_iterations: usize,
    /// Convergence threshold (fraction of changed pixels)
    pub epsilon: f32,
    /// Smoothness weight (beta) for neighbor penalty
    pub smoothness: f32,
}

impl Default for GrabCutParams {
    fn default() -> Self {
        Self {
            n_components: 5,
            max_iterations: 10,
            epsilon: 1e-3,
            smoothness: 50.0,
        }
    }
}

/// Result of GrabCut segmentation
#[derive(Debug, Clone)]
pub struct GrabCutResult {
    /// Binary foreground mask (height x width), true = foreground
    pub foreground_mask: Array2<bool>,
    /// Refined mask with GrabCutMask values
    pub mask: Array2<u8>,
    /// Number of iterations performed
    pub iterations: usize,
}

/// Perform GrabCut-like foreground extraction using a bounding box
///
/// The bounding box specifies the region that contains the foreground object.
/// Everything outside the box is considered definite background.
///
/// # Arguments
///
/// * `img` - Input RGB image
/// * `rect` - Bounding box (x, y, width, height) containing the foreground
/// * `params` - GrabCut parameters
///
/// # Returns
///
/// * Result containing the foreground mask
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_vision::segmentation::grabcut::{grabcut_rect, GrabCutParams};
/// use image::{DynamicImage, GenericImageView};
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("test.jpg").expect("Operation failed");
/// let (w, h) = img.dimensions();
/// // Assume the foreground is roughly in the center
/// let rect = (w / 4, h / 4, w / 2, h / 2);
/// let result = grabcut_rect(&img, rect, &GrabCutParams::default())?;
/// # Ok(())
/// # }
/// ```
pub fn grabcut_rect(
    img: &DynamicImage,
    rect: (u32, u32, u32, u32),
    params: &GrabCutParams,
) -> Result<GrabCutResult> {
    let (img_width, img_height) = img.dimensions();
    let (rx, ry, rw, rh) = rect;

    if rx + rw > img_width || ry + rh > img_height {
        return Err(VisionError::InvalidParameter(
            "Bounding box exceeds image dimensions".to_string(),
        ));
    }

    if rw == 0 || rh == 0 {
        return Err(VisionError::InvalidParameter(
            "Bounding box must have non-zero dimensions".to_string(),
        ));
    }

    // Initialize mask: everything outside rect is BG, inside is ProbableFG
    let h = img_height as usize;
    let w = img_width as usize;
    let mut mask = Array2::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let is_inside = x >= rx as usize
                && x < (rx + rw) as usize
                && y >= ry as usize
                && y < (ry + rh) as usize;

            mask[[y, x]] = if is_inside {
                GrabCutMask::ProbableForeground as u8
            } else {
                GrabCutMask::Background as u8
            };
        }
    }

    grabcut_with_mask(img, &mut mask, params)
}

/// Perform GrabCut-like foreground extraction using an initial mask
///
/// # Arguments
///
/// * `img` - Input image
/// * `mask` - Initial mask (values from GrabCutMask enum, modified in place)
/// * `params` - GrabCut parameters
///
/// # Returns
///
/// * Result containing the refined segmentation
pub fn grabcut_with_mask(
    img: &DynamicImage,
    mask: &mut Array2<u8>,
    params: &GrabCutParams,
) -> Result<GrabCutResult> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let h = height as usize;
    let w = width as usize;

    // Extract pixel colors
    let mut pixels: Vec<[f32; 3]> = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x as u32, y as u32);
            pixels.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }
    }

    // Build foreground and background pixel sets from initial mask
    let mut iterations = 0;

    for _iter in 0..params.max_iterations {
        iterations = _iter + 1;

        // Collect foreground and background samples
        let mut fg_pixels = Vec::new();
        let mut bg_pixels = Vec::new();

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                let m = mask[[y, x]];
                if m == GrabCutMask::Foreground as u8 || m == GrabCutMask::ProbableForeground as u8
                {
                    fg_pixels.push(pixels[idx]);
                } else {
                    bg_pixels.push(pixels[idx]);
                }
            }
        }

        if fg_pixels.is_empty() || bg_pixels.is_empty() {
            break;
        }

        // Build simplified GMMs (using K-means clusters as Gaussian components)
        let fg_model = build_gmm(&fg_pixels, params.n_components);
        let bg_model = build_gmm(&bg_pixels, params.n_components);

        // Reassign probable pixels based on likelihood
        let mut changed = 0usize;
        let total_probable = mask
            .iter()
            .filter(|&&m| {
                m == GrabCutMask::ProbableForeground as u8
                    || m == GrabCutMask::ProbableBackground as u8
            })
            .count();

        for y in 0..h {
            for x in 0..w {
                let m = mask[[y, x]];
                // Only update probable pixels
                if m != GrabCutMask::ProbableForeground as u8
                    && m != GrabCutMask::ProbableBackground as u8
                {
                    continue;
                }

                let idx = y * w + x;
                let pixel = &pixels[idx];

                // Compute data terms (negative log-likelihood)
                let fg_cost = gmm_nll(&fg_model, pixel);
                let bg_cost = gmm_nll(&bg_model, pixel);

                // Add smoothness term based on neighboring labels
                let smooth_penalty =
                    compute_smoothness_penalty(&pixels, mask, x, y, w, h, params.smoothness);

                let total_fg_cost = fg_cost - smooth_penalty;
                let total_bg_cost = bg_cost + smooth_penalty;

                let new_mask = if total_fg_cost < total_bg_cost {
                    GrabCutMask::ProbableForeground as u8
                } else {
                    GrabCutMask::ProbableBackground as u8
                };

                if new_mask != mask[[y, x]] {
                    changed += 1;
                }
                mask[[y, x]] = new_mask;
            }
        }

        // Check convergence
        if total_probable > 0 && (changed as f32 / total_probable as f32) < params.epsilon {
            break;
        }
    }

    // Build foreground mask
    let foreground_mask = Array2::from_shape_fn((h, w), |(y, x)| {
        let m = mask[[y, x]];
        m == GrabCutMask::Foreground as u8 || m == GrabCutMask::ProbableForeground as u8
    });

    Ok(GrabCutResult {
        foreground_mask,
        mask: mask.clone(),
        iterations,
    })
}

/// Simplified Gaussian Mixture Model for GrabCut
#[derive(Debug, Clone)]
struct SimplifiedGMM {
    /// Component means (n_components x 3)
    means: Vec<[f32; 3]>,
    /// Component variances (diagonal covariance, n_components x 3)
    variances: Vec<[f32; 3]>,
    /// Component weights
    weights: Vec<f32>,
}

/// Build a simplified GMM from pixel samples using K-means
fn build_gmm(pixels: &[[f32; 3]], n_components: usize) -> SimplifiedGMM {
    let k = n_components.min(pixels.len());
    if k == 0 {
        return SimplifiedGMM {
            means: vec![[128.0; 3]],
            variances: vec![[1000.0; 3]],
            weights: vec![1.0],
        };
    }

    // Simple K-means to find components
    let mut centers: Vec<[f32; 3]> = Vec::with_capacity(k);

    // Initialize centers by uniform sampling
    for i in 0..k {
        let idx = (i * pixels.len()) / k;
        centers.push(pixels[idx]);
    }

    let mut assignments = vec![0usize; pixels.len()];

    // Run a few K-means iterations
    for _ in 0..20 {
        // Assign
        for (i, pixel) in pixels.iter().enumerate() {
            let mut min_dist = f32::MAX;
            let mut best = 0;
            for (c, center) in centers.iter().enumerate() {
                let d = (pixel[0] - center[0]).powi(2)
                    + (pixel[1] - center[1]).powi(2)
                    + (pixel[2] - center[2]).powi(2);
                if d < min_dist {
                    min_dist = d;
                    best = c;
                }
            }
            assignments[i] = best;
        }

        // Update centers
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0usize; k];

        for (i, pixel) in pixels.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            sums[c][0] += pixel[0] as f64;
            sums[c][1] += pixel[1] as f64;
            sums[c][2] += pixel[2] as f64;
        }

        for c in 0..k {
            if counts[c] > 0 {
                centers[c] = [
                    (sums[c][0] / counts[c] as f64) as f32,
                    (sums[c][1] / counts[c] as f64) as f32,
                    (sums[c][2] / counts[c] as f64) as f32,
                ];
            }
        }
    }

    // Compute variances and weights
    let mut variances = vec![[0.0f64; 3]; k];
    let mut counts = vec![0usize; k];

    for (i, pixel) in pixels.iter().enumerate() {
        let c = assignments[i];
        counts[c] += 1;
        for ch in 0..3 {
            let diff = pixel[ch] - centers[c][ch];
            variances[c][ch] += (diff * diff) as f64;
        }
    }

    let total = pixels.len() as f32;
    let mut means = Vec::with_capacity(k);
    let mut vars = Vec::with_capacity(k);
    let mut weights = Vec::with_capacity(k);

    for c in 0..k {
        if counts[c] > 0 {
            means.push(centers[c]);
            vars.push([
                (variances[c][0] / counts[c] as f64).max(1.0) as f32,
                (variances[c][1] / counts[c] as f64).max(1.0) as f32,
                (variances[c][2] / counts[c] as f64).max(1.0) as f32,
            ]);
            weights.push(counts[c] as f32 / total);
        }
    }

    if means.is_empty() {
        means.push([128.0; 3]);
        vars.push([1000.0; 3]);
        weights.push(1.0);
    }

    SimplifiedGMM {
        means,
        variances: vars,
        weights,
    }
}

/// Compute negative log-likelihood under a simplified GMM
fn gmm_nll(gmm: &SimplifiedGMM, pixel: &[f32; 3]) -> f32 {
    let mut max_log_prob = f32::NEG_INFINITY;

    for (i, (mean, var)) in gmm.means.iter().zip(gmm.variances.iter()).enumerate() {
        let weight = gmm.weights[i];
        if weight <= 0.0 {
            continue;
        }

        // Log probability under diagonal Gaussian
        let mut log_prob = weight.ln();
        for ch in 0..3 {
            let diff = pixel[ch] - mean[ch];
            log_prob -= 0.5 * (diff * diff) / var[ch];
            log_prob -= 0.5 * var[ch].ln();
        }

        if log_prob > max_log_prob {
            max_log_prob = log_prob;
        }
    }

    -max_log_prob
}

/// Compute smoothness penalty based on neighboring pixel labels
fn compute_smoothness_penalty(
    pixels: &[[f32; 3]],
    mask: &Array2<u8>,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    smoothness: f32,
) -> f32 {
    let idx = y * w + x;
    let pixel = &pixels[idx];
    let mut penalty = 0.0f32;
    let mut neighbor_count = 0;

    // Check 4-connected neighbors
    let neighbors: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    for &(dx, dy) in &neighbors {
        let nx = x as i32 + dx;
        let ny = y as i32 + dy;

        if nx < 0 || nx >= w as i32 || ny < 0 || ny >= h as i32 {
            continue;
        }

        let nx = nx as usize;
        let ny = ny as usize;
        let n_idx = ny * w + nx;
        let n_pixel = &pixels[n_idx];

        // Color difference between current and neighbor
        let color_dist = ((pixel[0] - n_pixel[0]).powi(2)
            + (pixel[1] - n_pixel[1]).powi(2)
            + (pixel[2] - n_pixel[2]).powi(2))
        .sqrt();

        // Smoothness penalty: encourage same label for similar colors
        let n_mask = mask[[ny, nx]];
        let n_is_fg = n_mask == GrabCutMask::Foreground as u8
            || n_mask == GrabCutMask::ProbableForeground as u8;

        // High penalty for different labels when colors are similar
        let similarity = (-color_dist / smoothness).exp();

        if n_is_fg {
            penalty += similarity; // Encourage foreground
        } else {
            penalty -= similarity; // Encourage background
        }
        neighbor_count += 1;
    }

    if neighbor_count > 0 {
        penalty / neighbor_count as f32
    } else {
        0.0
    }
}

/// Convert a GrabCut foreground mask to a grayscale image
pub fn grabcut_mask_to_image(mask: &Array2<bool>) -> GrayImage {
    let (h, w) = mask.dim();
    let mut img = GrayImage::new(w as u32, h as u32);

    for y in 0..h {
        for x in 0..w {
            let val = if mask[[y, x]] { 255 } else { 0 };
            img.put_pixel(x as u32, y as u32, Luma([val]));
        }
    }

    img
}

/// Apply a foreground mask to extract the foreground from an image
///
/// Pixels not in the foreground are set to black (or a specified background color).
pub fn apply_foreground_mask(
    img: &DynamicImage,
    mask: &Array2<bool>,
    bg_color: Option<[u8; 3]>,
) -> Result<RgbImage> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let h = height as usize;
    let w = width as usize;

    if mask.dim() != (h, w) {
        return Err(VisionError::DimensionMismatch(
            "Mask dimensions must match image dimensions".to_string(),
        ));
    }

    let bg = bg_color.unwrap_or([0, 0, 0]);
    let mut result = RgbImage::new(width, height);

    for y in 0..h {
        for x in 0..w {
            if mask[[y, x]] {
                result.put_pixel(x as u32, y as u32, *rgb.get_pixel(x as u32, y as u32));
            } else {
                result.put_pixel(x as u32, y as u32, image::Rgb(bg));
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_fg_bg_image() -> DynamicImage {
        // Create an image with a bright center (foreground) and dark border (background)
        let mut img = image::RgbImage::new(20, 20);
        for y in 0..20u32 {
            for x in 0..20u32 {
                let is_center = x >= 5 && x < 15 && y >= 5 && y < 15;
                let color = if is_center {
                    [220u8, 220, 220] // Bright foreground
                } else {
                    [20u8, 20, 20] // Dark background
                };
                img.put_pixel(x, y, image::Rgb(color));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_grabcut_rect_basic() {
        let img = create_fg_bg_image();
        let result =
            grabcut_rect(&img, (4, 4, 12, 12), &GrabCutParams::default()).expect("GrabCut failed");

        assert_eq!(result.foreground_mask.dim(), (20, 20));
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_grabcut_mask_values() {
        let img = create_fg_bg_image();
        let result =
            grabcut_rect(&img, (4, 4, 12, 12), &GrabCutParams::default()).expect("GrabCut failed");

        // Corners should be background (outside the rect)
        assert!(
            !result.foreground_mask[[0, 0]],
            "Corner (0,0) should be background"
        );
        assert!(
            !result.foreground_mask[[19, 19]],
            "Corner (19,19) should be background"
        );
    }

    #[test]
    fn test_grabcut_with_mask() {
        let img = create_fg_bg_image();
        let mut mask = Array2::zeros((20, 20));

        // Mark center as probable foreground, rest as background
        for y in 0..20 {
            for x in 0..20 {
                let is_center = x >= 5 && x < 15 && y >= 5 && y < 15;
                mask[[y, x]] = if is_center {
                    GrabCutMask::ProbableForeground as u8
                } else {
                    GrabCutMask::Background as u8
                };
            }
        }

        let result =
            grabcut_with_mask(&img, &mut mask, &GrabCutParams::default()).expect("GrabCut failed");
        assert_eq!(result.foreground_mask.dim(), (20, 20));
    }

    #[test]
    fn test_grabcut_reject_invalid_rect() {
        let img = DynamicImage::new_rgb8(20, 20);
        // Rect extends beyond image
        let result = grabcut_rect(&img, (15, 15, 10, 10), &GrabCutParams::default());
        assert!(result.is_err());

        // Zero-size rect
        let result = grabcut_rect(&img, (5, 5, 0, 10), &GrabCutParams::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_grabcut_mask_to_image() {
        let mask = Array2::from_shape_fn((10, 10), |(y, x)| y >= 3 && y < 7 && x >= 3 && x < 7);
        let img = grabcut_mask_to_image(&mask);
        assert_eq!(img.dimensions(), (10, 10));

        assert_eq!(img.get_pixel(0, 0)[0], 0); // background
        assert_eq!(img.get_pixel(5, 5)[0], 255); // foreground
    }

    #[test]
    fn test_apply_foreground_mask() {
        let img = create_fg_bg_image();
        let mask = Array2::from_shape_fn((20, 20), |(y, x)| y >= 5 && y < 15 && x >= 5 && x < 15);

        let result =
            apply_foreground_mask(&img, &mask, None).expect("apply_foreground_mask failed");
        assert_eq!(result.dimensions(), (20, 20));

        // Background pixel should be black
        let bg_pixel = result.get_pixel(0, 0);
        assert_eq!(bg_pixel[0], 0);
        assert_eq!(bg_pixel[1], 0);
        assert_eq!(bg_pixel[2], 0);

        // Foreground pixel should keep original color
        let fg_pixel = result.get_pixel(10, 10);
        assert_eq!(fg_pixel[0], 220);
    }

    #[test]
    fn test_apply_foreground_mask_custom_bg() {
        let img = DynamicImage::new_rgb8(10, 10);
        let mask = Array2::from_elem((10, 10), false);
        let result = apply_foreground_mask(&img, &mask, Some([100, 150, 200]))
            .expect("apply_foreground_mask failed");

        let pixel = result.get_pixel(5, 5);
        assert_eq!(pixel[0], 100);
        assert_eq!(pixel[1], 150);
        assert_eq!(pixel[2], 200);
    }

    #[test]
    fn test_apply_foreground_mask_dimension_mismatch() {
        let img = DynamicImage::new_rgb8(10, 10);
        let mask = Array2::from_elem((5, 5), true);
        let result = apply_foreground_mask(&img, &mask, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_grabcut_mask_enum() {
        assert!(GrabCutMask::Foreground.is_foreground());
        assert!(GrabCutMask::ProbableForeground.is_foreground());
        assert!(!GrabCutMask::Background.is_foreground());
        assert!(!GrabCutMask::ProbableBackground.is_foreground());

        assert!(GrabCutMask::Background.is_background());
        assert!(GrabCutMask::ProbableBackground.is_background());
        assert!(!GrabCutMask::Foreground.is_background());
    }
}
