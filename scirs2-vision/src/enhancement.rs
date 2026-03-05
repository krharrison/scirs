//! Image enhancement algorithms
//!
//! This module provides contrast enhancement and filtering techniques:
//! - Contrast stretching (linear, logarithmic, power-law / gamma)
//! - Homomorphic filtering
//!
//! For additional enhancement algorithms, see [`preprocessing`](crate::preprocessing):
//! - Unsharp masking (`unsharp_mask`)
//! - Bilateral filtering (`bilateral_filter`, `bilateral_filter_advanced`)
//! - Non-local means denoising (`nlm_denoise`)
//! - Guided filtering (`guided_filter`)
//! - Retinex (single-scale / multi-scale) (`single_scale_retinex`, `multi_scale_retinex`)
//! - Gamma correction (`gamma_correction`, `auto_gamma_correction`)
//! - CLAHE (`clahe`)

use crate::error::{Result, VisionError};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};

// ---------------------------------------------------------------------------
// Contrast stretching
// ---------------------------------------------------------------------------

/// Apply linear contrast stretching (min-max normalisation)
///
/// Maps intensity values from `[in_low, in_high]` to `[out_low, out_high]`.
/// Values outside `[in_low, in_high]` are clamped.
///
/// # Arguments
///
/// * `img` - Input image
/// * `in_low` - Lower bound of the input range (0-255)
/// * `in_high` - Upper bound of the input range (0-255)
/// * `out_low` - Lower bound of the output range (0-255)
/// * `out_high` - Upper bound of the output range (0-255)
///
/// # Returns
///
/// Contrast-stretched image
pub fn contrast_stretch_linear(
    img: &DynamicImage,
    in_low: u8,
    in_high: u8,
    out_low: u8,
    out_high: u8,
) -> Result<DynamicImage> {
    if in_low >= in_high {
        return Err(VisionError::InvalidParameter(
            "in_low must be less than in_high".to_string(),
        ));
    }
    if out_low >= out_high {
        return Err(VisionError::InvalidParameter(
            "out_low must be less than out_high".to_string(),
        ));
    }

    let in_range = (in_high - in_low) as f32;
    let out_range = (out_high - out_low) as f32;

    // Build LUT
    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let v = i as f32;
        let clamped = v.clamp(in_low as f32, in_high as f32);
        let normalised = (clamped - in_low as f32) / in_range;
        *entry = (out_low as f32 + normalised * out_range).clamp(0.0, 255.0) as u8;
    }

    apply_lut_to_image(img, &lut)
}

/// Apply automatic linear contrast stretching
///
/// Determines `in_low` and `in_high` from the image histogram (percentile-based)
/// and stretches to the full [0, 255] range.
///
/// # Arguments
///
/// * `img` - Input image
/// * `low_percentile` - Lower percentile for input range (e.g. 2.0)
/// * `high_percentile` - Upper percentile for input range (e.g. 98.0)
///
/// # Returns
///
/// Auto-stretched image
pub fn contrast_stretch_auto(
    img: &DynamicImage,
    low_percentile: f32,
    high_percentile: f32,
) -> Result<DynamicImage> {
    if low_percentile < 0.0 || low_percentile >= high_percentile || high_percentile > 100.0 {
        return Err(VisionError::InvalidParameter(
            "Percentiles must satisfy 0 <= low < high <= 100".to_string(),
        ));
    }

    let gray = img.to_luma8();
    let total = (gray.width() as u64) * (gray.height() as u64);
    if total == 0 {
        return Err(VisionError::InvalidParameter(
            "Image has zero pixels".to_string(),
        ));
    }

    // Build histogram
    let mut hist = [0u64; 256];
    for pixel in gray.pixels() {
        hist[pixel[0] as usize] += 1;
    }

    // CDF
    let mut cdf = [0u64; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    let low_count = (total as f64 * low_percentile as f64 / 100.0) as u64;
    let high_count = (total as f64 * high_percentile as f64 / 100.0) as u64;

    let in_low = cdf.iter().position(|&c| c >= low_count).unwrap_or(0) as u8;

    let in_high = cdf
        .iter()
        .rposition(|&c| c <= high_count)
        .map(|p| (p as u8).max(in_low.saturating_add(1)))
        .unwrap_or(255);

    contrast_stretch_linear(img, in_low, in_high, 0, 255)
}

/// Apply logarithmic contrast enhancement
///
/// `s = c * log(1 + r)` where `c` is chosen so that the maximum maps to 255.
/// Expands dark intensities while compressing bright ones.
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// Log-transformed image
pub fn contrast_stretch_log(img: &DynamicImage) -> Result<DynamicImage> {
    // c = 255 / log(1 + max_input)
    // For 8-bit images max_input = 255, so c = 255 / log(256)
    let c = 255.0 / (256.0f32).ln();

    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let val = c * (1.0 + i as f32).ln();
        *entry = val.clamp(0.0, 255.0) as u8;
    }

    apply_lut_to_image(img, &lut)
}

/// Apply power-law (gamma) contrast transformation
///
/// `s = c * r^gamma` where `c = 255 / 255^gamma` to normalise.
/// gamma < 1 brightens (expands dark range), gamma > 1 darkens.
///
/// # Arguments
///
/// * `img` - Input image
/// * `gamma` - Gamma value (must be positive)
///
/// # Returns
///
/// Gamma-transformed image
pub fn contrast_stretch_power(img: &DynamicImage, gamma: f32) -> Result<DynamicImage> {
    if gamma <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "Gamma must be positive".to_string(),
        ));
    }

    let mut lut = [0u8; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        let normalised = i as f32 / 255.0;
        let val = normalised.powf(gamma) * 255.0;
        *entry = val.clamp(0.0, 255.0) as u8;
    }

    apply_lut_to_image(img, &lut)
}

// ---------------------------------------------------------------------------
// Homomorphic filtering
// ---------------------------------------------------------------------------

/// Apply homomorphic filtering for illumination-reflectance separation
///
/// Homomorphic filtering works in the frequency domain on the log of the image.
/// It reduces the effect of non-uniform illumination and enhances contrast.
///
/// Algorithm:
/// 1. Take log of image
/// 2. Apply high-pass Gaussian filter in spatial domain (approximation)
/// 3. Exponentiate
///
/// # Arguments
///
/// * `img` - Input image
/// * `gamma_low` - Gain for low-frequency components (< 1.0 compresses illumination)
/// * `gamma_high` - Gain for high-frequency components (> 1.0 enhances reflectance)
/// * `sigma` - Standard deviation for the Gaussian filter
/// * `cutoff` - Cutoff frequency (normalised, 0.0 to 1.0)
///
/// # Returns
///
/// Enhanced image
pub fn homomorphic_filter(
    img: &DynamicImage,
    gamma_low: f32,
    gamma_high: f32,
    sigma: f32,
    cutoff: f32,
) -> Result<DynamicImage> {
    if sigma <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "Sigma must be positive".to_string(),
        ));
    }
    if cutoff <= 0.0 || cutoff > 1.0 {
        return Err(VisionError::InvalidParameter(
            "Cutoff must be in (0, 1]".to_string(),
        ));
    }
    if gamma_low < 0.0 || gamma_high < 0.0 {
        return Err(VisionError::InvalidParameter(
            "Gamma values must be non-negative".to_string(),
        ));
    }

    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let w = width as usize;
    let h = height as usize;

    // Step 1: ln(image + 1)
    let mut log_img = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let val = gray.get_pixel(x as u32, y as u32)[0] as f32;
            log_img[y * w + x] = (val + 1.0).ln();
        }
    }

    // Step 2: Spatial-domain approximation of homomorphic filter
    // Apply a Gaussian low-pass filter, then reconstruct high-pass
    let kernel_radius = (3.0 * sigma).ceil() as usize;
    let ksize = 2 * kernel_radius + 1;

    // Build 1D Gaussian kernel
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut kernel_1d = vec![0.0f32; ksize];
    let mut ksum = 0.0f32;
    for (i, val) in kernel_1d.iter_mut().enumerate() {
        let d = (i as f32 - kernel_radius as f32) * cutoff;
        *val = (-d * d / two_sigma_sq).exp();
        ksum += *val;
    }
    for v in kernel_1d.iter_mut() {
        *v /= ksum;
    }

    // Separable convolution: horizontal pass
    let mut horiz = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for (k, &weight) in kernel_1d.iter().enumerate() {
                let ix = x as isize + k as isize - kernel_radius as isize;
                if ix >= 0 && ix < w as isize {
                    sum += log_img[y * w + ix as usize] * weight;
                    wsum += weight;
                }
            }
            if wsum > 0.0 {
                horiz[y * w + x] = sum / wsum;
            }
        }
    }

    // Vertical pass => low-pass result
    let mut low_pass = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for (k, &weight) in kernel_1d.iter().enumerate() {
                let iy = y as isize + k as isize - kernel_radius as isize;
                if iy >= 0 && iy < h as isize {
                    sum += horiz[iy as usize * w + x] * weight;
                    wsum += weight;
                }
            }
            if wsum > 0.0 {
                low_pass[y * w + x] = sum / wsum;
            }
        }
    }

    // High-pass = original - low-pass
    // Filtered = gamma_low * low_pass + gamma_high * high_pass
    let mut filtered = vec![0.0f32; w * h];
    for i in 0..w * h {
        let lp = low_pass[i];
        let hp = log_img[i] - lp;
        filtered[i] = gamma_low * lp + gamma_high * hp;
    }

    // Step 3: exp(filtered) - 1
    // Normalise to [0, 255]
    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    for &v in &filtered {
        let ev = v.exp() - 1.0;
        if ev < min_val {
            min_val = ev;
        }
        if ev > max_val {
            max_val = ev;
        }
    }

    let range = (max_val - min_val).max(1e-6);

    let mut result = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let v = filtered[y as usize * w + x as usize].exp() - 1.0;
            let normalised = ((v - min_val) / range * 255.0).clamp(0.0, 255.0) as u8;
            result.put_pixel(x, y, Luma([normalised]));
        }
    }

    Ok(DynamicImage::ImageLuma8(result))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply a 256-entry lookup table to every pixel of an image
///
/// Handles both grayscale and colour images.
fn apply_lut_to_image(img: &DynamicImage, lut: &[u8; 256]) -> Result<DynamicImage> {
    match img.color() {
        image::ColorType::L8 | image::ColorType::L16 | image::ColorType::La8 => {
            let gray = img.to_luma8();
            let (w, h) = gray.dimensions();
            let mut out = GrayImage::new(w, h);
            for (x, y, pixel) in gray.enumerate_pixels() {
                out.put_pixel(x, y, Luma([lut[pixel[0] as usize]]));
            }
            Ok(DynamicImage::ImageLuma8(out))
        }
        _ => {
            let rgb = img.to_rgb8();
            let (w, h) = rgb.dimensions();
            let mut out = RgbImage::new(w, h);
            for (x, y, pixel) in rgb.enumerate_pixels() {
                out.put_pixel(
                    x,
                    y,
                    Rgb([
                        lut[pixel[0] as usize],
                        lut[pixel[1] as usize],
                        lut[pixel[2] as usize],
                    ]),
                );
            }
            Ok(DynamicImage::ImageRgb8(out))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gray(width: u32, height: u32, val: u8) -> DynamicImage {
        let mut img = GrayImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = Luma([val]);
        }
        DynamicImage::ImageLuma8(img)
    }

    fn make_gradient(width: u32, height: u32) -> DynamicImage {
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = ((x as f32 / width as f32) * 255.0) as u8;
                img.put_pixel(x, y, Luma([val]));
            }
        }
        DynamicImage::ImageLuma8(img)
    }

    fn make_low_contrast(width: u32, height: u32) -> DynamicImage {
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                // Values only in [100, 150]
                let val = 100 + ((x as f32 / width as f32) * 50.0) as u8;
                img.put_pixel(x, y, Luma([val]));
            }
        }
        DynamicImage::ImageLuma8(img)
    }

    // --- Linear contrast stretch ---
    #[test]
    fn test_contrast_stretch_linear_identity() {
        let img = make_gradient(256, 1);
        let result = contrast_stretch_linear(&img, 0, 255, 0, 255).expect("stretch should succeed");
        let orig = img.to_luma8();
        let res = result.to_luma8();
        for x in 0..256u32 {
            assert_eq!(orig.get_pixel(x, 0)[0], res.get_pixel(x, 0)[0]);
        }
    }

    #[test]
    fn test_contrast_stretch_linear_expands() {
        let img = make_low_contrast(100, 10);
        let result =
            contrast_stretch_linear(&img, 100, 150, 0, 255).expect("stretch should succeed");
        let gray = result.to_luma8();
        let min_v = gray.pixels().map(|p| p[0]).min().unwrap_or(255);
        let max_v = gray.pixels().map(|p| p[0]).max().unwrap_or(0);
        assert!(
            max_v - min_v > 100,
            "Range should be expanded: got {min_v}-{max_v}"
        );
    }

    #[test]
    fn test_contrast_stretch_linear_invalid() {
        let img = make_gray(10, 10, 100);
        assert!(contrast_stretch_linear(&img, 100, 50, 0, 255).is_err());
        assert!(contrast_stretch_linear(&img, 0, 255, 200, 100).is_err());
    }

    // --- Auto contrast stretch ---
    #[test]
    fn test_contrast_stretch_auto() {
        let img = make_low_contrast(100, 10);
        let result = contrast_stretch_auto(&img, 1.0, 99.0).expect("auto stretch should succeed");
        let gray = result.to_luma8();
        let min_v = gray.pixels().map(|p| p[0]).min().unwrap_or(255);
        let max_v = gray.pixels().map(|p| p[0]).max().unwrap_or(0);
        // Should be close to full range
        assert!(min_v <= 10, "Min should be near 0, got {min_v}");
        assert!(max_v >= 245, "Max should be near 255, got {max_v}");
    }

    #[test]
    fn test_contrast_stretch_auto_invalid() {
        let img = make_gray(10, 10, 100);
        assert!(contrast_stretch_auto(&img, -1.0, 99.0).is_err());
        assert!(contrast_stretch_auto(&img, 50.0, 30.0).is_err());
        assert!(contrast_stretch_auto(&img, 0.0, 101.0).is_err());
    }

    // --- Log stretch ---
    #[test]
    fn test_contrast_stretch_log() {
        let img = make_gradient(256, 1);
        let result = contrast_stretch_log(&img).expect("log stretch should succeed");
        let gray = result.to_luma8();
        // Log transform should expand dark tones, compress bright
        let mid_orig = 128u8;
        let mid_log = gray.get_pixel(128, 0)[0];
        assert!(
            mid_log > mid_orig,
            "Log should brighten mid-tones: orig={mid_orig}, log={mid_log}"
        );
    }

    // --- Power-law stretch ---
    #[test]
    fn test_contrast_stretch_power_identity() {
        let img = make_gradient(256, 1);
        let result = contrast_stretch_power(&img, 1.0).expect("power stretch should succeed");
        let orig = img.to_luma8();
        let res = result.to_luma8();
        for x in 0..256u32 {
            assert!(
                (orig.get_pixel(x, 0)[0] as i16 - res.get_pixel(x, 0)[0] as i16).unsigned_abs()
                    <= 1,
                "Gamma=1.0 should be identity"
            );
        }
    }

    #[test]
    fn test_contrast_stretch_power_brighten() {
        let img = make_gray(10, 10, 50);
        let result = contrast_stretch_power(&img, 0.5).expect("power stretch should succeed");
        let gray = result.to_luma8();
        let val = gray.get_pixel(0, 0)[0];
        assert!(val > 50, "Gamma<1 should brighten: got {val}");
    }

    #[test]
    fn test_contrast_stretch_power_invalid() {
        let img = make_gray(10, 10, 100);
        assert!(contrast_stretch_power(&img, 0.0).is_err());
        assert!(contrast_stretch_power(&img, -1.0).is_err());
    }

    // --- Homomorphic filter ---
    #[test]
    fn test_homomorphic_filter_basic() {
        let img = make_gradient(64, 64);
        let result = homomorphic_filter(&img, 0.5, 2.0, 5.0, 0.5)
            .expect("homomorphic filter should succeed");
        assert_eq!(result.width(), 64);
        assert_eq!(result.height(), 64);
    }

    #[test]
    fn test_homomorphic_filter_uniform() {
        let img = make_gray(32, 32, 128);
        let result = homomorphic_filter(&img, 0.5, 2.0, 3.0, 0.5)
            .expect("homomorphic filter should succeed");
        assert_eq!(result.width(), 32);
        assert_eq!(result.height(), 32);
    }

    #[test]
    fn test_homomorphic_filter_invalid_sigma() {
        let img = make_gray(10, 10, 100);
        assert!(homomorphic_filter(&img, 0.5, 2.0, 0.0, 0.5).is_err());
        assert!(homomorphic_filter(&img, 0.5, 2.0, -1.0, 0.5).is_err());
    }

    #[test]
    fn test_homomorphic_filter_invalid_cutoff() {
        let img = make_gray(10, 10, 100);
        assert!(homomorphic_filter(&img, 0.5, 2.0, 5.0, 0.0).is_err());
        assert!(homomorphic_filter(&img, 0.5, 2.0, 5.0, 1.5).is_err());
    }

    #[test]
    fn test_homomorphic_filter_invalid_gamma() {
        let img = make_gray(10, 10, 100);
        assert!(homomorphic_filter(&img, -0.5, 2.0, 5.0, 0.5).is_err());
        assert!(homomorphic_filter(&img, 0.5, -2.0, 5.0, 0.5).is_err());
    }

    // --- LUT helper ---
    #[test]
    fn test_apply_lut_color() {
        let mut img = RgbImage::new(5, 5);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([100, 150, 200]);
        }
        let dyn_img = DynamicImage::ImageRgb8(img);

        // Identity LUT
        let mut lut = [0u8; 256];
        for (i, entry) in lut.iter_mut().enumerate() {
            *entry = i as u8;
        }

        let result = apply_lut_to_image(&dyn_img, &lut).expect("LUT should succeed");
        let px = result.to_rgb8().get_pixel(0, 0).0;
        assert_eq!(px, [100, 150, 200]);
    }
}
