//! Color transformation module
//!
//! This module provides functionality for working with different color spaces
//! and performing color transformations.

pub mod octree_quantization;
pub mod quantization;

use crate::error::Result;
use image::{DynamicImage, ImageBuffer, Rgb};
// Note: Array2 might be needed in future implementations

pub use octree_quantization::{adaptive_octree_quantize, extract_palette, octree_quantize};
pub use quantization::{kmeans_quantize, median_cut_quantize, InitMethod, KMeansParams};

/// Represents a color space
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorSpace {
    /// RGB color space
    RGB,
    /// HSV (Hue, Saturation, Value) color space
    HSV,
    /// HSL (Hue, Saturation, Lightness) color space
    HSL,
    /// LAB color space (CIE L*a*b*)
    LAB,
    /// CIE XYZ color space
    XYZ,
    /// YCbCr color space (ITU-R BT.601)
    YCbCr,
    /// Grayscale
    Gray,
}

/// Convert an image from RGB to HSV
///
/// # Arguments
///
/// * `img` - Input RGB image
///
/// # Returns
///
/// * Result containing an HSV image
#[allow(dead_code)]
pub fn rgb_to_hsv(img: &DynamicImage) -> Result<DynamicImage> {
    // Ensure input is RGB
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    // Create output buffer
    let mut hsv_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let rgb = rgb_img.get_pixel(x, y);
            let r = rgb[0] as f32 / 255.0;
            let g = rgb[1] as f32 / 255.0;
            let b = rgb[2] as f32 / 255.0;

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let delta = max - min;

            // Hue calculation
            let h = if delta == 0.0 {
                0.0
            } else if max == r {
                60.0 * (((g - b) / delta) % 6.0)
            } else if max == g {
                60.0 * (((b - r) / delta) + 2.0)
            } else {
                60.0 * (((r - g) / delta) + 4.0)
            };

            // Normalize hue to [0, 360)
            let h = if h < 0.0 { h + 360.0 } else { h };

            // Saturation calculation
            let s = if max == 0.0 { 0.0 } else { delta / max };

            // Value calculation
            let v = max;

            // Store HSV as RGB values for visualization
            // Hue [0, 360) -> [0, 255]
            // Saturation [0, 1] -> [0, 255]
            // Value [0, 1] -> [0, 255]
            hsv_img.put_pixel(
                x,
                y,
                Rgb([
                    (h / 360.0 * 255.0) as u8,
                    (s * 255.0) as u8,
                    (v * 255.0) as u8,
                ]),
            );
        }
    }

    Ok(DynamicImage::ImageRgb8(hsv_img))
}

/// Convert an image from HSV to RGB
///
/// # Arguments
///
/// * `img` - Input HSV image (represented as RGB buffer where channels are H, S, V)
///
/// # Returns
///
/// * Result containing an RGB image
#[allow(dead_code)]
pub fn hsv_to_rgb(_hsvimg: &DynamicImage) -> Result<DynamicImage> {
    let hsv = _hsvimg.to_rgb8();
    let (width, height) = hsv.dimensions();

    let mut rgb_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let hsv_pixel = hsv.get_pixel(x, y);

            // Convert back to HSV range
            let h = hsv_pixel[0] as f32 / 255.0 * 360.0;
            let s = hsv_pixel[1] as f32 / 255.0;
            let v = hsv_pixel[2] as f32 / 255.0;

            // HSV to RGB conversion
            let c = v * s;
            let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
            let m = v - c;

            let (r1, g1, b1) = if h < 60.0 {
                (c, x, 0.0)
            } else if h < 120.0 {
                (x, c, 0.0)
            } else if h < 180.0 {
                (0.0, c, x)
            } else if h < 240.0 {
                (0.0, x, c)
            } else if h < 300.0 {
                (x, 0.0, c)
            } else {
                (c, 0.0, x)
            };

            let r = ((r1 + m) * 255.0) as u8;
            let g = ((g1 + m) * 255.0) as u8;
            let b = ((b1 + m) * 255.0) as u8;

            #[allow(clippy::unnecessary_cast)]
            rgb_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(rgb_img))
}

/// Convert RGB to grayscale using weighted average
///
/// # Arguments
///
/// * `img` - Input RGB image
/// * `weights` - Optional RGB weights (default: [0.2989, 0.5870, 0.1140] - standard luminance)
///
/// # Returns
///
/// * Result containing a grayscale image
#[allow(dead_code)]
pub fn rgb_to_grayscale(img: &DynamicImage, weights: Option<[f32; 3]>) -> Result<DynamicImage> {
    // Default weights based on human perception of color
    let weights = weights.unwrap_or([0.2989, 0.5870, 0.1140]);

    // Get RGB image
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    // Create grayscale image
    let mut gray_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let rgb = rgb_img.get_pixel(x, y);

            // Apply weighted average
            let gray_value = (weights[0] * rgb[0] as f32
                + weights[1] * rgb[1] as f32
                + weights[2] * rgb[2] as f32)
                .clamp(0.0, 255.0) as u8;

            gray_img.put_pixel(x, y, image::Luma([gray_value]));
        }
    }

    Ok(DynamicImage::ImageLuma8(gray_img))
}

/// Convert RGB to LAB color space
///
/// # Arguments
///
/// * `img` - Input RGB image
///
/// # Returns
///
/// * Result containing a LAB image (represented as RGB buffer where channels are L, a, b)
#[allow(dead_code)]
pub fn rgb_to_lab(img: &DynamicImage) -> Result<DynamicImage> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    let mut lab_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let rgb = rgb_img.get_pixel(x, y);

            // Convert RGB to XYZ
            let r = rgb[0] as f32 / 255.0;
            let g = rgb[1] as f32 / 255.0;
            let b = rgb[2] as f32 / 255.0;

            // Gamma correction (sRGB to linear RGB)
            let r_lin = if r > 0.04045 {
                ((r + 0.055) / 1.055).powf(2.4)
            } else {
                r / 12.92
            };
            let g_lin = if g > 0.04045 {
                ((g + 0.055) / 1.055).powf(2.4)
            } else {
                g / 12.92
            };
            let b_lin = if b > 0.04045 {
                ((b + 0.055) / 1.055).powf(2.4)
            } else {
                b / 12.92
            };

            // RGB to XYZ conversion (using sRGB D65 matrix)
            let x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375;
            let y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750;
            let z = r_lin * 0.0193339 + g_lin * 0.119192 + b_lin * 0.9503041;

            // XYZ to LAB
            // Reference white point (D65)
            let x_n = 0.95047;
            let y_n = 1.0;
            let z_n = 1.08883;

            // Scale XYZ values relative to reference white
            let x_r = x / x_n;
            let y_r = y / y_n;
            let z_r = z / z_n;

            // XYZ to LAB helper function
            let f = |t: f32| -> f32 {
                if t > 0.008856 {
                    t.powf(1.0 / 3.0)
                } else {
                    (7.787 * t) + (16.0 / 116.0)
                }
            };

            let fx = f(x_r);
            let fy = f(y_r);
            let fz = f(z_r);

            // Calculate LAB values
            let l = (116.0 * fy) - 16.0;
            let a = 500.0 * (fx - fy);
            let b_val = 200.0 * (fy - fz);

            // Scale to fit in 8-bit channels
            // L: [0, 100] -> [0, 255]
            // a: [-128, 127] -> [0, 255]
            // b: [-128, 127] -> [0, 255]
            let l_scaled = (l * 2.55).clamp(0.0, 255.0) as u8;
            let a_scaled = ((a + 128.0).clamp(0.0, 255.0)) as u8;
            let b_scaled = ((b_val + 128.0).clamp(0.0, 255.0)) as u8;

            lab_img.put_pixel(x as u32, y as u32, Rgb([l_scaled, a_scaled, b_scaled]));
        }
    }

    Ok(DynamicImage::ImageRgb8(lab_img))
}

/// Convert LAB to RGB color space
///
/// # Arguments
///
/// * `img` - Input LAB image (represented as RGB buffer where channels are L, a, b)
///
/// # Returns
///
/// * Result containing an RGB image
#[allow(dead_code)]
pub fn lab_to_rgb(_labimg: &DynamicImage) -> Result<DynamicImage> {
    let lab = _labimg.to_rgb8();
    let (width, height) = lab.dimensions();

    let mut rgb_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let lab_pixel = lab.get_pixel(x, y);

            // Scale back from 8-bit to LAB range
            let l = lab_pixel[0] as f32 / 2.55; // [0, 255] -> [0, 100]
            let a = lab_pixel[1] as f32 - 128.0; // [0, 255] -> [-128, 127]
            let b = lab_pixel[2] as f32 - 128.0; // [0, 255] -> [-128, 127]

            // LAB to XYZ
            let fy = (l + 16.0) / 116.0;
            let fx = a / 500.0 + fy;
            let fz = fy - b / 200.0;

            // Reference white point (D65)
            let x_n = 0.95047;
            let y_n = 1.0;
            let z_n = 1.08883;

            // LAB to XYZ helper function
            let f = |t: f32| -> f32 {
                if t > 0.206893 {
                    t.powi(3)
                } else {
                    (t - 16.0 / 116.0) / 7.787
                }
            };

            let x = x_n * f(fx);
            let y = y_n * f(fy);
            let z = z_n * f(fz);

            // XYZ to linear RGB (using inverse sRGB D65 matrix)
            let r_lin = x * 3.2404542 - y * 1.5371385 - z * 0.4985314;
            let g_lin = -x * 0.969266 + y * 1.8760108 + z * 0.0415560;
            let b_lin = x * 0.0556434 - y * 0.2040259 + z * 1.0572252;

            // Linear RGB to sRGB
            let r = if r_lin > 0.0031308 {
                1.055 * r_lin.powf(1.0 / 2.4) - 0.055
            } else {
                12.92 * r_lin
            };

            let g = if g_lin > 0.0031308 {
                1.055 * g_lin.powf(1.0 / 2.4) - 0.055
            } else {
                12.92 * g_lin
            };

            let b = if b_lin > 0.0031308 {
                1.055 * b_lin.powf(1.0 / 2.4) - 0.055
            } else {
                12.92 * b_lin
            };

            // Convert to 8-bit and clamp to valid range
            let r = (r * 255.0).clamp(0.0, 255.0) as u8;
            let g = (g * 255.0).clamp(0.0, 255.0) as u8;
            let b = (b * 255.0).clamp(0.0, 255.0) as u8;

            #[allow(clippy::unnecessary_cast)]
            rgb_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(rgb_img))
}

/// Split an RGB image into separate channels
///
/// # Arguments
///
/// * `img` - Input RGB image
///
/// # Returns
///
/// * Result containing a tuple of grayscale images (r, g, b)
#[allow(dead_code)]
pub fn split_channels(img: &DynamicImage) -> Result<(DynamicImage, DynamicImage, DynamicImage)> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    let mut r_channel = ImageBuffer::new(width, height);
    let mut g_channel = ImageBuffer::new(width, height);
    let mut b_channel = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let rgb = rgb_img.get_pixel(x, y);

            r_channel.put_pixel(x, y, image::Luma([rgb[0]]));
            g_channel.put_pixel(x, y, image::Luma([rgb[1]]));
            b_channel.put_pixel(x, y, image::Luma([rgb[2]]));
        }
    }

    Ok((
        DynamicImage::ImageLuma8(r_channel),
        DynamicImage::ImageLuma8(g_channel),
        DynamicImage::ImageLuma8(b_channel),
    ))
}

/// Merge separate channels into an RGB image
///
/// # Arguments
///
/// * `r_channel` - Red channel image
/// * `g_channel` - Green channel image
/// * `b_channel` - Blue channel image
///
/// # Returns
///
/// * Result containing an RGB image
#[allow(dead_code)]
pub fn merge_channels(
    r_channel: &DynamicImage,
    g_channel: &DynamicImage,
    b_channel: &DynamicImage,
) -> Result<DynamicImage> {
    let r_img = r_channel.to_luma8();
    let g_img = g_channel.to_luma8();
    let b_img = b_channel.to_luma8();

    let (width, height) = r_img.dimensions();

    // Check dimensions
    if g_img.dimensions() != (width, height) || b_img.dimensions() != (width, height) {
        return Err(crate::error::VisionError::InvalidParameter(
            "Channel dimensions do not match".to_string(),
        ));
    }

    let mut rgb_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let r = r_img.get_pixel(x, y)[0];
            let g = g_img.get_pixel(x, y)[0];
            let b = b_img.get_pixel(x, y)[0];

            #[allow(clippy::unnecessary_cast)]
            rgb_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(rgb_img))
}

// ---------------------------------------------------------------------------
// RGB <-> HSL
// ---------------------------------------------------------------------------

/// Convert an image from RGB to HSL
///
/// Channels are packed into an RGB buffer as:
/// - Channel 0: Hue `[0,360)` mapped to `[0,255]`
/// - Channel 1: Saturation `[0,1]` mapped to `[0,255]`
/// - Channel 2: Lightness `[0,1]` mapped to `[0,255]`
#[allow(dead_code)]
pub fn rgb_to_hsl(img: &DynamicImage) -> Result<DynamicImage> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();
    let mut hsl_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let px = rgb_img.get_pixel(x, y);
            let r = px[0] as f32 / 255.0;
            let g = px[1] as f32 / 255.0;
            let b = px[2] as f32 / 255.0;

            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let delta = max_c - min_c;
            let l = (max_c + min_c) / 2.0;

            let s = if delta < 1e-6 {
                0.0
            } else {
                delta / (1.0 - (2.0 * l - 1.0).abs())
            };

            let h = if delta < 1e-6 {
                0.0
            } else if (max_c - r).abs() < 1e-6 {
                let mut hh = 60.0 * ((g - b) / delta);
                if hh < 0.0 {
                    hh += 360.0;
                }
                hh
            } else if (max_c - g).abs() < 1e-6 {
                60.0 * ((b - r) / delta + 2.0)
            } else {
                60.0 * ((r - g) / delta + 4.0)
            };

            hsl_img.put_pixel(
                x,
                y,
                Rgb([
                    (h / 360.0 * 255.0).clamp(0.0, 255.0) as u8,
                    (s * 255.0).clamp(0.0, 255.0) as u8,
                    (l * 255.0).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    Ok(DynamicImage::ImageRgb8(hsl_img))
}

/// Convert an image from HSL to RGB
///
/// Expects an HSL image packed in an RGB buffer (see `rgb_to_hsl`).
#[allow(dead_code)]
pub fn hsl_to_rgb(hsl_img: &DynamicImage) -> Result<DynamicImage> {
    let hsl = hsl_img.to_rgb8();
    let (width, height) = hsl.dimensions();
    let mut rgb_out = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let px = hsl.get_pixel(x, y);
            let h = px[0] as f32 / 255.0 * 360.0;
            let s = px[1] as f32 / 255.0;
            let l = px[2] as f32 / 255.0;

            let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
            let hp = h / 60.0;
            let x_val = c * (1.0 - (hp % 2.0 - 1.0).abs());
            let m = l - c / 2.0;

            let (r1, g1, b1) = if hp < 1.0 {
                (c, x_val, 0.0)
            } else if hp < 2.0 {
                (x_val, c, 0.0)
            } else if hp < 3.0 {
                (0.0, c, x_val)
            } else if hp < 4.0 {
                (0.0, x_val, c)
            } else if hp < 5.0 {
                (x_val, 0.0, c)
            } else {
                (c, 0.0, x_val)
            };

            rgb_out.put_pixel(
                x,
                y,
                Rgb([
                    ((r1 + m) * 255.0).clamp(0.0, 255.0) as u8,
                    ((g1 + m) * 255.0).clamp(0.0, 255.0) as u8,
                    ((b1 + m) * 255.0).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    Ok(DynamicImage::ImageRgb8(rgb_out))
}

// ---------------------------------------------------------------------------
// RGB <-> YCbCr (ITU-R BT.601)
// ---------------------------------------------------------------------------

/// Convert an image from RGB to YCbCr (ITU-R BT.601)
#[allow(dead_code)]
pub fn rgb_to_ycbcr(img: &DynamicImage) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut out = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let px = rgb.get_pixel(x, y);
            let r = px[0] as f32;
            let g = px[1] as f32;
            let b = px[2] as f32;

            let y_val = 0.299 * r + 0.587 * g + 0.114 * b;
            let cb = 128.0 + (-0.168736 * r - 0.331264 * g + 0.5 * b);
            let cr = 128.0 + (0.5 * r - 0.418688 * g - 0.081312 * b);

            out.put_pixel(
                x,
                y,
                Rgb([
                    y_val.clamp(0.0, 255.0) as u8,
                    cb.clamp(0.0, 255.0) as u8,
                    cr.clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    Ok(DynamicImage::ImageRgb8(out))
}

/// Convert an image from YCbCr to RGB
#[allow(dead_code)]
pub fn ycbcr_to_rgb(ycbcr_img: &DynamicImage) -> Result<DynamicImage> {
    let ycbcr = ycbcr_img.to_rgb8();
    let (width, height) = ycbcr.dimensions();
    let mut out = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let px = ycbcr.get_pixel(x, y);
            let y_val = px[0] as f32;
            let cb = px[1] as f32 - 128.0;
            let cr = px[2] as f32 - 128.0;

            let r = y_val + 1.402 * cr;
            let g = y_val - 0.344136 * cb - 0.714136 * cr;
            let b = y_val + 1.772 * cb;

            out.put_pixel(
                x,
                y,
                Rgb([
                    r.clamp(0.0, 255.0) as u8,
                    g.clamp(0.0, 255.0) as u8,
                    b.clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    Ok(DynamicImage::ImageRgb8(out))
}

// ---------------------------------------------------------------------------
// RGB <-> CIE XYZ
// ---------------------------------------------------------------------------

/// sRGB gamma linearisation helper
#[inline]
fn srgb_to_linear(v: f32) -> f32 {
    if v > 0.04045 {
        ((v + 0.055) / 1.055).powf(2.4)
    } else {
        v / 12.92
    }
}

/// sRGB inverse gamma helper
#[inline]
fn linear_to_srgb(v: f32) -> f32 {
    if v > 0.0031308 {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    } else {
        12.92 * v
    }
}

/// Convert an image from RGB to CIE XYZ (D65 illuminant)
#[allow(dead_code)]
pub fn rgb_to_xyz(img: &DynamicImage) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut out = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let px = rgb.get_pixel(x, y);
            let r = srgb_to_linear(px[0] as f32 / 255.0);
            let g = srgb_to_linear(px[1] as f32 / 255.0);
            let b = srgb_to_linear(px[2] as f32 / 255.0);

            let x_val = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
            let y_val = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
            let z_val = r * 0.0193339 + g * 0.119_192 + b * 0.9503041;

            out.put_pixel(
                x,
                y,
                Rgb([
                    (x_val * 255.0).clamp(0.0, 255.0) as u8,
                    (y_val * 255.0).clamp(0.0, 255.0) as u8,
                    (z_val * 255.0).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    Ok(DynamicImage::ImageRgb8(out))
}

/// Convert an image from CIE XYZ to RGB
#[allow(dead_code)]
pub fn xyz_to_rgb(xyz_img: &DynamicImage) -> Result<DynamicImage> {
    let xyz = xyz_img.to_rgb8();
    let (width, height) = xyz.dimensions();
    let mut out = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let px = xyz.get_pixel(x, y);
            let x_val = px[0] as f32 / 255.0;
            let y_val = px[1] as f32 / 255.0;
            let z_val = px[2] as f32 / 255.0;

            let r_lin = x_val * 3.2404542 - y_val * 1.5371385 - z_val * 0.4985314;
            let g_lin = -x_val * 0.969_266 + y_val * 1.8760108 + z_val * 0.0415560;
            let b_lin = x_val * 0.0556434 - y_val * 0.2040259 + z_val * 1.0572252;

            let r = linear_to_srgb(r_lin);
            let g = linear_to_srgb(g_lin);
            let b = linear_to_srgb(b_lin);

            out.put_pixel(
                x,
                y,
                Rgb([
                    (r * 255.0).clamp(0.0, 255.0) as u8,
                    (g * 255.0).clamp(0.0, 255.0) as u8,
                    (b * 255.0).clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }

    Ok(DynamicImage::ImageRgb8(out))
}

// ---------------------------------------------------------------------------
// Dominant color extraction
// ---------------------------------------------------------------------------

/// A dominant color with its weight (proportion of pixels)
#[derive(Debug, Clone)]
pub struct DominantColor {
    /// RGB values
    pub rgb: [u8; 3],
    /// Proportion of pixels represented by this colour (0.0 to 1.0)
    pub weight: f32,
}

/// Extract dominant colours from an image using median-cut quantisation
///
/// # Arguments
///
/// * `img` - Input image
/// * `num_colors` - Number of dominant colours to extract (typically 3-8)
///
/// # Returns
///
/// * Result containing a vector of `DominantColor` sorted by weight (most dominant first)
pub fn extract_dominant_colors(
    img: &DynamicImage,
    num_colors: usize,
) -> Result<Vec<DominantColor>> {
    if num_colors == 0 {
        return Err(crate::error::VisionError::InvalidParameter(
            "Number of colors must be positive".to_string(),
        ));
    }

    let rgb = img.to_rgb8();
    let total_pixels = (rgb.width() as usize) * (rgb.height() as usize);
    if total_pixels == 0 {
        return Ok(Vec::new());
    }

    let mut pixels: Vec<[u8; 3]> = Vec::with_capacity(total_pixels);
    for pixel in rgb.pixels() {
        pixels.push([pixel[0], pixel[1], pixel[2]]);
    }

    let mut boxes: Vec<Vec<[u8; 3]>> = vec![pixels];

    while boxes.len() < num_colors {
        let mut best_idx = 0;
        let mut best_range = 0u16;
        for (i, bx) in boxes.iter().enumerate() {
            if bx.len() < 2 {
                continue;
            }
            let range = box_max_range(bx);
            if range > best_range {
                best_range = range;
                best_idx = i;
            }
        }
        if best_range == 0 {
            break;
        }

        let bx = boxes.remove(best_idx);
        let (a, b) = split_box(bx);
        boxes.push(a);
        boxes.push(b);
    }

    let mut result: Vec<DominantColor> = boxes
        .iter()
        .filter(|bx| !bx.is_empty())
        .map(|bx| {
            let count = bx.len();
            let mut r_sum = 0u64;
            let mut g_sum = 0u64;
            let mut b_sum = 0u64;
            for px in bx {
                r_sum += px[0] as u64;
                g_sum += px[1] as u64;
                b_sum += px[2] as u64;
            }
            DominantColor {
                rgb: [
                    (r_sum / count as u64) as u8,
                    (g_sum / count as u64) as u8,
                    (b_sum / count as u64) as u8,
                ],
                weight: count as f32 / total_pixels as f32,
            }
        })
        .collect();

    result.sort_by(|a, b| {
        b.weight
            .partial_cmp(&a.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(result)
}

/// Compute max colour-space range of a pixel box across R, G, B
fn box_max_range(pixels: &[[u8; 3]]) -> u16 {
    let mut ranges = [0u16; 3];
    for ch in 0..3 {
        let min_v = pixels.iter().map(|p| p[ch]).min().unwrap_or(0);
        let max_v = pixels.iter().map(|p| p[ch]).max().unwrap_or(0);
        ranges[ch] = (max_v as u16).saturating_sub(min_v as u16);
    }
    *ranges.iter().max().unwrap_or(&0)
}

/// Split a pixel box along the channel with the greatest range
fn split_box(mut pixels: Vec<[u8; 3]>) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut best_ch = 0usize;
    let mut best_range = 0u16;
    for ch in 0..3 {
        let min_v = pixels.iter().map(|p| p[ch]).min().unwrap_or(0);
        let max_v = pixels.iter().map(|p| p[ch]).max().unwrap_or(0);
        let range = (max_v as u16).saturating_sub(min_v as u16);
        if range > best_range {
            best_range = range;
            best_ch = ch;
        }
    }

    pixels.sort_by_key(|p| p[best_ch]);
    let mid = pixels.len() / 2;
    let second = pixels.split_off(mid);
    (pixels, second)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgb(width: u32, height: u32, r: u8, g: u8, b: u8) -> DynamicImage {
        let mut img = image::RgbImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = Rgb([r, g, b]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_rgb_to_hsl_and_back() {
        let img = make_rgb(10, 10, 200, 100, 50);
        let hsl = rgb_to_hsl(&img).expect("rgb_to_hsl failed");
        let recovered = hsl_to_rgb(&hsl).expect("hsl_to_rgb failed");
        let orig = img.to_rgb8();
        let rec = recovered.to_rgb8();
        let px_o = orig.get_pixel(0, 0);
        let px_r = rec.get_pixel(0, 0);
        for ch in 0..3 {
            assert!(
                (px_o[ch] as i16 - px_r[ch] as i16).unsigned_abs() <= 3,
                "Channel {ch} mismatch: orig={}, recovered={}",
                px_o[ch],
                px_r[ch]
            );
        }
    }

    #[test]
    fn test_rgb_to_hsl_gray() {
        let img = make_rgb(5, 5, 128, 128, 128);
        let hsl = rgb_to_hsl(&img).expect("rgb_to_hsl failed");
        let px = hsl.to_rgb8().get_pixel(0, 0).0;
        assert_eq!(px[1], 0, "Saturation should be 0 for gray");
    }

    #[test]
    fn test_rgb_to_ycbcr_and_back() {
        let img = make_rgb(10, 10, 180, 90, 40);
        let ycbcr = rgb_to_ycbcr(&img).expect("rgb_to_ycbcr failed");
        let recovered = ycbcr_to_rgb(&ycbcr).expect("ycbcr_to_rgb failed");
        let orig = img.to_rgb8();
        let rec = recovered.to_rgb8();
        let px_o = orig.get_pixel(0, 0);
        let px_r = rec.get_pixel(0, 0);
        for ch in 0..3 {
            assert!(
                (px_o[ch] as i16 - px_r[ch] as i16).unsigned_abs() <= 2,
                "Channel {ch} mismatch: orig={}, recovered={}",
                px_o[ch],
                px_r[ch]
            );
        }
    }

    #[test]
    fn test_rgb_to_ycbcr_white() {
        let img = make_rgb(5, 5, 255, 255, 255);
        let ycbcr = rgb_to_ycbcr(&img).expect("rgb_to_ycbcr failed");
        let px = ycbcr.to_rgb8().get_pixel(0, 0).0;
        assert!(px[0] >= 250, "Y of white should be near 255, got {}", px[0]);
        assert!(
            (px[1] as i16 - 128).unsigned_abs() <= 2,
            "Cb of white should be ~128, got {}",
            px[1]
        );
    }

    #[test]
    fn test_rgb_to_xyz_and_back() {
        let img = make_rgb(10, 10, 150, 100, 80);
        let xyz = rgb_to_xyz(&img).expect("rgb_to_xyz failed");
        let recovered = xyz_to_rgb(&xyz).expect("xyz_to_rgb failed");
        let orig = img.to_rgb8();
        let rec = recovered.to_rgb8();
        let px_o = orig.get_pixel(0, 0);
        let px_r = rec.get_pixel(0, 0);
        for ch in 0..3 {
            assert!(
                (px_o[ch] as i16 - px_r[ch] as i16).unsigned_abs() <= 3,
                "Channel {ch} mismatch: orig={}, recovered={}",
                px_o[ch],
                px_r[ch]
            );
        }
    }

    #[test]
    fn test_rgb_to_xyz_black() {
        let img = make_rgb(5, 5, 0, 0, 0);
        let xyz = rgb_to_xyz(&img).expect("rgb_to_xyz failed");
        let px = xyz.to_rgb8().get_pixel(0, 0).0;
        assert_eq!(px[0], 0);
        assert_eq!(px[1], 0);
        assert_eq!(px[2], 0);
    }

    #[test]
    fn test_extract_dominant_colors_uniform() {
        let img = make_rgb(20, 20, 100, 50, 200);
        let colors = extract_dominant_colors(&img, 3).expect("dominant extraction failed");
        assert!(!colors.is_empty());
        let dc = &colors[0];
        assert!((dc.rgb[0] as i16 - 100).unsigned_abs() <= 1);
        assert!((dc.rgb[1] as i16 - 50).unsigned_abs() <= 1);
        assert!((dc.rgb[2] as i16 - 200).unsigned_abs() <= 1);
    }

    #[test]
    fn test_extract_dominant_colors_two_regions() {
        let mut img = image::RgbImage::new(20, 10);
        for y in 0..10 {
            for x in 0..10 {
                img.put_pixel(x, y, Rgb([255, 0, 0]));
            }
            for x in 10..20 {
                img.put_pixel(x, y, Rgb([0, 0, 255]));
            }
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let colors = extract_dominant_colors(&dyn_img, 2).expect("dominant extraction failed");
        assert_eq!(colors.len(), 2);
        for c in &colors {
            assert!((c.weight - 0.5).abs() < 0.05);
        }
    }

    #[test]
    fn test_extract_dominant_colors_invalid() {
        let img = make_rgb(5, 5, 100, 100, 100);
        assert!(extract_dominant_colors(&img, 0).is_err());
    }

    #[test]
    fn test_rgb_to_hsv_and_back() {
        let img = make_rgb(10, 10, 200, 100, 50);
        let hsv = rgb_to_hsv(&img).expect("rgb_to_hsv failed");
        let recovered = hsv_to_rgb(&hsv).expect("hsv_to_rgb failed");
        let orig = img.to_rgb8();
        let rec = recovered.to_rgb8();
        let px_o = orig.get_pixel(0, 0);
        let px_r = rec.get_pixel(0, 0);
        for ch in 0..3 {
            assert!(
                (px_o[ch] as i16 - px_r[ch] as i16).unsigned_abs() <= 5,
                "Channel {ch} mismatch: orig={}, recovered={}",
                px_o[ch],
                px_r[ch]
            );
        }
    }

    #[test]
    fn test_rgb_to_grayscale_test() {
        let img = make_rgb(10, 10, 100, 100, 100);
        let gray = rgb_to_grayscale(&img, None).expect("grayscale failed");
        let luma = gray.to_luma8();
        let val = luma.get_pixel(0, 0)[0];
        assert!((val as i16 - 100).unsigned_abs() <= 2);
    }

    #[test]
    fn test_split_merge_channels_roundtrip() {
        let img = make_rgb(5, 5, 100, 150, 200);
        let (r, g, b) = split_channels(&img).expect("split failed");
        let merged = merge_channels(&r, &g, &b).expect("merge failed");
        let orig = img.to_rgb8();
        let rec = merged.to_rgb8();
        for yy in 0..5u32 {
            for xx in 0..5u32 {
                assert_eq!(orig.get_pixel(xx, yy), rec.get_pixel(xx, yy));
            }
        }
    }
}
