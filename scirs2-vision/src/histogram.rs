//! Histogram operations for image analysis and enhancement
//!
//! This module provides comprehensive histogram-based image processing:
//! - Histogram computation (1D grayscale and per-channel color)
//! - Histogram equalization (global and CLAHE/adaptive)
//! - Histogram matching/specification
//! - Cumulative distribution function (CDF)
//! - Otsu's threshold selection (single and multi-level)
//! - Histogram backprojection

use crate::error::{Result, VisionError};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};

/// Number of bins for 8-bit histograms
const NUM_BINS: usize = 256;

/// A 1D histogram with 256 bins for 8-bit images
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bin counts
    pub bins: [u64; NUM_BINS],
    /// Total number of pixels
    pub total: u64,
}

impl Histogram {
    /// Create a new empty histogram
    pub fn new() -> Self {
        Self {
            bins: [0u64; NUM_BINS],
            total: 0,
        }
    }

    /// Get the normalized histogram (probability distribution)
    pub fn normalized(&self) -> [f64; NUM_BINS] {
        let mut result = [0.0f64; NUM_BINS];
        if self.total > 0 {
            for (i, bin) in self.bins.iter().enumerate() {
                result[i] = *bin as f64 / self.total as f64;
            }
        }
        result
    }

    /// Compute the cumulative distribution function
    pub fn cdf(&self) -> [f64; NUM_BINS] {
        let norm = self.normalized();
        let mut cdf = [0.0f64; NUM_BINS];
        cdf[0] = norm[0];
        for i in 1..NUM_BINS {
            cdf[i] = cdf[i - 1] + norm[i];
        }
        cdf
    }

    /// Compute the cumulative distribution function (unnormalized, as counts)
    pub fn cdf_counts(&self) -> [u64; NUM_BINS] {
        let mut cdf = [0u64; NUM_BINS];
        cdf[0] = self.bins[0];
        for i in 1..NUM_BINS {
            cdf[i] = cdf[i - 1] + self.bins[i];
        }
        cdf
    }

    /// Get the mean intensity
    pub fn mean(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let mut sum = 0u64;
        for (i, &count) in self.bins.iter().enumerate() {
            sum += i as u64 * count;
        }
        sum as f64 / self.total as f64
    }

    /// Get the variance of intensity
    pub fn variance(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let mean = self.mean();
        let mut var_sum = 0.0f64;
        for (i, &count) in self.bins.iter().enumerate() {
            let diff = i as f64 - mean;
            var_sum += diff * diff * count as f64;
        }
        var_sum / self.total as f64
    }

    /// Get the minimum non-zero bin index
    pub fn min_value(&self) -> Option<u8> {
        self.bins.iter().position(|&c| c > 0).map(|i| i as u8)
    }

    /// Get the maximum non-zero bin index
    pub fn max_value(&self) -> Option<u8> {
        self.bins.iter().rposition(|&c| c > 0).map(|i| i as u8)
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-channel color histogram
#[derive(Debug, Clone)]
pub struct ColorHistogram {
    /// Red channel histogram
    pub red: Histogram,
    /// Green channel histogram
    pub green: Histogram,
    /// Blue channel histogram
    pub blue: Histogram,
}

impl ColorHistogram {
    /// Create a new empty color histogram
    pub fn new() -> Self {
        Self {
            red: Histogram::new(),
            green: Histogram::new(),
            blue: Histogram::new(),
        }
    }
}

impl Default for ColorHistogram {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Histogram computation
// ---------------------------------------------------------------------------

/// Compute histogram of a grayscale image
///
/// # Arguments
///
/// * `img` - Input image (will be converted to grayscale if needed)
///
/// # Returns
///
/// Histogram with 256 bins
pub fn compute_histogram(img: &DynamicImage) -> Histogram {
    let gray = img.to_luma8();
    let mut hist = Histogram::new();
    for pixel in gray.pixels() {
        hist.bins[pixel[0] as usize] += 1;
        hist.total += 1;
    }
    hist
}

/// Compute per-channel histogram of a color image
///
/// # Arguments
///
/// * `img` - Input image (will be converted to RGB if needed)
///
/// # Returns
///
/// ColorHistogram with separate R, G, B histograms
pub fn compute_color_histogram(img: &DynamicImage) -> ColorHistogram {
    let rgb = img.to_rgb8();
    let mut hist = ColorHistogram::new();
    for pixel in rgb.pixels() {
        hist.red.bins[pixel[0] as usize] += 1;
        hist.red.total += 1;
        hist.green.bins[pixel[1] as usize] += 1;
        hist.green.total += 1;
        hist.blue.bins[pixel[2] as usize] += 1;
        hist.blue.total += 1;
    }
    hist
}

// ---------------------------------------------------------------------------
// Histogram equalization
// ---------------------------------------------------------------------------

/// Apply global histogram equalization to a grayscale image
///
/// Redistributes intensity values to achieve a more uniform histogram,
/// improving contrast especially in images with narrow intensity ranges.
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// Equalized grayscale image
pub fn equalize_histogram(img: &DynamicImage) -> Result<DynamicImage> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    let total_pixels = (width as u64) * (height as u64);

    if total_pixels == 0 {
        return Err(VisionError::InvalidParameter(
            "Image has zero pixels".to_string(),
        ));
    }

    // Build histogram
    let mut hist = Histogram::new();
    for pixel in gray.pixels() {
        hist.bins[pixel[0] as usize] += 1;
    }
    hist.total = total_pixels;

    let cdf = hist.cdf_counts();

    // Find first non-zero CDF value
    let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);

    // Build lookup table
    let mut lut = [0u8; NUM_BINS];
    let denom = total_pixels.saturating_sub(cdf_min);
    if denom > 0 {
        for i in 0..NUM_BINS {
            let mapped = ((cdf[i].saturating_sub(cdf_min)) as f64 / denom as f64 * 255.0)
                .round()
                .clamp(0.0, 255.0);
            lut[i] = mapped as u8;
        }
    }

    // Apply LUT
    let mut result = ImageBuffer::new(width, height);
    for (x, y, pixel) in gray.enumerate_pixels() {
        result.put_pixel(x, y, Luma([lut[pixel[0] as usize]]));
    }

    Ok(DynamicImage::ImageLuma8(result))
}

/// Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
///
/// Divides the image into tiles, applies contrast-limited equalization
/// per tile, and bilinearly interpolates at boundaries.
///
/// # Arguments
///
/// * `img` - Input image
/// * `tile_size` - Width/height of each tile in pixels (typically 8)
/// * `clip_limit` - Contrast limit factor (>= 1.0; 1.0 = no clipping)
///
/// # Returns
///
/// Enhanced image
pub fn clahe(img: &DynamicImage, tile_size: u32, clip_limit: f32) -> Result<DynamicImage> {
    if tile_size == 0 {
        return Err(VisionError::InvalidParameter(
            "Tile size must be positive".to_string(),
        ));
    }
    if clip_limit < 1.0 {
        return Err(VisionError::InvalidParameter(
            "Clip limit must be >= 1.0".to_string(),
        ));
    }

    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    let nx = width.div_ceil(tile_size);
    let ny = height.div_ceil(tile_size);

    // Build per-tile histograms
    let mut histograms = vec![vec![[0u32; NUM_BINS]; nx as usize]; ny as usize];

    for y in 0..height {
        for x in 0..width {
            let tx = (x / tile_size) as usize;
            let ty = (y / tile_size) as usize;
            histograms[ty][tx][gray.get_pixel(x, y)[0] as usize] += 1;
        }
    }

    // Clip and redistribute
    #[allow(clippy::needless_range_loop)]
    for ty in 0..ny as usize {
        for tx in 0..nx as usize {
            let tw = tile_size.min(width.saturating_sub(tx as u32 * tile_size));
            let th = tile_size.min(height.saturating_sub(ty as u32 * tile_size));
            let area = tw * th;
            if area == 0 {
                continue;
            }

            let limit = (clip_limit * area as f32 / NUM_BINS as f32) as u32;
            let mut excess = 0u32;

            for bin in histograms[ty][tx].iter_mut() {
                if *bin > limit {
                    excess += *bin - limit;
                    *bin = limit;
                }
            }

            let per_bin = excess / NUM_BINS as u32;
            let mut residual = excess % NUM_BINS as u32;
            for bin in histograms[ty][tx].iter_mut() {
                *bin += per_bin;
                if residual > 0 {
                    *bin += 1;
                    residual -= 1;
                }
            }
        }
    }

    // Compute CDFs per tile
    let mut cdfs = vec![vec![[0u32; NUM_BINS]; nx as usize]; ny as usize];
    for ty in 0..ny as usize {
        for tx in 0..nx as usize {
            let tw = tile_size.min(width.saturating_sub(tx as u32 * tile_size));
            let th = tile_size.min(height.saturating_sub(ty as u32 * tile_size));
            let area = tw * th;

            cdfs[ty][tx][0] = histograms[ty][tx][0];
            for i in 1..NUM_BINS {
                cdfs[ty][tx][i] = cdfs[ty][tx][i - 1] + histograms[ty][tx][i];
            }

            if area > 0 {
                for v in cdfs[ty][tx].iter_mut() {
                    *v = (*v * 255) / area;
                }
            }
        }
    }

    // Bilinear interpolation of tile CDFs
    let mut result = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let val = gray.get_pixel(x, y)[0] as usize;
            let tx = x / tile_size;
            let ty = y / tile_size;

            let fx = (x % tile_size) as f32 / tile_size as f32;
            let fy = (y % tile_size) as f32 / tile_size as f32;

            let mapped = if tx >= nx - 1 && ty >= ny - 1 {
                cdfs[ty as usize][tx as usize][val] as f32
            } else if tx >= nx - 1 {
                let top = cdfs[ty as usize][tx as usize][val] as f32;
                let bot = cdfs[((ty + 1) as usize).min((ny - 1) as usize)][tx as usize][val] as f32;
                (1.0 - fy) * top + fy * bot
            } else if ty >= ny - 1 {
                let left = cdfs[ty as usize][tx as usize][val] as f32;
                let right =
                    cdfs[ty as usize][((tx + 1) as usize).min((nx - 1) as usize)][val] as f32;
                (1.0 - fx) * left + fx * right
            } else {
                let tl = cdfs[ty as usize][tx as usize][val] as f32;
                let tr = cdfs[ty as usize][(tx + 1) as usize][val] as f32;
                let bl = cdfs[(ty + 1) as usize][tx as usize][val] as f32;
                let br = cdfs[(ty + 1) as usize][(tx + 1) as usize][val] as f32;

                let top = (1.0 - fx) * tl + fx * tr;
                let bot = (1.0 - fx) * bl + fx * br;
                (1.0 - fy) * top + fy * bot
            };

            result.put_pixel(x, y, Luma([mapped.clamp(0.0, 255.0) as u8]));
        }
    }

    Ok(DynamicImage::ImageLuma8(result))
}

// ---------------------------------------------------------------------------
// Histogram matching / specification
// ---------------------------------------------------------------------------

/// Match the histogram of a source image to a reference histogram
///
/// The output image has the same spatial structure as `source` but with
/// an intensity distribution that approximates `reference_hist`.
///
/// # Arguments
///
/// * `source` - Input image
/// * `reference_hist` - Target histogram to match
///
/// # Returns
///
/// Image with matched histogram
pub fn match_histogram(source: &DynamicImage, reference_hist: &Histogram) -> Result<DynamicImage> {
    let gray = source.to_luma8();
    let (width, height) = gray.dimensions();

    // CDF of the source
    let src_hist = compute_histogram(source);
    let src_cdf = src_hist.cdf();

    // CDF of the reference
    let ref_cdf = reference_hist.cdf();

    // Build mapping: for each source level, find the reference level whose
    // CDF value is closest to the source CDF value.
    let mut lut = [0u8; NUM_BINS];
    for i in 0..NUM_BINS {
        let target_cdf = src_cdf[i];
        let mut best_j = 0usize;
        let mut best_diff = f64::INFINITY;
        for (j, &rcdf_val) in ref_cdf.iter().enumerate() {
            let diff = (rcdf_val - target_cdf).abs();
            if diff < best_diff {
                best_diff = diff;
                best_j = j;
            }
        }
        lut[i] = best_j as u8;
    }

    let mut result = ImageBuffer::new(width, height);
    for (x, y, pixel) in gray.enumerate_pixels() {
        result.put_pixel(x, y, Luma([lut[pixel[0] as usize]]));
    }

    Ok(DynamicImage::ImageLuma8(result))
}

/// Match the histogram of a source image to a reference image
///
/// Convenience wrapper that computes the reference histogram automatically.
///
/// # Arguments
///
/// * `source` - Input image to transform
/// * `reference` - Reference image whose histogram to match
///
/// # Returns
///
/// Image with histogram matched to reference
pub fn match_histogram_image(
    source: &DynamicImage,
    reference: &DynamicImage,
) -> Result<DynamicImage> {
    let ref_hist = compute_histogram(reference);
    match_histogram(source, &ref_hist)
}

// ---------------------------------------------------------------------------
// CDF helpers
// ---------------------------------------------------------------------------

/// Compute the cumulative distribution function for a grayscale image
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// Array of 256 CDF values in [0.0, 1.0]
pub fn compute_cdf(img: &DynamicImage) -> [f64; NUM_BINS] {
    let hist = compute_histogram(img);
    hist.cdf()
}

// ---------------------------------------------------------------------------
// Otsu's threshold
// ---------------------------------------------------------------------------

/// Find the optimal threshold using Otsu's method
///
/// Minimises intra-class variance (equivalently maximises inter-class variance)
/// to find the best binarisation threshold.
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// Optimal threshold value [0, 255]
pub fn otsu_threshold(img: &DynamicImage) -> u8 {
    let hist = compute_histogram(img);
    otsu_threshold_from_histogram(&hist)
}

/// Compute Otsu threshold from a pre-computed histogram
pub fn otsu_threshold_from_histogram(hist: &Histogram) -> u8 {
    if hist.total == 0 {
        return 128;
    }

    let prob = hist.normalized();

    // Total mean
    let mut total_mean = 0.0f64;
    for (i, &p) in prob.iter().enumerate() {
        total_mean += i as f64 * p;
    }

    let mut best_threshold = 0u8;
    let mut best_variance = 0.0f64;

    let mut w0 = 0.0f64; // weight of class 0
    let mut sum0 = 0.0f64; // cumulative sum for class 0

    for (t, &prob_t) in prob.iter().enumerate().take(255) {
        w0 += prob_t;
        if w0 == 0.0 {
            continue;
        }
        let w1 = 1.0 - w0;
        if w1 == 0.0 {
            break;
        }

        sum0 += t as f64 * prob_t;
        let mu0 = sum0 / w0;
        let mu1 = (total_mean - sum0) / w1;

        let between_var = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
        if between_var > best_variance {
            best_variance = between_var;
            best_threshold = t as u8;
        }
    }

    best_threshold
}

/// Find multiple thresholds using multi-level Otsu's method
///
/// Extends Otsu's binarization to multiple classes by exhaustively
/// searching for the thresholds that maximise inter-class variance.
///
/// # Arguments
///
/// * `img` - Input image
/// * `num_thresholds` - Number of thresholds (number of classes = num_thresholds + 1)
///
/// # Returns
///
/// Sorted vector of threshold values
pub fn multi_otsu_threshold(img: &DynamicImage, num_thresholds: usize) -> Result<Vec<u8>> {
    if num_thresholds == 0 {
        return Err(VisionError::InvalidParameter(
            "Number of thresholds must be positive".to_string(),
        ));
    }
    if num_thresholds > 4 {
        return Err(VisionError::InvalidParameter(
            "Multi-Otsu supports up to 4 thresholds (5 classes)".to_string(),
        ));
    }

    let hist = compute_histogram(img);
    if hist.total == 0 {
        return Ok(vec![128; num_thresholds]);
    }

    let prob = hist.normalized();

    if num_thresholds == 1 {
        return Ok(vec![otsu_threshold_from_histogram(&hist)]);
    }

    // Precompute partial sums for efficiency
    // P[k] = sum(prob[0..=k]), S[k] = sum(i*prob[i] for i in 0..=k)
    let mut p = [0.0f64; NUM_BINS];
    let mut s = [0.0f64; NUM_BINS];
    p[0] = prob[0];
    s[0] = 0.0;
    for i in 1..NUM_BINS {
        p[i] = p[i - 1] + prob[i];
        s[i] = s[i - 1] + i as f64 * prob[i];
    }

    let total_mean = s[NUM_BINS - 1];

    // Helper: inter-class variance contribution for a range [lo..hi]
    // Mean of class: s_range / p_range
    // Weight: p_range
    let class_variance = |lo: usize, hi: usize| -> f64 {
        let p_lo = if lo == 0 { 0.0 } else { p[lo - 1] };
        let s_lo = if lo == 0 { 0.0 } else { s[lo - 1] };
        let p_range = p[hi] - p_lo;
        let s_range = s[hi] - s_lo;
        if p_range < 1e-15 {
            return 0.0;
        }
        let mu = s_range / p_range;
        p_range * (mu - total_mean) * (mu - total_mean)
    };

    match num_thresholds {
        2 => {
            let mut best = 0.0f64;
            let mut best_t = (0u8, 0u8);
            for t1 in 1..254u8 {
                for t2 in (t1 + 1)..255u8 {
                    let var = class_variance(0, t1 as usize)
                        + class_variance(t1 as usize + 1, t2 as usize)
                        + class_variance(t2 as usize + 1, 255);
                    if var > best {
                        best = var;
                        best_t = (t1, t2);
                    }
                }
            }
            Ok(vec![best_t.0, best_t.1])
        }
        3 => {
            let mut best = 0.0f64;
            let mut best_t = (0u8, 0u8, 0u8);
            // Use step to reduce exhaustive search for 3 thresholds
            for t1 in (1..253u8).step_by(2) {
                for t2 in ((t1 + 1)..254u8).step_by(2) {
                    for t3 in ((t2 + 1)..255u8).step_by(2) {
                        let var = class_variance(0, t1 as usize)
                            + class_variance(t1 as usize + 1, t2 as usize)
                            + class_variance(t2 as usize + 1, t3 as usize)
                            + class_variance(t3 as usize + 1, 255);
                        if var > best {
                            best = var;
                            best_t = (t1, t2, t3);
                        }
                    }
                }
            }
            // Refine around best found with step 1
            let refine_range = 3i16;
            for d1 in -refine_range..=refine_range {
                let t1 = (best_t.0 as i16 + d1).clamp(1, 252) as u8;
                for d2 in -refine_range..=refine_range {
                    let t2 = (best_t.1 as i16 + d2).clamp(t1 as i16 + 1, 253) as u8;
                    for d3 in -refine_range..=refine_range {
                        let t3 = (best_t.2 as i16 + d3).clamp(t2 as i16 + 1, 254) as u8;
                        let var = class_variance(0, t1 as usize)
                            + class_variance(t1 as usize + 1, t2 as usize)
                            + class_variance(t2 as usize + 1, t3 as usize)
                            + class_variance(t3 as usize + 1, 255);
                        if var > best {
                            best = var;
                            best_t = (t1, t2, t3);
                        }
                    }
                }
            }
            Ok(vec![best_t.0, best_t.1, best_t.2])
        }
        4 => {
            // 4 thresholds: coarse search with step 4, then refine
            let mut best = 0.0f64;
            let mut best_t = (0u8, 0u8, 0u8, 0u8);
            for t1 in (1..252u8).step_by(4) {
                for t2 in ((t1 + 1)..253u8).step_by(4) {
                    for t3 in ((t2 + 1)..254u8).step_by(4) {
                        for t4 in ((t3 + 1)..255u8).step_by(4) {
                            let var = class_variance(0, t1 as usize)
                                + class_variance(t1 as usize + 1, t2 as usize)
                                + class_variance(t2 as usize + 1, t3 as usize)
                                + class_variance(t3 as usize + 1, t4 as usize)
                                + class_variance(t4 as usize + 1, 255);
                            if var > best {
                                best = var;
                                best_t = (t1, t2, t3, t4);
                            }
                        }
                    }
                }
            }
            // Refine
            let refine_range = 4i16;
            for d1 in -refine_range..=refine_range {
                let t1 = (best_t.0 as i16 + d1).clamp(1, 251) as u8;
                for d2 in -refine_range..=refine_range {
                    let t2 = (best_t.1 as i16 + d2).clamp(t1 as i16 + 1, 252) as u8;
                    for d3 in -refine_range..=refine_range {
                        let t3 = (best_t.2 as i16 + d3).clamp(t2 as i16 + 1, 253) as u8;
                        for d4 in -refine_range..=refine_range {
                            let t4 = (best_t.3 as i16 + d4).clamp(t3 as i16 + 1, 254) as u8;
                            let var = class_variance(0, t1 as usize)
                                + class_variance(t1 as usize + 1, t2 as usize)
                                + class_variance(t2 as usize + 1, t3 as usize)
                                + class_variance(t3 as usize + 1, t4 as usize)
                                + class_variance(t4 as usize + 1, 255);
                            if var > best {
                                best = var;
                                best_t = (t1, t2, t3, t4);
                            }
                        }
                    }
                }
            }
            Ok(vec![best_t.0, best_t.1, best_t.2, best_t.3])
        }
        _ => Err(VisionError::InvalidParameter(format!(
            "Unsupported number of thresholds: {num_thresholds}"
        ))),
    }
}

/// Apply Otsu's threshold to binarise an image
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// (binary image, threshold value)
pub fn binarize_otsu(img: &DynamicImage) -> Result<(DynamicImage, u8)> {
    let threshold = otsu_threshold(img);
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    let mut result = ImageBuffer::new(width, height);
    for (x, y, pixel) in gray.enumerate_pixels() {
        let val = if pixel[0] > threshold { 255u8 } else { 0u8 };
        result.put_pixel(x, y, Luma([val]));
    }

    Ok((DynamicImage::ImageLuma8(result), threshold))
}

// ---------------------------------------------------------------------------
// Histogram backprojection
// ---------------------------------------------------------------------------

/// Compute histogram backprojection
///
/// Projects a histogram model onto an image, producing a probability map
/// indicating how well each pixel matches the model histogram.
/// Commonly used for object detection/tracking.
///
/// # Arguments
///
/// * `img` - Input image
/// * `model_hist` - Model histogram (should be normalized)
/// * `channel` - Which channel to use for backprojection (0=R/Gray, 1=G, 2=B)
///
/// # Returns
///
/// Grayscale probability map
pub fn backproject_histogram(
    img: &DynamicImage,
    model_hist: &Histogram,
    channel: usize,
) -> Result<DynamicImage> {
    if channel > 2 {
        return Err(VisionError::InvalidParameter(
            "Channel must be 0 (R/Gray), 1 (G), or 2 (B)".to_string(),
        ));
    }

    let norm = model_hist.normalized();

    // Find max for normalisation to [0, 255]
    let max_prob = norm.iter().copied().fold(0.0f64, f64::max);
    if max_prob < 1e-15 {
        // Empty histogram, return black image
        let gray = img.to_luma8();
        let (w, h) = gray.dimensions();
        return Ok(DynamicImage::ImageLuma8(ImageBuffer::new(w, h)));
    }

    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut result = ImageBuffer::new(width, height);

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let idx = pixel[channel] as usize;
        let prob = norm[idx] / max_prob;
        let val = (prob * 255.0).clamp(0.0, 255.0) as u8;
        result.put_pixel(x, y, Luma([val]));
    }

    Ok(DynamicImage::ImageLuma8(result))
}

/// Compute histogram backprojection using hue-saturation model
///
/// Uses the H and S channels of HSV for more robust colour-based
/// backprojection.
///
/// # Arguments
///
/// * `img` - Input image
/// * `hue_hist` - Histogram for hue channel
/// * `sat_hist` - Histogram for saturation channel
///
/// # Returns
///
/// Grayscale probability map
pub fn backproject_hs_histogram(
    img: &DynamicImage,
    hue_hist: &Histogram,
    sat_hist: &Histogram,
) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let h_norm = hue_hist.normalized();
    let s_norm = sat_hist.normalized();

    let h_max = h_norm.iter().copied().fold(0.0f64, f64::max).max(1e-15);
    let s_max = s_norm.iter().copied().fold(0.0f64, f64::max).max(1e-15);

    let mut result = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;

            let max_c = r.max(g).max(b);
            let min_c = r.min(g).min(b);
            let delta = max_c - min_c;

            // Hue
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

            // Saturation
            let s = if max_c < 1e-6 { 0.0 } else { delta / max_c };

            let h_idx = ((h / 360.0 * 255.0) as usize).min(255);
            let s_idx = ((s * 255.0) as usize).min(255);

            let h_prob = h_norm[h_idx] / h_max;
            let s_prob = s_norm[s_idx] / s_max;

            let combined = (h_prob * s_prob * 255.0).clamp(0.0, 255.0) as u8;
            result.put_pixel(x, y, Luma([combined]));
        }
    }

    Ok(DynamicImage::ImageLuma8(result))
}

// ---------------------------------------------------------------------------
// Equalize histogram for color images (per-channel)
// ---------------------------------------------------------------------------

/// Apply histogram equalization to each channel of a colour image independently
///
/// # Arguments
///
/// * `img` - Input colour image
///
/// # Returns
///
/// Equalized colour image
pub fn equalize_histogram_color(img: &DynamicImage) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let total = (width as u64) * (height as u64);

    if total == 0 {
        return Err(VisionError::InvalidParameter(
            "Image has zero pixels".to_string(),
        ));
    }

    // Build per-channel histograms
    let mut hists = [[0u64; NUM_BINS]; 3];
    for pixel in rgb.pixels() {
        hists[0][pixel[0] as usize] += 1;
        hists[1][pixel[1] as usize] += 1;
        hists[2][pixel[2] as usize] += 1;
    }

    // CDFs and LUTs
    let mut luts = [[0u8; NUM_BINS]; 3];
    for ch in 0..3 {
        let mut cdf = [0u64; NUM_BINS];
        cdf[0] = hists[ch][0];
        for i in 1..NUM_BINS {
            cdf[i] = cdf[i - 1] + hists[ch][i];
        }
        let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0);
        let denom = total.saturating_sub(cdf_min);
        if denom > 0 {
            for i in 0..NUM_BINS {
                luts[ch][i] = ((cdf[i].saturating_sub(cdf_min) as f64 / denom as f64) * 255.0)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }

    let mut result = RgbImage::new(width, height);
    for (x, y, pixel) in rgb.enumerate_pixels() {
        result.put_pixel(
            x,
            y,
            Rgb([
                luts[0][pixel[0] as usize],
                luts[1][pixel[1] as usize],
                luts[2][pixel[2] as usize],
            ]),
        );
    }

    Ok(DynamicImage::ImageRgb8(result))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gray_image(width: u32, height: u32, fill: u8) -> DynamicImage {
        let mut img = GrayImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = Luma([fill]);
        }
        DynamicImage::ImageLuma8(img)
    }

    fn make_gradient_image(width: u32, height: u32) -> DynamicImage {
        let mut img = GrayImage::new(width, height);
        let divisor = if width > 1 { (width - 1) as f32 } else { 1.0 };
        for y in 0..height {
            for x in 0..width {
                let val = ((x as f32 / divisor) * 255.0).min(255.0) as u8;
                img.put_pixel(x, y, Luma([val]));
            }
        }
        DynamicImage::ImageLuma8(img)
    }

    fn make_bimodal_image(width: u32, height: u32) -> DynamicImage {
        let mut img = GrayImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let val = if x < width / 2 { 50u8 } else { 200u8 };
                img.put_pixel(x, y, Luma([val]));
            }
        }
        DynamicImage::ImageLuma8(img)
    }

    fn make_rgb_image(width: u32, height: u32) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let r = ((x as f32 / width as f32) * 255.0) as u8;
                let g = ((y as f32 / height as f32) * 255.0) as u8;
                let b = 128u8;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_compute_histogram_uniform() {
        let img = make_gray_image(10, 10, 128);
        let hist = compute_histogram(&img);
        assert_eq!(hist.total, 100);
        assert_eq!(hist.bins[128], 100);
        assert_eq!(hist.bins[0], 0);
    }

    #[test]
    fn test_compute_histogram_gradient() {
        let img = make_gradient_image(256, 1);
        let hist = compute_histogram(&img);
        assert_eq!(hist.total, 256);
        // Gradient should spread across bins
        let non_zero = hist.bins.iter().filter(|&&c| c > 0).count();
        assert!(non_zero > 100);
    }

    #[test]
    fn test_histogram_mean_uniform() {
        let img = make_gray_image(10, 10, 100);
        let hist = compute_histogram(&img);
        assert!((hist.mean() - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_histogram_variance_uniform() {
        let img = make_gray_image(10, 10, 100);
        let hist = compute_histogram(&img);
        assert!(hist.variance() < 0.01);
    }

    #[test]
    fn test_histogram_min_max() {
        let img = make_gradient_image(256, 1);
        let hist = compute_histogram(&img);
        assert_eq!(hist.min_value(), Some(0));
        assert_eq!(hist.max_value(), Some(255));
    }

    #[test]
    fn test_histogram_cdf() {
        let img = make_gray_image(10, 10, 100);
        let hist = compute_histogram(&img);
        let cdf = hist.cdf();
        // CDF should be 0 before 100, 1.0 at and after 100
        assert!(cdf[99] < 0.01);
        assert!((cdf[100] - 1.0).abs() < 0.01);
        assert!((cdf[255] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_color_histogram() {
        let img = make_rgb_image(20, 20);
        let hist = compute_color_histogram(&img);
        assert_eq!(hist.red.total, 400);
        assert_eq!(hist.green.total, 400);
        assert_eq!(hist.blue.total, 400);
    }

    #[test]
    fn test_equalize_histogram() {
        let img = make_gradient_image(256, 10);
        let eq = equalize_histogram(&img).expect("equalization should succeed");
        assert_eq!(eq.width(), 256);
        assert_eq!(eq.height(), 10);
    }

    #[test]
    fn test_equalize_histogram_uniform_stays_similar() {
        let img = make_gray_image(10, 10, 128);
        let eq = equalize_histogram(&img).expect("equalization should succeed");
        let gray = eq.to_luma8();
        // All pixels should map to same value
        let val = gray.get_pixel(0, 0)[0];
        for pixel in gray.pixels() {
            assert_eq!(pixel[0], val);
        }
    }

    #[test]
    fn test_clahe_basic() {
        let img = make_gradient_image(64, 64);
        let result = clahe(&img, 8, 2.0).expect("CLAHE should succeed");
        assert_eq!(result.width(), 64);
        assert_eq!(result.height(), 64);
    }

    #[test]
    fn test_clahe_invalid_params() {
        let img = make_gray_image(10, 10, 100);
        assert!(clahe(&img, 0, 2.0).is_err());
        assert!(clahe(&img, 8, 0.5).is_err());
    }

    #[test]
    fn test_match_histogram() {
        // Source: dark image, Reference: bright histogram
        let source = make_gray_image(20, 20, 50);
        let mut ref_hist = Histogram::new();
        ref_hist.bins[200] = 100;
        ref_hist.total = 100;

        let matched = match_histogram(&source, &ref_hist).expect("matching should succeed");
        let gray = matched.to_luma8();
        // All pixels should be near 200
        let val = gray.get_pixel(0, 0)[0];
        assert!(val >= 180, "Expected bright pixels, got {val}");
    }

    #[test]
    fn test_match_histogram_image() {
        let source = make_gray_image(10, 10, 50);
        let reference = make_gray_image(10, 10, 200);
        let matched = match_histogram_image(&source, &reference).expect("matching should succeed");
        let gray = matched.to_luma8();
        let val = gray.get_pixel(0, 0)[0];
        assert!(val >= 180, "Expected bright pixels, got {val}");
    }

    #[test]
    fn test_compute_cdf() {
        let img = make_gray_image(10, 10, 100);
        let cdf = compute_cdf(&img);
        assert!((cdf[255] - 1.0).abs() < 1e-10);
        assert!(cdf[99] < 0.01);
    }

    #[test]
    fn test_otsu_bimodal() {
        let img = make_bimodal_image(100, 10);
        let t = otsu_threshold(&img);
        // Threshold should fall between the two peaks (inclusive bounds)
        assert!(
            t >= 50 && t <= 200,
            "Otsu threshold {t} not between 50 and 200"
        );
    }

    #[test]
    fn test_binarize_otsu() {
        let img = make_bimodal_image(100, 10);
        let (binary, threshold) = binarize_otsu(&img).expect("binarize should succeed");
        assert!(
            threshold >= 50 && threshold <= 200,
            "Threshold {threshold} not in [50, 200]"
        );
        let gray = binary.to_luma8();
        // Left half (value 50) should be <= threshold -> 0,
        // right half (value 200) should be > threshold -> 255
        assert_eq!(gray.get_pixel(10, 5)[0], 0);
        assert_eq!(gray.get_pixel(90, 5)[0], 255);
    }

    #[test]
    fn test_multi_otsu_single() {
        let img = make_bimodal_image(100, 10);
        let thresholds = multi_otsu_threshold(&img, 1).expect("single threshold should work");
        assert_eq!(thresholds.len(), 1);
        assert!(
            thresholds[0] >= 50 && thresholds[0] <= 200,
            "Threshold {} not in [50, 200]",
            thresholds[0]
        );
    }

    #[test]
    fn test_multi_otsu_two() {
        // Create a trimodal image
        let mut img = GrayImage::new(300, 10);
        for y in 0..10 {
            for x in 0..100 {
                img.put_pixel(x, y, Luma([30]));
            }
            for x in 100..200 {
                img.put_pixel(x, y, Luma([128]));
            }
            for x in 200..300 {
                img.put_pixel(x, y, Luma([220]));
            }
        }
        let dyn_img = DynamicImage::ImageLuma8(img);
        let thresholds = multi_otsu_threshold(&dyn_img, 2).expect("two thresholds should work");
        assert_eq!(thresholds.len(), 2);
        assert!(thresholds[0] < thresholds[1]);
        // First threshold between 30 and 128 (inclusive)
        assert!(
            thresholds[0] >= 30 && thresholds[0] <= 128,
            "First threshold {} not in [30, 128]",
            thresholds[0]
        );
        // Second threshold between 100 and 220 (inclusive)
        assert!(
            thresholds[1] >= 100 && thresholds[1] <= 220,
            "Second threshold {} not in [100, 220]",
            thresholds[1]
        );
    }

    #[test]
    fn test_multi_otsu_invalid() {
        let img = make_gray_image(10, 10, 100);
        assert!(multi_otsu_threshold(&img, 0).is_err());
        assert!(multi_otsu_threshold(&img, 5).is_err());
    }

    #[test]
    fn test_backproject_histogram() {
        let img = make_rgb_image(20, 20);
        let mut model = Histogram::new();
        model.bins[128] = 100;
        model.total = 100;

        let result = backproject_histogram(&img, &model, 2).expect("backproject should succeed");
        assert_eq!(result.width(), 20);
        assert_eq!(result.height(), 20);
        // Blue channel is constant at 128, so all should be bright
        let gray = result.to_luma8();
        for pixel in gray.pixels() {
            assert_eq!(pixel[0], 255);
        }
    }

    #[test]
    fn test_backproject_invalid_channel() {
        let img = make_gray_image(10, 10, 100);
        let model = Histogram::new();
        assert!(backproject_histogram(&img, &model, 5).is_err());
    }

    #[test]
    fn test_equalize_histogram_color() {
        let img = make_rgb_image(20, 20);
        let result = equalize_histogram_color(&img).expect("color equalization should succeed");
        assert_eq!(result.width(), 20);
        assert_eq!(result.height(), 20);
    }

    #[test]
    fn test_histogram_normalized_sums_to_one() {
        let img = make_gradient_image(128, 4);
        let hist = compute_histogram(&img);
        let norm = hist.normalized();
        let sum: f64 = norm.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Normalized histogram sum = {sum}"
        );
    }

    #[test]
    fn test_backproject_hs_histogram() {
        let img = make_rgb_image(20, 20);
        let hue_hist = Histogram {
            bins: [1u64; NUM_BINS],
            total: NUM_BINS as u64,
        };
        let sat_hist = Histogram {
            bins: [1u64; NUM_BINS],
            total: NUM_BINS as u64,
        };
        let result = backproject_hs_histogram(&img, &hue_hist, &sat_hist)
            .expect("HS backproject should succeed");
        assert_eq!(result.width(), 20);
        assert_eq!(result.height(), 20);
    }
}
