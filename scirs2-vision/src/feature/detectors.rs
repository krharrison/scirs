//! Classical feature detectors for computer vision
//!
//! This module provides classical computer vision feature detectors:
//!
//! - **FAST**: Features from Accelerated Segment Test — very fast corner detector
//! - **Harris response**: Harris & Stephens corner response map
//! - **Shi-Tomasi**: Minimum eigenvalue corner score (Good Features to Track)
//! - **NMS**: Non-maximum suppression on generic response maps
//!
//! The detectors operate on raw `ndarray` arrays for tight integration with the
//! SciRS2 numeric stack, and expose a unified [`KeyPoint`] type that matches the
//! convention used by OpenCV.
//!
//! # References
//!
//! - Rosten & Drummond, 2006 — FAST corner detector
//! - Harris & Stephens, 1988 — Combined corner and edge detector
//! - Shi & Tomasi, 1994 — Good features to track

use crate::error::{Result, VisionError};
use image::GrayImage;
use scirs2_core::ndarray::Array2;

// ─── KeyPoint ───────────────────────────────────────────────────────────────

/// A detected keypoint with rich metadata, compatible with the OpenCV `KeyPoint` convention.
///
/// Fields:
/// - `x`, `y`       — sub-pixel position (column, row)
/// - `size`         — diameter of the meaningful keypoint neighbourhood (0 if not computed)
/// - `angle`        — dominant orientation in degrees [0, 360), −1 if not computed
/// - `response`     — detector response at this keypoint
/// - `octave`       — pyramid octave in which the keypoint was detected (−1 if N/A)
#[derive(Debug, Clone, PartialEq)]
pub struct KeyPoint {
    /// X coordinate (column), sub-pixel
    pub x: f32,
    /// Y coordinate (row), sub-pixel
    pub y: f32,
    /// Diameter of the meaningful neighbourhood in pixels (0 = unknown)
    pub size: f32,
    /// Dominant orientation in degrees, or -1 if not computed
    pub angle: f32,
    /// Detector response strength
    pub response: f32,
    /// Pyramid octave (-1 if not applicable)
    pub octave: i32,
}

impl KeyPoint {
    /// Create a minimal keypoint (size=0, angle=−1, octave=−1).
    pub fn new(x: f32, y: f32, response: f32) -> Self {
        Self {
            x,
            y,
            size: 0.0,
            angle: -1.0,
            response,
            octave: -1,
        }
    }
}

// ─── FAST detector ──────────────────────────────────────────────────────────

/// FAST corner detector — Features from Accelerated Segment Test.
///
/// For each candidate pixel *p* the algorithm tests a Bresenham circle of 16
/// pixels (radius 3).  A pixel is declared a corner when *N* or more
/// consecutive circle pixels are all brighter than `p + threshold` **or** all
/// darker than `p - threshold`.
///
/// # Typical usage
///
/// ```rust
/// use scirs2_vision::feature::detectors::FastDetector;
/// use image::{GrayImage, Luma};
///
/// let mut img = GrayImage::new(32, 32);
/// // paint a bright square so the corners become detectable
/// for y in 8u32..24 { for x in 8u32..24 {
///     img.put_pixel(x, y, Luma([220u8]));
/// }}
///
/// let det = FastDetector::new(20, 9, true);
/// let kps = det.detect(&img).unwrap();
/// // corners of the square should be detected
/// assert!(!kps.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct FastDetector {
    /// Intensity difference threshold
    pub threshold: u8,
    /// Number of consecutive bright/dark pixels required (typically 9, 10, or 12)
    pub n_consecutive: usize,
    /// Apply non-maximum suppression to the raw detections
    pub suppress_nonmax: bool,
}

impl FastDetector {
    /// Construct a new FAST detector.
    ///
    /// # Arguments
    ///
    /// * `threshold`       — intensity difference threshold (0–255)
    /// * `n_consecutive`   — minimum run length on the Bresenham circle (9 is standard)
    /// * `suppress_nonmax` — whether to apply non-maximum suppression on corner scores
    pub fn new(threshold: u8, n_consecutive: usize, suppress_nonmax: bool) -> Self {
        Self {
            threshold,
            n_consecutive,
            suppress_nonmax,
        }
    }

    /// Run the detector on a grayscale image.
    ///
    /// Returns a vector of [`KeyPoint`]s sorted by descending response.
    pub fn detect(&self, image: &GrayImage) -> Result<Vec<KeyPoint>> {
        let (width, height) = image.dimensions();
        let w = width as usize;
        let h = height as usize;

        if w < 7 || h < 7 {
            return Err(VisionError::InvalidParameter(
                "Image must be at least 7×7 for FAST detection".to_string(),
            ));
        }

        // Pixel intensities in a flat row-major buffer for fast access
        let pixels: Vec<u8> = image.pixels().map(|p| p[0]).collect();

        let pix = |row: usize, col: usize| -> u8 { pixels[row * w + col] };

        // Bresenham circle offsets (dx, dy) for a circle of radius 3
        // Indexed 0..15 counter-clockwise starting from top-centre
        const CIRCLE16: [(i32, i32); 16] = [
            (0, -3),
            (1, -3),
            (2, -2),
            (3, -1),
            (3, 0),
            (3, 1),
            (2, 2),
            (1, 3),
            (0, 3),
            (-1, 3),
            (-2, 2),
            (-3, 1),
            (-3, 0),
            (-3, -1),
            (-2, -2),
            (-1, -3),
        ];

        let n = self.n_consecutive;
        let t = self.threshold as i16;

        // Score array (row-major); 0 means "not a corner"
        let mut scores: Vec<f32> = vec![0.0; w * h];

        for row in 3..(h - 3) {
            for col in 3..(w - 3) {
                let center = pix(row, col) as i16;

                // Fast rejection: at least 2 of the 4 cardinal pixels must agree (2 allows detection of 90° corner pixels)
                let cardinals = [
                    CIRCLE16[0],  // top
                    CIRCLE16[4],  // right
                    CIRCLE16[8],  // bottom
                    CIRCLE16[12], // left
                ];
                let mut n_brighter = 0u8;
                let mut n_darker = 0u8;
                for (dx, dy) in cardinals {
                    let v = pix((row as i32 + dy) as usize, (col as i32 + dx) as usize) as i16;
                    if v > center + t {
                        n_brighter += 1;
                    } else if v < center - t {
                        n_darker += 1;
                    }
                }
                if n_brighter < 2 && n_darker < 2 {
                    continue;
                }

                // Full circle test — gather all 16 values
                let mut circle = [0i16; 16];
                for (i, (dx, dy)) in CIRCLE16.iter().enumerate() {
                    circle[i] = pix((row as i32 + dy) as usize, (col as i32 + dx) as usize) as i16;
                }

                if consecutive_run(&circle, center, t, n) {
                    let score = corner_score(&circle, center, t);
                    scores[row * w + col] = score;
                }
            }
        }

        // Optional non-maximum suppression (3×3 window)
        if self.suppress_nonmax {
            scores = nms_scores(&scores, w, h, 3);
        }

        // Harvest keypoints
        let mut keypoints: Vec<KeyPoint> = Vec::new();
        for row in 0..h {
            for col in 0..w {
                let s = scores[row * w + col];
                if s > 0.0 {
                    keypoints.push(KeyPoint {
                        x: col as f32,
                        y: row as f32,
                        size: 7.0, // FAST uses a circle of diameter ~7
                        angle: -1.0,
                        response: s,
                        octave: 0,
                    });
                }
            }
        }

        // Sort by descending response
        keypoints.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(keypoints)
    }
}

/// Check whether `circle[i]` has a run of >= `n` consecutive entries that are
/// all `> center + t` (brighter) **or** all `< center - t` (darker).
///
/// The double-loop trick handles the circular wrap-around without allocations.
fn consecutive_run(circle: &[i16; 16], center: i16, t: i16, n: usize) -> bool {
    let hi = center + t;
    let lo = center - t;

    // Check bright run
    let mut run = 0usize;
    for &v in circle.iter().chain(circle.iter()) {
        if v > hi {
            run += 1;
            if run >= n {
                return true;
            }
        } else {
            run = 0;
        }
    }

    // Check dark run
    run = 0;
    for &v in circle.iter().chain(circle.iter()) {
        if v < lo {
            run += 1;
            if run >= n {
                return true;
            }
        } else {
            run = 0;
        }
    }

    false
}

/// Compute a corner score = sum of absolute differences exceeding threshold.
fn corner_score(circle: &[i16; 16], center: i16, t: i16) -> f32 {
    let hi = center + t;
    let lo = center - t;
    let mut score = 0.0f32;
    for &v in circle {
        if v > hi {
            score += (v - hi) as f32;
        } else if v < lo {
            score += (lo - v) as f32;
        }
    }
    score
}

/// Apply non-maximum suppression over a flat score buffer.
/// Pixels that are not the maximum in their `win_size×win_size` neighbourhood
/// are set to zero.
fn nms_scores(scores: &[f32], w: usize, h: usize, win_size: usize) -> Vec<f32> {
    let r = win_size / 2;
    let mut result = vec![0.0f32; w * h];

    for row in r..(h.saturating_sub(r)) {
        for col in r..(w.saturating_sub(r)) {
            let s = scores[row * w + col];
            if s == 0.0 {
                continue;
            }
            let mut is_max = true;
            'outer: for dr in 0..=(2 * r) {
                for dc in 0..=(2 * r) {
                    let nr = row + dr - r;
                    let nc = col + dc - r;
                    if nr == row && nc == col {
                        continue;
                    }
                    if scores[nr * w + nc] > s {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max {
                result[row * w + col] = s;
            }
        }
    }

    result
}

// ─── Harris response map ─────────────────────────────────────────────────────

/// Compute the Harris corner response map.
///
/// The response at each pixel is `det(M) − k·trace(M)²`, where `M` is the
/// 2×2 structure tensor computed with a Gaussian window of standard deviation
/// `sigma`.
///
/// # Arguments
///
/// * `image` — input grayscale image (pixel values 0–255)
/// * `k`     — Harris free parameter (typically 0.04–0.06)
/// * `sigma` — Gaussian window standard deviation for structure tensor smoothing
///
/// # Returns
///
/// A 2-D array of Harris response values (same dimensions as the input).  
/// Positive values indicate corner-like locations.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::detectors::harris_response;
/// use image::{GrayImage, Luma};
///
/// let mut img = GrayImage::new(32, 32);
/// for y in 8u32..24 { for x in 8u32..24 {
///     img.put_pixel(x, y, Luma([200u8]));
/// }}
/// let resp = harris_response(&img, 0.04, 1.0).unwrap();
/// assert_eq!(resp.dim(), (32, 32));
/// ```
pub fn harris_response(image: &GrayImage, k: f64, sigma: f64) -> Result<Array2<f64>> {
    if sigma <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "sigma must be positive".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&k) {
        return Err(VisionError::InvalidParameter(
            "k must be in [0, 1)".to_string(),
        ));
    }

    let (width, height) = image.dimensions();
    let w = width as usize;
    let h = height as usize;

    // Convert to f64 array
    let mut arr = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            arr[[y, x]] = image.get_pixel(x as u32, y as u32)[0] as f64 / 255.0;
        }
    }

    // Compute image gradients via central differences (clamped at border)
    let mut ix = Array2::<f64>::zeros((h, w));
    let mut iy = Array2::<f64>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            let xl = if x > 0 { x - 1 } else { 0 };
            let xr = if x + 1 < w { x + 1 } else { w - 1 };
            let yt = if y > 0 { y - 1 } else { 0 };
            let yb = if y + 1 < h { y + 1 } else { h - 1 };
            ix[[y, x]] = (arr[[y, xr]] - arr[[y, xl]]) / 2.0;
            iy[[y, x]] = (arr[[yb, x]] - arr[[yt, x]]) / 2.0;
        }
    }

    // Structure tensor products
    let mut ix2 = Array2::<f64>::zeros((h, w));
    let mut iy2 = Array2::<f64>::zeros((h, w));
    let mut ixy = Array2::<f64>::zeros((h, w));

    for y in 0..h {
        for x in 0..w {
            ix2[[y, x]] = ix[[y, x]] * ix[[y, x]];
            iy2[[y, x]] = iy[[y, x]] * iy[[y, x]];
            ixy[[y, x]] = ix[[y, x]] * iy[[y, x]];
        }
    }

    // Gaussian smoothing of structure tensor
    let ix2s = gaussian_smooth_2d(&ix2, sigma)?;
    let iy2s = gaussian_smooth_2d(&iy2, sigma)?;
    let ixys = gaussian_smooth_2d(&ixy, sigma)?;

    // Compute Harris response
    let mut response = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let a = ix2s[[y, x]];
            let b = iy2s[[y, x]];
            let c = ixys[[y, x]];
            let det = a * b - c * c;
            let trace = a + b;
            response[[y, x]] = det - k * trace * trace;
        }
    }

    Ok(response)
}

// ─── Shi-Tomasi score ────────────────────────────────────────────────────────

/// Compute the Shi-Tomasi (minimum eigenvalue) corner score map.
///
/// The score at each pixel is the smaller eigenvalue of the 2×2 structure
/// tensor smoothed with a Gaussian window of standard deviation `sigma`.
/// This is the criterion used by Shi & Tomasi's "Good Features to Track".
///
/// # Arguments
///
/// * `image` — input grayscale image
/// * `sigma` — Gaussian window standard deviation
///
/// # Returns
///
/// A 2-D array of minimum eigenvalue scores (same dimensions as the input).
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::detectors::shi_tomasi_score;
/// use image::{GrayImage, Luma};
///
/// let mut img = GrayImage::new(32, 32);
/// for y in 8u32..24 { for x in 8u32..24 {
///     img.put_pixel(x, y, Luma([200u8]));
/// }}
/// let scores = shi_tomasi_score(&img, 1.0).unwrap();
/// assert_eq!(scores.dim(), (32, 32));
/// ```
pub fn shi_tomasi_score(image: &GrayImage, sigma: f64) -> Result<Array2<f64>> {
    if sigma <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "sigma must be positive".to_string(),
        ));
    }

    let (width, height) = image.dimensions();
    let w = width as usize;
    let h = height as usize;

    // Convert to f64
    let mut arr = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            arr[[y, x]] = image.get_pixel(x as u32, y as u32)[0] as f64 / 255.0;
        }
    }

    // Gradients
    let mut ix = Array2::<f64>::zeros((h, w));
    let mut iy = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let xl = if x > 0 { x - 1 } else { 0 };
            let xr = if x + 1 < w { x + 1 } else { w - 1 };
            let yt = if y > 0 { y - 1 } else { 0 };
            let yb = if y + 1 < h { y + 1 } else { h - 1 };
            ix[[y, x]] = (arr[[y, xr]] - arr[[y, xl]]) / 2.0;
            iy[[y, x]] = (arr[[yb, x]] - arr[[yt, x]]) / 2.0;
        }
    }

    // Structure tensor
    let mut ix2 = Array2::<f64>::zeros((h, w));
    let mut iy2 = Array2::<f64>::zeros((h, w));
    let mut ixy = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            ix2[[y, x]] = ix[[y, x]] * ix[[y, x]];
            iy2[[y, x]] = iy[[y, x]] * iy[[y, x]];
            ixy[[y, x]] = ix[[y, x]] * iy[[y, x]];
        }
    }

    let ix2s = gaussian_smooth_2d(&ix2, sigma)?;
    let iy2s = gaussian_smooth_2d(&iy2, sigma)?;
    let ixys = gaussian_smooth_2d(&ixy, sigma)?;

    // Minimum eigenvalue = (a+b)/2 - sqrt(((a-b)/2)^2 + c^2)
    let mut scores = Array2::<f64>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let a = ix2s[[y, x]];
            let b = iy2s[[y, x]];
            let c = ixys[[y, x]];
            let half_trace = (a + b) / 2.0;
            let half_diff = (a - b) / 2.0;
            let disc = (half_diff * half_diff + c * c).sqrt();
            let min_eig = half_trace - disc;
            scores[[y, x]] = min_eig.max(0.0);
        }
    }

    Ok(scores)
}

// ─── Non-maximum suppression ─────────────────────────────────────────────────

/// Apply non-maximum suppression to a response map and return peak locations.
///
/// A pixel at `(row, col)` is returned if:
/// 1. `response[[row, col]] >= threshold`, **and**
/// 2. `response[[row, col]]` is strictly greater than all neighbours within a
///    circular window of the given `radius`.
///
/// # Arguments
///
/// * `response`  — 2-D response map
/// * `radius`    — suppression radius in pixels
/// * `threshold` — minimum response value (anything below is ignored)
///
/// # Returns
///
/// Vector of `(row, col)` pairs sorted by descending response value.
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::detectors::{harris_response, nms_response};
/// use image::{GrayImage, Luma};
///
/// let mut img = GrayImage::new(32, 32);
/// for y in 8u32..24 { for x in 8u32..24 {
///     img.put_pixel(x, y, Luma([200u8]));
/// }}
/// let resp = harris_response(&img, 0.04, 1.0).unwrap();
/// let peaks = nms_response(&resp, 3, 1e-4);
/// assert!(!peaks.is_empty());
/// ```
pub fn nms_response(response: &Array2<f64>, radius: usize, threshold: f64) -> Vec<(usize, usize)> {
    let (h, w) = response.dim();
    let r = radius as isize;
    let mut peaks: Vec<(usize, usize, f64)> = Vec::new();

    for row in 0..h {
        for col in 0..w {
            let val = response[[row, col]];
            if val < threshold {
                continue;
            }

            let mut is_max = true;
            'check: for dr in -r..=r {
                for dc in -r..=r {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = row as isize + dr;
                    let nc = col as isize + dc;
                    if nr < 0 || nr >= h as isize || nc < 0 || nc >= w as isize {
                        continue;
                    }
                    if response[[nr as usize, nc as usize]] > val {
                        is_max = false;
                        break 'check;
                    }
                }
            }

            if is_max {
                peaks.push((row, col, val));
            }
        }
    }

    // Sort by descending response
    peaks.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    peaks.into_iter().map(|(r, c, _)| (r, c)).collect()
}

// ─── Gaussian helper ─────────────────────────────────────────────────────────

/// Separable Gaussian smoothing of a 2-D array.
///
/// Uses reflected boundary conditions to avoid border artefacts.
fn gaussian_smooth_2d(arr: &Array2<f64>, sigma: f64) -> Result<Array2<f64>> {
    let kernel = gaussian_kernel_1d(sigma);
    let (h, w) = arr.dim();

    // Horizontal pass
    let mut tmp = Array2::<f64>::zeros((h, w));
    let r = kernel.len() / 2;
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let c = col as isize + ki as isize - r as isize;
                // Reflect at border
                let c = c.clamp(0, w as isize - 1) as usize;
                acc += arr[[row, c]] * kv;
            }
            tmp[[row, col]] = acc;
        }
    }

    // Vertical pass
    let mut out = Array2::<f64>::zeros((h, w));
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let rr = row as isize + ki as isize - r as isize;
                let rr = rr.clamp(0, h as isize - 1) as usize;
                acc += tmp[[rr, col]] * kv;
            }
            out[[row, col]] = acc;
        }
    }

    Ok(out)
}

/// Build a normalised 1-D Gaussian kernel with half-width = ceil(3·sigma).
fn gaussian_kernel_1d(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;
    let mut kernel = vec![0.0f64; size];
    let two_sigma2 = 2.0 * sigma * sigma;
    let mut sum = 0.0;

    for (i, val) in kernel.iter_mut().enumerate() {
        let d = i as f64 - radius as f64;
        let v = (-d * d / two_sigma2).exp();
        *val = v;
        sum += v;
    }

    // Normalise
    for v in &mut kernel {
        *v /= sum;
    }

    kernel
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    /// Synthesise a 64×64 image with a bright white square (corners are
    /// canonical test targets for any corner detector).
    fn bright_square_image(side: u32) -> GrayImage {
        let size = 64u32;
        let start = (size - side) / 2;
        let end = start + side;
        let mut img = GrayImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let val: u8 = if x >= start && x < end && y >= start && y < end {
                    220
                } else {
                    20
                };
                img.put_pixel(x, y, Luma([val]));
            }
        }
        img
    }

    // ── FastDetector ──────────────────────────────────────────────────────

    #[test]
    fn fast_detects_corners_of_square() {
        let img = bright_square_image(32);
        let det = FastDetector::new(30, 9, true);
        let kps = det.detect(&img).expect("FAST should succeed");
        assert!(
            !kps.is_empty(),
            "FAST should detect corners on a bright square"
        );
        // All keypoints must be in-bounds
        let (w, h) = img.dimensions();
        for kp in &kps {
            assert!(kp.x >= 0.0 && (kp.x as u32) < w);
            assert!(kp.y >= 0.0 && (kp.y as u32) < h);
        }
    }

    #[test]
    fn fast_sorted_by_descending_response() {
        let img = bright_square_image(24);
        let det = FastDetector::new(20, 9, false);
        let kps = det.detect(&img).expect("FAST should succeed");
        for win in kps.windows(2) {
            assert!(
                win[0].response >= win[1].response,
                "Keypoints must be sorted by descending response"
            );
        }
    }

    #[test]
    fn fast_uniform_image_no_corners() {
        let img = GrayImage::from_pixel(64, 64, Luma([128u8]));
        let det = FastDetector::new(10, 9, true);
        let kps = det
            .detect(&img)
            .expect("FAST on uniform image should succeed");
        assert!(kps.is_empty(), "Uniform image should have no corners");
    }

    #[test]
    fn fast_n12_is_more_selective_than_n9() {
        let img = bright_square_image(32);
        let det9 = FastDetector::new(20, 9, false);
        let det12 = FastDetector::new(20, 12, false);
        let kps9 = det9.detect(&img).expect("FAST-9 should succeed");
        let kps12 = det12.detect(&img).expect("FAST-12 should succeed");
        assert!(
            kps9.len() >= kps12.len(),
            "FAST-12 should not detect more corners than FAST-9"
        );
    }

    #[test]
    fn fast_too_small_image_returns_error() {
        let img = GrayImage::new(4, 4);
        let det = FastDetector::new(20, 9, true);
        assert!(det.detect(&img).is_err(), "Tiny image should return error");
    }

    // ── KeyPoint ─────────────────────────────────────────────────────────

    #[test]
    fn keypoint_new_defaults() {
        let kp = KeyPoint::new(10.0, 20.0, 0.5);
        assert_eq!(kp.x, 10.0);
        assert_eq!(kp.y, 20.0);
        assert_eq!(kp.size, 0.0);
        assert_eq!(kp.angle, -1.0);
        assert_eq!(kp.octave, -1);
        assert_eq!(kp.response, 0.5);
    }

    // ── Harris response ───────────────────────────────────────────────────

    #[test]
    fn harris_response_correct_shape() {
        let img = bright_square_image(32);
        let resp = harris_response(&img, 0.04, 1.5).expect("Harris should succeed");
        assert_eq!(resp.dim(), (64, 64));
    }

    #[test]
    fn harris_response_corner_areas_have_positive_values() {
        let img = bright_square_image(32);
        let resp = harris_response(&img, 0.04, 1.5).expect("Harris should succeed");
        let max_resp = resp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_resp > 0.0,
            "Harris response should be positive near corners"
        );
    }

    #[test]
    fn harris_response_uniform_image_near_zero() {
        let img = GrayImage::from_pixel(32, 32, Luma([128u8]));
        let resp = harris_response(&img, 0.04, 1.0).expect("Harris on uniform should succeed");
        let max_resp = resp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_resp < 1e-8,
            "Uniform image should produce near-zero Harris response"
        );
    }

    #[test]
    fn harris_response_invalid_sigma_returns_error() {
        let img = GrayImage::new(16, 16);
        assert!(harris_response(&img, 0.04, 0.0).is_err());
        assert!(harris_response(&img, 0.04, -1.0).is_err());
    }

    #[test]
    fn harris_response_invalid_k_returns_error() {
        let img = GrayImage::new(16, 16);
        assert!(harris_response(&img, -0.1, 1.0).is_err());
        assert!(harris_response(&img, 1.5, 1.0).is_err());
    }

    // ── Shi-Tomasi score ─────────────────────────────────────────────────

    #[test]
    fn shi_tomasi_correct_shape() {
        let img = bright_square_image(32);
        let scores = shi_tomasi_score(&img, 1.5).expect("Shi-Tomasi should succeed");
        assert_eq!(scores.dim(), (64, 64));
    }

    #[test]
    fn shi_tomasi_nonnegative() {
        let img = bright_square_image(32);
        let scores = shi_tomasi_score(&img, 1.5).expect("Shi-Tomasi should succeed");
        for &v in scores.iter() {
            assert!(v >= 0.0, "Minimum eigenvalue must be >= 0");
        }
    }

    #[test]
    fn shi_tomasi_uniform_near_zero() {
        let img = GrayImage::from_pixel(32, 32, Luma([100u8]));
        let scores = shi_tomasi_score(&img, 1.0).expect("Shi-Tomasi on uniform should succeed");
        let max_score = scores.iter().copied().fold(0.0f64, f64::max);
        assert!(
            max_score < 1e-8,
            "Uniform image: Shi-Tomasi scores should be near zero"
        );
    }

    #[test]
    fn shi_tomasi_invalid_sigma_returns_error() {
        let img = GrayImage::new(16, 16);
        assert!(shi_tomasi_score(&img, 0.0).is_err());
        assert!(shi_tomasi_score(&img, -0.5).is_err());
    }

    // ── NMS response ─────────────────────────────────────────────────────

    #[test]
    fn nms_response_finds_peaks() {
        let img = bright_square_image(32);
        let resp = harris_response(&img, 0.04, 1.5).expect("Harris should succeed");
        let peaks = nms_response(&resp, 3, 1e-5);
        assert!(!peaks.is_empty(), "NMS should find peaks in a Harris map");
    }

    #[test]
    fn nms_response_minimum_distance_honoured() {
        let img = bright_square_image(32);
        let resp = harris_response(&img, 0.04, 1.5).expect("Harris should succeed");
        let radius = 5;
        let peaks = nms_response(&resp, radius, 1e-6);
        // No two peaks should be within `radius` pixels of each other
        for (i, &(r1, c1)) in peaks.iter().enumerate() {
            for &(r2, c2) in &peaks[i + 1..] {
                let dr = r1 as isize - r2 as isize;
                let dc = c1 as isize - c2 as isize;
                let dist_sq = dr * dr + dc * dc;
                assert!(
                    dist_sq > (radius as isize * radius as isize),
                    "Peaks at ({r1},{c1}) and ({r2},{c2}) are too close (radius={radius})"
                );
            }
        }
    }

    #[test]
    fn nms_response_high_threshold_filters_all() {
        let img = bright_square_image(32);
        let resp = harris_response(&img, 0.04, 1.5).expect("Harris should succeed");
        let peaks = nms_response(&resp, 3, 1e10); // impossibly high
        assert!(
            peaks.is_empty(),
            "Impossibly high threshold should filter all peaks"
        );
    }

    #[test]
    fn nms_response_all_in_bounds() {
        let img = bright_square_image(32);
        let resp = harris_response(&img, 0.04, 1.5).expect("Harris should succeed");
        let (h, w) = resp.dim();
        let peaks = nms_response(&resp, 3, 1e-6);
        for (r, c) in peaks {
            assert!(r < h && c < w, "Peak ({r},{c}) is out of bounds ({h}×{w})");
        }
    }
}
