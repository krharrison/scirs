//! Unified corner detection API
//!
//! Provides a single entry point for corner detection using multiple methods
//! (Harris, Shi-Tomasi, FAST). Each method has its own strengths:
//!
//! - **Harris**: Good for detecting corners with a well-defined mathematical
//!   response based on the structure tensor eigenvalues.
//! - **Shi-Tomasi**: An improvement over Harris that uses the minimum eigenvalue
//!   criterion, often yielding more robust corners.
//! - **FAST**: Very fast corner detection using a segment test on a circle of
//!   pixels around a candidate. Best for real-time applications.

use crate::error::{Result, VisionError};
use image::DynamicImage;

/// A detected corner point with position and score
#[derive(Debug, Clone, Copy)]
pub struct CornerPoint {
    /// X coordinate (column)
    pub x: f32,
    /// Y coordinate (row)
    pub y: f32,
    /// Corner response score (interpretation depends on method)
    pub score: f32,
}

/// Corner detection method selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CornerMethod {
    /// Harris corner detector (uses structure tensor determinant/trace ratio)
    Harris,
    /// Shi-Tomasi (Good Features to Track, uses minimum eigenvalue)
    ShiTomasi,
    /// FAST (Features from Accelerated Segment Test)
    Fast,
}

/// Parameters for unified corner detection
#[derive(Debug, Clone)]
pub struct CornerDetectParams {
    /// Corner detection method
    pub method: CornerMethod,
    /// Detection threshold (method-specific interpretation)
    /// - Harris: Harris response threshold (typical: 0.0001 to 0.01)
    /// - Shi-Tomasi: minimum eigenvalue threshold (typical: 0.01 to 0.1)
    /// - FAST: intensity difference threshold (typical: 10.0 to 50.0)
    pub threshold: f32,
    /// Maximum number of corners to return (0 for unlimited)
    pub max_corners: usize,
    /// Minimum distance between detected corners
    pub min_distance: usize,
    /// Block/window size for corner computation (Harris and Shi-Tomasi, must be odd >= 3)
    pub block_size: usize,
    /// Harris detector free parameter k (only for Harris, typical: 0.04-0.06)
    pub harris_k: f32,
    /// Number of consecutive pixels for FAST (typically 9 or 12)
    pub fast_n: usize,
    /// Apply non-maximum suppression
    pub non_max_suppression: bool,
}

impl Default for CornerDetectParams {
    fn default() -> Self {
        Self {
            method: CornerMethod::Harris,
            threshold: 0.01,
            max_corners: 500,
            min_distance: 10,
            block_size: 3,
            harris_k: 0.04,
            fast_n: 9,
            non_max_suppression: true,
        }
    }
}

/// Detect corners in an image using the specified method
///
/// This is the unified API for corner detection. It dispatches to the
/// appropriate detector based on the `params.method` field.
///
/// # Arguments
///
/// * `img` - Input image
/// * `params` - Corner detection parameters including method selection
///
/// # Returns
///
/// * Vector of detected corner points, sorted by descending score
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::corners::{detect_corners, CornerDetectParams, CornerMethod};
/// use image::{DynamicImage, GrayImage, Luma};
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// // Create a test image with some structure
/// let mut buf = GrayImage::new(32, 32);
/// for y in 0..32u32 {
///     for x in 0..32u32 {
///         let val = if x > 10 && x < 22 && y > 10 && y < 22 { 200u8 } else { 50u8 };
///         buf.put_pixel(x, y, Luma([val]));
///     }
/// }
/// let img = DynamicImage::ImageLuma8(buf);
///
/// let params = CornerDetectParams {
///     method: CornerMethod::Harris,
///     threshold: 0.0001,
///     ..CornerDetectParams::default()
/// };
/// let corners = detect_corners(&img, &params)?;
/// # Ok(())
/// # }
/// ```
pub fn detect_corners(img: &DynamicImage, params: &CornerDetectParams) -> Result<Vec<CornerPoint>> {
    match params.method {
        CornerMethod::Harris => detect_harris(img, params),
        CornerMethod::ShiTomasi => detect_shi_tomasi(img, params),
        CornerMethod::Fast => detect_fast(img, params),
    }
}

/// Harris corner detection returning CornerPoints
fn detect_harris(img: &DynamicImage, params: &CornerDetectParams) -> Result<Vec<CornerPoint>> {
    let array = crate::feature::image_to_array(img)?;
    let (height, width) = array.dim();

    // Ensure block_size is odd and >= 3
    let block_size = ensure_odd_block_size(params.block_size);
    let radius = block_size / 2;
    let k = params.harris_k;

    // Compute gradients
    let mut ix2 = scirs2_core::ndarray::Array2::<f32>::zeros((height, width));
    let mut iy2 = scirs2_core::ndarray::Array2::<f32>::zeros((height, width));
    let mut ixy = scirs2_core::ndarray::Array2::<f32>::zeros((height, width));

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let dx = (array[[y, x + 1]] - array[[y, x - 1]]) / 2.0;
            let dy = (array[[y + 1, x]] - array[[y - 1, x]]) / 2.0;
            ix2[[y, x]] = dx * dx;
            iy2[[y, x]] = dy * dy;
            ixy[[y, x]] = dx * dy;
        }
    }

    // Apply box filter and compute Harris response
    let mut response = scirs2_core::ndarray::Array2::<f32>::zeros((height, width));

    for y in radius..height.saturating_sub(radius) {
        for x in radius..width.saturating_sub(radius) {
            let mut sum_ix2 = 0.0f32;
            let mut sum_iy2 = 0.0f32;
            let mut sum_ixy = 0.0f32;

            for dy in y.saturating_sub(radius)..=(y + radius).min(height - 1) {
                for dx in x.saturating_sub(radius)..=(x + radius).min(width - 1) {
                    sum_ix2 += ix2[[dy, dx]];
                    sum_iy2 += iy2[[dy, dx]];
                    sum_ixy += ixy[[dy, dx]];
                }
            }

            let det = sum_ix2 * sum_iy2 - sum_ixy * sum_ixy;
            let trace = sum_ix2 + sum_iy2;
            response[[y, x]] = det - k * trace * trace;
        }
    }

    // Extract corners above threshold with NMS
    extract_corners_from_response(&response, params)
}

/// Shi-Tomasi corner detection using good_features_to_track
fn detect_shi_tomasi(img: &DynamicImage, params: &CornerDetectParams) -> Result<Vec<CornerPoint>> {
    let block_size = ensure_odd_block_size(params.block_size);

    let features = crate::feature::shi_tomasi::good_features_to_track(
        img,
        block_size,
        params.threshold,
        params.max_corners,
        params.min_distance,
    )?;

    let corners: Vec<CornerPoint> = features
        .iter()
        .map(|&(x, y, score)| CornerPoint { x, y, score })
        .collect();

    Ok(corners)
}

/// FAST corner detection returning CornerPoints
fn detect_fast(img: &DynamicImage, params: &CornerDetectParams) -> Result<Vec<CornerPoint>> {
    // FAST returns a GrayImage; we extract corner coordinates from it
    let corner_img = crate::feature::fast::fast_corners(
        img,
        params.threshold,
        params.fast_n,
        params.non_max_suppression,
    )?;

    let (width, height) = corner_img.dimensions();
    let mut corners = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let val = corner_img.get_pixel(x, y)[0];
            if val > 0 {
                corners.push(CornerPoint {
                    x: x as f32,
                    y: y as f32,
                    score: val as f32,
                });
            }
        }
    }

    // Sort by score descending
    corners.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply max_corners limit
    if params.max_corners > 0 && corners.len() > params.max_corners {
        corners.truncate(params.max_corners);
    }

    // Apply minimum distance filtering
    if params.min_distance > 0 {
        corners = filter_by_distance(&corners, params.min_distance as f32);
    }

    Ok(corners)
}

/// Extract corner points from a response map with non-maximum suppression
fn extract_corners_from_response(
    response: &scirs2_core::ndarray::Array2<f32>,
    params: &CornerDetectParams,
) -> Result<Vec<CornerPoint>> {
    let (height, width) = response.dim();
    let nms_radius = (params.block_size / 2).max(1);
    let mut corners = Vec::new();

    for y in nms_radius..height.saturating_sub(nms_radius) {
        for x in nms_radius..width.saturating_sub(nms_radius) {
            let r = response[[y, x]];
            if r <= params.threshold {
                continue;
            }

            // Non-maximum suppression in local neighborhood
            if params.non_max_suppression {
                let mut is_max = true;

                'nms: for dy in y.saturating_sub(nms_radius)..=(y + nms_radius).min(height - 1) {
                    for dx in x.saturating_sub(nms_radius)..=(x + nms_radius).min(width - 1) {
                        if dy == y && dx == x {
                            continue;
                        }
                        if response[[dy, dx]] >= r {
                            is_max = false;
                            break 'nms;
                        }
                    }
                }

                if !is_max {
                    continue;
                }
            }

            corners.push(CornerPoint {
                x: x as f32,
                y: y as f32,
                score: r,
            });
        }
    }

    // Sort by score descending
    corners.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply max corners limit
    if params.max_corners > 0 && corners.len() > params.max_corners {
        corners.truncate(params.max_corners);
    }

    // Apply minimum distance filtering
    if params.min_distance > 0 {
        corners = filter_by_distance(&corners, params.min_distance as f32);
    }

    Ok(corners)
}

/// Filter corners to maintain minimum distance between them
fn filter_by_distance(corners: &[CornerPoint], min_dist: f32) -> Vec<CornerPoint> {
    let min_dist_sq = min_dist * min_dist;
    let mut filtered = Vec::new();

    for corner in corners {
        let too_close = filtered.iter().any(|existing: &CornerPoint| {
            let dx = corner.x - existing.x;
            let dy = corner.y - existing.y;
            dx * dx + dy * dy < min_dist_sq
        });

        if !too_close {
            filtered.push(*corner);
        }
    }

    filtered
}

/// Ensure block_size is odd and >= 3
fn ensure_odd_block_size(block_size: usize) -> usize {
    let bs = block_size.max(3);
    if bs.is_multiple_of(2) {
        bs + 1
    } else {
        bs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn create_corner_image() -> DynamicImage {
        // Create an image with sharp corners (a bright square on dark background)
        let mut buf = GrayImage::new(40, 40);
        for y in 0..40u32 {
            for x in 0..40u32 {
                let val = if (10..30).contains(&x) && (10..30).contains(&y) {
                    200u8
                } else {
                    30u8
                };
                buf.put_pixel(x, y, Luma([val]));
            }
        }
        DynamicImage::ImageLuma8(buf)
    }

    #[test]
    fn test_detect_corners_harris() {
        let img = create_corner_image();
        let params = CornerDetectParams {
            method: CornerMethod::Harris,
            threshold: 0.00001,
            max_corners: 100,
            min_distance: 3,
            block_size: 3,
            harris_k: 0.04,
            non_max_suppression: true,
            ..CornerDetectParams::default()
        };

        let corners = detect_corners(&img, &params).expect("Harris detection failed");
        assert!(
            !corners.is_empty(),
            "Harris should detect corners of the square"
        );
    }

    #[test]
    fn test_detect_corners_shi_tomasi() {
        let img = create_corner_image();
        let params = CornerDetectParams {
            method: CornerMethod::ShiTomasi,
            threshold: 0.001,
            max_corners: 50,
            min_distance: 5,
            block_size: 3,
            ..CornerDetectParams::default()
        };

        let corners = detect_corners(&img, &params).expect("Shi-Tomasi detection failed");
        assert!(
            !corners.is_empty(),
            "Shi-Tomasi should detect corners of the square"
        );
    }

    #[test]
    fn test_detect_corners_fast() {
        let img = create_corner_image();
        let params = CornerDetectParams {
            method: CornerMethod::Fast,
            threshold: 20.0,
            max_corners: 50,
            min_distance: 5,
            fast_n: 9,
            non_max_suppression: true,
            ..CornerDetectParams::default()
        };

        let corners = detect_corners(&img, &params).expect("FAST detection failed");
        // Just verify it runs without error
        assert!(corners.len() <= 50, "Should respect max_corners limit");
    }

    #[test]
    fn test_detect_corners_no_corners_uniform() {
        // Uniform image should have no corners
        let img = DynamicImage::new_luma8(30, 30);
        let params = CornerDetectParams {
            method: CornerMethod::Harris,
            threshold: 0.01,
            ..CornerDetectParams::default()
        };

        let corners = detect_corners(&img, &params).expect("Harris on uniform failed");
        assert!(corners.is_empty(), "Uniform image should have no corners");
    }

    #[test]
    fn test_corner_point_scores_sorted() {
        let img = create_corner_image();
        let params = CornerDetectParams {
            method: CornerMethod::Harris,
            threshold: 0.000001,
            max_corners: 0,
            min_distance: 0,
            non_max_suppression: false,
            ..CornerDetectParams::default()
        };

        let corners = detect_corners(&img, &params).expect("Harris detection failed");
        // Verify sorted by descending score
        for window in corners.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "Corners should be sorted by descending score"
            );
        }
    }

    #[test]
    fn test_min_distance_filtering() {
        let corners = vec![
            CornerPoint {
                x: 0.0,
                y: 0.0,
                score: 1.0,
            },
            CornerPoint {
                x: 1.0,
                y: 0.0,
                score: 0.9,
            },
            CornerPoint {
                x: 10.0,
                y: 10.0,
                score: 0.8,
            },
            CornerPoint {
                x: 11.0,
                y: 10.0,
                score: 0.7,
            },
        ];

        let filtered = filter_by_distance(&corners, 5.0);
        assert_eq!(filtered.len(), 2, "Should keep 2 clusters");
        assert!((filtered[0].x - 0.0).abs() < 0.01);
        assert!((filtered[1].x - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_corner_detect_params_default() {
        let params = CornerDetectParams::default();
        assert_eq!(params.method, CornerMethod::Harris);
        assert!(params.threshold > 0.0);
        assert!(params.max_corners > 0);
        assert!(params.block_size >= 3);
    }

    #[test]
    fn test_ensure_odd_block_size() {
        assert_eq!(ensure_odd_block_size(3), 3);
        assert_eq!(ensure_odd_block_size(4), 5);
        assert_eq!(ensure_odd_block_size(5), 5);
        assert_eq!(ensure_odd_block_size(1), 3);
        assert_eq!(ensure_odd_block_size(2), 3);
    }
}
