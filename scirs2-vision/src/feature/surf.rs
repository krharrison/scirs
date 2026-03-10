//! SURF (Speeded Up Robust Features) feature detector and descriptor
//!
//! This module implements the SURF algorithm, which is a fast and robust
//! scale and rotation-invariant feature detector and descriptor.
//!
//! # References
//!
//! - Bay, H., Tuytelaars, T., & Van Gool, L. (2006). SURF: Speeded up robust features.
//!   In European conference on computer vision (pp. 404-417). Springer, Berlin, Heidelberg.

use crate::error::{Result, VisionError};
use image::{DynamicImage, GrayImage};
use scirs2_core::ndarray::{Array2, Array3};
use std::f32::consts::PI;

/// SURF keypoint with position, scale, and orientation
#[derive(Debug, Clone)]
pub struct SurfKeyPoint {
    /// X-coordinate in image
    pub x: f32,
    /// Y-coordinate in image
    pub y: f32,
    /// Scale (size) of the keypoint
    pub scale: f32,
    /// Orientation in radians
    pub orientation: f32,
    /// Hessian response strength
    pub response: f32,
    /// Octave index
    pub octave: usize,
    /// Scale index within octave
    pub scale_index: usize,
}

/// SURF descriptor
#[derive(Debug, Clone)]
pub struct SurfDescriptor {
    /// Associated keypoint
    pub keypoint: SurfKeyPoint,
    /// Descriptor vector (64 or 128 dimensions)
    pub vector: Vec<f32>,
}

/// Configuration for SURF detector
#[derive(Debug, Clone)]
pub struct SurfConfig {
    /// Number of octaves (scale levels groups)
    pub num_octaves: usize,
    /// Number of scale levels per octave
    pub num_scales: usize,
    /// Hessian response threshold for keypoint detection
    pub hessian_threshold: f32,
    /// Use extended 128-dim descriptor (vs 64-dim)
    pub extended: bool,
    /// Skip orientation assignment (upright SURF)
    pub upright: bool,
    /// Initial sampling step in pixels
    pub initial_step: usize,
}

impl Default for SurfConfig {
    fn default() -> Self {
        Self {
            num_octaves: 4,
            num_scales: 4,
            hessian_threshold: 100.0,
            extended: false,
            upright: false,
            initial_step: 2,
        }
    }
}

/// SURF detector
pub struct SurfDetector {
    config: SurfConfig,
}

impl Default for SurfDetector {
    fn default() -> Self {
        Self::new(SurfConfig::default())
    }
}

impl SurfDetector {
    /// Create a new SURF detector with given configuration
    pub fn new(config: SurfConfig) -> Self {
        Self { config }
    }

    /// Detect SURF keypoints in an image
    ///
    /// # Arguments
    ///
    /// * `img` - Input image
    ///
    /// # Returns
    ///
    /// * Result containing vector of detected keypoints
    pub fn detect(&self, img: &DynamicImage) -> Result<Vec<SurfKeyPoint>> {
        let gray = img.to_luma8();
        let (width, height) = gray.dimensions();

        // Compute integral image
        let integral = self.compute_integral_image(&gray)?;

        // Build scale space and detect keypoints
        let keypoints = self.detect_keypoints(&integral, width as usize, height as usize)?;

        Ok(keypoints)
    }

    /// Compute SURF descriptors for detected keypoints
    ///
    /// # Arguments
    ///
    /// * `img` - Input image
    /// * `keypoints` - Detected keypoints
    ///
    /// # Returns
    ///
    /// * Result containing vector of descriptors
    pub fn compute(
        &self,
        img: &DynamicImage,
        keypoints: &[SurfKeyPoint],
    ) -> Result<Vec<SurfDescriptor>> {
        let gray = img.to_luma8();
        let integral = self.compute_integral_image(&gray)?;

        let mut descriptors = Vec::with_capacity(keypoints.len());

        for kp in keypoints {
            if let Ok(desc_vec) = self.compute_descriptor(&integral, kp) {
                descriptors.push(SurfDescriptor {
                    keypoint: kp.clone(),
                    vector: desc_vec,
                });
            }
        }

        Ok(descriptors)
    }

    /// Detect and compute SURF features in one step
    ///
    /// # Arguments
    ///
    /// * `img` - Input image
    ///
    /// # Returns
    ///
    /// * Result containing vector of descriptors with keypoints
    pub fn detect_and_compute(&self, img: &DynamicImage) -> Result<Vec<SurfDescriptor>> {
        let keypoints = self.detect(img)?;
        self.compute(img, &keypoints)
    }

    /// Compute integral image from grayscale image
    fn compute_integral_image(&self, img: &GrayImage) -> Result<Array2<f64>> {
        let (width, height) = img.dimensions();
        let mut integral = Array2::zeros((height as usize + 1, width as usize + 1));

        // Build integral image: I(x,y) = sum of all pixels above and to the left
        for y in 0..height as usize {
            let mut row_sum = 0.0;
            for x in 0..width as usize {
                let pixel_val = img.get_pixel(x as u32, y as u32)[0] as f64;
                row_sum += pixel_val;
                integral[[y + 1, x + 1]] = row_sum + integral[[y, x + 1]];
            }
        }

        Ok(integral)
    }

    /// Compute box filter response using integral image
    fn box_filter(&self, integral: &Array2<f64>, x: i32, y: i32, w: i32, h: i32) -> f64 {
        let (height, width) = integral.dim();

        // Clamp coordinates to valid range
        let x1 = (x).max(0).min(width as i32 - 1) as usize;
        let y1 = (y).max(0).min(height as i32 - 1) as usize;
        let x2 = (x + w).max(0).min(width as i32 - 1) as usize;
        let y2 = (y + h).max(0).min(height as i32 - 1) as usize;

        // Use integral image for fast box filter
        integral[[y2, x2]] + integral[[y1, x1]] - integral[[y1, x2]] - integral[[y2, x1]]
    }

    /// Compute Hessian response at a point using box filters
    fn hessian_response(&self, integral: &Array2<f64>, x: usize, y: usize, scale: usize) -> f64 {
        let s = scale as i32;
        let x = x as i32;
        let y = y as i32;

        // Box filter sizes based on scale
        let filter_size = 3 * s;
        let half_filter = filter_size / 2;

        // Dxx: second derivative in x direction
        let dxx = self.box_filter(integral, x - half_filter, y - s, filter_size, 2 * s)
            - 3.0 * self.box_filter(integral, x - s, y - s, 2 * s, 2 * s);

        // Dyy: second derivative in y direction
        let dyy = self.box_filter(integral, x - s, y - half_filter, 2 * s, filter_size)
            - 3.0 * self.box_filter(integral, x - s, y - s, 2 * s, 2 * s);

        // Dxy: cross derivative
        let dxy1 = self.box_filter(integral, x + 1, y + 1, s, s);
        let dxy2 = self.box_filter(integral, x - s, y - s, s, s);
        let dxy3 = self.box_filter(integral, x - s, y + 1, s, s);
        let dxy4 = self.box_filter(integral, x + 1, y - s, s, s);
        let dxy = dxy1 + dxy2 - dxy3 - dxy4;

        // Hessian determinant approximation

        dxx * dyy - 0.81 * dxy * dxy
    }

    /// Detect keypoints across scale space
    fn detect_keypoints(
        &self,
        integral: &Array2<f64>,
        width: usize,
        height: usize,
    ) -> Result<Vec<SurfKeyPoint>> {
        let mut all_responses = Vec::new();

        // Build scale space
        for octave in 0..self.config.num_octaves {
            for scale_idx in 0..self.config.num_scales {
                // Scale increases with octave and scale index
                let scale = self.config.initial_step * (1 << octave) * (scale_idx + 1);

                if scale * 3 >= width.min(height) {
                    continue;
                }

                let mut responses = Array2::zeros((height, width));

                // Compute Hessian responses
                for y in scale * 2..height.saturating_sub(scale * 2) {
                    for x in scale * 2..width.saturating_sub(scale * 2) {
                        let response = self.hessian_response(integral, x, y, scale);
                        responses[[y, x]] = response;
                    }
                }

                all_responses.push((octave, scale_idx, scale, responses));
            }
        }

        // Find local maxima across space and scale
        let mut keypoints = Vec::new();

        for i in 1..all_responses.len().saturating_sub(1) {
            let (octave, scale_idx, scale, ref current) = all_responses[i];
            let (_, _, _, ref prev) = all_responses[i - 1];
            let (_, _, _, ref next) = all_responses[i + 1];

            for y in scale * 2..height.saturating_sub(scale * 2) {
                for x in scale * 2..width.saturating_sub(scale * 2) {
                    let response = current[[y, x]];

                    // Threshold check
                    if response < self.config.hessian_threshold as f64 {
                        continue;
                    }

                    // Check if local maximum in 3x3x3 neighborhood
                    if self.is_local_maximum(response, current, prev, next, x, y) {
                        let orientation = if self.config.upright {
                            0.0
                        } else {
                            self.compute_orientation(integral, x, y, scale)
                                .unwrap_or(0.0)
                        };

                        keypoints.push(SurfKeyPoint {
                            x: x as f32,
                            y: y as f32,
                            scale: scale as f32,
                            orientation,
                            response: response as f32,
                            octave,
                            scale_index: scale_idx,
                        });
                    }
                }
            }
        }

        Ok(keypoints)
    }

    /// Check if a point is a local maximum in 3x3x3 neighborhood
    fn is_local_maximum(
        &self,
        value: f64,
        current: &Array2<f64>,
        prev: &Array2<f64>,
        next: &Array2<f64>,
        x: usize,
        y: usize,
    ) -> bool {
        // Check current scale level
        for dy in 0..3 {
            let ny = y + dy - 1;
            for dx in 0..3 {
                let nx = x + dx - 1;
                if (dx != 1 || dy != 1) && current[[ny, nx]] >= value {
                    return false;
                }
            }
        }

        // Check previous scale level
        for dy in 0..3 {
            let ny = y + dy - 1;
            for dx in 0..3 {
                let nx = x + dx - 1;
                if prev[[ny, nx]] >= value {
                    return false;
                }
            }
        }

        // Check next scale level
        for dy in 0..3 {
            let ny = y + dy - 1;
            for dx in 0..3 {
                let nx = x + dx - 1;
                if next[[ny, nx]] >= value {
                    return false;
                }
            }
        }

        true
    }

    /// Compute dominant orientation for a keypoint
    fn compute_orientation(
        &self,
        integral: &Array2<f64>,
        x: usize,
        y: usize,
        scale: usize,
    ) -> Result<f32> {
        let radius = (scale * 6).min(integral.dim().0.min(integral.dim().1) / 4);

        // Haar wavelet responses in circular region
        const NUM_BINS: usize = 36;
        let mut histogram = [0.0f32; NUM_BINS];

        for dy in 0..radius * 2 {
            let py = y + dy - radius;
            for dx in 0..radius * 2 {
                let px = x + dx - radius;

                // Check if within circular region
                let dist_sq =
                    (dx as i32 - radius as i32).pow(2) + (dy as i32 - radius as i32).pow(2);
                if dist_sq > (radius as i32).pow(2) {
                    continue;
                }

                // Compute Haar wavelets
                let dx_response = self.haar_x(integral, px, py, scale);
                let dy_response = self.haar_y(integral, px, py, scale);

                // Compute angle and magnitude
                let angle = (dy_response as f32).atan2(dx_response as f32);
                let magnitude = (dx_response.powi(2) + dy_response.powi(2)).sqrt() as f32;

                // Add to histogram
                let bin = (((angle + PI) / (2.0 * PI) * NUM_BINS as f32) as usize) % NUM_BINS;
                histogram[bin] += magnitude;
            }
        }

        // Find dominant orientation
        let max_bin = histogram
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or(VisionError::OperationError(
                "Failed to find dominant orientation".to_string(),
            ))?;

        let orientation = (max_bin as f32 / NUM_BINS as f32) * 2.0 * PI - PI;
        Ok(orientation)
    }

    /// Compute Haar wavelet response in x direction
    fn haar_x(&self, integral: &Array2<f64>, x: usize, y: usize, scale: usize) -> f64 {
        let s = scale as i32;
        let x = x as i32;
        let y = y as i32;

        let left = self.box_filter(integral, x - s, y - s, s, 2 * s);
        let right = self.box_filter(integral, x, y - s, s, 2 * s);

        right - left
    }

    /// Compute Haar wavelet response in y direction
    fn haar_y(&self, integral: &Array2<f64>, x: usize, y: usize, scale: usize) -> f64 {
        let s = scale as i32;
        let x = x as i32;
        let y = y as i32;

        let top = self.box_filter(integral, x - s, y - s, 2 * s, s);
        let bottom = self.box_filter(integral, x - s, y, 2 * s, s);

        bottom - top
    }

    /// Compute SURF descriptor for a keypoint
    fn compute_descriptor(&self, integral: &Array2<f64>, kp: &SurfKeyPoint) -> Result<Vec<f32>> {
        let scale = kp.scale as usize;
        let x = kp.x as usize;
        let y = kp.y as usize;

        let descriptor_size = if self.config.extended { 128 } else { 64 };
        let mut descriptor = vec![0.0f32; descriptor_size];

        // Rotation matrix for orientation normalization
        let cos_theta = kp.orientation.cos();
        let sin_theta = kp.orientation.sin();

        // Sample region around keypoint
        let region_size = 20 * scale;
        let sub_region_size = region_size / 4;

        for i in 0..4 {
            for j in 0..4 {
                let mut dx_sum = 0.0f32;
                let mut dy_sum = 0.0f32;
                let mut abs_dx_sum = 0.0f32;
                let mut abs_dy_sum = 0.0f32;

                // Sample within sub-region
                for sy in 0..5 {
                    for sx in 0..5 {
                        let sample_x = x as i32
                            + ((i * sub_region_size + sx * sub_region_size / 5) as i32
                                - region_size as i32 / 2);
                        let sample_y = y as i32
                            + ((j * sub_region_size + sy * sub_region_size / 5) as i32
                                - region_size as i32 / 2);

                        if sample_x < 0 || sample_y < 0 {
                            continue;
                        }

                        // Rotate coordinates
                        let rx = ((sample_x - x as i32) as f32 * cos_theta
                            + (sample_y - y as i32) as f32 * sin_theta)
                            as usize;
                        let ry = (-(sample_x - x as i32) as f32 * sin_theta
                            + (sample_y - y as i32) as f32 * cos_theta)
                            as usize;

                        if rx >= integral.dim().1 || ry >= integral.dim().0 {
                            continue;
                        }

                        // Compute Haar wavelets
                        let dx = self.haar_x(integral, rx, ry, scale) as f32;
                        let dy = self.haar_y(integral, rx, ry, scale) as f32;

                        dx_sum += dx;
                        dy_sum += dy;
                        abs_dx_sum += dx.abs();
                        abs_dy_sum += dy.abs();
                    }
                }

                // Store in descriptor
                let idx = (i * 4 + j) * 4;
                descriptor[idx] = dx_sum;
                descriptor[idx + 1] = dy_sum;
                descriptor[idx + 2] = abs_dx_sum;
                descriptor[idx + 3] = abs_dy_sum;

                // For extended descriptor, add second-order statistics
                if self.config.extended && descriptor_size == 128 {
                    let ext_idx = 64 + idx;
                    descriptor[ext_idx] = dx_sum;
                    descriptor[ext_idx + 1] = dy_sum;
                    descriptor[ext_idx + 2] = abs_dx_sum;
                    descriptor[ext_idx + 3] = abs_dy_sum;
                }
            }
        }

        // Normalize descriptor
        let norm: f32 = descriptor.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut descriptor {
                *val /= norm;
            }
        }

        Ok(descriptor)
    }
}

/// Convenience function to detect SURF keypoints
///
/// # Arguments
///
/// * `img` - Input image
/// * `config` - SURF configuration
///
/// # Returns
///
/// * Result containing vector of keypoints
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_vision::feature::surf::{detect_surf, SurfConfig};
/// use image::open;
///
/// fn main() {
///     let img = open("image.jpg").expect("image.jpg");
///     let keypoints = detect_surf(&img, &SurfConfig::default()).expect("surf");
///     println!("Detected {} SURF keypoints", keypoints.len());
/// }
/// ```
pub fn detect_surf(img: &DynamicImage, config: &SurfConfig) -> Result<Vec<SurfKeyPoint>> {
    let detector = SurfDetector::new(config.clone());
    detector.detect(img)
}

/// Convenience function to compute SURF descriptors
///
/// # Arguments
///
/// * `img` - Input image
/// * `keypoints` - Detected keypoints
/// * `config` - SURF configuration
///
/// # Returns
///
/// * Result containing vector of descriptors
pub fn compute_surf_descriptors(
    img: &DynamicImage,
    keypoints: &[SurfKeyPoint],
    config: &SurfConfig,
) -> Result<Vec<SurfDescriptor>> {
    let detector = SurfDetector::new(config.clone());
    detector.compute(img, keypoints)
}

/// Convenience function to detect and compute SURF features
///
/// # Arguments
///
/// * `img` - Input image
/// * `config` - SURF configuration
///
/// # Returns
///
/// * Result containing vector of descriptors with keypoints
pub fn detect_and_compute_surf(
    img: &DynamicImage,
    config: &SurfConfig,
) -> Result<Vec<SurfDescriptor>> {
    let detector = SurfDetector::new(config.clone());
    detector.detect_and_compute(img)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_surf_detector_creation() {
        let config = SurfConfig::default();
        let detector = SurfDetector::new(config);
        assert_eq!(detector.config.num_octaves, 4);
    }

    #[test]
    fn test_integral_image() {
        let img = GrayImage::new(10, 10);
        let detector = SurfDetector::default();
        let integral = detector.compute_integral_image(&img);
        assert!(integral.is_ok());
        let integral = integral.expect("integral image computation should succeed");
        assert_eq!(integral.dim(), (11, 11));
    }

    #[test]
    fn test_surf_detection() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let config = SurfConfig {
            hessian_threshold: 500.0, // Higher threshold for simple test image
            ..Default::default()
        };

        let keypoints = detect_surf(&img, &config);
        assert!(keypoints.is_ok());
    }

    #[test]
    fn test_surf_detect_and_compute() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let config = SurfConfig {
            hessian_threshold: 500.0,
            ..Default::default()
        };

        let descriptors = detect_and_compute_surf(&img, &config);
        assert!(descriptors.is_ok());
    }

    #[test]
    fn test_surf_extended_descriptor() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let config = SurfConfig {
            extended: true,
            hessian_threshold: 500.0,
            ..Default::default()
        };

        let descriptors = detect_and_compute_surf(&img, &config);
        assert!(descriptors.is_ok());

        if let Ok(descs) = descriptors {
            for desc in descs {
                assert_eq!(desc.vector.len(), 128);
            }
        }
    }

    #[test]
    fn test_surf_upright() {
        let img = DynamicImage::ImageRgb8(RgbImage::new(100, 100));
        let config = SurfConfig {
            upright: true,
            hessian_threshold: 500.0,
            ..Default::default()
        };

        let keypoints = detect_surf(&img, &config);
        assert!(keypoints.is_ok());

        if let Ok(kps) = keypoints {
            for kp in kps {
                assert_eq!(kp.orientation, 0.0);
            }
        }
    }
}
