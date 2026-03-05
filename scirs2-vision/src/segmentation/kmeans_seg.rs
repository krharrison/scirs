//! K-means color segmentation
//!
//! Segments an image by clustering pixel colors into K groups using the
//! K-means algorithm. Each pixel is assigned a label corresponding to
//! its nearest cluster center.
//!
//! # References
//!
//! - MacQueen, J., 1967. Some methods for classification and analysis of
//!   multivariate observations. In Proceedings of the fifth Berkeley symposium
//!   on mathematical statistics and probability (Vol. 1, No. 14, pp. 281-297).

use crate::error::{Result, VisionError};
use image::{DynamicImage, GenericImageView, GrayImage, Luma};
use scirs2_core::ndarray::Array2;

/// Parameters for K-means color segmentation
#[derive(Debug, Clone)]
pub struct KMeansSegParams {
    /// Number of clusters (K)
    pub k: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence threshold (change in centroids)
    pub epsilon: f32,
    /// Number of random restarts (best result kept)
    pub n_init: usize,
    /// Whether to use RGB (3-channel) or grayscale (1-channel)
    pub use_color: bool,
}

impl Default for KMeansSegParams {
    fn default() -> Self {
        Self {
            k: 3,
            max_iterations: 100,
            epsilon: 1e-4,
            n_init: 3,
            use_color: true,
        }
    }
}

/// Result of K-means segmentation
#[derive(Debug, Clone)]
pub struct KMeansSegResult {
    /// Label map (height x width), each value in 0..k
    pub labels: Array2<u32>,
    /// Cluster centers (k x channels)
    pub centers: Vec<Vec<f32>>,
    /// Total inertia (sum of squared distances to assigned centers)
    pub inertia: f64,
    /// Number of iterations performed
    pub iterations: usize,
}

/// Segment an image using K-means clustering on pixel colors
///
/// # Arguments
///
/// * `img` - Input image
/// * `params` - K-means segmentation parameters
///
/// # Returns
///
/// * Result containing the segmentation result with labels and cluster centers
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_vision::segmentation::kmeans_seg::{kmeans_segment, KMeansSegParams};
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("test.jpg").expect("Operation failed");
/// let result = kmeans_segment(&img, &KMeansSegParams::default())?;
/// println!("Segmented into {} clusters", result.centers.len());
/// # Ok(())
/// # }
/// ```
pub fn kmeans_segment(img: &DynamicImage, params: &KMeansSegParams) -> Result<KMeansSegResult> {
    if params.k < 2 {
        return Err(VisionError::InvalidParameter(
            "K must be at least 2 for K-means segmentation".to_string(),
        ));
    }

    let (width, height) = img.dimensions();
    let n_pixels = (width * height) as usize;

    if n_pixels < params.k {
        return Err(VisionError::InvalidParameter(
            "Image has fewer pixels than requested clusters".to_string(),
        ));
    }

    // Extract pixel features
    let channels = if params.use_color { 3 } else { 1 };
    let pixels = extract_pixel_features(img, params.use_color);

    // Run K-means with multiple restarts
    let mut best_result: Option<KMeansSegResult> = None;

    for restart in 0..params.n_init {
        let result = run_kmeans_once(
            &pixels,
            width as usize,
            height as usize,
            channels,
            params,
            restart,
        )?;

        let is_better = match &best_result {
            None => true,
            Some(prev) => result.inertia < prev.inertia,
        };

        if is_better {
            best_result = Some(result);
        }
    }

    best_result.ok_or_else(|| {
        VisionError::OperationError("K-means failed to produce a result".to_string())
    })
}

/// Extract pixel features from an image
fn extract_pixel_features(img: &DynamicImage, use_color: bool) -> Vec<Vec<f32>> {
    let (width, height) = img.dimensions();
    let mut pixels = Vec::with_capacity((width * height) as usize);

    if use_color {
        let rgb = img.to_rgb8();
        for y in 0..height {
            for x in 0..width {
                let p = rgb.get_pixel(x, y);
                pixels.push(vec![p[0] as f32, p[1] as f32, p[2] as f32]);
            }
        }
    } else {
        let gray = img.to_luma8();
        for y in 0..height {
            for x in 0..width {
                pixels.push(vec![gray.get_pixel(x, y)[0] as f32]);
            }
        }
    }

    pixels
}

/// Run a single K-means iteration
fn run_kmeans_once(
    pixels: &[Vec<f32>],
    width: usize,
    height: usize,
    channels: usize,
    params: &KMeansSegParams,
    seed: usize,
) -> Result<KMeansSegResult> {
    let n_pixels = pixels.len();
    let k = params.k;

    // Initialize centers using K-means++ style initialization
    let mut centers = initialize_centers(pixels, k, channels, seed);
    let mut labels = vec![0u32; n_pixels];
    let mut iterations = 0;

    for _iter in 0..params.max_iterations {
        iterations = _iter + 1;

        // Assignment step: assign each pixel to nearest center
        for (i, pixel) in pixels.iter().enumerate() {
            let mut min_dist = f32::MAX;
            let mut best_label = 0u32;

            for (c, center) in centers.iter().enumerate() {
                let dist = squared_distance(pixel, center);
                if dist < min_dist {
                    min_dist = dist;
                    best_label = c as u32;
                }
            }

            labels[i] = best_label;
        }

        // Update step: recompute centers
        let mut new_centers = vec![vec![0.0f64; channels]; k];
        let mut counts = vec![0usize; k];

        for (i, pixel) in pixels.iter().enumerate() {
            let label = labels[i] as usize;
            counts[label] += 1;
            for c in 0..channels {
                new_centers[label][c] += pixel[c] as f64;
            }
        }

        let mut max_shift: f32 = 0.0;
        for (c_idx, new_center) in new_centers.iter_mut().enumerate() {
            if counts[c_idx] > 0 {
                #[allow(clippy::needless_range_loop)]
                for ch in 0..channels {
                    new_center[ch] /= counts[c_idx] as f64;
                }
            }

            // Compute shift
            #[allow(clippy::needless_range_loop)]
            for ch in 0..channels {
                let diff = new_center[ch] as f32 - centers[c_idx][ch];
                if diff.abs() > max_shift {
                    max_shift = diff.abs();
                }
            }

            // Update center
            for ch in 0..channels {
                centers[c_idx][ch] = new_center[ch] as f32;
            }
        }

        if max_shift < params.epsilon {
            break;
        }
    }

    // Compute inertia
    let mut inertia = 0.0f64;
    for (i, pixel) in pixels.iter().enumerate() {
        let label = labels[i] as usize;
        inertia += squared_distance(pixel, &centers[label]) as f64;
    }

    // Reshape labels into 2D array
    let mut label_map = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            label_map[[y, x]] = labels[y * width + x];
        }
    }

    Ok(KMeansSegResult {
        labels: label_map,
        centers,
        inertia,
        iterations,
    })
}

/// Initialize cluster centers using K-means++ strategy
fn initialize_centers(
    pixels: &[Vec<f32>],
    k: usize,
    channels: usize,
    seed: usize,
) -> Vec<Vec<f32>> {
    let n = pixels.len();
    let mut centers = Vec::with_capacity(k);

    // Use a simple deterministic seeding based on the restart index
    // First center: pick based on seed
    let first_idx = (seed * 7919 + 1) % n; // prime-based hashing
    centers.push(pixels[first_idx].clone());

    // Remaining centers: K-means++ probabilistic selection (deterministic variant)
    for _ in 1..k {
        // Compute distance to nearest existing center for each pixel
        let mut distances = vec![f32::MAX; n];
        let mut total_dist = 0.0f64;

        for (i, pixel) in pixels.iter().enumerate() {
            for center in &centers {
                let d = squared_distance(pixel, center);
                if d < distances[i] {
                    distances[i] = d;
                }
            }
            total_dist += distances[i] as f64;
        }

        // Pick the point that is farthest from all existing centers (deterministic)
        // This is a simplified D^2 weighting: we pick based on cumulative distribution
        if total_dist > 0.0 {
            // Use a deterministic threshold based on centers.len()
            let threshold = total_dist * ((centers.len() as f64 * 0.618033988) % 1.0);
            let mut cumulative = 0.0f64;
            let mut chosen = 0;

            for (i, &d) in distances.iter().enumerate() {
                cumulative += d as f64;
                if cumulative >= threshold {
                    chosen = i;
                    break;
                }
            }

            centers.push(pixels[chosen].clone());
        } else {
            // All distances are 0, just pick uniformly
            let idx = (centers.len() * 6271) % n;
            centers.push(pixels[idx].clone());
        }
    }

    centers
}

/// Compute squared Euclidean distance between two vectors
#[inline]
fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Convert K-means labels to a colored image for visualization
///
/// Each cluster is assigned a distinct color.
///
/// # Arguments
///
/// * `labels` - Label map from K-means segmentation
/// * `k` - Number of clusters
///
/// # Returns
///
/// * RGB image with each segment colored differently
pub fn kmeans_labels_to_color(labels: &Array2<u32>, k: usize) -> image::RgbImage {
    let (height, width) = labels.dim();
    let mut result = image::RgbImage::new(width as u32, height as u32);

    // Generate distinct colors for each cluster
    let colors = generate_distinct_colors(k);

    for y in 0..height {
        for x in 0..width {
            let label = labels[[y, x]] as usize;
            let color = if label < colors.len() {
                colors[label]
            } else {
                [128, 128, 128]
            };
            result.put_pixel(x as u32, y as u32, image::Rgb(color));
        }
    }

    result
}

/// Generate N visually distinct colors using HSV spacing
fn generate_distinct_colors(n: usize) -> Vec<[u8; 3]> {
    let mut colors = Vec::with_capacity(n);

    for i in 0..n {
        let hue = (i as f32 / n as f32) * 360.0;
        let sat = 0.8;
        let val = 0.9;

        let c = val * sat;
        let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
        let m = val - c;

        let (r, g, b) = if hue < 60.0 {
            (c, x, 0.0)
        } else if hue < 120.0 {
            (x, c, 0.0)
        } else if hue < 180.0 {
            (0.0, c, x)
        } else if hue < 240.0 {
            (0.0, x, c)
        } else if hue < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        colors.push([
            ((r + m) * 255.0) as u8,
            ((g + m) * 255.0) as u8,
            ((b + m) * 255.0) as u8,
        ]);
    }

    colors
}

/// Convert K-means labels to a grayscale label image
///
/// Labels are scaled to fill the 0-255 range.
pub fn kmeans_labels_to_gray(labels: &Array2<u32>, k: usize) -> GrayImage {
    let (height, width) = labels.dim();
    let mut result = GrayImage::new(width as u32, height as u32);

    let scale = if k > 1 { 255.0 / (k - 1) as f32 } else { 0.0 };

    for y in 0..height {
        for x in 0..width {
            let label = labels[[y, x]] as f32;
            let val = (label * scale).clamp(0.0, 255.0) as u8;
            result.put_pixel(x as u32, y as u32, Luma([val]));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_two_region_image(width: u32, height: u32) -> DynamicImage {
        let mut img = image::RgbImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let color = if x < width / 2 {
                    [200u8, 50, 50] // Red region
                } else {
                    [50u8, 50, 200] // Blue region
                };
                img.put_pixel(x, y, image::Rgb(color));
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_kmeans_segment_basic() {
        let img = create_two_region_image(20, 20);
        let params = KMeansSegParams {
            k: 2,
            max_iterations: 50,
            epsilon: 1e-3,
            n_init: 1,
            use_color: true,
        };

        let result = kmeans_segment(&img, &params).expect("K-means failed");
        assert_eq!(result.labels.dim(), (20, 20));
        assert_eq!(result.centers.len(), 2);
        assert!(result.inertia >= 0.0);
    }

    #[test]
    fn test_kmeans_grayscale_mode() {
        let img = create_two_region_image(16, 16);
        let params = KMeansSegParams {
            k: 2,
            use_color: false,
            ..KMeansSegParams::default()
        };

        let result = kmeans_segment(&img, &params).expect("K-means grayscale failed");
        assert_eq!(result.centers[0].len(), 1); // Single channel
    }

    #[test]
    fn test_kmeans_two_regions_correct_labels() {
        let img = create_two_region_image(20, 20);
        let params = KMeansSegParams {
            k: 2,
            max_iterations: 100,
            epsilon: 1e-4,
            n_init: 3,
            use_color: true,
        };

        let result = kmeans_segment(&img, &params).expect("K-means failed");

        // All pixels in the left half should have the same label
        let left_label = result.labels[[0, 0]];
        for y in 0..20 {
            for x in 0..10 {
                assert_eq!(
                    result.labels[[y, x]],
                    left_label,
                    "Left region label inconsistent at ({}, {})",
                    x,
                    y
                );
            }
        }

        // All pixels in the right half should have the same (different) label
        let right_label = result.labels[[0, 19]];
        assert_ne!(left_label, right_label);
        for y in 0..20 {
            for x in 10..20 {
                assert_eq!(
                    result.labels[[y, x]],
                    right_label,
                    "Right region label inconsistent at ({}, {})",
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_kmeans_multiple_clusters() {
        let mut img = image::RgbImage::new(30, 10);
        for y in 0..10u32 {
            for x in 0..10u32 {
                img.put_pixel(x, y, image::Rgb([255, 0, 0]));
                img.put_pixel(x + 10, y, image::Rgb([0, 255, 0]));
                img.put_pixel(x + 20, y, image::Rgb([0, 0, 255]));
            }
        }

        let dyn_img = DynamicImage::ImageRgb8(img);
        let params = KMeansSegParams {
            k: 3,
            max_iterations: 100,
            epsilon: 1e-4,
            n_init: 3,
            use_color: true,
        };

        let result = kmeans_segment(&dyn_img, &params).expect("K-means 3-cluster failed");
        assert_eq!(result.centers.len(), 3);

        // Verify distinct labels for three regions
        let l0 = result.labels[[5, 5]];
        let l1 = result.labels[[5, 15]];
        let l2 = result.labels[[5, 25]];
        assert_ne!(l0, l1);
        assert_ne!(l1, l2);
        assert_ne!(l0, l2);
    }

    #[test]
    fn test_kmeans_reject_invalid_k() {
        let img = DynamicImage::new_luma8(10, 10);
        let params = KMeansSegParams {
            k: 1,
            ..KMeansSegParams::default()
        };
        assert!(kmeans_segment(&img, &params).is_err());
    }

    #[test]
    fn test_kmeans_labels_to_color() {
        let labels = Array2::from_shape_fn((10, 10), |(y, _x)| if y < 5 { 0u32 } else { 1u32 });
        let color_img = kmeans_labels_to_color(&labels, 2);
        assert_eq!(color_img.dimensions(), (10, 10));

        // Top and bottom should have different colors
        let top = color_img.get_pixel(5, 2);
        let bot = color_img.get_pixel(5, 7);
        assert_ne!(top, bot);
    }

    #[test]
    fn test_kmeans_labels_to_gray() {
        let labels = Array2::from_shape_fn((10, 10), |(y, _x)| if y < 5 { 0u32 } else { 1u32 });
        let gray_img = kmeans_labels_to_gray(&labels, 2);
        assert_eq!(gray_img.dimensions(), (10, 10));

        // Label 0 -> 0, Label 1 -> 255
        assert_eq!(gray_img.get_pixel(0, 0)[0], 0);
        assert_eq!(gray_img.get_pixel(0, 9)[0], 255);
    }

    #[test]
    fn test_kmeans_convergence() {
        let img = create_two_region_image(20, 20);
        let params = KMeansSegParams {
            k: 2,
            max_iterations: 1000,
            epsilon: 1e-6,
            n_init: 1,
            use_color: true,
        };

        let result = kmeans_segment(&img, &params).expect("K-means convergence failed");
        // Should converge well before max_iterations for a simple 2-region image
        assert!(
            result.iterations < 50,
            "Expected convergence in <50 iterations, got {}",
            result.iterations
        );
    }
}
