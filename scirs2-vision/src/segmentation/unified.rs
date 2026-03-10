//! Unified image segmentation API
//!
//! Provides a single entry point for segmenting images using multiple methods:
//! Otsu thresholding, adaptive thresholding, K-means color segmentation,
//! GrabCut foreground extraction, and watershed segmentation.

use crate::error::{Result, VisionError};
use image::DynamicImage;
use scirs2_core::ndarray::Array2;

/// Available segmentation methods
#[derive(Debug, Clone)]
pub enum SegmentMethod {
    /// Otsu's automatic thresholding (binary segmentation)
    Otsu,
    /// Adaptive thresholding with local neighborhood
    Adaptive {
        /// Block size for local thresholding (must be odd >= 3)
        block_size: usize,
        /// Constant subtracted from the mean
        c: f32,
        /// Thresholding method (Mean or Gaussian)
        method: super::AdaptiveMethod,
    },
    /// K-means color segmentation
    KMeans {
        /// Number of clusters
        k: usize,
        /// Maximum iterations
        max_iterations: usize,
    },
    /// GrabCut foreground extraction with bounding box
    GrabCut {
        /// Bounding box (x, y, width, height) containing the foreground
        rect: (u32, u32, u32, u32),
        /// Number of GMM components
        n_components: usize,
    },
    /// Watershed segmentation
    Watershed {
        /// Number of initial markers (None for auto)
        n_markers: Option<usize>,
        /// Connectivity (4 or 8)
        connectivity: u8,
    },
}

/// Result of unified segmentation
#[derive(Debug, Clone)]
pub struct SegmentResult {
    /// Label map (height x width). Each pixel has an integer label.
    /// For binary methods (Otsu, Adaptive): 0 = background, 1 = foreground
    /// For multi-label methods: 0..n_segments
    pub labels: Array2<u32>,
    /// Number of unique segments found
    pub n_segments: usize,
    /// Method that was used
    pub method_name: String,
}

/// Segment an image using the specified method
///
/// This is the unified API for image segmentation. It dispatches to the
/// appropriate algorithm based on the `method` parameter.
///
/// # Arguments
///
/// * `img` - Input image
/// * `method` - Segmentation method and its parameters
///
/// # Returns
///
/// * Segmentation result containing the label map
///
/// # Example
///
/// ```rust
/// use scirs2_vision::segmentation::unified::{segment, SegmentMethod};
/// use image::{DynamicImage, GrayImage, Luma};
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let mut buf = GrayImage::new(32, 32);
/// for y in 0..32u32 {
///     for x in 0..32u32 {
///         let val = if x < 16 { 200u8 } else { 50u8 };
///         buf.put_pixel(x, y, Luma([val]));
///     }
/// }
/// let img = DynamicImage::ImageLuma8(buf);
///
/// let result = segment(&img, SegmentMethod::Otsu)?;
/// assert_eq!(result.n_segments, 2);
/// # Ok(())
/// # }
/// ```
pub fn segment(img: &DynamicImage, method: SegmentMethod) -> Result<SegmentResult> {
    match method {
        SegmentMethod::Otsu => segment_otsu(img),
        SegmentMethod::Adaptive {
            block_size,
            c,
            method: adaptive_method,
        } => segment_adaptive(img, block_size, c, adaptive_method),
        SegmentMethod::KMeans { k, max_iterations } => segment_kmeans(img, k, max_iterations),
        SegmentMethod::GrabCut { rect, n_components } => segment_grabcut(img, rect, n_components),
        SegmentMethod::Watershed {
            n_markers,
            connectivity,
        } => segment_watershed(img, n_markers, connectivity),
    }
}

/// Otsu segmentation
fn segment_otsu(img: &DynamicImage) -> Result<SegmentResult> {
    let (binary, _threshold) = super::otsu_threshold(img)?;
    let (width, height) = binary.dimensions();
    let h = height as usize;
    let w = width as usize;

    let mut labels = Array2::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            labels[[y, x]] = if binary.get_pixel(x as u32, y as u32)[0] > 0 {
                1
            } else {
                0
            };
        }
    }

    Ok(SegmentResult {
        labels,
        n_segments: 2,
        method_name: "Otsu".to_string(),
    })
}

/// Adaptive thresholding segmentation
fn segment_adaptive(
    img: &DynamicImage,
    block_size: usize,
    c: f32,
    method: super::AdaptiveMethod,
) -> Result<SegmentResult> {
    let binary = super::adaptive_threshold(img, block_size, c, method)?;
    let (width, height) = binary.dimensions();
    let h = height as usize;
    let w = width as usize;

    let mut labels = Array2::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            labels[[y, x]] = if binary.get_pixel(x as u32, y as u32)[0] > 0 {
                1
            } else {
                0
            };
        }
    }

    Ok(SegmentResult {
        labels,
        n_segments: 2,
        method_name: "Adaptive".to_string(),
    })
}

/// K-means color segmentation
fn segment_kmeans(img: &DynamicImage, k: usize, max_iterations: usize) -> Result<SegmentResult> {
    let params = super::kmeans_seg::KMeansSegParams {
        k,
        max_iterations,
        epsilon: 1e-4,
        n_init: 3,
        use_color: true,
    };

    let result = super::kmeans_seg::kmeans_segment(img, &params)?;

    Ok(SegmentResult {
        labels: result.labels,
        n_segments: k,
        method_name: format!("KMeans(k={})", k),
    })
}

/// GrabCut foreground extraction
fn segment_grabcut(
    img: &DynamicImage,
    rect: (u32, u32, u32, u32),
    n_components: usize,
) -> Result<SegmentResult> {
    let params = super::grabcut::GrabCutParams {
        n_components,
        max_iterations: 10,
        epsilon: 1e-3,
        smoothness: 50.0,
    };

    let result = super::grabcut::grabcut_rect(img, rect, &params)?;
    let (h, w) = result.foreground_mask.dim();

    let mut labels = Array2::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            labels[[y, x]] = if result.foreground_mask[[y, x]] { 1 } else { 0 };
        }
    }

    Ok(SegmentResult {
        labels,
        n_segments: 2,
        method_name: "GrabCut".to_string(),
    })
}

/// Watershed segmentation
fn segment_watershed(
    img: &DynamicImage,
    _n_markers: Option<usize>,
    connectivity: u8,
) -> Result<SegmentResult> {
    let conn = if connectivity == 4 { 4 } else { 8 };

    let labels = super::watershed::watershed(img, None, conn)?;

    // Count unique labels
    let mut unique_labels = std::collections::HashSet::new();
    for &label in labels.iter() {
        unique_labels.insert(label);
    }

    Ok(SegmentResult {
        labels,
        n_segments: unique_labels.len(),
        method_name: "Watershed".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn create_bimodal_image() -> DynamicImage {
        // Create a bimodal image with a gradient band in the middle.
        // Left half is bright (220), right half is dark (20), with a smooth
        // transition zone at the boundary. This ensures Otsu picks a threshold
        // between the two modes rather than at either mode value.
        let mut buf = GrayImage::new(40, 32);
        for y in 0..32u32 {
            for x in 0..40u32 {
                let val = if x < 15 {
                    220u8
                } else if x > 24 {
                    20u8
                } else {
                    // Transition zone: linearly interpolate
                    let t = (x - 15) as f32 / 10.0;
                    (220.0 * (1.0 - t) + 20.0 * t) as u8
                };
                buf.put_pixel(x, y, Luma([val]));
            }
        }
        DynamicImage::ImageLuma8(buf)
    }

    #[test]
    fn test_segment_otsu() {
        let img = create_bimodal_image();
        let result = segment(&img, SegmentMethod::Otsu).expect("Otsu failed");
        assert_eq!(result.n_segments, 2);
        assert_eq!(result.labels.dim(), (32, 40));
        assert_eq!(result.method_name, "Otsu");

        // Bright side should be foreground (label 1), dark side background (label 0)
        let bright = result.labels[[16, 5]]; // x=5, value 220
        let dark = result.labels[[16, 35]]; // x=35, value 20
        assert_ne!(bright, dark, "Otsu should separate bright and dark regions");
    }

    #[test]
    fn test_segment_adaptive_mean() {
        let img = create_bimodal_image();
        let result = segment(
            &img,
            SegmentMethod::Adaptive {
                block_size: 7,
                c: 0.0,
                method: super::super::AdaptiveMethod::Mean,
            },
        )
        .expect("Adaptive mean failed");

        assert_eq!(result.n_segments, 2);
        assert_eq!(result.labels.dim(), (32, 40));
    }

    #[test]
    fn test_segment_adaptive_gaussian() {
        let img = create_bimodal_image();
        let result = segment(
            &img,
            SegmentMethod::Adaptive {
                block_size: 7,
                c: 0.0,
                method: super::super::AdaptiveMethod::Gaussian,
            },
        )
        .expect("Adaptive gaussian failed");

        assert_eq!(result.n_segments, 2);
    }

    #[test]
    fn test_segment_kmeans() {
        let mut buf = image::RgbImage::new(20, 20);
        for y in 0..20u32 {
            for x in 0..20u32 {
                let color = if x < 10 {
                    [200u8, 50, 50]
                } else {
                    [50u8, 50, 200]
                };
                buf.put_pixel(x, y, image::Rgb(color));
            }
        }
        let img = DynamicImage::ImageRgb8(buf);

        let result = segment(
            &img,
            SegmentMethod::KMeans {
                k: 2,
                max_iterations: 100,
            },
        )
        .expect("KMeans failed");

        assert_eq!(result.n_segments, 2);
        assert_eq!(result.labels.dim(), (20, 20));
    }

    #[test]
    fn test_segment_grabcut() {
        // Create image with bright center
        let mut buf = image::RgbImage::new(20, 20);
        for y in 0..20u32 {
            for x in 0..20u32 {
                let is_center = (5..15).contains(&x) && (5..15).contains(&y);
                let color = if is_center {
                    [220u8, 220, 220]
                } else {
                    [20u8, 20, 20]
                };
                buf.put_pixel(x, y, image::Rgb(color));
            }
        }
        let img = DynamicImage::ImageRgb8(buf);

        let result = segment(
            &img,
            SegmentMethod::GrabCut {
                rect: (4, 4, 12, 12),
                n_components: 3,
            },
        )
        .expect("GrabCut failed");

        assert_eq!(result.n_segments, 2);
        assert_eq!(result.labels.dim(), (20, 20));
    }

    #[test]
    fn test_segment_result_labels_range() {
        // Use a simple 32x32 image for this test
        let mut buf = GrayImage::new(32, 32);
        for y in 0..32u32 {
            for x in 0..32u32 {
                buf.put_pixel(x, y, Luma([if x < 16 { 240u8 } else { 10u8 }]));
            }
        }
        // Add transition pixels so Otsu threshold is between the modes
        for y in 0..32u32 {
            buf.put_pixel(15, y, Luma([125u8]));
            buf.put_pixel(16, y, Luma([125u8]));
        }
        let img = DynamicImage::ImageLuma8(buf);
        let result = segment(&img, SegmentMethod::Otsu).expect("Otsu failed");

        // All labels should be 0 or 1 for binary segmentation
        for &label in result.labels.iter() {
            assert!(label <= 1, "Label should be 0 or 1, got {}", label);
        }
    }

    #[test]
    fn test_segment_kmeans_three_regions() {
        let mut buf = image::RgbImage::new(30, 10);
        for y in 0..10u32 {
            for x in 0..10u32 {
                buf.put_pixel(x, y, image::Rgb([255, 0, 0]));
                buf.put_pixel(x + 10, y, image::Rgb([0, 255, 0]));
                buf.put_pixel(x + 20, y, image::Rgb([0, 0, 255]));
            }
        }
        let img = DynamicImage::ImageRgb8(buf);

        let result = segment(
            &img,
            SegmentMethod::KMeans {
                k: 3,
                max_iterations: 100,
            },
        )
        .expect("KMeans 3-cluster failed");

        assert_eq!(result.n_segments, 3);

        // Three regions should have different labels
        let l0 = result.labels[[5, 5]];
        let l1 = result.labels[[5, 15]];
        let l2 = result.labels[[5, 25]];
        assert_ne!(l0, l1);
        assert_ne!(l1, l2);
        assert_ne!(l0, l2);
    }
}
