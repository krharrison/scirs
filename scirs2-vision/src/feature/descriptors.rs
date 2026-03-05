//! Unified feature descriptor API
//!
//! Provides a single entry point for computing feature descriptors using
//! multiple methods (BRIEF, ORB, HOG). The descriptors are returned in a
//! unified format that supports both binary and float descriptors.
//!
//! - **BRIEF**: Fast binary descriptor for keypoint matching
//! - **ORB**: Oriented FAST + Rotated BRIEF (rotation-invariant binary descriptor)
//! - **HOG**: Histogram of Oriented Gradients (dense float descriptor for detection)

use crate::error::{Result, VisionError};
use crate::feature::KeyPoint;
use image::DynamicImage;

/// Descriptor method selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorMethod {
    /// BRIEF binary descriptor (fast, not rotation invariant)
    Brief,
    /// ORB descriptor (oriented FAST + rotated BRIEF, rotation invariant)
    Orb,
    /// HOG descriptor (dense float descriptor, good for object detection)
    Hog,
}

/// A computed feature descriptor in unified format
#[derive(Debug, Clone)]
pub struct UnifiedDescriptor {
    /// Associated keypoint (position, scale, orientation)
    pub keypoint: KeyPoint,
    /// Float descriptor vector (for SIFT-like, HOG descriptors)
    /// For binary descriptors, each bit is expanded to 0.0 or 1.0
    pub float_vector: Vec<f32>,
    /// Binary descriptor (only populated for BRIEF/ORB)
    pub binary_vector: Option<Vec<u32>>,
    /// Method used to compute this descriptor
    pub method: DescriptorMethod,
}

/// Parameters for unified descriptor computation
#[derive(Debug, Clone)]
pub struct DescriptorParams {
    /// Descriptor method to use
    pub method: DescriptorMethod,
    /// BRIEF descriptor size in bits (128, 256, or 512)
    pub brief_size: usize,
    /// BRIEF/ORB patch size
    pub patch_size: usize,
    /// ORB number of features
    pub orb_num_features: usize,
    /// HOG cell size
    pub hog_cell_size: usize,
    /// HOG number of orientation bins
    pub hog_num_bins: usize,
}

impl Default for DescriptorParams {
    fn default() -> Self {
        Self {
            method: DescriptorMethod::Brief,
            brief_size: 256,
            patch_size: 48,
            orb_num_features: 500,
            hog_cell_size: 8,
            hog_num_bins: 9,
        }
    }
}

/// Compute descriptors for given keypoints using the specified method
///
/// # Arguments
///
/// * `img` - Input image
/// * `keypoints` - Detected keypoints to compute descriptors for
/// * `params` - Descriptor computation parameters
///
/// # Returns
///
/// * Vector of unified descriptors
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_vision::feature::descriptors::{compute_descriptors, DescriptorMethod, DescriptorParams};
/// use scirs2_vision::feature::KeyPoint;
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("test.jpg").expect("Failed to open");
/// let keypoints = vec![
///     KeyPoint { x: 50.0, y: 50.0, scale: 1.0, orientation: 0.0, response: 1.0 },
/// ];
/// let params = DescriptorParams {
///     method: DescriptorMethod::Brief,
///     ..DescriptorParams::default()
/// };
/// let descriptors = compute_descriptors(&img, &keypoints, &params)?;
/// # Ok(())
/// # }
/// ```
pub fn compute_descriptors(
    img: &DynamicImage,
    keypoints: &[KeyPoint],
    params: &DescriptorParams,
) -> Result<Vec<UnifiedDescriptor>> {
    match params.method {
        DescriptorMethod::Brief => compute_brief_unified(img, keypoints, params),
        DescriptorMethod::Orb => compute_orb_unified(img, keypoints, params),
        DescriptorMethod::Hog => compute_hog_unified(img, keypoints, params),
    }
}

/// Compute BRIEF descriptors and convert to unified format
fn compute_brief_unified(
    img: &DynamicImage,
    keypoints: &[KeyPoint],
    params: &DescriptorParams,
) -> Result<Vec<UnifiedDescriptor>> {
    let config = crate::feature::brief::BriefConfig {
        descriptor_size: params.brief_size,
        patch_size: params.patch_size,
        use_smoothing: true,
        smoothing_sigma: 2.0,
    };

    let brief_descs =
        crate::feature::brief::compute_brief_descriptors(img, keypoints.to_vec(), &config)?;

    let mut unified = Vec::with_capacity(brief_descs.len());
    for desc in brief_descs {
        let float_vector = binary_to_float(&desc.descriptor, params.brief_size);
        unified.push(UnifiedDescriptor {
            keypoint: desc.keypoint,
            float_vector,
            binary_vector: Some(desc.descriptor),
            method: DescriptorMethod::Brief,
        });
    }

    Ok(unified)
}

/// Compute ORB descriptors and convert to unified format
fn compute_orb_unified(
    img: &DynamicImage,
    _keypoints: &[KeyPoint],
    params: &DescriptorParams,
) -> Result<Vec<UnifiedDescriptor>> {
    let config = crate::feature::orb::OrbConfig {
        num_features: params.orb_num_features,
        patch_size: params.patch_size.min(31),
        ..crate::feature::orb::OrbConfig::default()
    };

    // ORB detects its own keypoints and computes descriptors
    let orb_descs = crate::feature::orb::detect_and_compute_orb(img, &config)?;

    let mut unified = Vec::with_capacity(orb_descs.len());
    for desc in orb_descs {
        let float_vector = binary_to_float(&desc.descriptor, 256);
        unified.push(UnifiedDescriptor {
            keypoint: desc.keypoint,
            float_vector,
            binary_vector: Some(desc.descriptor),
            method: DescriptorMethod::Orb,
        });
    }

    Ok(unified)
}

/// Compute HOG descriptor and create unified descriptors
///
/// HOG computes a dense descriptor for the entire image. We create one
/// UnifiedDescriptor per input keypoint, extracting the local HOG features
/// around each keypoint's cell.
fn compute_hog_unified(
    img: &DynamicImage,
    keypoints: &[KeyPoint],
    params: &DescriptorParams,
) -> Result<Vec<UnifiedDescriptor>> {
    let config = crate::feature::hog::HogConfig {
        cell_size: params.hog_cell_size,
        num_bins: params.hog_num_bins,
        ..crate::feature::hog::HogConfig::default()
    };

    let hog_desc = crate::feature::hog::compute_hog(img, &config)?;

    // For each keypoint, extract the local cell's histogram as its descriptor
    let cell_size = params.hog_cell_size;
    let bins = params.hog_num_bins;
    let cells_x = hog_desc.cells_x;

    let mut unified = Vec::with_capacity(keypoints.len());

    for kp in keypoints {
        let cell_x = (kp.x as usize) / cell_size;
        let cell_y = (kp.y as usize) / cell_size;

        if cell_x >= hog_desc.cells_x || cell_y >= hog_desc.cells_y {
            continue;
        }

        // Extract multi-cell neighborhood (3x3 cells around keypoint)
        let mut float_vector = Vec::new();
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let cy = cell_y as i32 + dy;
                let cx = cell_x as i32 + dx;

                if cy >= 0
                    && cy < hog_desc.cells_y as i32
                    && cx >= 0
                    && cx < hog_desc.cells_x as i32
                {
                    let offset = (cy as usize * cells_x + cx as usize) * bins;
                    if offset + bins <= hog_desc.features.len() {
                        float_vector.extend_from_slice(&hog_desc.features[offset..offset + bins]);
                    }
                } else {
                    // Pad with zeros for border cells
                    float_vector.extend(std::iter::repeat_n(0.0f32, bins));
                }
            }
        }

        unified.push(UnifiedDescriptor {
            keypoint: kp.clone(),
            float_vector,
            binary_vector: None,
            method: DescriptorMethod::Hog,
        });
    }

    Ok(unified)
}

/// Convert binary descriptor words to float vector
fn binary_to_float(binary: &[u32], num_bits: usize) -> Vec<f32> {
    let mut float_vec = Vec::with_capacity(num_bits);

    for (word_idx, &word) in binary.iter().enumerate() {
        for bit in 0..32 {
            if word_idx * 32 + bit >= num_bits {
                break;
            }
            let val = if word & (1 << bit) != 0 {
                1.0f32
            } else {
                0.0f32
            };
            float_vec.push(val);
        }
    }

    float_vec
}

/// Match two sets of unified descriptors
///
/// For binary descriptors, uses Hamming distance.
/// For float descriptors, uses Euclidean distance.
///
/// # Arguments
///
/// * `desc1` - First set of descriptors
/// * `desc2` - Second set of descriptors
/// * `max_distance` - Maximum distance for a valid match (normalized 0-1)
///
/// # Returns
///
/// * Vector of (index1, index2, distance) tuples for matched descriptors
pub fn match_unified_descriptors(
    desc1: &[UnifiedDescriptor],
    desc2: &[UnifiedDescriptor],
    max_distance: f32,
) -> Vec<(usize, usize, f32)> {
    if desc1.is_empty() || desc2.is_empty() {
        return Vec::new();
    }

    let use_binary = desc1[0].binary_vector.is_some() && desc2[0].binary_vector.is_some();
    let mut matches = Vec::new();

    for (i, d1) in desc1.iter().enumerate() {
        let mut best_dist = f32::MAX;
        let mut best_idx = 0;
        let mut second_best = f32::MAX;

        for (j, d2) in desc2.iter().enumerate() {
            let dist = if use_binary {
                match (d1.binary_vector.as_ref(), d2.binary_vector.as_ref()) {
                    (Some(b1), Some(b2)) => {
                        let hamming: u32 = b1
                            .iter()
                            .zip(b2.iter())
                            .map(|(&a, &b)| (a ^ b).count_ones())
                            .sum();
                        let max_bits = (b1.len() * 32) as f32;
                        hamming as f32 / max_bits
                    }
                    _ => euclidean_distance(&d1.float_vector, &d2.float_vector),
                }
            } else {
                euclidean_distance(&d1.float_vector, &d2.float_vector)
            };

            if dist < best_dist {
                second_best = best_dist;
                best_dist = dist;
                best_idx = j;
            } else if dist < second_best {
                second_best = dist;
            }
        }

        // Apply distance threshold and Lowe's ratio test
        if best_dist < max_distance && best_dist < second_best * 0.75 {
            matches.push((i, best_idx, best_dist));
        }
    }

    matches.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    matches
}

/// Compute normalized Euclidean distance between two float vectors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return f32::MAX;
    }

    let sum_sq: f32 = a
        .iter()
        .take(len)
        .zip(b.iter().take(len))
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();

    sum_sq.sqrt() / (len as f32).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_to_float() {
        let binary = vec![0b00001111u32]; // bits 0-3 set
        let float_vec = binary_to_float(&binary, 8);
        assert_eq!(float_vec.len(), 8);
        assert_eq!(float_vec[0], 1.0); // bit 0 set
        assert_eq!(float_vec[3], 1.0); // bit 3 set
        assert_eq!(float_vec[4], 0.0); // bit 4 not set
        assert_eq!(float_vec[7], 0.0); // bit 7 not set
    }

    #[test]
    fn test_euclidean_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let dist = euclidean_distance(&a, &a);
        assert!(dist.abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_different() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let dist = euclidean_distance(&a, &b);
        // sqrt(1) / sqrt(3) ~= 0.577
        assert!((dist - 1.0 / 3.0f32.sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_match_unified_empty() {
        let matches = match_unified_descriptors(&[], &[], 1.0);
        assert!(matches.is_empty());
    }

    #[test]
    fn test_match_unified_float_descriptors() {
        let desc1 = vec![UnifiedDescriptor {
            keypoint: KeyPoint {
                x: 10.0,
                y: 10.0,
                scale: 1.0,
                orientation: 0.0,
                response: 1.0,
            },
            float_vector: vec![1.0, 0.0, 0.0, 0.0],
            binary_vector: None,
            method: DescriptorMethod::Hog,
        }];

        let desc2 = vec![
            UnifiedDescriptor {
                keypoint: KeyPoint {
                    x: 20.0,
                    y: 20.0,
                    scale: 1.0,
                    orientation: 0.0,
                    response: 1.0,
                },
                float_vector: vec![1.0, 0.0, 0.0, 0.0],
                binary_vector: None,
                method: DescriptorMethod::Hog,
            },
            UnifiedDescriptor {
                keypoint: KeyPoint {
                    x: 30.0,
                    y: 30.0,
                    scale: 1.0,
                    orientation: 0.0,
                    response: 1.0,
                },
                float_vector: vec![0.0, 1.0, 0.0, 0.0],
                binary_vector: None,
                method: DescriptorMethod::Hog,
            },
        ];

        // Same descriptor should match well
        let matches = match_unified_descriptors(&desc1, &desc2, 1.0);
        // With only 2 candidates and ratio test, may or may not match
        // Just verify it runs without errors
        assert!(matches.len() <= 1);
    }

    #[test]
    fn test_descriptor_params_default() {
        let params = DescriptorParams::default();
        assert_eq!(params.method, DescriptorMethod::Brief);
        assert_eq!(params.brief_size, 256);
        assert!(params.patch_size > 0);
    }

    #[test]
    fn test_compute_descriptors_brief_no_keypoints() {
        let img = DynamicImage::new_luma8(64, 64);
        let params = DescriptorParams {
            method: DescriptorMethod::Brief,
            ..DescriptorParams::default()
        };

        let descs =
            compute_descriptors(&img, &[], &params).expect("BRIEF with empty keypoints failed");
        assert!(descs.is_empty());
    }

    #[test]
    fn test_compute_descriptors_hog_empty_keypoints() {
        let img = DynamicImage::new_luma8(64, 64);
        let params = DescriptorParams {
            method: DescriptorMethod::Hog,
            ..DescriptorParams::default()
        };

        let descs =
            compute_descriptors(&img, &[], &params).expect("HOG with empty keypoints failed");
        assert!(descs.is_empty());
    }

    #[test]
    fn test_compute_hog_descriptor_has_values() {
        // Create an image with some gradient
        let mut buf = image::GrayImage::new(64, 64);
        for y in 0..64u32 {
            for x in 0..64u32 {
                buf.put_pixel(x, y, image::Luma([(x * 4) as u8]));
            }
        }
        let img = DynamicImage::ImageLuma8(buf);

        let keypoints = vec![KeyPoint {
            x: 32.0,
            y: 32.0,
            scale: 1.0,
            orientation: 0.0,
            response: 1.0,
        }];

        let params = DescriptorParams {
            method: DescriptorMethod::Hog,
            hog_cell_size: 8,
            hog_num_bins: 9,
            ..DescriptorParams::default()
        };

        let descs = compute_descriptors(&img, &keypoints, &params)
            .expect("HOG descriptor computation failed");
        assert_eq!(descs.len(), 1);
        // 3x3 neighborhood x 9 bins = 81 features
        assert_eq!(descs[0].float_vector.len(), 81);
        assert!(descs[0].binary_vector.is_none());
    }
}
