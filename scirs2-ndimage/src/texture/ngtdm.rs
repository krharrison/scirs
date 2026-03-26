//! Neighborhood Gray-Tone Difference Matrix (NGTDM) Computation and Features
//!
//! The NGTDM captures the difference between a pixel's gray level and the
//! average gray level of its neighbors. For each gray level `i`, it accumulates:
//!
//! - `s[i]`: sum of absolute differences between pixels of gray level `i`
//!   and their neighborhood averages
//! - `n[i]`: count of pixels with gray level `i` that have a complete
//!   neighborhood
//!
//! # Features
//!
//! Five features are derived:
//! - **Coarseness**: inverse measure of local gray-level variation
//! - **Contrast**: measures dynamic range and spatial frequency of intensity changes
//! - **Busyness**: measures rapid changes between gray levels
//! - **Complexity**: measures non-uniformity of gray-level transitions
//! - **Strength**: measures definite texture primitives
//!
//! # References
//!
//! - Amadasun, M. & King, R. (1989). "Textural features corresponding to
//!   textural properties." IEEE Transactions on Systems, Man, and Cybernetics,
//!   19(5), 1264-1274.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array1, Array2};

/// NGTDM result containing the s and n vectors.
#[derive(Debug, Clone)]
pub struct NgtdmResult {
    /// `s[i]`: sum of absolute differences for gray level `i`
    pub s: Array1<f64>,
    /// `n[i]`: count of pixels with gray level `i` having complete neighborhoods
    pub n: Array1<f64>,
    /// Number of gray levels
    pub n_levels: usize,
    /// Neighborhood distance (radius)
    pub distance: usize,
}

/// Features extracted from an NGTDM.
#[derive(Debug, Clone)]
pub struct NgtdmFeatures {
    /// Coarseness: inverse of sum of p(i)*s(i). Large value => coarse texture.
    pub coarseness: f64,
    /// Contrast: dynamic range and spatial frequency of changes.
    pub contrast: f64,
    /// Busyness: rapid changes between neighboring gray levels.
    pub busyness: f64,
    /// Complexity: non-uniformity and non-periodicity.
    pub complexity: f64,
    /// Strength: definiteness of texture primitives.
    pub strength: f64,
}

/// Compute the NGTDM from an 8-bit image.
///
/// For each pixel with a complete neighborhood (all neighbors within bounds),
/// the average gray level of the neighborhood is computed and the absolute
/// difference from the pixel's own gray level is accumulated.
///
/// # Parameters
/// - `image` - 8-bit grayscale image with values in `[0, n_levels-1]`
/// - `n_levels` - number of gray levels (>= 2)
/// - `distance` - neighborhood radius (1 = 3x3, 2 = 5x5, etc.)
///
/// # Errors
/// Returns error if `n_levels < 2`, `distance == 0`, or image is too small.
pub fn compute_ngtdm(
    image: &Array2<u8>,
    n_levels: usize,
    distance: usize,
) -> NdimageResult<NgtdmResult> {
    if n_levels < 2 {
        return Err(NdimageError::InvalidInput(
            "n_levels must be at least 2".into(),
        ));
    }
    if distance == 0 {
        return Err(NdimageError::InvalidInput(
            "distance must be at least 1".into(),
        ));
    }
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(NdimageError::InvalidInput("Image must not be empty".into()));
    }
    let min_dim = 2 * distance + 1;
    if rows < min_dim || cols < min_dim {
        return Err(NdimageError::InvalidInput(format!(
            "Image must be at least {}x{} for distance={}",
            min_dim, min_dim, distance
        )));
    }

    let mut s = Array1::<f64>::zeros(n_levels);
    let mut n = Array1::<f64>::zeros(n_levels);

    let d = distance as i64;
    let n_neighbors = ((2 * distance + 1) * (2 * distance + 1) - 1) as f64;

    for r in distance..rows - distance {
        for c in distance..cols - distance {
            let g = (image[[r, c]] as usize).min(n_levels - 1);

            // Compute average of neighbors
            let mut neighbor_sum = 0.0f64;
            for dr in -d..=d {
                for dc in -d..=d {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = (r as i64 + dr) as usize;
                    let nc = (c as i64 + dc) as usize;
                    neighbor_sum += image[[nr, nc]] as f64;
                }
            }
            let neighbor_avg = neighbor_sum / n_neighbors;

            s[g] += (g as f64 - neighbor_avg).abs();
            n[g] += 1.0;
        }
    }

    Ok(NgtdmResult {
        s,
        n,
        n_levels,
        distance,
    })
}

/// Extract NGTDM features from a pre-computed result.
///
/// # Parameters
/// - `ngtdm` - NGTDM result from `compute_ngtdm`
///
/// # Errors
/// Returns error if total pixel count is zero.
pub fn ngtdm_features(ngtdm: &NgtdmResult) -> NdimageResult<NgtdmFeatures> {
    let n_levels = ngtdm.n_levels;
    let total_n: f64 = ngtdm.n.iter().sum();

    if total_n < 1e-15 {
        return Err(NdimageError::InvalidInput(
            "No valid pixels in NGTDM (total count is 0)".into(),
        ));
    }

    // Probabilities p[i] = n[i] / total_n
    let p: Vec<f64> = (0..n_levels).map(|i| ngtdm.n[i] / total_n).collect();

    // Number of gray levels with non-zero count
    let n_nonzero = p.iter().filter(|&&pi| pi > 1e-15).count();

    // Coarseness: 1 / (eps + sum_i p(i)*s(i))
    let coarseness_denom: f64 = (0..n_levels).map(|i| p[i] * ngtdm.s[i]).sum::<f64>();
    let coarseness = if coarseness_denom < 1e-15 {
        // Uniform image: coarseness is maximal
        1e10_f64.min(1.0 / 1e-15)
    } else {
        1.0 / coarseness_denom
    };

    // Contrast
    let contrast = if n_nonzero < 2 {
        0.0
    } else {
        let n_nz = n_nonzero as f64;
        // sum of squared differences between gray levels
        let mut sum_diff_sq = 0.0f64;
        for i in 0..n_levels {
            if p[i] < 1e-15 {
                continue;
            }
            for j in 0..n_levels {
                if p[j] < 1e-15 {
                    continue;
                }
                sum_diff_sq += p[i] * p[j] * (i as f64 - j as f64).powi(2);
            }
        }
        let sum_s: f64 = ngtdm.s.iter().sum();
        (1.0 / (n_nz * (n_nz - 1.0))) * sum_diff_sq * sum_s / total_n
    };

    // Busyness
    let busyness = if n_nonzero < 2 {
        0.0
    } else {
        let numerator: f64 = (0..n_levels)
            .filter(|&i| p[i] > 1e-15)
            .map(|i| p[i] * ngtdm.s[i])
            .sum();

        let mut denominator = 0.0f64;
        for i in 0..n_levels {
            if p[i] < 1e-15 {
                continue;
            }
            for j in 0..n_levels {
                if p[j] < 1e-15 {
                    continue;
                }
                denominator += (i as f64 * p[i] - j as f64 * p[j]).abs();
            }
        }
        if denominator < 1e-15 {
            0.0
        } else {
            numerator / denominator
        }
    };

    // Complexity
    let complexity = if n_nonzero < 2 {
        0.0
    } else {
        let mut val = 0.0f64;
        for i in 0..n_levels {
            if p[i] < 1e-15 {
                continue;
            }
            for j in 0..n_levels {
                if p[j] < 1e-15 {
                    continue;
                }
                let diff = (i as f64 - j as f64).abs();
                let term = diff * (p[i] * ngtdm.s[i] + p[j] * ngtdm.s[j]) / (p[i] + p[j]);
                val += term;
            }
        }
        val / total_n
    };

    // Strength
    let strength = if n_nonzero < 2 {
        0.0
    } else {
        let mut numerator = 0.0f64;
        for i in 0..n_levels {
            if p[i] < 1e-15 {
                continue;
            }
            for j in 0..n_levels {
                if p[j] < 1e-15 {
                    continue;
                }
                numerator += (p[i] + p[j]) * (i as f64 - j as f64).powi(2);
            }
        }
        let sum_s: f64 = ngtdm.s.iter().sum();
        if sum_s < 1e-15 {
            0.0
        } else {
            numerator / sum_s
        }
    };

    Ok(NgtdmFeatures {
        coarseness,
        contrast,
        busyness,
        complexity,
        strength,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_ngtdm_uniform_image() {
        // Uniform image: all neighbors equal the center pixel.
        // s[i] should be 0 for all i, meaning coarseness should be maximal.
        let img = Array2::from_elem((5, 5), 2u8);
        let result = compute_ngtdm(&img, 4, 1).expect("ngtdm");

        // s should be all zeros (no difference from neighbors)
        let total_s: f64 = result.s.iter().sum();
        assert!(
            total_s < 1e-10,
            "Uniform image should have zero s values, got {}",
            total_s
        );

        let feats = ngtdm_features(&result).expect("features");
        // Coarseness should be very large for uniform image
        assert!(
            feats.coarseness > 1e8,
            "Coarseness should be very large for uniform image, got {}",
            feats.coarseness
        );
        // Contrast should be 0 (only one gray level present)
        assert!(
            feats.contrast.abs() < 1e-10,
            "Contrast should be 0 for uniform image, got {}",
            feats.contrast
        );
    }

    #[test]
    fn test_ngtdm_two_level_image() {
        // Image with two gray levels in vertical stripes
        let img = Array2::from_shape_fn((5, 6), |(_, c)| if c < 3 { 0u8 } else { 3u8 });
        let result = compute_ngtdm(&img, 4, 1).expect("ngtdm");

        // Only levels 0 and 3 should have counts
        assert!(result.n[0] > 0.0);
        assert!(result.n[3] > 0.0);
        assert!(result.n[1] < 1e-15);
        assert!(result.n[2] < 1e-15);

        let feats = ngtdm_features(&result).expect("features");
        // With two gray levels and large differences, contrast should be positive
        assert!(feats.contrast > 0.0, "Contrast should be positive");
    }

    #[test]
    fn test_ngtdm_gradient_image() {
        // Gradient: each row increases by 1
        let img = Array2::from_shape_fn((7, 7), |(_, c)| (c as u8).min(3));
        let result = compute_ngtdm(&img, 4, 1).expect("ngtdm");

        let feats = ngtdm_features(&result).expect("features");
        // All features should be finite and non-negative
        assert!(feats.coarseness.is_finite() && feats.coarseness >= 0.0);
        assert!(feats.contrast.is_finite() && feats.contrast >= 0.0);
        assert!(feats.busyness.is_finite() && feats.busyness >= 0.0);
        assert!(feats.complexity.is_finite() && feats.complexity >= 0.0);
        assert!(feats.strength.is_finite() && feats.strength >= 0.0);
    }

    #[test]
    fn test_ngtdm_larger_distance() {
        let img = Array2::from_shape_fn((9, 9), |(i, j)| ((i + j) % 4) as u8);
        let result = compute_ngtdm(&img, 4, 2).expect("ngtdm");

        // Should use 5x5 neighborhoods (distance=2)
        assert_eq!(result.distance, 2);
        let total_n: f64 = result.n.iter().sum();
        // Only pixels with r in [2..7), c in [2..7) have complete neighborhoods
        // That's 5*5 = 25 pixels
        assert!(
            (total_n - 25.0).abs() < 1e-10,
            "Expected 25 valid pixels, got {}",
            total_n
        );
    }

    #[test]
    fn test_ngtdm_errors() {
        let img = Array2::from_elem((5, 5), 0u8);
        // n_levels < 2
        assert!(compute_ngtdm(&img, 1, 1).is_err());
        // distance = 0
        assert!(compute_ngtdm(&img, 4, 0).is_err());
        // Image too small for distance
        let small = Array2::from_elem((2, 2), 0u8);
        assert!(compute_ngtdm(&small, 4, 1).is_err());
        // Empty image
        let empty = Array2::<u8>::zeros((0, 0));
        assert!(compute_ngtdm(&empty, 4, 1).is_err());
    }

    #[test]
    fn test_ngtdm_known_3x3() {
        // 3x3 image with distance=1: only center pixel (1,1) has a complete neighborhood
        // Image:
        // 0 1 2
        // 3 4 5
        // 6 7 0
        let img = Array2::from_shape_vec((3, 3), vec![0, 1, 2, 3, 4, 5, 6, 7, 0]).expect("ok");
        let result = compute_ngtdm(&img, 8, 1).expect("ngtdm");

        // Only pixel (1,1)=4 has a complete neighborhood
        let total_n: f64 = result.n.iter().sum();
        assert!(
            (total_n - 1.0).abs() < 1e-10,
            "Only 1 pixel should have a complete neighborhood"
        );
        assert!((result.n[4] - 1.0).abs() < 1e-10);

        // Neighbor average = (0+1+2+3+5+6+7+0)/8 = 24/8 = 3.0
        // |4 - 3| = 1.0
        assert!(
            (result.s[4] - 1.0).abs() < 1e-10,
            "s[4] should be 1.0, got {}",
            result.s[4]
        );
    }
}
