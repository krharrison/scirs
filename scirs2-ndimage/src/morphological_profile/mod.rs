//! Morphological Attribute Profiles (MAP) and related operations.
//!
//! Provides:
//! - `MorphologicalProfile`: openings/closings by reconstruction at multiple scales.
//! - `build_morphological_profile`: builds the DMP (difference of morphological profiles).
//! - `area_opening` / `area_closing`: remove components smaller/larger than `min_area`.
//! - `attribute_profile`: generic attribute-based filtering.

use crate::error::{NdimageError, NdimageResult};
use crate::reconstruction_ops::{
    flat_dilation, opening_by_reconstruction, closing_by_reconstruction,
};

// ─── Data structures ──────────────────────────────────────────────────────────

/// Result of building a morphological profile at multiple scales (lambdas).
#[derive(Debug, Clone)]
pub struct MorphologicalProfile {
    /// Opening by reconstruction at each scale: `openings[i]` corresponds to
    /// `lambdas[i]`.
    pub openings: Vec<Vec<Vec<f64>>>,
    /// Closing by reconstruction at each scale: `closings[i]` corresponds to
    /// `lambdas[i]`.
    pub closings: Vec<Vec<Vec<f64>>>,
    /// The scale values used.
    pub lambdas: Vec<usize>,
}

/// Structuring element type for granulometry / profile operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SEType {
    /// Disk (approximated as an octagon by combining horizontal and diagonal square SEs).
    Disk,
    /// Axis-aligned square / rectangle.
    Square,
    /// Horizontal line.
    Line,
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Build a morphological attribute profile (MAP) for the given image.
///
/// For each scale λ in `lambda_range`:
/// 1. Erode with a square SE of radius λ.
/// 2. Reconstruct by dilation → opening by reconstruction.
/// 3. Dilate with a square SE of radius λ.
/// 4. Reconstruct by erosion → closing by reconstruction.
///
/// # Errors
/// Returns an error if image dimensions are inconsistent.
pub fn build_morphological_profile(
    image: &[Vec<f64>],
    lambda_range: &[usize],
) -> NdimageResult<MorphologicalProfile> {
    if image.is_empty() {
        return Err(NdimageError::InvalidInput("image must not be empty".into()));
    }
    let rows = image.len();
    let cols = image[0].len();
    if image.iter().any(|r| r.len() != cols) {
        return Err(NdimageError::DimensionError("image rows must have equal length".into()));
    }
    if lambda_range.is_empty() {
        return Err(NdimageError::InvalidInput("lambda_range must not be empty".into()));
    }

    let mut openings = Vec::with_capacity(lambda_range.len());
    let mut closings = Vec::with_capacity(lambda_range.len());

    for &lambda in lambda_range {
        let eroded = flat_erosion_2d(image, lambda)?;
        let opened = opening_by_reconstruction(image, &eroded)?;
        openings.push(opened);

        let dilated = flat_dilation(image, lambda)?;
        let closed  = closing_by_reconstruction(image, &dilated)?;
        closings.push(closed);
    }

    Ok(MorphologicalProfile {
        openings,
        closings,
        lambdas: lambda_range.to_vec(),
    })
}

/// Difference of Morphological Profiles (DMP) — opening component.
///
/// Returns `opening[i-1] - opening[i]` for i in 1..n (change between
/// consecutive scales). Length = `lambdas.len() - 1`.
pub fn dmp_openings(profile: &MorphologicalProfile) -> Vec<Vec<Vec<f64>>> {
    difference_of_profiles(&profile.openings, false)
}

/// Difference of Morphological Profiles (DMP) — closing component.
///
/// Returns `closing[i] - closing[i-1]` for i in 1..n.
/// Length = `lambdas.len() - 1`.
pub fn dmp_closings(profile: &MorphologicalProfile) -> Vec<Vec<Vec<f64>>> {
    difference_of_profiles(&profile.closings, true)
}

fn difference_of_profiles(
    profiles: &[Vec<Vec<f64>>],
    reverse: bool,
) -> Vec<Vec<Vec<f64>>> {
    if profiles.len() < 2 { return vec![]; }
    let n = profiles.len();
    (1..n).map(|i| {
        let (a_idx, b_idx) = if reverse { (i, i - 1) } else { (i - 1, i) };
        let a = &profiles[a_idx];
        let b = &profiles[b_idx];
        a.iter().zip(b.iter()).map(|(ar, br)| {
            ar.iter().zip(br.iter()).map(|(&av, &bv)| av - bv).collect()
        }).collect()
    }).collect()
}

// ─── Area opening / closing ───────────────────────────────────────────────────

/// Area opening: remove connected components with area (pixel count) < `min_area`.
///
/// Uses 4-connectivity for component labeling. Pixels belonging to
/// components that survive are kept at their original intensity; pixels
/// in removed components are replaced by the local background computed via
/// dilation + reconstruction (flat background extension).
///
/// # Errors
/// Returns an error if image dimensions are inconsistent.
pub fn area_opening(image: &[Vec<f64>], min_area: usize) -> NdimageResult<Vec<Vec<f64>>> {
    if image.is_empty() { return Ok(vec![]); }
    let rows = image.len();
    let cols = image[0].len();
    if image.iter().any(|r| r.len() != cols) {
        return Err(NdimageError::DimensionError("image rows must have equal length".into()));
    }

    // Threshold-free area opening: iterative reconstruction.
    // We use a level-by-level (h-maxima based) approach:
    // For each pixel, check the area of the regional maximum it belongs to.
    // Simpler approach: union-find on level sets.

    // Sort unique intensity values descending.
    let mut levels: Vec<f64> = image.iter().flat_map(|r| r.iter().copied()).collect();
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    levels.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    levels.reverse(); // process from highest to lowest

    let mut result = image.iter().map(|r| r.to_vec()).collect::<Vec<_>>();

    // For each level, label connected components and suppress small ones.
    for &level in &levels {
        // Create binary image: pixels >= level.
        let binary: Vec<Vec<bool>> = image.iter()
            .map(|r| r.iter().map(|&v| v >= level).collect())
            .collect();
        let (labels, areas) = label_components_4(&binary)?;

        for r in 0..rows {
            for c in 0..cols {
                let label = labels[r][c];
                if label == 0 { continue; }
                // If this component is too small and at exactly this level,
                // we need to suppress it.
                if areas[label - 1] < min_area && (image[r][c] - level).abs() < 1e-15 {
                    // Replace with the maximum of 4-neighbours (background fill).
                    let mut max_nb = f64::NEG_INFINITY;
                    for (nr, nc) in neighbours4(r, c, rows, cols) {
                        if image[nr][nc] < level && result[nr][nc] > max_nb {
                            max_nb = result[nr][nc];
                        }
                    }
                    if max_nb > f64::NEG_INFINITY {
                        result[r][c] = max_nb;
                    }
                }
            }
        }
    }

    Ok(result)
}

/// Area closing: fill holes (connected components of low intensity) with area < `max_area`.
///
/// Equivalent to area opening applied to the inverted image, then re-inversion.
///
/// # Errors
/// Returns an error if image dimensions are inconsistent.
pub fn area_closing(image: &[Vec<f64>], max_area: usize) -> NdimageResult<Vec<Vec<f64>>> {
    if image.is_empty() { return Ok(vec![]); }
    if image.iter().any(|r| r.len() != image[0].len()) {
        return Err(NdimageError::DimensionError("image rows must have equal length".into()));
    }

    // Invert image.
    let max_val = image.iter().flat_map(|r| r.iter().copied()).fold(f64::NEG_INFINITY, f64::max);
    let inverted: Vec<Vec<f64>> = image.iter()
        .map(|r| r.iter().map(|&v| max_val - v).collect())
        .collect();

    // Area opening on inverted.
    let opened = area_opening(&inverted, max_area)?;

    // Re-invert.
    let result = opened.iter()
        .map(|r| r.iter().map(|&v| max_val - v).collect())
        .collect();

    Ok(result)
}

/// Generic attribute profile filtering.
///
/// Recursively removes regional maxima whose computed `attribute_fn` value
/// falls below `threshold`.  The function receives the sub-image of a
/// connected component and returns a scalar attribute.
///
/// # Errors
/// Returns an error if image dimensions are inconsistent.
pub fn attribute_profile(
    image: &[Vec<f64>],
    attribute_fn: impl Fn(&[Vec<f64>]) -> f64,
    threshold: f64,
) -> NdimageResult<Vec<Vec<f64>>> {
    if image.is_empty() { return Ok(vec![]); }
    let rows = image.len();
    let cols = image[0].len();
    if image.iter().any(|r| r.len() != cols) {
        return Err(NdimageError::DimensionError("image rows must have equal length".into()));
    }

    let mut result = image.iter().map(|r| r.to_vec()).collect::<Vec<_>>();

    // Sort unique levels descending.
    let mut levels: Vec<f64> = image.iter().flat_map(|r| r.iter().copied()).collect();
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    levels.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    levels.reverse();

    for &level in &levels {
        let binary: Vec<Vec<bool>> = image.iter()
            .map(|r| r.iter().map(|&v| v >= level).collect())
            .collect();
        let (labels, _) = label_components_4(&binary)?;

        // Collect pixel groups per label.
        let n_labels = labels.iter().flat_map(|r| r.iter().copied()).max().unwrap_or(0);
        for lbl in 1..=n_labels {
            // Extract sub-image for this component.
            let sub: Vec<Vec<f64>> = (0..rows).map(|r| {
                (0..cols).map(|c| {
                    if labels[r][c] == lbl && (image[r][c] - level).abs() < 1e-15 {
                        image[r][c]
                    } else {
                        0.0
                    }
                }).collect()
            }).collect();

            let attr = attribute_fn(&sub);
            if attr < threshold {
                // Suppress: replace with max 4-neighbour below level.
                for r in 0..rows {
                    for c in 0..cols {
                        if labels[r][c] == lbl && (image[r][c] - level).abs() < 1e-15 {
                            let mut max_nb = f64::NEG_INFINITY;
                            for (nr, nc) in neighbours4(r, c, rows, cols) {
                                if image[nr][nc] < level && result[nr][nc] > max_nb {
                                    max_nb = result[nr][nc];
                                }
                            }
                            if max_nb > f64::NEG_INFINITY {
                                result[r][c] = max_nb;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(result)
}

// ─── Internal utilities ───────────────────────────────────────────────────────

/// 4-connected neighbours.
fn neighbours4(r: usize, c: usize, rows: usize, cols: usize) -> impl Iterator<Item = (usize, usize)> {
    let mut nb = Vec::with_capacity(4);
    if r > 0       { nb.push((r - 1, c)); }
    if r + 1 < rows { nb.push((r + 1, c)); }
    if c > 0       { nb.push((r, c - 1)); }
    if c + 1 < cols { nb.push((r, c + 1)); }
    nb.into_iter()
}

/// Label 4-connected components of a binary image.
/// Returns `(label_map, areas)` where `label_map[r][c]` is 0 (background)
/// or a 1-based component index, and `areas[i]` is the area of component `i+1`.
fn label_components_4(
    binary: &[Vec<bool>],
) -> NdimageResult<(Vec<Vec<usize>>, Vec<usize>)> {
    let rows = binary.len();
    if rows == 0 { return Ok((vec![], vec![])); }
    let cols = binary[0].len();

    let mut labels = vec![vec![0usize; cols]; rows];
    let mut next_label = 1usize;
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut areas: Vec<usize> = Vec::new();

    for sr in 0..rows {
        for sc in 0..cols {
            if binary[sr][sc] && labels[sr][sc] == 0 {
                let lbl = next_label;
                next_label += 1;
                labels[sr][sc] = lbl;
                let mut area = 1usize;
                stack.push((sr, sc));
                while let Some((r, c)) = stack.pop() {
                    for (nr, nc) in neighbours4(r, c, rows, cols) {
                        if binary[nr][nc] && labels[nr][nc] == 0 {
                            labels[nr][nc] = lbl;
                            area += 1;
                            stack.push((nr, nc));
                        }
                    }
                }
                areas.push(area);
            }
        }
    }

    Ok((labels, areas))
}

/// Flat erosion with square SE of given radius.
fn flat_erosion_2d(image: &[Vec<f64>], radius: usize) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = image.len();
    if rows == 0 { return Ok(vec![]); }
    let cols = image[0].len();

    // Row-wise.
    let mut row_eroded = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let c0 = c.saturating_sub(radius);
            let c1 = (c + radius + 1).min(cols);
            let mut min_v = f64::INFINITY;
            for k in c0..c1 {
                if image[r][k] < min_v { min_v = image[r][k]; }
            }
            row_eroded[r][c] = min_v;
        }
    }

    // Column-wise.
    let mut result = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let r0 = r.saturating_sub(radius);
            let r1 = (r + radius + 1).min(rows);
            let mut min_v = f64::INFINITY;
            for k in r0..r1 {
                if row_eroded[k][c] < min_v { min_v = row_eroded[k][c]; }
            }
            result[r][c] = min_v;
        }
    }
    Ok(result)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_image(data: &[&[f64]]) -> Vec<Vec<f64>> {
        data.iter().map(|row| row.to_vec()).collect()
    }

    #[test]
    fn test_build_morphological_profile() {
        let image = make_image(&[
            &[0.0, 0.0, 1.0, 0.0, 0.0],
            &[0.0, 1.0, 2.0, 1.0, 0.0],
            &[1.0, 2.0, 3.0, 2.0, 1.0],
            &[0.0, 1.0, 2.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0, 0.0],
        ]);
        let lambdas = vec![1, 2];
        let profile = build_morphological_profile(&image, &lambdas).expect("profile failed");
        assert_eq!(profile.lambdas.len(), 2);
        assert_eq!(profile.openings.len(), 2);
        assert_eq!(profile.closings.len(), 2);
        assert_eq!(profile.openings[0].len(), 5);
        assert_eq!(profile.openings[0][0].len(), 5);
    }

    #[test]
    fn test_dmp_openings_length() {
        let image = make_image(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 9.0],
        ]);
        let profile = build_morphological_profile(&image, &[1, 2, 3]).expect("profile failed");
        let dmp = dmp_openings(&profile);
        assert_eq!(dmp.len(), 2, "DMP should have n-1 entries");
    }

    #[test]
    fn test_area_opening_removes_small_components() {
        // Single bright pixel on a flat background.
        let mut image = vec![vec![0.0f64; 10]; 10];
        image[5][5] = 1.0;
        let opened = area_opening(&image, 4).expect("area_opening failed");
        // The single pixel should be suppressed.
        assert!(opened[5][5] < 0.5, "small component should be removed");
    }

    #[test]
    fn test_area_opening_preserves_large_components() {
        let mut image = vec![vec![0.0f64; 10]; 10];
        // A 3×3 block (area=9).
        for r in 3..6 {
            for c in 3..6 {
                image[r][c] = 1.0;
            }
        }
        let opened = area_opening(&image, 5).expect("area_opening failed");
        // Large block (area=9 >= 5) should be preserved.
        assert!(opened[4][4] > 0.5, "large component should be preserved");
    }

    #[test]
    fn test_area_closing_fills_hole() {
        // A flat bright image with a tiny dark hole.
        let mut image = vec![vec![1.0f64; 10]; 10];
        image[5][5] = 0.0;
        let closed = area_closing(&image, 4).expect("area_closing failed");
        assert!(closed[5][5] > 0.5, "small dark hole should be filled");
    }

    #[test]
    fn test_attribute_profile_area() {
        let mut image = vec![vec![0.0f64; 10]; 10];
        image[0][0] = 1.0; // single pixel, area=1
        for r in 4..8 {
            for c in 4..8 {
                image[r][c] = 1.0; // 4×4 block, area=16
            }
        }
        // Area attribute: count non-zero pixels in the sub-image.
        let attr_fn = |sub: &[Vec<f64>]| -> f64 {
            sub.iter().flat_map(|r| r.iter().copied()).filter(|&v| v > 0.0).count() as f64
        };
        // Threshold 5: remove single pixel, keep 4×4 block.
        let result = attribute_profile(&image, attr_fn, 5.0).expect("attribute_profile failed");
        assert!(result[0][0] < 0.5, "single pixel should be removed");
        assert!(result[5][5] > 0.5, "large block should survive");
    }
}
