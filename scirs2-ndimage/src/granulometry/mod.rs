//! Morphological granulometry (size distribution analysis).
//!
//! Computes the pattern spectrum (size distribution) of an image using
//! morphological openings at increasing scales.

use crate::error::{NdimageError, NdimageResult};
use crate::reconstruction_ops::opening_by_reconstruction;

// ─── Data structures ──────────────────────────────────────────────────────────

/// Result of granulometry analysis.
#[derive(Debug, Clone)]
pub struct GranulometryResult {
    /// Scale values (SE radii) used.
    pub scales: Vec<f64>,
    /// Pattern spectrum: normalised rate of change of opened area per scale.
    /// `pattern_spectrum[i]` corresponds to `scales[i]`.
    pub pattern_spectrum: Vec<f64>,
}

/// Structuring element type for granulometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SEType {
    /// Disk (octagonal approximation).
    Disk,
    /// Axis-aligned square.
    Square,
    /// Horizontal line.
    Line(f64),
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Compute the granulometry (size distribution) of a 2-D image.
///
/// Applies opening with a structuring element of increasing scale and computes
/// the pattern spectrum as the discrete derivative of the normalised area.
///
/// # Arguments
/// * `image`     – 2-D image as `Vec<Vec<f64>>`
/// * `max_scale` – maximum SE radius (inclusive)
/// * `n_scales`  – number of scale steps (evenly spaced from 1 to `max_scale`)
/// * `se_type`   – structuring element type
///
/// # Errors
/// Returns an error if the image dimensions are inconsistent.
pub fn granulometry(
    image: &[Vec<f64>],
    max_scale: usize,
    n_scales: usize,
    se_type: SEType,
) -> NdimageResult<GranulometryResult> {
    if image.is_empty() {
        return Err(NdimageError::InvalidInput("image must not be empty".into()));
    }
    let cols = image[0].len();
    if image.iter().any(|r| r.len() != cols) {
        return Err(NdimageError::DimensionError("image rows must have equal length".into()));
    }
    if n_scales == 0 {
        return Err(NdimageError::InvalidInput("n_scales must be at least 1".into()));
    }
    if max_scale == 0 {
        return Err(NdimageError::InvalidInput("max_scale must be at least 1".into()));
    }

    // Total image "area" = sum of all pixel intensities (generalised area).
    let total_area: f64 = image.iter().flat_map(|r| r.iter().copied()).sum();
    if total_area.abs() < 1e-300 {
        // Uniform zero image → flat spectrum.
        let scales: Vec<f64> = (0..n_scales)
            .map(|i| 1.0 + i as f64 * (max_scale as f64 - 1.0) / n_scales.max(1) as f64)
            .collect();
        let ps = vec![0.0f64; n_scales];
        return Ok(GranulometryResult { scales, pattern_spectrum: ps });
    }

    // Compute scales (radii) as floating-point for output, integer for SE.
    let scales: Vec<f64> = (0..n_scales)
        .map(|i| 1.0 + i as f64 * (max_scale as f64 - 1.0) / (n_scales as f64))
        .collect();
    let radii: Vec<usize> = scales.iter().map(|&s| s.ceil() as usize).collect();

    // Compute opened area at scale 0 (original image).
    let mut prev_area = total_area;

    let mut pattern_spectrum = Vec::with_capacity(n_scales);

    for &radius in &radii {
        let eroded = apply_erosion(image, radius, se_type)?;
        let opened = opening_by_reconstruction(image, &eroded)?;
        let opened_area: f64 = opened.iter().flat_map(|r| r.iter().copied()).sum();
        let ps = ((prev_area - opened_area) / total_area).max(0.0);
        pattern_spectrum.push(ps);
        prev_area = opened_area;
    }

    Ok(GranulometryResult { scales, pattern_spectrum })
}

/// Discrete size distribution: fraction of connected components suppressed at each scale.
///
/// For each scale, applies area opening and measures the fraction of component
/// pixels removed compared to the original.
///
/// # Arguments
/// * `image`     – 2-D binary or grayscale image
/// * `n_scales`  – number of scale steps to analyse
///
/// # Errors
/// Returns an error if the image dimensions are inconsistent.
pub fn size_distribution(
    image: &[Vec<f64>],
    n_scales: usize,
) -> NdimageResult<Vec<f64>> {
    if image.is_empty() { return Ok(vec![]); }
    let cols = image[0].len();
    if image.iter().any(|r| r.len() != cols) {
        return Err(NdimageError::DimensionError("image rows must have equal length".into()));
    }
    if n_scales == 0 {
        return Err(NdimageError::InvalidInput("n_scales must be at least 1".into()));
    }

    let total_fg: f64 = image.iter().flat_map(|r| r.iter().copied())
        .filter(|&v| v > 0.5)
        .count() as f64;
    if total_fg < 1.0 {
        return Ok(vec![0.0; n_scales]);
    }

    let mut result = Vec::with_capacity(n_scales);
    for i in 0..n_scales {
        let min_area = i + 1; // 1-based
        let opened = apply_area_opening_simple(image, min_area)?;
        let remaining: f64 = opened.iter().flat_map(|r| r.iter().copied())
            .filter(|&v| v > 0.5)
            .count() as f64;
        result.push(((total_fg - remaining) / total_fg).clamp(0.0, 1.0));
    }

    Ok(result)
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Apply erosion with the given SE type and radius.
fn apply_erosion(
    image: &[Vec<f64>],
    radius: usize,
    se_type: SEType,
) -> NdimageResult<Vec<Vec<f64>>> {
    match se_type {
        SEType::Square => flat_erosion_square(image, radius),
        SEType::Disk   => flat_erosion_disk(image, radius),
        SEType::Line(_) => flat_erosion_line(image, radius),
    }
}

/// Flat erosion with a square SE.
fn flat_erosion_square(image: &[Vec<f64>], radius: usize) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = image.len();
    let cols = image[0].len();
    // Row-wise.
    let mut tmp = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let c0 = c.saturating_sub(radius);
            let c1 = (c + radius + 1).min(cols);
            let mut mv = f64::INFINITY;
            for k in c0..c1 { if image[r][k] < mv { mv = image[r][k]; } }
            tmp[r][c] = mv;
        }
    }
    // Column-wise.
    let mut result = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let r0 = r.saturating_sub(radius);
            let r1 = (r + radius + 1).min(rows);
            let mut mv = f64::INFINITY;
            for k in r0..r1 { if tmp[k][c] < mv { mv = tmp[k][c]; } }
            result[r][c] = mv;
        }
    }
    Ok(result)
}

/// Flat erosion with a disk SE (Euclidean distance).
fn flat_erosion_disk(image: &[Vec<f64>], radius: usize) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = image.len();
    let cols = image[0].len();
    let r = radius as isize;
    let mut result = vec![vec![f64::INFINITY; cols]; rows];
    for row in 0..rows {
        for col in 0..cols {
            let mut mv = f64::INFINITY;
            for dr in -r..=r {
                for dc in -r..=r {
                    if (dr * dr + dc * dc) as f64 <= (radius * radius) as f64 {
                        let nr = row as isize + dr;
                        let nc = col as isize + dc;
                        if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                            let v = image[nr as usize][nc as usize];
                            if v < mv { mv = v; }
                        }
                    }
                }
            }
            result[row][col] = mv;
        }
    }
    Ok(result)
}

/// Flat erosion with a horizontal line SE.
fn flat_erosion_line(image: &[Vec<f64>], radius: usize) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = image.len();
    let cols = image[0].len();
    let mut result = vec![vec![0.0f64; cols]; rows];
    for r in 0..rows {
        for c in 0..cols {
            let c0 = c.saturating_sub(radius);
            let c1 = (c + radius + 1).min(cols);
            let mut mv = f64::INFINITY;
            for k in c0..c1 { if image[r][k] < mv { mv = image[r][k]; } }
            result[r][c] = mv;
        }
    }
    Ok(result)
}

/// Simple area opening for size distribution (removes fg components < min_area).
fn apply_area_opening_simple(image: &[Vec<f64>], min_area: usize) -> NdimageResult<Vec<Vec<f64>>> {
    let rows = image.len();
    let cols = image[0].len();
    // Label foreground components.
    let binary: Vec<Vec<bool>> = image.iter()
        .map(|r| r.iter().map(|&v| v > 0.5).collect())
        .collect();
    let (labels, areas) = label_4(&binary);

    let mut result = image.iter().map(|r| r.to_vec()).collect::<Vec<_>>();
    for r in 0..rows {
        for c in 0..cols {
            let lbl = labels[r][c];
            if lbl > 0 && areas[lbl - 1] < min_area {
                result[r][c] = 0.0;
            }
        }
    }
    Ok(result)
}

/// 4-connected labeling for binary images.
fn label_4(binary: &[Vec<bool>]) -> (Vec<Vec<usize>>, Vec<usize>) {
    let rows = binary.len();
    if rows == 0 { return (vec![], vec![]); }
    let cols = binary[0].len();
    let mut labels = vec![vec![0usize; cols]; rows];
    let mut areas: Vec<usize> = Vec::new();
    let mut next = 1usize;
    let mut stack: Vec<(usize, usize)> = Vec::new();

    for sr in 0..rows {
        for sc in 0..cols {
            if binary[sr][sc] && labels[sr][sc] == 0 {
                labels[sr][sc] = next;
                let mut area = 1usize;
                stack.push((sr, sc));
                while let Some((r, c)) = stack.pop() {
                    macro_rules! try_nb {
                        ($nr:expr, $nc:expr) => {
                            if binary[$nr][$nc] && labels[$nr][$nc] == 0 {
                                labels[$nr][$nc] = next;
                                area += 1;
                                stack.push(($nr, $nc));
                            }
                        };
                    }
                    if r > 0        { try_nb!(r - 1, c); }
                    if r + 1 < rows { try_nb!(r + 1, c); }
                    if c > 0        { try_nb!(r, c - 1); }
                    if c + 1 < cols { try_nb!(r, c + 1); }
                }
                areas.push(area);
                next += 1;
            }
        }
    }
    (labels, areas)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_gradient(rows: usize, cols: usize) -> Vec<Vec<f64>> {
        (0..rows).map(|r| (0..cols).map(|c| (r + c) as f64).collect()).collect()
    }

    #[test]
    fn test_granulometry_basic() {
        let image = build_gradient(20, 20);
        let result = granulometry(&image, 5, 5, SEType::Square).expect("granulometry failed");
        assert_eq!(result.scales.len(), 5);
        assert_eq!(result.pattern_spectrum.len(), 5);
        for &ps in &result.pattern_spectrum {
            assert!(ps >= 0.0 && ps <= 1.0, "PS out of [0,1]: {}", ps);
        }
    }

    #[test]
    fn test_granulometry_disk() {
        let image = build_gradient(15, 15);
        let result = granulometry(&image, 3, 3, SEType::Disk).expect("granulometry disk failed");
        assert_eq!(result.pattern_spectrum.len(), 3);
    }

    #[test]
    fn test_granulometry_line() {
        let image = build_gradient(10, 10);
        let result = granulometry(&image, 3, 3, SEType::Line(0.0)).expect("granulometry line failed");
        assert_eq!(result.pattern_spectrum.len(), 3);
    }

    #[test]
    fn test_size_distribution_monotone() {
        // Binary image with various sized blobs.
        let mut image = vec![vec![0.0f64; 20]; 20];
        // 1-pixel blob
        image[1][1] = 1.0;
        // 2×2 blob
        for r in 5..7 { for c in 5..7 { image[r][c] = 1.0; } }
        // 4×4 blob
        for r in 10..14 { for c in 10..14 { image[r][c] = 1.0; } }

        let dist = size_distribution(&image, 10).expect("size_distribution failed");
        assert_eq!(dist.len(), 10);
        // Should be non-decreasing (more small components removed at larger scales).
        for i in 1..dist.len() {
            assert!(dist[i] >= dist[i - 1] - 1e-10, "expected non-decreasing at {}: {} < {}", i, dist[i], dist[i-1]);
        }
    }

    #[test]
    fn test_granulometry_empty_image() {
        let image: Vec<Vec<f64>> = vec![];
        assert!(granulometry(&image, 5, 5, SEType::Square).is_err());
    }

    #[test]
    fn test_granulometry_zero_image() {
        let image = vec![vec![0.0f64; 10]; 10];
        let result = granulometry(&image, 3, 3, SEType::Square).expect("zero image failed");
        for &ps in &result.pattern_spectrum {
            assert_eq!(ps, 0.0);
        }
    }
}
