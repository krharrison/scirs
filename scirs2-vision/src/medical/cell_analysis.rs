//! Cell detection and counting for fluorescence and brightfield microscopy images.
//!
//! This module provides:
//!
//! - [`Cell`] – descriptor of a detected cell (centre, size, shape metrics)
//! - [`detect_cells_log`] – multi-scale Laplacian-of-Gaussian blob detector
//! - [`count_cells_threshold`] – threshold + connected-component cell counter
//! - [`segment_cells_watershed`] – marker-controlled watershed cell segmentation
//! - [`analyze_cell_shapes`] – shape and intensity statistics from a label image
//! - [`nuclear_cytoplasm_ratio`] – N/C ratio from nuclear and cell masks

use crate::error::VisionError;
use scirs2_core::ndarray::{Array2, ArrayView2};
use std::collections::VecDeque;

// ── Cell descriptor ───────────────────────────────────────────────────────────

/// Descriptor for a single detected cell.
#[derive(Debug, Clone)]
pub struct Cell {
    /// Centroid position `(row, col)` in pixels.
    pub center: (f64, f64),
    /// Estimated radius in pixels.
    pub radius: f64,
    /// Pixel area of the cell.
    pub area: f64,
    /// Shape circularity: `4π × area / perimeter²` (1.0 = perfect circle).
    pub circularity: f64,
    /// Mean pixel intensity over the cell region.
    pub mean_intensity: f64,
    /// Perimeter length in pixels.
    pub perimeter: f64,
}

// ── LoG blob detector ─────────────────────────────────────────────────────────

/// Detect circular cell blobs using a multi-scale Laplacian-of-Gaussian (LoG) detector.
///
/// The algorithm:
/// 1. Build a scale-space stack by convolving the image with Gaussian kernels at
///    `num_sigma` logarithmically-spaced scales between `min_sigma` and `max_sigma`.
/// 2. Compute the Laplacian (finite-difference approximation) at each scale and
///    scale-normalise by σ².
/// 3. Find local minima in the 3-D (row × col × scale) stack that are below
///    `−threshold`.
/// 4. Convert scale → radius: `radius = σ × √2`.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] if parameters are out of range.
pub fn detect_cells_log(
    image: ArrayView2<f64>,
    min_sigma: f64,
    max_sigma: f64,
    num_sigma: usize,
    threshold: f64,
) -> Result<Vec<Cell>, VisionError> {
    if min_sigma <= 0.0 || max_sigma <= 0.0 || min_sigma > max_sigma {
        return Err(VisionError::InvalidParameter(
            "min_sigma and max_sigma must be positive and min_sigma <= max_sigma".to_string(),
        ));
    }
    if num_sigma < 1 {
        return Err(VisionError::InvalidParameter(
            "num_sigma must be at least 1".to_string(),
        ));
    }

    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Ok(Vec::new());
    }

    // Build logarithmically-spaced sigma values
    let sigmas: Vec<f64> = if num_sigma == 1 {
        vec![min_sigma]
    } else {
        let log_min = min_sigma.ln();
        let log_max = max_sigma.ln();
        (0..num_sigma)
            .map(|i| {
                let t = i as f64 / (num_sigma - 1) as f64;
                (log_min + t * (log_max - log_min)).exp()
            })
            .collect()
    };

    // Scale-space stack: responses[scale][row][col]
    let mut responses: Vec<Array2<f64>> = Vec::with_capacity(num_sigma);
    for &sigma in &sigmas {
        let blurred = gaussian_blur_2d(image, sigma);
        let laplacian = laplacian_2d(blurred.view());
        // Scale normalisation: σ² * ΔL (sign-inverted to find bright blobs as maxima)
        let normalised = laplacian.mapv(|v| -v * sigma * sigma);
        responses.push(normalised);
    }

    // Find local maxima in scale-space (3-D non-maximum suppression)
    let mut cells = Vec::new();
    for s in 0..num_sigma {
        for r in 1..(rows.saturating_sub(1)) {
            for c in 1..(cols.saturating_sub(1)) {
                let val = responses[s][[r, c]];
                if val < threshold {
                    continue;
                }
                // Check spatial neighbours at same scale
                let mut is_max = true;
                'spatial: for dr in -1_i32..=1 {
                    for dc in -1_i32..=1 {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = (r as i32 + dr) as usize;
                        let nc = (c as i32 + dc) as usize;
                        if responses[s][[nr, nc]] > val {
                            is_max = false;
                            break 'spatial;
                        }
                    }
                }
                // Check scale neighbours
                if is_max {
                    if s > 0 && responses[s - 1][[r, c]] > val {
                        is_max = false;
                    }
                    if is_max && s + 1 < num_sigma && responses[s + 1][[r, c]] > val {
                        is_max = false;
                    }
                }
                if is_max {
                    let radius = sigmas[s] * std::f64::consts::SQRT_2;
                    cells.push(Cell {
                        center: (r as f64, c as f64),
                        radius,
                        area: std::f64::consts::PI * radius * radius,
                        circularity: 1.0,
                        mean_intensity: image[[r, c]],
                        perimeter: 2.0 * std::f64::consts::PI * radius,
                    });
                }
            }
        }
    }

    Ok(cells)
}

// ── Threshold-based cell counting ─────────────────────────────────────────────

/// Count cells by thresholding the image and extracting connected components.
///
/// Returns `(count, cells)` where `count` equals `cells.len()`.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] if `min_area > max_area` or `threshold` is NaN.
pub fn count_cells_threshold(
    image: ArrayView2<f64>,
    threshold: f64,
    min_area: f64,
    max_area: f64,
) -> Result<(usize, Vec<Cell>), VisionError> {
    if min_area > max_area {
        return Err(VisionError::InvalidParameter(
            "min_area must be <= max_area".to_string(),
        ));
    }
    if threshold.is_nan() {
        return Err(VisionError::InvalidParameter(
            "threshold must not be NaN".to_string(),
        ));
    }

    let (rows, cols) = image.dim();
    let binary: Array2<bool> = image.mapv(|v| v > threshold);

    let (labels, n_labels) = label_connected_components(&binary);
    let cells = region_props(&labels, image, n_labels, min_area, max_area);
    let count = cells.len();
    Ok((count, cells))
}

// ── Watershed cell segmentation ───────────────────────────────────────────────

/// Segment touching cells using a distance-transform + marker watershed approach.
///
/// Returns a label image where background pixels are −1 and cell pixels carry
/// a non-negative integer label.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] for zero-sized images.
pub fn segment_cells_watershed(
    image: ArrayView2<f64>,
    min_distance: usize,
) -> Result<Array2<i32>, VisionError> {
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Err(VisionError::InvalidParameter(
            "Image must be non-empty".to_string(),
        ));
    }

    // Step 1 – threshold to get foreground
    let max_val = image.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let threshold = if max_val > 0.0 { max_val * 0.3 } else { 0.1 };
    let binary: Array2<bool> = image.mapv(|v| v > threshold);

    // Step 2 – Euclidean distance transform of the foreground
    let dist = distance_transform(&binary);

    // Step 3 – find local maxima in distance map (cell centres)
    let markers = find_local_maxima_with_suppression(&dist, min_distance);

    // Step 4 – BFS-based flood-fill from markers
    let mut labels = Array2::<i32>::from_elem((rows, cols), -1);
    let mut queue: VecDeque<(usize, usize, i32)> = VecDeque::new();

    for (idx, &(mr, mc)) in markers.iter().enumerate() {
        if binary[[mr, mc]] {
            labels[[mr, mc]] = idx as i32;
            queue.push_back((mr, mc, idx as i32));
        }
    }

    while let Some((r, c, lbl)) = queue.pop_front() {
        for (nr, nc) in neighbours_4(r, c, rows, cols) {
            if binary[[nr, nc]] && labels[[nr, nc]] == -1 {
                labels[[nr, nc]] = lbl;
                queue.push_back((nr, nc, lbl));
            }
        }
    }

    Ok(labels)
}

// ── Shape analysis ────────────────────────────────────────────────────────────

/// Compute shape and intensity statistics for each label in a segmented image.
pub fn analyze_cell_shapes(
    label_image: &Array2<i32>,
    intensity_image: ArrayView2<f64>,
) -> Vec<Cell> {
    let (rows, cols) = label_image.dim();

    // Group pixels by label
    let mut label_pixels: std::collections::HashMap<i32, Vec<(usize, usize, f64)>> =
        std::collections::HashMap::new();
    for r in 0..rows {
        for c in 0..cols {
            let l = label_image[[r, c]];
            if l >= 0 {
                let intensity = intensity_image[[r, c]];
                label_pixels.entry(l).or_default().push((r, c, intensity));
            }
        }
    }

    let mut cells = Vec::new();
    for pixels in label_pixels.values() {
        let area = pixels.len() as f64;
        if area == 0.0 {
            continue;
        }
        let sum_r: f64 = pixels.iter().map(|&(r, _, _)| r as f64).sum();
        let sum_c: f64 = pixels.iter().map(|&(_, c, _)| c as f64).sum();
        let center = (sum_r / area, sum_c / area);

        let mean_intensity = pixels.iter().map(|&(_, _, i)| i).sum::<f64>() / area;

        // Perimeter: count boundary pixels (those on the image border or with
        // at least one background/out-of-set neighbour).
        let pixel_set: std::collections::HashSet<(usize, usize)> =
            pixels.iter().map(|&(r, c, _)| (r, c)).collect();
        let perimeter = pixels
            .iter()
            .filter(|&&(r, c, _)| {
                let nbrs = neighbours_4(r, c, rows, cols);
                nbrs.len() < 4 || nbrs.iter().any(|nb| !pixel_set.contains(nb))
            })
            .count() as f64;

        let circularity = if perimeter > 0.0 {
            (4.0 * std::f64::consts::PI * area / (perimeter * perimeter)).min(1.0)
        } else {
            1.0
        };
        let radius = (area / std::f64::consts::PI).sqrt();

        cells.push(Cell {
            center,
            radius,
            area,
            circularity,
            mean_intensity,
            perimeter,
        });
    }

    cells
}

// ── Nuclear-to-cytoplasm ratio ────────────────────────────────────────────────

/// Estimate the nuclear-to-cytoplasm (N/C) area ratio.
///
/// Cytoplasm area is computed as `cell_area − nuclear_area`.  Returns 0.0 when
/// the cytoplasm area is zero to avoid division by zero.
pub fn nuclear_cytoplasm_ratio(nuclear_mask: &Array2<bool>, cell_mask: &Array2<bool>) -> f64 {
    let nuclear_area = nuclear_mask.iter().filter(|&&v| v).count() as f64;
    let cell_area = cell_mask.iter().filter(|&&v| v).count() as f64;
    let cytoplasm_area = (cell_area - nuclear_area).max(0.0);
    if cytoplasm_area == 0.0 {
        0.0
    } else {
        nuclear_area / cytoplasm_area
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn neighbours_4(r: usize, c: usize, rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let mut n = Vec::with_capacity(4);
    if r > 0 {
        n.push((r - 1, c));
    }
    if r + 1 < rows {
        n.push((r + 1, c));
    }
    if c > 0 {
        n.push((r, c - 1));
    }
    if c + 1 < cols {
        n.push((r, c + 1));
    }
    n
}

/// 4-connectivity connected-component labelling.
/// Returns `(label_image, n_labels)` where labels are 0-based.
fn label_connected_components(binary: &Array2<bool>) -> (Array2<i32>, usize) {
    let (rows, cols) = binary.dim();
    let mut labels = Array2::<i32>::from_elem((rows, cols), -1);
    let mut n = 0i32;
    for sr in 0..rows {
        for sc in 0..cols {
            if binary[[sr, sc]] && labels[[sr, sc]] == -1 {
                let mut queue = VecDeque::new();
                queue.push_back((sr, sc));
                labels[[sr, sc]] = n;
                while let Some((r, c)) = queue.pop_front() {
                    for (nr, nc) in neighbours_4(r, c, rows, cols) {
                        if binary[[nr, nc]] && labels[[nr, nc]] == -1 {
                            labels[[nr, nc]] = n;
                            queue.push_back((nr, nc));
                        }
                    }
                }
                n += 1;
            }
        }
    }
    (labels, n as usize)
}

/// Extract region properties for labels whose area is within `[min_area, max_area]`.
fn region_props(
    labels: &Array2<i32>,
    intensity: ArrayView2<f64>,
    n_labels: usize,
    min_area: f64,
    max_area: f64,
) -> Vec<Cell> {
    let (rows, cols) = labels.dim();
    let mut buckets: Vec<Vec<(usize, usize, f64)>> = vec![Vec::new(); n_labels];
    for r in 0..rows {
        for c in 0..cols {
            let l = labels[[r, c]];
            if l >= 0 {
                buckets[l as usize].push((r, c, intensity[[r, c]]));
            }
        }
    }

    let mut cells = Vec::new();
    for pixels in buckets {
        let area = pixels.len() as f64;
        if area < min_area || area > max_area {
            continue;
        }
        let sum_r: f64 = pixels.iter().map(|&(r, _, _)| r as f64).sum();
        let sum_c: f64 = pixels.iter().map(|&(_, c, _)| c as f64).sum();
        let center = (sum_r / area, sum_c / area);
        let mean_intensity = pixels.iter().map(|&(_, _, i)| i).sum::<f64>() / area;
        let pixel_set: std::collections::HashSet<(usize, usize)> =
            pixels.iter().map(|&(r, c, _)| (r, c)).collect();
        let perimeter = pixels
            .iter()
            .filter(|&&(r, c, _)| {
                neighbours_4(r, c, rows, cols)
                    .iter()
                    .any(|nb| !pixel_set.contains(nb))
            })
            .count() as f64;
        let circularity = if perimeter > 0.0 {
            4.0 * std::f64::consts::PI * area / (perimeter * perimeter)
        } else {
            1.0
        };
        let radius = (area / std::f64::consts::PI).sqrt();
        cells.push(Cell {
            center,
            radius,
            area,
            circularity,
            mean_intensity,
            perimeter,
        });
    }
    cells
}

/// Separable Gaussian blur using a 1-D kernel.
pub(crate) fn gaussian_blur_2d(image: ArrayView2<f64>, sigma: f64) -> Array2<f64> {
    let (rows, cols) = image.dim();
    let radius = (3.0 * sigma).ceil() as usize;
    let kernel: Vec<f64> = {
        let mut k: Vec<f64> = (-(radius as i64)..=radius as i64)
            .map(|x| (-(x as f64 * x as f64) / (2.0 * sigma * sigma)).exp())
            .collect();
        let s: f64 = k.iter().sum();
        k.iter_mut().for_each(|v| *v /= s);
        k
    };

    // Row pass
    let mut row_pass = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let offset = ki as i64 - radius as i64;
                let nc = c as i64 + offset;
                let nc = nc.clamp(0, cols as i64 - 1) as usize;
                acc += image[[r, nc]] * kv;
            }
            row_pass[[r, c]] = acc;
        }
    }

    // Column pass
    let mut col_pass = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for (ki, &kv) in kernel.iter().enumerate() {
                let offset = ki as i64 - radius as i64;
                let nr = r as i64 + offset;
                let nr = nr.clamp(0, rows as i64 - 1) as usize;
                acc += row_pass[[nr, c]] * kv;
            }
            col_pass[[r, c]] = acc;
        }
    }
    col_pass
}

/// Discrete Laplacian (5-point stencil: finite differences).
fn laplacian_2d(image: ArrayView2<f64>) -> Array2<f64> {
    let (rows, cols) = image.dim();
    let mut out = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let v = image[[r, c]];
            let up = if r > 0 { image[[r - 1, c]] } else { v };
            let dn = if r + 1 < rows { image[[r + 1, c]] } else { v };
            let lt = if c > 0 { image[[r, c - 1]] } else { v };
            let rt = if c + 1 < cols { image[[r, c + 1]] } else { v };
            out[[r, c]] = up + dn + lt + rt - 4.0 * v;
        }
    }
    out
}

/// Approximate Euclidean distance transform via iterative chamfer passes.
fn distance_transform(binary: &Array2<bool>) -> Array2<f64> {
    let (rows, cols) = binary.dim();
    let inf = (rows + cols) as f64 * 10.0;
    let mut dist = Array2::from_elem((rows, cols), inf);

    // Initialise background pixels to 0 (foreground stays at inf).
    // The result gives each foreground pixel the distance to the nearest
    // background pixel, which is useful for finding cell centres.
    for r in 0..rows {
        for c in 0..cols {
            if !binary[[r, c]] {
                dist[[r, c]] = 0.0;
            }
        }
    }

    // Forward pass
    for r in 0..rows {
        for c in 0..cols {
            let d = dist[[r, c]];
            if r > 0 && dist[[r - 1, c]] + 1.0 < d {
                dist[[r, c]] = dist[[r - 1, c]] + 1.0;
            }
            if c > 0 && dist[[r, c - 1]] + 1.0 < dist[[r, c]] {
                dist[[r, c]] = dist[[r, c - 1]] + 1.0;
            }
        }
    }
    // Backward pass
    for r in (0..rows).rev() {
        for c in (0..cols).rev() {
            if r + 1 < rows && dist[[r + 1, c]] + 1.0 < dist[[r, c]] {
                dist[[r, c]] = dist[[r + 1, c]] + 1.0;
            }
            if c + 1 < cols && dist[[r, c + 1]] + 1.0 < dist[[r, c]] {
                dist[[r, c]] = dist[[r, c + 1]] + 1.0;
            }
        }
    }
    dist
}

/// Find local maxima in a 2-D image with non-maximum suppression radius `min_distance`.
fn find_local_maxima_with_suppression(
    image: &Array2<f64>,
    min_distance: usize,
) -> Vec<(usize, usize)> {
    let (rows, cols) = image.dim();
    let md = min_distance.max(1);
    let mut maxima = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let val = image[[r, c]];
            if val <= 0.0 {
                continue;
            }
            let mut is_max = true;
            'outer: for dr in -(md as i64)..=(md as i64) {
                for dc in -(md as i64)..=(md as i64) {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = r as i64 + dr;
                    let nc = c as i64 + dc;
                    if nr < 0 || nr >= rows as i64 || nc < 0 || nc >= cols as i64 {
                        continue;
                    }
                    if image[[nr as usize, nc as usize]] > val {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max {
                maxima.push((r, c));
            }
        }
    }
    maxima
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    // ── detect_cells_log ─────────────────────────────────────────────────────

    #[test]
    fn test_detect_cells_log_invalid_sigma() {
        let img = Array2::zeros((10, 10));
        assert!(detect_cells_log(img.view(), -1.0, 3.0, 3, 0.01).is_err());
        assert!(detect_cells_log(img.view(), 3.0, 1.0, 3, 0.01).is_err());
    }

    #[test]
    fn test_detect_cells_log_invalid_num_sigma() {
        let img = Array2::zeros((10, 10));
        assert!(detect_cells_log(img.view(), 1.0, 3.0, 0, 0.01).is_err());
    }

    #[test]
    fn test_detect_cells_log_empty_image() {
        let img: Array2<f64> = Array2::zeros((0, 0));
        let cells = detect_cells_log(img.view(), 1.0, 3.0, 3, 0.1).expect("Should return empty");
        assert!(cells.is_empty());
    }

    #[test]
    fn test_detect_cells_log_single_blob() {
        // Place a bright circular blob at (10, 10) in a 20×20 image
        let mut img = Array2::<f64>::zeros((20, 20));
        for dr in -2_i32..=2 {
            for dc in -2_i32..=2 {
                if dr * dr + dc * dc <= 4 {
                    let r = (10 + dr) as usize;
                    let c = (10 + dc) as usize;
                    img[[r, c]] = 1.0;
                }
            }
        }
        let cells = detect_cells_log(img.view(), 0.5, 3.0, 5, 0.01).expect("Should succeed");
        // We expect at least one detection near the blob
        assert!(!cells.is_empty(), "Expected at least one blob detection");
        let closest = cells.iter().min_by(|a, b| {
            let da = (a.center.0 - 10.0).hypot(a.center.1 - 10.0);
            let db = (b.center.0 - 10.0).hypot(b.center.1 - 10.0);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        let closest = closest.expect("Non-empty");
        let dist = (closest.center.0 - 10.0).hypot(closest.center.1 - 10.0);
        assert!(
            dist < 4.0,
            "Closest detection too far from blob center: {dist}"
        );
    }

    // ── count_cells_threshold ────────────────────────────────────────────────

    #[test]
    fn test_count_cells_threshold_invalid_params() {
        let img = Array2::zeros((10, 10));
        assert!(count_cells_threshold(img.view(), 0.5, 10.0, 5.0).is_err()); // min > max
        assert!(count_cells_threshold(img.view(), f64::NAN, 1.0, 100.0).is_err());
    }

    #[test]
    fn test_count_cells_threshold_zero_cells() {
        let img = Array2::from_elem((10, 10), 0.0_f64);
        let (count, cells) =
            count_cells_threshold(img.view(), 0.5, 1.0, 100.0).expect("Should succeed");
        assert_eq!(count, 0);
        assert!(cells.is_empty());
    }

    #[test]
    fn test_count_cells_threshold_single_cell() {
        // 5×5 bright square in a 20×20 image
        let mut img = Array2::<f64>::zeros((20, 20));
        for r in 7..12 {
            for c in 7..12 {
                img[[r, c]] = 1.0;
            }
        }
        let (count, cells) =
            count_cells_threshold(img.view(), 0.5, 10.0, 200.0).expect("Should succeed");
        assert_eq!(count, 1);
        assert_eq!(cells.len(), 1);
        let cell = &cells[0];
        // Centroid should be near (9, 9)
        assert!(
            (cell.center.0 - 9.0).abs() < 1.5,
            "row centroid: {}",
            cell.center.0
        );
        assert!(
            (cell.center.1 - 9.0).abs() < 1.5,
            "col centroid: {}",
            cell.center.1
        );
    }

    #[test]
    fn test_count_cells_threshold_multiple_cells() {
        let mut img = Array2::<f64>::zeros((40, 40));
        // Two separate 3×3 blobs
        for r in 5..8 {
            for c in 5..8 {
                img[[r, c]] = 1.0;
            }
        }
        for r in 30..33 {
            for c in 30..33 {
                img[[r, c]] = 1.0;
            }
        }
        let (count, _) = count_cells_threshold(img.view(), 0.5, 5.0, 50.0).expect("Should succeed");
        assert_eq!(count, 2, "Expected 2 cells, got {count}");
    }

    // ── segment_cells_watershed ───────────────────────────────────────────────

    #[test]
    fn test_segment_cells_watershed_empty_error() {
        let img: Array2<f64> = Array2::zeros((0, 0));
        assert!(segment_cells_watershed(img.view(), 3).is_err());
    }

    #[test]
    fn test_segment_cells_watershed_returns_label_image() {
        let mut img = Array2::<f64>::zeros((20, 20));
        for r in 5..10 {
            for c in 5..10 {
                img[[r, c]] = 1.0;
            }
        }
        let labels = segment_cells_watershed(img.view(), 3).expect("Should succeed");
        assert_eq!(labels.dim(), (20, 20));
        // Background pixels should be -1
        assert_eq!(labels[[0, 0]], -1);
        // Interior pixels should have a valid label
        let interior_label = labels[[7, 7]];
        assert!(
            interior_label >= 0,
            "interior should have non-negative label"
        );
    }

    // ── analyze_cell_shapes ───────────────────────────────────────────────────

    #[test]
    fn test_analyze_cell_shapes_empty_label() {
        let labels = Array2::from_elem((10, 10), -1_i32);
        let intensity = Array2::from_elem((10, 10), 0.5_f64);
        let cells = analyze_cell_shapes(&labels, intensity.view());
        assert!(cells.is_empty());
    }

    #[test]
    fn test_analyze_cell_shapes_single_cell() {
        let mut labels = Array2::from_elem((10, 10), -1_i32);
        let mut intensity = Array2::from_elem((10, 10), 0.0_f64);
        // 3×3 cell at top-left
        for r in 0..3 {
            for c in 0..3 {
                labels[[r, c]] = 0;
                intensity[[r, c]] = 2.0;
            }
        }
        let cells = analyze_cell_shapes(&labels, intensity.view());
        assert_eq!(cells.len(), 1);
        let cell = &cells[0];
        assert!((cell.area - 9.0).abs() < 1e-10, "area={}", cell.area);
        assert!((cell.mean_intensity - 2.0).abs() < 1e-10);
        assert!(cell.circularity > 0.0 && cell.circularity <= 1.0 + 1e-6);
    }

    // ── nuclear_cytoplasm_ratio ───────────────────────────────────────────────

    #[test]
    fn test_nc_ratio_basic() {
        let mut nuclear = Array2::from_elem((10, 10), false);
        let mut cell = Array2::from_elem((10, 10), false);
        // Nuclear 2×2, cell 4×4
        for r in 0..2 {
            for c in 0..2 {
                nuclear[[r, c]] = true;
            }
        }
        for r in 0..4 {
            for c in 0..4 {
                cell[[r, c]] = true;
            }
        }
        let ratio = nuclear_cytoplasm_ratio(&nuclear, &cell);
        // nuclear=4, cell=16, cytoplasm=12, ratio=4/12=1/3
        assert!((ratio - 1.0 / 3.0).abs() < 1e-10, "ratio={ratio}");
    }

    #[test]
    fn test_nc_ratio_zero_cytoplasm() {
        let mask = Array2::from_elem((5, 5), true);
        // Same mask for both → cytoplasm area = 0
        let ratio = nuclear_cytoplasm_ratio(&mask, &mask);
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn test_nc_ratio_no_nuclear() {
        let nuclear = Array2::from_elem((5, 5), false);
        let cell = Array2::from_elem((5, 5), true);
        let ratio = nuclear_cytoplasm_ratio(&nuclear, &cell);
        assert_eq!(ratio, 0.0);
    }
}
