//! Watershed segmentation algorithm
//!
//! This module provides the watershed segmentation algorithm for image segmentation.
//! Implements Meyer's flooding algorithm with priority queue, supporting configurable
//! 4-connectivity and 8-connectivity for 2D images.

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::{Array, Array2, Ix2};
use scirs2_core::numeric::{Float, NumAssign};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// Connectivity mode for watershed segmentation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatershedConnectivity {
    /// 4-connectivity: only face-adjacent neighbors (up, down, left, right)
    Four,
    /// 8-connectivity: face-adjacent plus diagonal neighbors
    Eight,
}

impl Default for WatershedConnectivity {
    fn default() -> Self {
        WatershedConnectivity::Eight
    }
}

/// Configuration for watershed segmentation
#[derive(Debug, Clone)]
pub struct WatershedConfig {
    /// Connectivity mode (4 or 8)
    pub connectivity: WatershedConnectivity,
    /// Whether to create watershed lines (barrier pixels with label -1)
    pub watershed_line: bool,
    /// Whether to compact the labels (remove gaps in label numbering)
    pub compact_labels: bool,
}

impl Default for WatershedConfig {
    fn default() -> Self {
        WatershedConfig {
            connectivity: WatershedConnectivity::Eight,
            watershed_line: false,
            compact_labels: false,
        }
    }
}

/// Sentinel value for watershed ridge lines
const WATERSHED_LABEL: i32 = -1;
/// Sentinel for pixels in the queue but not yet labeled
const IN_QUEUE: i32 = -2;

/// Internal priority point for the min-heap
#[derive(Clone, Debug)]
struct QueueEntry {
    /// Row position
    row: usize,
    /// Column position
    col: usize,
    /// Priority value (image intensity); lower = higher priority
    priority: f64,
    /// Insertion order for tie-breaking (FIFO)
    order: u64,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row && self.col == other.col
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse the ordering so BinaryHeap pops smallest first
        // First compare by priority (lower first)
        match other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => {
                // Tie-break by insertion order (FIFO: smaller order = higher priority)
                other.order.cmp(&self.order)
            }
            ord => ord,
        }
    }
}

/// Get neighbor offsets based on connectivity
fn get_offsets(connectivity: WatershedConnectivity) -> &'static [(isize, isize)] {
    match connectivity {
        WatershedConnectivity::Four => &[(-1, 0), (0, -1), (0, 1), (1, 0)],
        WatershedConnectivity::Eight => &[
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ],
    }
}

/// Check if coordinates are in bounds
#[inline]
fn in_bounds(r: isize, c: isize, rows: usize, cols: usize) -> bool {
    r >= 0 && r < rows as isize && c >= 0 && c < cols as isize
}

/// Watershed segmentation for 2D arrays (Meyer's flooding algorithm)
///
/// The watershed algorithm treats the image as a topographic surface,
/// where bright pixels are high and dark pixels are low. It segments
/// the image into catchment basins starting from the given markers.
///
/// This implements Meyer's flooding algorithm using a priority queue,
/// which processes pixels in order of increasing image intensity.
///
/// # Arguments
///
/// * `image` - Input array (intensity/gradient image)
/// * `markers` - Initial markers array (same shape as input, with unique positive values
///   for each region to be segmented, and 0 for unknown regions)
///
/// # Returns
///
/// * Result containing the labeled segmented image
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::segmentation::watershed;
///
/// let image = array![
///     [0.5, 0.6, 0.7],
///     [0.4, 0.1, 0.2],
///     [0.3, 0.4, 0.5],
/// ];
///
/// let markers = array![
///     [0, 0, 0],
///     [0, 1, 0],
///     [0, 0, 2],
/// ];
///
/// let segmented = watershed(&image, &markers).expect("Operation failed");
/// ```
pub fn watershed<T>(
    image: &Array<T, Ix2>,
    markers: &Array<i32, Ix2>,
) -> NdimageResult<Array<i32, Ix2>>
where
    T: Float + NumAssign + std::fmt::Debug + 'static,
{
    watershed_with_config(image, markers, &WatershedConfig::default())
}

/// Marker-controlled watershed for 2D arrays with configurable connectivity
///
/// A variant of the watershed algorithm that uses specified markers
/// as seeds and a gradient image to find the boundaries.
///
/// # Arguments
///
/// * `image` - Input array (intensity image)
/// * `markers` - Initial markers array (same shape as input, with unique positive values
///   for each region to be segmented, and 0 for unknown regions)
/// * `connectivity` - Connectivity for considering neighbors (1: 4-connected, 2: 8-connected)
///
/// # Returns
///
/// * Result containing the labeled segmented image
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::segmentation::marker_watershed;
///
/// let image = array![
///     [0.5, 0.5, 0.5],
///     [0.5, 0.2, 0.5],
///     [0.5, 0.5, 0.5],
/// ];
///
/// let markers = array![
///     [0, 0, 0],
///     [1, 0, 2],
///     [0, 0, 0],
/// ];
///
/// let segmented = marker_watershed(&image, &markers, 1).expect("Operation failed");
/// ```
pub fn marker_watershed<T>(
    image: &Array<T, Ix2>,
    markers: &Array<i32, Ix2>,
    connectivity: usize,
) -> NdimageResult<Array<i32, Ix2>>
where
    T: Float + NumAssign + std::fmt::Debug + 'static,
{
    // Check connectivity
    if connectivity != 1 && connectivity != 2 {
        return Err(NdimageError::InvalidInput(
            "Connectivity must be 1 (4-connected) or 2 (8-connected)".to_string(),
        ));
    }

    let conn = if connectivity == 1 {
        WatershedConnectivity::Four
    } else {
        WatershedConnectivity::Eight
    };

    let config = WatershedConfig {
        connectivity: conn,
        watershed_line: false,
        compact_labels: false,
    };

    watershed_with_config(image, markers, &config)
}

/// Full watershed segmentation with configuration
///
/// # Arguments
///
/// * `image` - Input array (intensity/gradient image)
/// * `markers` - Initial markers array
/// * `config` - Watershed configuration
///
/// # Returns
///
/// * Result containing the labeled segmented image
pub fn watershed_with_config<T>(
    image: &Array<T, Ix2>,
    markers: &Array<i32, Ix2>,
    config: &WatershedConfig,
) -> NdimageResult<Array<i32, Ix2>>
where
    T: Float + NumAssign + std::fmt::Debug + 'static,
{
    // Check shapes match
    if image.shape() != markers.shape() {
        return Err(NdimageError::DimensionError(
            "Input image and markers must have the same shape".to_string(),
        ));
    }

    let rows = image.nrows();
    let cols = image.ncols();

    if rows == 0 || cols == 0 {
        return Ok(markers.clone());
    }

    let offsets = get_offsets(config.connectivity);

    // Initialize output with markers
    let mut output = markers.clone();

    // Create priority queue and insert boundary pixels of each marker region
    let mut queue = BinaryHeap::new();
    let mut insertion_order: u64 = 0;

    // Mark all initial marker pixels and enqueue their unlabeled neighbors
    for r in 0..rows {
        for c in 0..cols {
            let marker = markers[[r, c]];
            if marker > 0 {
                // This pixel is a seed. Check its neighbors.
                for &(dr, dc) in offsets {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;

                    if in_bounds(nr, nc, rows, cols) {
                        let nr = nr as usize;
                        let nc = nc as usize;

                        if output[[nr, nc]] == 0 {
                            // Mark as in-queue so we don't enqueue it again
                            output[[nr, nc]] = IN_QUEUE;
                            let priority = image[[nr, nc]].to_f64().unwrap_or(f64::INFINITY);
                            queue.push(QueueEntry {
                                row: nr,
                                col: nc,
                                priority,
                                order: insertion_order,
                            });
                            insertion_order += 1;
                        }
                    }
                }
            }
        }
    }

    // Meyer's flooding: process pixels in order of increasing intensity
    while let Some(entry) = queue.pop() {
        let r = entry.row;
        let c = entry.col;

        // Collect distinct labels from already-labeled neighbors
        let mut neighbor_labels: HashMap<i32, usize> = HashMap::new();
        let mut _has_watershed_neighbor = false;

        for &(dr, dc) in offsets {
            let nr = r as isize + dr;
            let nc = c as isize + dc;

            if in_bounds(nr, nc, rows, cols) {
                let nr = nr as usize;
                let nc = nc as usize;
                let label = output[[nr, nc]];

                if label > 0 {
                    *neighbor_labels.entry(label).or_insert(0) += 1;
                } else if label == WATERSHED_LABEL {
                    _has_watershed_neighbor = true;
                }
            }
        }

        if neighbor_labels.is_empty() {
            // No labeled neighbors yet; leave as IN_QUEUE (will be revisited if a neighbor gets labeled later)
            // Actually this shouldn't happen in a proper BFS from markers,
            // but reset to 0 so future passes can pick it up
            output[[r, c]] = 0;
            continue;
        }

        // Check if this pixel is on a watershed line
        let distinct_labels: Vec<i32> = neighbor_labels.keys().copied().collect();

        if config.watershed_line && distinct_labels.len() > 1 {
            // Multiple distinct labels: this is a watershed ridge
            output[[r, c]] = WATERSHED_LABEL;
        } else {
            // Assign the most frequent neighbor label
            let best_label = neighbor_labels
                .iter()
                .max_by_key(|&(_, count)| count)
                .map(|(&lbl, _)| lbl)
                .unwrap_or(0);

            if best_label > 0 {
                output[[r, c]] = best_label;
            } else {
                // Fallback: no valid label found
                output[[r, c]] = 0;
                continue;
            }
        }

        // Enqueue unlabeled neighbors
        for &(dr, dc) in offsets {
            let nr = r as isize + dr;
            let nc = c as isize + dc;

            if in_bounds(nr, nc, rows, cols) {
                let nr = nr as usize;
                let nc = nc as usize;

                if output[[nr, nc]] == 0 {
                    output[[nr, nc]] = IN_QUEUE;
                    let priority = image[[nr, nc]].to_f64().unwrap_or(f64::INFINITY);
                    queue.push(QueueEntry {
                        row: nr,
                        col: nc,
                        priority,
                        order: insertion_order,
                    });
                    insertion_order += 1;
                }
            }
        }
    }

    // Clean up: any pixel still marked IN_QUEUE should be set to 0
    for val in output.iter_mut() {
        if *val == IN_QUEUE {
            *val = 0;
        }
    }

    Ok(output)
}

/// Automatic watershed segmentation from distance transform
///
/// Computes markers automatically from local maxima of the distance transform
/// of a binary image, then applies watershed on the negative distance transform.
///
/// # Arguments
///
/// * `binary_image` - Input binary image (true = foreground)
/// * `connectivity` - Connectivity mode for watershed
/// * `min_distance` - Minimum distance between markers (in pixels)
///
/// # Returns
///
/// * Result containing the labeled segmented image
pub fn watershed_from_distance<T>(
    binary_image: &Array2<bool>,
    connectivity: WatershedConnectivity,
    min_distance: usize,
) -> NdimageResult<Array2<i32>>
where
    T: Float + NumAssign + std::fmt::Debug + 'static,
{
    let rows = binary_image.nrows();
    let cols = binary_image.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    // Compute distance transform (simple city-block for efficiency)
    let mut distance = Array2::<f64>::zeros((rows, cols));
    // Forward pass
    for r in 0..rows {
        for c in 0..cols {
            if binary_image[[r, c]] {
                let mut d = f64::INFINITY;
                if r > 0 {
                    let above = distance[[r - 1, c]];
                    if above + 1.0 < d {
                        d = above + 1.0;
                    }
                }
                if c > 0 {
                    let left = distance[[r, c - 1]];
                    if left + 1.0 < d {
                        d = left + 1.0;
                    }
                }
                if !d.is_finite() {
                    d = (rows + cols) as f64; // large sentinel
                }
                distance[[r, c]] = d;
            }
            // else distance stays 0 for background
        }
    }
    // Backward pass
    for r in (0..rows).rev() {
        for c in (0..cols).rev() {
            if binary_image[[r, c]] {
                if r + 1 < rows {
                    let below = distance[[r + 1, c]] + 1.0;
                    if below < distance[[r, c]] {
                        distance[[r, c]] = below;
                    }
                }
                if c + 1 < cols {
                    let right = distance[[r, c + 1]] + 1.0;
                    if right < distance[[r, c]] {
                        distance[[r, c]] = right;
                    }
                }
            }
        }
    }

    // Find local maxima of distance as markers
    let offsets = get_offsets(connectivity);
    let mut markers = Array2::<i32>::zeros((rows, cols));
    let mut next_label = 1i32;
    let min_dist_f = min_distance as f64;

    for r in 0..rows {
        for c in 0..cols {
            if !binary_image[[r, c]] {
                continue;
            }
            let val = distance[[r, c]];
            if val < min_dist_f {
                continue;
            }

            let mut is_max = true;
            for &(dr, dc) in offsets {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if in_bounds(nr, nc, rows, cols) {
                    if distance[[nr as usize, nc as usize]] > val {
                        is_max = false;
                        break;
                    }
                }
            }

            if is_max {
                markers[[r, c]] = next_label;
                next_label += 1;
            }
        }
    }

    // Create negative distance transform as the "image" for watershed
    let neg_distance: Array2<f64> = distance.mapv(|v| -v);

    // Apply watershed
    let config = WatershedConfig {
        connectivity,
        watershed_line: false,
        compact_labels: false,
    };

    let mut result = watershed_with_config(&neg_distance, &markers, &config)?;

    // Mask out background
    for r in 0..rows {
        for c in 0..cols {
            if !binary_image[[r, c]] {
                result[[r, c]] = 0;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_watershed_basic() {
        let image = array![[0.5, 0.6, 0.7], [0.4, 0.1, 0.2], [0.3, 0.4, 0.5],];
        let markers = array![[0, 0, 0], [0, 1, 0], [0, 0, 2],];

        let result = watershed(&image, &markers).expect("watershed should succeed");
        // All pixels should be labeled (no zeros in a connected image)
        // The marker 1 at (1,1) and marker 2 at (2,2) should flood outward
        assert_eq!(result[[1, 1]], 1);
        assert_eq!(result[[2, 2]], 2);
    }

    #[test]
    fn test_watershed_shape_mismatch() {
        let image = array![[0.5, 0.6], [0.4, 0.1],];
        let markers = array![[0, 0, 0], [0, 1, 0],];

        let result = watershed(&image, &markers);
        assert!(result.is_err());
    }

    #[test]
    fn test_marker_watershed_4_connectivity() {
        // Create a simple gradient image with two basins
        let image = array![
            [5.0, 5.0, 9.0, 5.0, 5.0],
            [5.0, 3.0, 9.0, 3.0, 5.0],
            [5.0, 1.0, 9.0, 1.0, 5.0],
            [5.0, 3.0, 9.0, 3.0, 5.0],
            [5.0, 5.0, 9.0, 5.0, 5.0],
        ];
        let markers = array![
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ];

        let result =
            marker_watershed(&image, &markers, 1).expect("marker_watershed should succeed");
        // With 4-connectivity and a high ridge at column 2, basin 1 and 2 should separate
        assert_eq!(result[[2, 1]], 1);
        assert_eq!(result[[2, 3]], 2);
    }

    #[test]
    fn test_marker_watershed_8_connectivity() {
        let image = array![
            [5.0, 5.0, 9.0, 5.0, 5.0],
            [5.0, 3.0, 9.0, 3.0, 5.0],
            [5.0, 1.0, 9.0, 1.0, 5.0],
            [5.0, 3.0, 9.0, 3.0, 5.0],
            [5.0, 5.0, 9.0, 5.0, 5.0],
        ];
        let markers = array![
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ];

        let result =
            marker_watershed(&image, &markers, 2).expect("marker_watershed should succeed");
        assert_eq!(result[[2, 1]], 1);
        assert_eq!(result[[2, 3]], 2);
    }

    #[test]
    fn test_marker_watershed_invalid_connectivity() {
        let image = array![[1.0, 2.0], [3.0, 4.0],];
        let markers = array![[1, 0], [0, 2],];

        let result = marker_watershed(&image, &markers, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_watershed_with_config_watershed_line() {
        // Two basins separated by a high ridge
        let image = array![
            [1.0, 2.0, 9.0, 2.0, 1.0],
            [1.0, 2.0, 9.0, 2.0, 1.0],
            [1.0, 2.0, 9.0, 2.0, 1.0],
        ];
        let markers = array![[1, 0, 0, 0, 2], [0, 0, 0, 0, 0], [1, 0, 0, 0, 2],];

        let config = WatershedConfig {
            connectivity: WatershedConnectivity::Four,
            watershed_line: true,
            compact_labels: false,
        };

        let result = watershed_with_config(&image, &markers, &config)
            .expect("watershed with line should succeed");

        // Markers should remain
        assert_eq!(result[[0, 0]], 1);
        assert_eq!(result[[0, 4]], 2);
        // The high ridge should become watershed line or stay with one label
        // depending on the flooding dynamics
    }

    #[test]
    fn test_watershed_single_marker_floods_all() {
        // A single marker should flood the entire image
        let image = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],];
        let markers = array![[1, 0, 0], [0, 0, 0], [0, 0, 0],];

        let result = watershed(&image, &markers).expect("watershed should succeed");
        // All pixels should be labeled 1
        for val in result.iter() {
            assert_eq!(*val, 1);
        }
    }

    #[test]
    fn test_watershed_empty_image() {
        let image: Array2<f64> = Array2::zeros((0, 0));
        let markers: Array2<i32> = Array2::zeros((0, 0));
        let result = watershed(&image, &markers).expect("empty should succeed");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_watershed_all_foreground_markers() {
        // If all pixels are markers, output should be the same as markers
        let image = array![[1.0, 2.0], [3.0, 4.0],];
        let markers = array![[1, 2], [3, 4],];

        let result = watershed(&image, &markers).expect("watershed should succeed");
        assert_eq!(result, markers);
    }

    #[test]
    fn test_watershed_from_distance_basic() {
        // Simple binary image with two distinct objects
        let binary = array![
            [true, true, true, false, false, true, true, true],
            [true, true, true, false, false, true, true, true],
            [true, true, true, false, false, true, true, true],
            [false, false, false, false, false, false, false, false],
            [true, true, true, false, false, true, true, true],
            [true, true, true, false, false, true, true, true],
            [true, true, true, false, false, true, true, true],
        ];

        let result = watershed_from_distance::<f64>(&binary, WatershedConnectivity::Four, 1)
            .expect("watershed_from_distance should succeed");

        // Background pixels should be 0
        assert_eq!(result[[0, 3]], 0);
        assert_eq!(result[[3, 0]], 0);
    }
}
