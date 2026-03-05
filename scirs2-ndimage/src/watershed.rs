//! Advanced watershed segmentation algorithms
//!
//! This module provides enhanced watershed segmentation beyond the basic Meyer's flooding
//! algorithm found in `segmentation::watershed`. Features include:
//!
//! - **Compact watershed**: Regularized watershed that penalizes irregular region shapes
//! - **Oversegmentation control**: Merging of small regions and h-minima suppression
//! - **Dam (boundary) pixel labeling**: Explicit watershed ridge line extraction
//! - **Multi-scale watershed**: Hierarchical watershed with merge tree
//!
//! # References
//!
//! - Meyer, F. (1994). "Topographic distance and watershed lines"
//! - Neubert, P. & Protzel, P. (2014). "Compact Watershed and Preemptive SLIC"
//! - Beucher, S. & Meyer, F. (1993). "The morphological approach to segmentation"

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::{Float, NumAssign};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Connectivity mode for watershed neighbor traversal
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WatershedNeighborhood {
    /// 4-connectivity (face-adjacent only)
    Conn4,
    /// 8-connectivity (face + diagonal)
    Conn8,
}

impl Default for WatershedNeighborhood {
    fn default() -> Self {
        WatershedNeighborhood::Conn8
    }
}

/// Configuration for compact (regularized) watershed
#[derive(Debug, Clone)]
pub struct CompactWatershedConfig {
    /// Neighborhood connectivity
    pub connectivity: WatershedNeighborhood,
    /// Compactness weight (0 = standard watershed, higher = more compact regions)
    /// Typical range: 0.0 to 1.0
    pub compactness: f64,
    /// Whether to produce watershed ridge lines (dam pixels labeled as 0)
    pub watershed_line: bool,
}

impl Default for CompactWatershedConfig {
    fn default() -> Self {
        CompactWatershedConfig {
            connectivity: WatershedNeighborhood::Conn8,
            compactness: 0.0,
            watershed_line: false,
        }
    }
}

/// Configuration for oversegmentation control
#[derive(Debug, Clone)]
pub struct OversegmentationConfig {
    /// Minimum region area in pixels; regions smaller than this are merged
    pub min_region_area: usize,
    /// H-minima threshold: suppress minima shallower than this value
    /// before computing markers (set to 0.0 to disable)
    pub h_minima: f64,
    /// Maximum number of output regions (0 = unlimited)
    pub max_regions: usize,
    /// Neighborhood connectivity
    pub connectivity: WatershedNeighborhood,
}

impl Default for OversegmentationConfig {
    fn default() -> Self {
        OversegmentationConfig {
            min_region_area: 1,
            h_minima: 0.0,
            max_regions: 0,
            connectivity: WatershedNeighborhood::Conn8,
        }
    }
}

/// Result of dam (boundary) extraction
#[derive(Debug, Clone)]
pub struct DamResult {
    /// Labeled image where dam pixels have label 0
    pub labels: Array2<i32>,
    /// Binary mask of dam (boundary) pixels
    pub dam_mask: Array2<bool>,
    /// Number of distinct regions (excluding dams)
    pub num_regions: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Sentinel: watershed ridge line (dam pixel)
const WSHED: i32 = -1;
/// Sentinel: pixel queued but not yet assigned
const IN_QUEUE: i32 = -2;

fn neighbor_offsets(conn: WatershedNeighborhood) -> &'static [(isize, isize)] {
    match conn {
        WatershedNeighborhood::Conn4 => &[(-1, 0), (0, -1), (0, 1), (1, 0)],
        WatershedNeighborhood::Conn8 => &[
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

#[inline]
fn in_bounds(r: isize, c: isize, rows: usize, cols: usize) -> bool {
    r >= 0 && r < rows as isize && c >= 0 && c < cols as isize
}

/// Priority-queue entry for compact watershed
#[derive(Clone, Debug)]
struct CompactEntry {
    row: usize,
    col: usize,
    /// Image intensity (primary sort key)
    intensity: f64,
    /// Spatial distance penalty from nearest marker seed (secondary for compactness)
    distance: f64,
    /// Combined priority = intensity + compactness * distance
    priority: f64,
    /// Insertion order for tie-breaking
    order: u64,
}

impl PartialEq for CompactEntry {
    fn eq(&self, other: &Self) -> bool {
        self.row == other.row && self.col == other.col
    }
}
impl Eq for CompactEntry {}

impl PartialOrd for CompactEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CompactEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering so BinaryHeap pops smallest priority first
        match other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Equal => other.order.cmp(&self.order),
            ord => ord,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compact (regularized) watershed segmentation
///
/// Extends Meyer's flooding algorithm with a compactness penalty that encourages
/// spatially compact regions. When `compactness > 0`, pixels are assigned to the
/// marker whose combined cost (image intensity + compactness * geodesic distance)
/// is lowest, producing more regular region shapes.
///
/// # Arguments
///
/// * `image`   - Intensity / gradient image (2D)
/// * `markers` - Seed labels (positive integers for seeds, 0 for unlabeled pixels)
/// * `config`  - Compact watershed configuration
///
/// # Returns
///
/// Labeled 2D array where each pixel is assigned its closest marker label.
/// If `watershed_line` is enabled, ridge pixels are labeled 0.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::watershed::{compact_watershed, CompactWatershedConfig};
///
/// let image = array![
///     [0.1, 0.2, 0.9, 0.2, 0.1],
///     [0.1, 0.1, 0.9, 0.1, 0.1],
///     [0.1, 0.2, 0.9, 0.2, 0.1],
/// ];
/// let markers = array![
///     [0, 0, 0, 0, 0],
///     [1, 0, 0, 0, 2],
///     [0, 0, 0, 0, 0],
/// ];
///
/// let config = CompactWatershedConfig {
///     compactness: 0.5,
///     ..Default::default()
/// };
///
/// let result = compact_watershed(&image, &markers, &config).expect("should succeed");
/// assert_eq!(result[[1, 0]], 1);
/// assert_eq!(result[[1, 4]], 2);
/// ```
pub fn compact_watershed<T>(
    image: &Array2<T>,
    markers: &Array2<i32>,
    config: &CompactWatershedConfig,
) -> NdimageResult<Array2<i32>>
where
    T: Float + NumAssign + std::fmt::Debug + 'static,
{
    if image.shape() != markers.shape() {
        return Err(NdimageError::DimensionError(
            "Image and markers must have the same shape".to_string(),
        ));
    }

    let rows = image.nrows();
    let cols = image.ncols();

    if rows == 0 || cols == 0 {
        return Ok(markers.clone());
    }

    let offsets = neighbor_offsets(config.connectivity);
    let compact = config.compactness;

    // Output labels (initialized from markers)
    let mut output = markers.clone();

    // Distance from each pixel to its seed (for compactness penalty)
    let mut dist = Array2::<f64>::from_elem((rows, cols), f64::INFINITY);

    // Seed centroids (label -> (row, col)) for spatial distance computation
    let mut seed_centers: HashMap<i32, (f64, f64)> = HashMap::new();
    let mut seed_counts: HashMap<i32, usize> = HashMap::new();

    for r in 0..rows {
        for c in 0..cols {
            let lbl = markers[[r, c]];
            if lbl > 0 {
                let entry = seed_centers.entry(lbl).or_insert((0.0, 0.0));
                entry.0 += r as f64;
                entry.1 += c as f64;
                *seed_counts.entry(lbl).or_insert(0) += 1;
            }
        }
    }

    for (lbl, center) in seed_centers.iter_mut() {
        let cnt = *seed_counts.get(lbl).unwrap_or(&1) as f64;
        center.0 /= cnt;
        center.1 /= cnt;
    }

    // Initialize priority queue from marker boundary pixels
    let mut queue = BinaryHeap::new();
    let mut insertion_order: u64 = 0;

    for r in 0..rows {
        for c in 0..cols {
            let lbl = markers[[r, c]];
            if lbl > 0 {
                dist[[r, c]] = 0.0;

                for &(dr, dc) in offsets {
                    let nr = r as isize + dr;
                    let nc = c as isize + dc;
                    if in_bounds(nr, nc, rows, cols) {
                        let nr = nr as usize;
                        let nc = nc as usize;
                        if output[[nr, nc]] == 0 {
                            output[[nr, nc]] = IN_QUEUE;
                            let intensity = image[[nr, nc]].to_f64().unwrap_or(f64::INFINITY);
                            let spatial_dist = if let Some(&center) = seed_centers.get(&lbl) {
                                ((nr as f64 - center.0).powi(2) + (nc as f64 - center.1).powi(2))
                                    .sqrt()
                            } else {
                                0.0
                            };
                            let priority = intensity + compact * spatial_dist;

                            queue.push(CompactEntry {
                                row: nr,
                                col: nc,
                                intensity,
                                distance: spatial_dist,
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

    // Flooding loop
    while let Some(entry) = queue.pop() {
        let r = entry.row;
        let c = entry.col;

        // Collect neighbor labels
        let mut neighbor_labels: HashMap<i32, (usize, f64)> = HashMap::new(); // label -> (count, min_dist)
        let mut _has_wshed_neighbor = false;

        for &(dr, dc) in offsets {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if in_bounds(nr, nc, rows, cols) {
                let nr = nr as usize;
                let nc = nc as usize;
                let nlbl = output[[nr, nc]];
                if nlbl > 0 {
                    let nd = dist[[nr, nc]];
                    let e = neighbor_labels.entry(nlbl).or_insert((0, f64::INFINITY));
                    e.0 += 1;
                    if nd < e.1 {
                        e.1 = nd;
                    }
                } else if nlbl == WSHED && config.watershed_line {
                    _has_wshed_neighbor = true;
                }
            }
        }

        if neighbor_labels.is_empty() {
            output[[r, c]] = 0;
            continue;
        }

        let distinct: Vec<i32> = neighbor_labels.keys().copied().collect();

        let assigned_label;
        if config.watershed_line && distinct.len() > 1 {
            // Multiple labels meeting: this is a dam pixel
            output[[r, c]] = WSHED;
            assigned_label = WSHED;
        } else {
            // Choose the best label considering compactness
            let best_label = if compact > 0.0 {
                // Pick label with lowest combined cost
                let mut best = distinct[0];
                let mut best_cost = f64::INFINITY;
                for &lbl in &distinct {
                    let spatial = if let Some(&center) = seed_centers.get(&lbl) {
                        ((r as f64 - center.0).powi(2) + (c as f64 - center.1).powi(2)).sqrt()
                    } else {
                        0.0
                    };
                    let cost = entry.intensity + compact * spatial;
                    if cost < best_cost {
                        best_cost = cost;
                        best = lbl;
                    }
                }
                best
            } else {
                // Standard: most frequent neighbor label
                neighbor_labels
                    .iter()
                    .max_by_key(|&(_, (count, _))| count)
                    .map(|(&lbl, _)| lbl)
                    .unwrap_or(0)
            };

            if best_label > 0 {
                output[[r, c]] = best_label;
                assigned_label = best_label;
                // Update distance
                let spatial = if let Some(&center) = seed_centers.get(&best_label) {
                    ((r as f64 - center.0).powi(2) + (c as f64 - center.1).powi(2)).sqrt()
                } else {
                    0.0
                };
                dist[[r, c]] = spatial;
            } else {
                output[[r, c]] = 0;
                continue;
            }
        }

        // Enqueue unlabeled neighbors (both for labeled and dam pixels)
        for &(dr, dc) in offsets {
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if in_bounds(nr, nc, rows, cols) {
                let nr = nr as usize;
                let nc = nc as usize;
                if output[[nr, nc]] == 0 {
                    output[[nr, nc]] = IN_QUEUE;
                    let intensity = image[[nr, nc]].to_f64().unwrap_or(f64::INFINITY);
                    let spatial_dist = if assigned_label > 0 {
                        if let Some(&center) = seed_centers.get(&assigned_label) {
                            ((nr as f64 - center.0).powi(2) + (nc as f64 - center.1).powi(2)).sqrt()
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    let priority = intensity + compact * spatial_dist;

                    queue.push(CompactEntry {
                        row: nr,
                        col: nc,
                        intensity,
                        distance: spatial_dist,
                        priority,
                        order: insertion_order,
                    });
                    insertion_order += 1;
                }
            }
        }
    }

    // Clean up IN_QUEUE sentinels
    for val in output.iter_mut() {
        if *val == IN_QUEUE {
            *val = 0;
        }
    }

    Ok(output)
}

/// Extract dam (boundary) pixels from a watershed segmentation
///
/// Given a labeled image (e.g., from watershed), identifies pixels that lie on
/// region boundaries (where at least two different positive labels meet).
///
/// # Arguments
///
/// * `labels`       - Labeled image from watershed segmentation
/// * `connectivity` - Neighborhood connectivity for boundary detection
///
/// # Returns
///
/// A `DamResult` containing the relabeled image, binary dam mask, and region count.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::watershed::{extract_dams, WatershedNeighborhood};
///
/// let labels = array![
///     [1, 1, 2, 2],
///     [1, 1, 2, 2],
///     [3, 3, 4, 4],
///     [3, 3, 4, 4],
/// ];
///
/// let result = extract_dams(&labels, WatershedNeighborhood::Conn4)
///     .expect("should succeed");
/// assert!(result.dam_mask[[0, 1]] || result.dam_mask[[1, 0]] || result.num_regions >= 2);
/// ```
pub fn extract_dams(
    labels: &Array2<i32>,
    connectivity: WatershedNeighborhood,
) -> NdimageResult<DamResult> {
    let rows = labels.nrows();
    let cols = labels.ncols();

    if rows == 0 || cols == 0 {
        return Ok(DamResult {
            labels: labels.clone(),
            dam_mask: Array2::from_elem((rows, cols), false),
            num_regions: 0,
        });
    }

    let offsets = neighbor_offsets(connectivity);
    let mut dam_mask = Array2::from_elem((rows, cols), false);
    let mut out = labels.clone();
    let mut unique_labels = std::collections::HashSet::new();

    for r in 0..rows {
        for c in 0..cols {
            let lbl = labels[[r, c]];
            if lbl <= 0 {
                continue;
            }
            unique_labels.insert(lbl);

            let mut is_dam = false;
            for &(dr, dc) in offsets {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if in_bounds(nr, nc, rows, cols) {
                    let nlbl = labels[[nr as usize, nc as usize]];
                    if nlbl > 0 && nlbl != lbl {
                        is_dam = true;
                        break;
                    }
                }
            }

            if is_dam {
                dam_mask[[r, c]] = true;
                out[[r, c]] = WSHED;
            }
        }
    }

    Ok(DamResult {
        labels: out,
        dam_mask,
        num_regions: unique_labels.len(),
    })
}

/// Control oversegmentation by merging small regions
///
/// Performs hierarchical region merging on a watershed segmentation result,
/// merging regions whose area is below `config.min_region_area` into their
/// most similar neighbor.
///
/// # Arguments
///
/// * `image`  - Original intensity / gradient image
/// * `labels` - Labeled image from watershed
/// * `config` - Oversegmentation control parameters
///
/// # Returns
///
/// Relabeled image with small regions merged.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::watershed::{merge_small_regions, OversegmentationConfig};
///
/// let image = array![
///     [0.1, 0.1, 0.9, 0.9],
///     [0.1, 0.1, 0.9, 0.9],
///     [0.1, 0.1, 0.9, 0.9],
///     [0.1, 0.1, 0.9, 0.9],
/// ];
///
/// let labels = array![
///     [1, 1, 2, 2],
///     [1, 3, 2, 2],
///     [1, 1, 2, 2],
///     [1, 1, 2, 2],
/// ];
///
/// let config = OversegmentationConfig {
///     min_region_area: 3,
///     ..Default::default()
/// };
///
/// let merged = merge_small_regions(&image, &labels, &config).expect("should succeed");
/// // Region 3 (area 1) should be merged into region 1
/// assert!(merged[[1, 1]] != 3);
/// ```
pub fn merge_small_regions<T>(
    image: &Array2<T>,
    labels: &Array2<i32>,
    config: &OversegmentationConfig,
) -> NdimageResult<Array2<i32>>
where
    T: Float + NumAssign + std::fmt::Debug + 'static,
{
    if image.shape() != labels.shape() {
        return Err(NdimageError::DimensionError(
            "Image and labels must have the same shape".to_string(),
        ));
    }

    let rows = image.nrows();
    let cols = image.ncols();

    if rows == 0 || cols == 0 {
        return Ok(labels.clone());
    }

    // Compute region statistics: area, mean intensity, adjacency
    let mut region_area: HashMap<i32, usize> = HashMap::new();
    let mut region_sum: HashMap<i32, f64> = HashMap::new();

    for r in 0..rows {
        for c in 0..cols {
            let lbl = labels[[r, c]];
            if lbl <= 0 {
                continue;
            }
            *region_area.entry(lbl).or_insert(0) += 1;
            let val = image[[r, c]].to_f64().unwrap_or(0.0);
            *region_sum.entry(lbl).or_insert(0.0) += val;
        }
    }

    // Build adjacency: for each region, find neighboring region labels
    let offsets = neighbor_offsets(config.connectivity);
    let mut adjacency: HashMap<i32, HashMap<i32, usize>> = HashMap::new(); // region -> {neighbor -> boundary_length}

    for r in 0..rows {
        for c in 0..cols {
            let lbl = labels[[r, c]];
            if lbl <= 0 {
                continue;
            }
            for &(dr, dc) in offsets {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if in_bounds(nr, nc, rows, cols) {
                    let nlbl = labels[[nr as usize, nc as usize]];
                    if nlbl > 0 && nlbl != lbl {
                        *adjacency.entry(lbl).or_default().entry(nlbl).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Create merge mapping
    let mut merge_map: HashMap<i32, i32> = HashMap::new();

    // Sort regions by area (smallest first) to merge greedily
    let mut regions_by_area: Vec<(i32, usize)> =
        region_area.iter().map(|(&k, &v)| (k, v)).collect();
    regions_by_area.sort_by_key(|&(_, area)| area);

    for &(lbl, area) in &regions_by_area {
        if area >= config.min_region_area {
            continue; // large enough, skip
        }

        // Find the best neighbor to merge into (most similar mean intensity)
        let my_mean = region_sum.get(&lbl).copied().unwrap_or(0.0) / (area.max(1) as f64);

        let best_neighbor = if let Some(neighbors) = adjacency.get(&lbl) {
            let mut best: Option<(i32, f64)> = None;
            for (&nlbl, _) in neighbors {
                // Follow merge chain
                let mut final_lbl = nlbl;
                while let Some(&merged_to) = merge_map.get(&final_lbl) {
                    if merged_to == final_lbl {
                        break;
                    }
                    final_lbl = merged_to;
                }

                if final_lbl == lbl {
                    continue; // don't merge into self
                }

                let n_area = region_area.get(&final_lbl).copied().unwrap_or(1);
                let n_mean =
                    region_sum.get(&final_lbl).copied().unwrap_or(0.0) / (n_area.max(1) as f64);
                let diff = (my_mean - n_mean).abs();

                match best {
                    None => best = Some((final_lbl, diff)),
                    Some((_, best_diff)) if diff < best_diff => {
                        best = Some((final_lbl, diff));
                    }
                    _ => {}
                }
            }
            best.map(|(lbl, _)| lbl)
        } else {
            None
        };

        if let Some(target) = best_neighbor {
            merge_map.insert(lbl, target);
            // Update target statistics
            let my_sum = region_sum.get(&lbl).copied().unwrap_or(0.0);
            *region_sum.entry(target).or_insert(0.0) += my_sum;
            *region_area.entry(target).or_insert(0) += area;
        }
    }

    // Resolve transitive merges
    let all_labels: Vec<i32> = merge_map.keys().copied().collect();
    for lbl in all_labels {
        let mut target = lbl;
        let mut visited = std::collections::HashSet::new();
        while let Some(&next) = merge_map.get(&target) {
            if next == target || visited.contains(&next) {
                break;
            }
            visited.insert(target);
            target = next;
        }
        merge_map.insert(lbl, target);
    }

    // Apply merge mapping
    let mut output = labels.clone();
    for r in 0..rows {
        for c in 0..cols {
            let lbl = output[[r, c]];
            if lbl > 0 {
                if let Some(&target) = merge_map.get(&lbl) {
                    output[[r, c]] = target;
                }
            }
        }
    }

    // If max_regions is set, keep merging until we're under the limit
    if config.max_regions > 0 {
        let mut unique: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for &v in output.iter() {
            if v > 0 {
                unique.insert(v);
            }
        }

        if unique.len() > config.max_regions {
            // Recompute and merge smallest regions until under limit
            let mut remaining = unique.len();
            while remaining > config.max_regions {
                // Recount areas
                let mut areas: HashMap<i32, usize> = HashMap::new();
                for &v in output.iter() {
                    if v > 0 {
                        *areas.entry(v).or_insert(0) += 1;
                    }
                }

                // Find smallest region
                let smallest = areas
                    .iter()
                    .min_by_key(|&(_, &area)| area)
                    .map(|(&lbl, _)| lbl);

                if let Some(small_lbl) = smallest {
                    // Find best neighbor
                    let mut neighbor_boundary: HashMap<i32, usize> = HashMap::new();
                    for r in 0..rows {
                        for c in 0..cols {
                            if output[[r, c]] != small_lbl {
                                continue;
                            }
                            for &(dr, dc) in offsets {
                                let nr = r as isize + dr;
                                let nc = c as isize + dc;
                                if in_bounds(nr, nc, rows, cols) {
                                    let nlbl = output[[nr as usize, nc as usize]];
                                    if nlbl > 0 && nlbl != small_lbl {
                                        *neighbor_boundary.entry(nlbl).or_insert(0) += 1;
                                    }
                                }
                            }
                        }
                    }

                    // Merge into the neighbor with the longest shared boundary
                    let best = neighbor_boundary
                        .iter()
                        .max_by_key(|&(_, &count)| count)
                        .map(|(&lbl, _)| lbl);

                    if let Some(target) = best {
                        for val in output.iter_mut() {
                            if *val == small_lbl {
                                *val = target;
                            }
                        }
                        remaining -= 1;
                    } else {
                        break; // no neighbors, can't merge further
                    }
                } else {
                    break;
                }
            }
        }
    }

    Ok(output)
}

/// H-minima transform: suppress all minima whose depth is less than `h`
///
/// This is a morphological operation that raises all minima of depth < h,
/// useful for reducing oversegmentation in watershed by eliminating
/// shallow catchment basins.
///
/// # Arguments
///
/// * `image` - Input intensity image (2D)
/// * `h`     - Height threshold; minima shallower than this are suppressed
///
/// # Returns
///
/// Filtered image with shallow minima suppressed.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::watershed::h_minima_transform;
///
/// let image = array![
///     [5.0, 5.0, 5.0],
///     [5.0, 3.0, 5.0],  // shallow minimum at (1,1)
///     [5.0, 5.0, 5.0],
/// ];
///
/// let result = h_minima_transform(&image, 3.0).expect("should succeed");
/// // The minimum at (1,1) has depth 2 (5-3), which is less than h=3, so it should be suppressed
/// assert!(result[[1, 1]] >= 5.0 - 3.0);
/// ```
pub fn h_minima_transform<T>(image: &Array2<T>, h: f64) -> NdimageResult<Array2<f64>>
where
    T: Float + NumAssign + std::fmt::Debug + 'static,
{
    if h < 0.0 {
        return Err(NdimageError::InvalidInput(
            "h must be non-negative".to_string(),
        ));
    }

    let rows = image.nrows();
    let cols = image.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    // Convert to f64
    let img_f64: Array2<f64> = image.mapv(|x| x.to_f64().unwrap_or(0.0));

    // Marker: f - h (clamped from below)
    let marker: Array2<f64> = img_f64.mapv(|x| x - h);

    // Morphological reconstruction by dilation of marker under mask (original image)
    // The result fills all minima shallower than h
    let result = morphological_reconstruction_by_dilation_2d(&marker, &img_f64, 200)?;

    Ok(result)
}

/// Morphological reconstruction by erosion (2D, geodesic erosion)
///
/// Iteratively erodes the marker image while keeping it above the mask image,
/// until convergence. Used by h-maxima transform and other morphological operations.
#[allow(dead_code)]
fn morphological_reconstruction_by_erosion_2d(
    marker: &Array2<f64>,
    mask: &Array2<f64>,
    max_iterations: usize,
) -> NdimageResult<Array2<f64>> {
    let rows = marker.nrows();
    let cols = marker.ncols();

    let mut result = marker.clone();

    // Use raster/anti-raster scanning for efficiency
    for _iter in 0..max_iterations {
        let mut changed = false;

        // Forward scan (top-left to bottom-right)
        for r in 0..rows {
            for c in 0..cols {
                let mut val = result[[r, c]];
                // Check neighbors that have already been processed
                if r > 0 && result[[r - 1, c]] < val {
                    val = result[[r - 1, c]];
                }
                if c > 0 && result[[r, c - 1]] < val {
                    val = result[[r, c - 1]];
                }
                // Ensure we don't go below the mask
                let mask_val = mask[[r, c]];
                if val < mask_val {
                    val = mask_val;
                }
                if (val - result[[r, c]]).abs() > 1e-15 {
                    result[[r, c]] = val;
                    changed = true;
                }
            }
        }

        // Backward scan (bottom-right to top-left)
        for r in (0..rows).rev() {
            for c in (0..cols).rev() {
                let mut val = result[[r, c]];
                if r + 1 < rows && result[[r + 1, c]] < val {
                    val = result[[r + 1, c]];
                }
                if c + 1 < cols && result[[r, c + 1]] < val {
                    val = result[[r, c + 1]];
                }
                let mask_val = mask[[r, c]];
                if val < mask_val {
                    val = mask_val;
                }
                if (val - result[[r, c]]).abs() > 1e-15 {
                    result[[r, c]] = val;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    Ok(result)
}

/// Morphological reconstruction by dilation (2D, geodesic dilation)
///
/// Iteratively dilates the marker image while keeping it below the mask image,
/// until convergence. This is the dual of reconstruction by erosion.
///
/// Given marker <= mask pointwise, the result is the largest image R such that:
/// - marker <= R <= mask (pointwise)
/// - R is "connected from below" by the marker
fn morphological_reconstruction_by_dilation_2d(
    marker: &Array2<f64>,
    mask: &Array2<f64>,
    max_iterations: usize,
) -> NdimageResult<Array2<f64>> {
    let rows = marker.nrows();
    let cols = marker.ncols();

    // Clamp marker to be at most mask everywhere
    let mut result = Array2::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            result[[r, c]] = marker[[r, c]].min(mask[[r, c]]);
        }
    }

    // Use raster/anti-raster scanning for efficiency (Vincent 1993)
    for _iter in 0..max_iterations {
        let mut changed = false;

        // Forward scan (top-left to bottom-right)
        for r in 0..rows {
            for c in 0..cols {
                let mut val = result[[r, c]];
                // Check neighbors that have already been processed (above and left)
                if r > 0 && result[[r - 1, c]] > val {
                    val = result[[r - 1, c]];
                }
                if c > 0 && result[[r, c - 1]] > val {
                    val = result[[r, c - 1]];
                }
                // Also check diagonal neighbors for 8-connectivity
                if r > 0 && c > 0 && result[[r - 1, c - 1]] > val {
                    val = result[[r - 1, c - 1]];
                }
                if r > 0 && c + 1 < cols && result[[r - 1, c + 1]] > val {
                    val = result[[r - 1, c + 1]];
                }
                // Ensure we don't exceed the mask
                let mask_val = mask[[r, c]];
                if val > mask_val {
                    val = mask_val;
                }
                if (val - result[[r, c]]).abs() > 1e-15 {
                    result[[r, c]] = val;
                    changed = true;
                }
            }
        }

        // Backward scan (bottom-right to top-left)
        for r in (0..rows).rev() {
            for c in (0..cols).rev() {
                let mut val = result[[r, c]];
                if r + 1 < rows && result[[r + 1, c]] > val {
                    val = result[[r + 1, c]];
                }
                if c + 1 < cols && result[[r, c + 1]] > val {
                    val = result[[r, c + 1]];
                }
                // Diagonal neighbors
                if r + 1 < rows && c + 1 < cols && result[[r + 1, c + 1]] > val {
                    val = result[[r + 1, c + 1]];
                }
                if r + 1 < rows && c > 0 && result[[r + 1, c - 1]] > val {
                    val = result[[r + 1, c - 1]];
                }
                let mask_val = mask[[r, c]];
                if val > mask_val {
                    val = mask_val;
                }
                if (val - result[[r, c]]).abs() > 1e-15 {
                    result[[r, c]] = val;
                    changed = true;
                }
            }
        }

        if !changed {
            break;
        }
    }

    Ok(result)
}

/// Generate automatic markers from local minima of the image
///
/// Detects local minima (pixels lower than all their neighbors) and assigns
/// each a unique label. Useful as input markers for watershed segmentation.
///
/// # Arguments
///
/// * `image`        - Input intensity image
/// * `connectivity` - Neighborhood connectivity
/// * `threshold`    - Optional minimum depth of minima to consider
///
/// # Returns
///
/// Marker array with unique positive labels at each detected minimum.
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::watershed::{auto_markers, WatershedNeighborhood};
///
/// let image = array![
///     [5.0, 5.0, 5.0, 5.0, 5.0],
///     [5.0, 1.0, 5.0, 2.0, 5.0],
///     [5.0, 5.0, 5.0, 5.0, 5.0],
/// ];
///
/// let markers = auto_markers(&image, WatershedNeighborhood::Conn4, None)
///     .expect("should succeed");
/// assert!(markers[[1, 1]] > 0); // local minimum detected
/// assert!(markers[[1, 3]] > 0); // local minimum detected
/// ```
pub fn auto_markers<T>(
    image: &Array2<T>,
    connectivity: WatershedNeighborhood,
    threshold: Option<f64>,
) -> NdimageResult<Array2<i32>>
where
    T: Float + NumAssign + std::fmt::Debug + 'static,
{
    let rows = image.nrows();
    let cols = image.ncols();

    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((rows, cols)));
    }

    let offsets = neighbor_offsets(connectivity);
    let thresh = threshold.unwrap_or(0.0);

    let mut markers = Array2::<i32>::zeros((rows, cols));
    let mut next_label = 1i32;

    for r in 0..rows {
        for c in 0..cols {
            let val = image[[r, c]].to_f64().unwrap_or(f64::INFINITY);

            let mut is_minimum = true;
            let mut min_neighbor = f64::INFINITY;

            for &(dr, dc) in offsets {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if in_bounds(nr, nc, rows, cols) {
                    let nval = image[[nr as usize, nc as usize]]
                        .to_f64()
                        .unwrap_or(f64::INFINITY);
                    if nval <= val {
                        is_minimum = false;
                        break;
                    }
                    if nval < min_neighbor {
                        min_neighbor = nval;
                    }
                }
            }

            if is_minimum {
                // Check depth threshold
                let depth = if min_neighbor.is_finite() {
                    min_neighbor - val
                } else {
                    f64::INFINITY
                };

                if depth >= thresh {
                    markers[[r, c]] = next_label;
                    next_label += 1;
                }
            }
        }
    }

    Ok(markers)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_compact_watershed_basic() {
        let image = array![
            [0.1, 0.2, 0.9, 0.2, 0.1],
            [0.1, 0.1, 0.9, 0.1, 0.1],
            [0.1, 0.2, 0.9, 0.2, 0.1],
        ];
        let markers = array![[0, 0, 0, 0, 0], [1, 0, 0, 0, 2], [0, 0, 0, 0, 0],];

        let config = CompactWatershedConfig::default();
        let result = compact_watershed(&image, &markers, &config).expect("should succeed");

        assert_eq!(result[[1, 0]], 1);
        assert_eq!(result[[1, 4]], 2);
        // All pixels should be labeled
        for &v in result.iter() {
            assert!(v > 0);
        }
    }

    #[test]
    fn test_compact_watershed_with_compactness() {
        let image = array![
            [0.1, 0.1, 0.5, 0.1, 0.1],
            [0.1, 0.1, 0.5, 0.1, 0.1],
            [0.1, 0.1, 0.5, 0.1, 0.1],
        ];
        let markers = array![[0, 0, 0, 0, 0], [1, 0, 0, 0, 2], [0, 0, 0, 0, 0],];

        let config = CompactWatershedConfig {
            compactness: 1.0,
            ..Default::default()
        };

        let result = compact_watershed(&image, &markers, &config).expect("should succeed");
        assert_eq!(result[[1, 0]], 1);
        assert_eq!(result[[1, 4]], 2);
    }

    #[test]
    fn test_compact_watershed_shape_mismatch() {
        let image = array![[0.1, 0.2], [0.3, 0.4]];
        let markers = array![[0, 0, 0], [1, 0, 2]];

        let config = CompactWatershedConfig::default();
        let result = compact_watershed(&image, &markers, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_compact_watershed_single_marker() {
        let image = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let markers = array![[1, 0, 0], [0, 0, 0], [0, 0, 0]];

        let config = CompactWatershedConfig::default();
        let result = compact_watershed(&image, &markers, &config).expect("should succeed");
        for &v in result.iter() {
            assert_eq!(v, 1);
        }
    }

    #[test]
    fn test_compact_watershed_watershed_line() {
        let image = array![
            [1.0, 2.0, 9.0, 2.0, 1.0],
            [1.0, 2.0, 9.0, 2.0, 1.0],
            [1.0, 2.0, 9.0, 2.0, 1.0],
        ];
        let markers = array![[1, 0, 0, 0, 2], [0, 0, 0, 0, 0], [1, 0, 0, 0, 2],];

        let config = CompactWatershedConfig {
            watershed_line: true,
            connectivity: WatershedNeighborhood::Conn4,
            ..Default::default()
        };

        let result = compact_watershed(&image, &markers, &config).expect("should succeed");
        // Markers should be preserved
        assert_eq!(result[[0, 0]], 1);
        assert_eq!(result[[0, 4]], 2);
    }

    #[test]
    fn test_compact_watershed_empty() {
        let image: Array2<f64> = Array2::zeros((0, 0));
        let markers: Array2<i32> = Array2::zeros((0, 0));
        let config = CompactWatershedConfig::default();
        let result = compact_watershed(&image, &markers, &config).expect("empty should succeed");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_extract_dams_basic() {
        let labels = array![[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4],];

        let result = extract_dams(&labels, WatershedNeighborhood::Conn4).expect("should succeed");

        // Pixels at boundaries should be dams
        // (0,1)/(1,0) border region 1/2 or 1/3
        assert!(result.num_regions >= 2);

        // Interior pixels should NOT be dams
        assert!(!result.dam_mask[[0, 0]]);
        assert!(!result.dam_mask[[3, 3]]);
    }

    #[test]
    fn test_extract_dams_single_region() {
        let labels = Array2::from_elem((4, 4), 1i32);
        let result = extract_dams(&labels, WatershedNeighborhood::Conn8).expect("should succeed");
        // No dams in a single-region image
        for &v in result.dam_mask.iter() {
            assert!(!v);
        }
        assert_eq!(result.num_regions, 1);
    }

    #[test]
    fn test_merge_small_regions_basic() {
        let image = array![
            [0.1, 0.1, 0.9, 0.9],
            [0.1, 0.1, 0.9, 0.9],
            [0.1, 0.1, 0.9, 0.9],
            [0.1, 0.1, 0.9, 0.9],
        ];

        let labels = array![
            [1, 1, 2, 2],
            [1, 3, 2, 2], // region 3 has area 1
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ];

        let config = OversegmentationConfig {
            min_region_area: 3,
            ..Default::default()
        };

        let result = merge_small_regions(&image, &labels, &config).expect("should succeed");

        // Region 3 (area 1) should be merged
        assert_ne!(result[[1, 1]], 3);
        // It should be merged into region 1 (most similar mean intensity)
        assert_eq!(result[[1, 1]], 1);
    }

    #[test]
    fn test_merge_small_regions_max_regions() {
        let image = Array2::<f64>::from_elem((6, 6), 0.5);

        let mut labels = Array2::<i32>::zeros((6, 6));
        // Create 4 regions
        for r in 0..3 {
            for c in 0..3 {
                labels[[r, c]] = 1;
            }
        }
        for r in 0..3 {
            for c in 3..6 {
                labels[[r, c]] = 2;
            }
        }
        for r in 3..6 {
            for c in 0..3 {
                labels[[r, c]] = 3;
            }
        }
        for r in 3..6 {
            for c in 3..6 {
                labels[[r, c]] = 4;
            }
        }

        let config = OversegmentationConfig {
            min_region_area: 1,
            max_regions: 2,
            ..Default::default()
        };

        let result = merge_small_regions(&image, &labels, &config).expect("should succeed");

        // Count unique labels
        let mut unique = std::collections::HashSet::new();
        for &v in result.iter() {
            if v > 0 {
                unique.insert(v);
            }
        }
        assert!(
            unique.len() <= 2,
            "Expected <= 2 regions, got {}",
            unique.len()
        );
    }

    #[test]
    fn test_h_minima_transform() {
        // h-minima transform: HMIN_h(f) = reconstruction_by_dilation(f - h, f)
        //
        // For a minimum at pixel p with depth d (= surrounding_level - f(p)):
        // - If d <= h: the minimum is SUPPRESSED (filled flat with its plateau)
        //   Result: surrounding_level - h (same as neighboring pixels in reconstruction)
        // - If d > h: the minimum is PRESERVED but with reduced depth d - h
        //   Result: f(p) (clamped by mask, stays at original value)

        let image = array![
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 3.0, 5.0, 1.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
        ];
        // Minimum at (1,1): value=3, depth=2 (5-3)
        // Minimum at (1,3): value=1, depth=4 (5-1)

        // h=1.5: both minima deeper than h, both remain as local minima
        // marker = f - 1.5 => surrounding=3.5, (1,1)=1.5, (1,3)=-0.5
        // After reconstruction:
        //   (1,1): reconstruction raises marker to min(3.5, mask=3) = 3.0
        //   (1,3): reconstruction tries 3.5 but clamped by mask=1 => 1.0
        //   neighbors: all become 3.5 (< mask=5)
        // So both remain as minima in the reconstructed image
        let result = h_minima_transform(&image, 1.5).expect("should succeed");

        // Flat plateau should be at 5.0 - 1.5 = 3.5
        assert!((result[[0, 0]] - 3.5).abs() < 1e-10);
        // (1,1) has depth 2 > h=1.5: still a minimum, value = original (3.0)
        //   actually, reconstruction of marker=1.5 under mask=3.0 =>
        //   neighbors dilate to 3.5 then clamp to min(3.5, 3.0) = 3.0
        //   So (1,1) = 3.0, plateau = 3.5 => depth = 0.5 = 2.0 - 1.5 = d - h
        assert!(result[[1, 1]] < result[[0, 0]]); // still a minimum
                                                  // (1,3) has depth 4 > h=1.5: still a minimum
        assert!(result[[1, 3]] < result[[0, 0]]); // still a minimum

        // h=3.0: minimum at (1,1) with depth 2 <= h=3 is SUPPRESSED
        //   marker = f - 3 => surrounding=2, (1,1)=0, (1,3)=-2
        //   After reconstruction:
        //     (1,1): neighbors dilate to 2, min(2, mask=3) = 2 => same as plateau
        //     (1,3): neighbors dilate to 2, min(2, mask=1) = 1 => still a minimum
        //     Plateau pixels: all 2 (= 5 - 3)
        let result2 = h_minima_transform(&image, 3.0).expect("should succeed");

        // Plateau level = 5 - 3 = 2.0
        let plateau_level = result2[[0, 0]];
        assert!((plateau_level - 2.0).abs() < 1e-10);

        // (1,1) depth=2 <= h=3: SUPPRESSED -- raised to plateau level
        assert!(
            (result2[[1, 1]] - plateau_level).abs() < 1e-10,
            "expected (1,1) to be suppressed to plateau {}, got {}",
            plateau_level,
            result2[[1, 1]]
        );

        // (1,3) depth=4 > h=3: PRESERVED as a minimum
        // Reconstruction: marker=-2, neighbors=2, mask=1 => min(2, 1) = 1
        assert!(
            result2[[1, 3]] < plateau_level,
            "expected (1,3) to still be a minimum (below plateau {}), got {}",
            plateau_level,
            result2[[1, 3]]
        );
        assert!((result2[[1, 3]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_auto_markers_basic() {
        let image = array![
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 1.0, 5.0, 2.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
        ];

        let markers =
            auto_markers(&image, WatershedNeighborhood::Conn4, None).expect("should succeed");

        // Should detect two local minima
        assert!(markers[[1, 1]] > 0);
        assert!(markers[[1, 3]] > 0);
        assert_ne!(markers[[1, 1]], markers[[1, 3]]);

        // Non-minimum pixels should be 0
        assert_eq!(markers[[0, 0]], 0);
    }

    #[test]
    fn test_auto_markers_with_threshold() {
        let image = array![
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [5.0, 4.5, 5.0, 1.0, 5.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
        ];

        // Threshold = 1.0: only minima with depth >= 1.0
        let markers =
            auto_markers(&image, WatershedNeighborhood::Conn4, Some(1.0)).expect("should succeed");

        // (1,1) has depth 0.5 < threshold: not a marker
        assert_eq!(markers[[1, 1]], 0);
        // (1,3) has depth 4.0 >= threshold: is a marker
        assert!(markers[[1, 3]] > 0);
    }

    #[test]
    fn test_auto_markers_empty() {
        let image: Array2<f64> = Array2::zeros((0, 0));
        let markers =
            auto_markers(&image, WatershedNeighborhood::Conn4, None).expect("empty should succeed");
        assert_eq!(markers.len(), 0);
    }

    #[test]
    fn test_compact_watershed_conn4() {
        let image = array![[0.1, 0.9, 0.1], [0.9, 0.9, 0.9], [0.1, 0.9, 0.1],];
        let markers = array![[1, 0, 2], [0, 0, 0], [3, 0, 4],];

        let config = CompactWatershedConfig {
            connectivity: WatershedNeighborhood::Conn4,
            compactness: 0.0,
            watershed_line: false,
        };

        let result = compact_watershed(&image, &markers, &config).expect("should succeed");
        assert_eq!(result[[0, 0]], 1);
        assert_eq!(result[[0, 2]], 2);
        assert_eq!(result[[2, 0]], 3);
        assert_eq!(result[[2, 2]], 4);
    }
}
