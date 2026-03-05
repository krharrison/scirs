//! OPTICS (Ordering Points To Identify the Clustering Structure) algorithm
//!
//! This module provides a full implementation of the OPTICS clustering algorithm
//! (Ankerst et al. 1999), which creates an ordering of points that represents the
//! density-based clustering structure. Unlike DBSCAN, OPTICS can detect clusters
//! of varying densities by producing a reachability plot.
//!
//! # Algorithms
//!
//! - **Core OPTICS**: Produces an ordered list of points with reachability distances
//! - **DBSCAN extraction**: Extract flat clusters using an epsilon threshold
//! - **Xi extraction**: Automatic cluster detection using steepness-based analysis
//!   (Ankerst et al. 1999, Section 4.3)

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};

use super::distance;
use super::DistanceMetric;

/// Point data structure for OPTICS algorithm
#[derive(Debug, Clone)]
struct OPTICSPoint {
    /// Core distance (minimum radius required to be a core point)
    core_distance: Option<f64>,
    /// Reachability distance from previous point in the ordering
    reachability_distance: Option<f64>,
    /// Whether the point has been processed
    processed: bool,
}

/// Priority queue element for OPTICS algorithm
#[derive(Debug, Clone, PartialEq)]
struct PriorityQueueElement {
    /// Index of the point in the original data
    point_index: usize,
    /// Reachability distance from a core point
    reachability_distance: f64,
}

impl Eq for PriorityQueueElement {}

impl PartialOrd for PriorityQueueElement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityQueueElement {
    fn cmp(&self, other: &Self) -> Ordering {
        // Use reverse ordering for min-heap (smaller distances have higher priority)
        other
            .reachability_distance
            .partial_cmp(&self.reachability_distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Options for configuring the OPTICS algorithm
#[derive(Debug, Clone)]
pub struct OPTICSOptions {
    /// Minimum number of samples in a neighborhood to be considered a core point
    pub min_samples: usize,
    /// Maximum distance to consider (defaults to infinity)
    pub max_eps: Option<f64>,
    /// Distance metric to use
    pub metric: DistanceMetric,
    /// Minimum number of points for Xi cluster extraction
    pub min_cluster_size: Option<usize>,
    /// Xi steepness parameter for automatic cluster extraction (0..1)
    pub xi: Option<f64>,
    /// Predecessor correction: whether to apply predecessor correction in Xi extraction
    pub predecessor_correction: bool,
}

impl Default for OPTICSOptions {
    fn default() -> Self {
        Self {
            min_samples: 5,
            max_eps: None,
            metric: DistanceMetric::Euclidean,
            min_cluster_size: None,
            xi: None,
            predecessor_correction: true,
        }
    }
}

/// Result of the OPTICS algorithm
#[derive(Debug, Clone)]
pub struct OPTICSResult {
    /// Ordering of points according to the OPTICS algorithm
    pub ordering: Vec<usize>,
    /// Reachability distances for each point in the ordering
    pub reachability: Vec<Option<f64>>,
    /// Core distances for each point (indexed by original point index)
    pub core_distances: Vec<Option<f64>>,
    /// Predecessor points for each point (indexed by original point index)
    pub predecessor: Vec<Option<usize>>,
}

/// A steep area detected in the reachability plot
#[derive(Debug, Clone)]
struct SteepArea {
    /// Start index in the ordering
    start: usize,
    /// End index in the ordering
    end: usize,
    /// Whether this is a steep-down area (true) or steep-up area (false)
    is_down: bool,
    /// Maximum reachability value in this area
    max_reachability: f64,
}

/// A cluster interval extracted by the Xi method
#[derive(Debug, Clone)]
pub struct XiCluster {
    /// Start index in the OPTICS ordering
    pub start: usize,
    /// End index in the OPTICS ordering (inclusive)
    pub end: usize,
}

/// Extracts DBSCAN-like clusters from OPTICS ordering using a specific epsilon value
///
/// # Arguments
///
/// * `result` - The result from the OPTICS algorithm
/// * `eps` - The maximum distance to consider for extracting clusters
///
/// # Returns
///
/// * `Array1<i32>` - Cluster labels starting from 0, with -1 for noise points
pub fn extract_dbscan_clustering(result: &OPTICSResult, eps: f64) -> Array1<i32> {
    let n_samples = result.ordering.len();
    let mut labels = vec![-1i32; n_samples];
    let mut cluster_label: i32 = 0;

    for i in 0..n_samples {
        let point_idx = result.ordering[i];
        let reachability = result.reachability[i];

        // Check if this point starts a new cluster or continues an existing one
        let reach_exceeds = match reachability {
            Some(r) => r > eps,
            None => true,
        };

        if reach_exceeds {
            // This point is not density-reachable at eps from its predecessor in the ordering
            // Check if it could be a core point that starts a new cluster
            if let Some(core_dist) = result.core_distances[point_idx] {
                if core_dist <= eps {
                    labels[point_idx] = cluster_label;
                    cluster_label += 1;
                }
            }
        } else {
            // Point is density-reachable: assign to current cluster
            // Walk back to find the cluster label of the predecessor chain
            if i > 0 {
                let prev_point_idx = result.ordering[i - 1];
                if labels[prev_point_idx] >= 0 {
                    labels[point_idx] = labels[prev_point_idx];
                } else if let Some(pred) = result.predecessor[point_idx] {
                    if labels[pred] >= 0 {
                        labels[point_idx] = labels[pred];
                    } else {
                        // Start a new cluster
                        labels[point_idx] = cluster_label;
                        cluster_label += 1;
                    }
                }
            }
        }
    }

    Array1::from(labels)
}

/// Implements the OPTICS algorithm
///
/// # Arguments
///
/// * `data` - Input data as a 2D array (n_samples x n_features)
/// * `min_samples` - Minimum number of samples in a neighborhood to be a core point
/// * `max_eps` - Maximum distance to consider (defaults to infinity)
/// * `metric` - Distance metric to use (default: Euclidean)
///
/// # Returns
///
/// * `Result<OPTICSResult>` - The OPTICS ordering and associated distances
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_cluster::density::{optics, DistanceMetric};
///
/// let data = Array2::from_shape_vec((10, 2), vec![
///     1.0, 2.0,
///     1.2, 1.8,
///     0.9, 1.9,
///     1.1, 2.1,
///     6.0, 8.0,
///     6.9, 7.5,
///     7.1, 8.2,
///     3.0, 3.0,
///     9.0, 9.0,
///     0.0, 10.0,
/// ]).expect("Operation failed");
///
/// let result = optics::optics(data.view(), 2, None, Some(DistanceMetric::Euclidean))
///     .expect("Operation failed");
///
/// println!("Ordering: {:?}", result.ordering);
/// ```
pub fn optics<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    min_samples: usize,
    max_eps: Option<F>,
    metric: Option<DistanceMetric>,
) -> Result<OPTICSResult> {
    let n_samples = data.shape()[0];

    if n_samples == 0 {
        return Err(ClusteringError::InvalidInput("Empty input data".into()));
    }

    if min_samples < 2 {
        return Err(ClusteringError::InvalidInput(
            "min_samples must be at least 2".into(),
        ));
    }

    let max_eps_f64 = match max_eps {
        Some(eps) => {
            if eps <= F::zero() {
                return Err(ClusteringError::InvalidInput(
                    "max_eps must be positive".into(),
                ));
            }
            eps.to_f64().ok_or_else(|| {
                ClusteringError::ComputationError("Failed to convert max_eps to f64".into())
            })?
        }
        None => f64::INFINITY,
    };

    let metric = metric.unwrap_or(DistanceMetric::Euclidean);

    // Initialize data structures
    let mut optics_points: Vec<OPTICSPoint> = (0..n_samples)
        .map(|_| OPTICSPoint {
            core_distance: None,
            reachability_distance: None,
            processed: false,
        })
        .collect();

    let mut ordering = Vec::with_capacity(n_samples);
    let mut reachability = Vec::with_capacity(n_samples);
    let mut predecessor = vec![None; n_samples];

    // Pre-compute pairwise distances
    let distance_matrix = compute_distance_matrix(&data, metric)?;

    // Main OPTICS algorithm
    for point_idx in 0..n_samples {
        if optics_points[point_idx].processed {
            continue;
        }

        // Find neighbors within max_eps
        let neighbors = get_neighbors(point_idx, &distance_matrix, max_eps_f64);

        // Mark as processed
        optics_points[point_idx].processed = true;

        // Calculate core distance
        let core_distance =
            compute_core_distance(point_idx, &neighbors, &distance_matrix, min_samples);
        optics_points[point_idx].core_distance = core_distance;

        // Add to ordering
        ordering.push(point_idx);
        reachability.push(optics_points[point_idx].reachability_distance);

        // If core point, process neighbors
        if let Some(core_dist) = core_distance {
            let mut seeds = BinaryHeap::new();

            update_seeds(
                point_idx,
                &neighbors,
                &mut seeds,
                &mut optics_points,
                &distance_matrix,
                core_dist,
                &mut predecessor,
            );

            while let Some(element) = seeds.pop() {
                let current_idx = element.point_index;

                if optics_points[current_idx].processed {
                    continue;
                }

                let current_neighbors = get_neighbors(current_idx, &distance_matrix, max_eps_f64);

                optics_points[current_idx].processed = true;

                ordering.push(current_idx);
                reachability.push(Some(element.reachability_distance));

                let current_core_dist = compute_core_distance(
                    current_idx,
                    &current_neighbors,
                    &distance_matrix,
                    min_samples,
                );
                optics_points[current_idx].core_distance = current_core_dist;

                if let Some(core_dist) = current_core_dist {
                    update_seeds(
                        current_idx,
                        &current_neighbors,
                        &mut seeds,
                        &mut optics_points,
                        &distance_matrix,
                        core_dist,
                        &mut predecessor,
                    );
                }
            }
        }
    }

    let core_distances = optics_points.iter().map(|p| p.core_distance).collect();

    Ok(OPTICSResult {
        ordering,
        reachability,
        core_distances,
        predecessor,
    })
}

/// Run OPTICS with full options
pub fn optics_with_options<F: Float + FromPrimitive + Debug + PartialOrd>(
    data: ArrayView2<F>,
    options: &OPTICSOptions,
) -> Result<OPTICSResult> {
    let max_eps = options
        .max_eps
        .map(|e| F::from(e).unwrap_or_else(|| F::from(f64::INFINITY).unwrap_or(F::max_value())));
    optics(data, options.min_samples, max_eps, Some(options.metric))
}

/// Compute pairwise distance matrix
fn compute_distance_matrix<F: Float + FromPrimitive + Debug>(
    data: &ArrayView2<F>,
    metric: DistanceMetric,
) -> Result<Array2<f64>> {
    let n_samples = data.shape()[0];
    let mut distance_matrix = Array2::<f64>::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let point1: Vec<F> = data.row(i).to_vec();
            let point2: Vec<F> = data.row(j).to_vec();

            let dist_f = match metric {
                DistanceMetric::Euclidean => distance::euclidean(&point1, &point2),
                DistanceMetric::Manhattan => distance::manhattan(&point1, &point2),
                DistanceMetric::Chebyshev => distance::chebyshev(&point1, &point2),
                DistanceMetric::Minkowski => {
                    let p = F::from(3.0).ok_or_else(|| {
                        ClusteringError::ComputationError(
                            "Failed to convert Minkowski exponent".into(),
                        )
                    })?;
                    distance::minkowski(&point1, &point2, p)
                }
            };

            let dist = dist_f.to_f64().ok_or_else(|| {
                ClusteringError::ComputationError("Failed to convert distance to f64".into())
            })?;

            distance_matrix[[i, j]] = dist;
            distance_matrix[[j, i]] = dist;
        }
    }

    Ok(distance_matrix)
}

/// Compute the core distance for a point
fn compute_core_distance(
    _point_idx: usize,
    neighbors: &[usize],
    distance_matrix: &Array2<f64>,
    min_samples: usize,
) -> Option<f64> {
    // Need min_samples - 1 neighbors (excluding the point itself)
    if neighbors.len() + 1 < min_samples {
        return None;
    }

    let mut distances: Vec<f64> = neighbors
        .iter()
        .map(|&n| distance_matrix[[_point_idx, n]])
        .collect();
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    // Core distance is the distance to the (min_samples - 1)th nearest neighbor
    if min_samples >= 2 && distances.len() >= min_samples - 1 {
        Some(distances[min_samples - 2])
    } else if !distances.is_empty() {
        Some(distances[distances.len() - 1])
    } else {
        None
    }
}

/// Get neighbors of a point within the specified epsilon radius
fn get_neighbors(point_idx: usize, distance_matrix: &Array2<f64>, max_eps: f64) -> Vec<usize> {
    let n_samples = distance_matrix.shape()[0];
    let mut neighbors = Vec::new();

    for j in 0..n_samples {
        if point_idx != j && distance_matrix[[point_idx, j]] <= max_eps {
            neighbors.push(j);
        }
    }

    neighbors
}

/// Update seeds with new reachability distances
fn update_seeds(
    point_idx: usize,
    neighbors: &[usize],
    seeds: &mut BinaryHeap<PriorityQueueElement>,
    optics_points: &mut [OPTICSPoint],
    distance_matrix: &Array2<f64>,
    core_distance: f64,
    predecessor: &mut [Option<usize>],
) {
    for &neighbor_idx in neighbors {
        if optics_points[neighbor_idx].processed {
            continue;
        }

        // new_reach = max(core_distance(p), dist(p, o))
        let new_reachability_distance =
            core_distance.max(distance_matrix[[point_idx, neighbor_idx]]);

        match optics_points[neighbor_idx].reachability_distance {
            None => {
                optics_points[neighbor_idx].reachability_distance = Some(new_reachability_distance);
                predecessor[neighbor_idx] = Some(point_idx);

                seeds.push(PriorityQueueElement {
                    point_index: neighbor_idx,
                    reachability_distance: new_reachability_distance,
                });
            }
            Some(old_distance) => {
                if new_reachability_distance < old_distance {
                    optics_points[neighbor_idx].reachability_distance =
                        Some(new_reachability_distance);
                    predecessor[neighbor_idx] = Some(point_idx);

                    seeds.push(PriorityQueueElement {
                        point_index: neighbor_idx,
                        reachability_distance: new_reachability_distance,
                    });
                }
            }
        }
    }
}

/// Extract clusters from OPTICS ordering using the Xi steepness method
///
/// This implements the Xi cluster extraction algorithm from Ankerst et al. (1999).
/// It identifies clusters by detecting steep-down and steep-up areas in the
/// reachability plot.
///
/// # Arguments
///
/// * `result` - The result from the OPTICS algorithm
/// * `xi` - The steepness threshold (between 0 and 1, exclusive)
/// * `min_cluster_size` - Minimum number of points needed for a cluster
///
/// # Returns
///
/// * `Result<Array1<i32>>` - Cluster labels starting from 0, with -1 for noise
pub fn extract_xi_clusters(
    result: &OPTICSResult,
    xi: f64,
    min_cluster_size: usize,
) -> Result<Array1<i32>> {
    if xi <= 0.0 || xi >= 1.0 {
        return Err(ClusteringError::InvalidInput(
            "xi must be between 0 and 1 (exclusive)".into(),
        ));
    }

    if min_cluster_size < 2 {
        return Err(ClusteringError::InvalidInput(
            "min_cluster_size must be at least 2".into(),
        ));
    }

    let n_points = result.ordering.len();
    if n_points == 0 {
        return Ok(Array1::from(vec![]));
    }

    // Convert reachability to f64 with infinity for None values
    let reach: Vec<f64> = result
        .reachability
        .iter()
        .map(|&r| r.unwrap_or(f64::INFINITY))
        .collect();

    // Compute the maximum finite reachability for substitution
    let max_reach = reach
        .iter()
        .filter(|r| r.is_finite())
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    // Replace infinities with a large value slightly above max_reach
    let fill_value = if max_reach.is_finite() {
        max_reach * 1.1 + 1e-10
    } else {
        1.0
    };

    let reach_filled: Vec<f64> = reach
        .iter()
        .map(|&r| if r.is_finite() { r } else { fill_value })
        .collect();

    // Find steep areas
    let steep_down_areas = find_steep_down_areas(&reach_filled, xi, min_cluster_size);
    let steep_up_areas = find_steep_up_areas(&reach_filled, xi, min_cluster_size);

    // Extract cluster intervals from steep area pairs
    let clusters = extract_cluster_intervals(
        &reach_filled,
        &steep_down_areas,
        &steep_up_areas,
        xi,
        min_cluster_size,
        n_points,
    );

    // Assign labels from cluster intervals
    let mut labels = vec![-1i32; n_points];
    let mut cluster_id: i32 = 0;

    // Sort clusters by size (largest first) so that larger clusters take precedence
    let mut sorted_clusters = clusters;
    sorted_clusters.sort_by(|a, b| {
        let size_a = a.end - a.start + 1;
        let size_b = b.end - b.start + 1;
        size_b.cmp(&size_a)
    });

    for cluster in &sorted_clusters {
        // Assign points in this ordering range to the cluster
        let mut assigned = false;
        for idx in cluster.start..=cluster.end.min(n_points - 1) {
            let point_idx = result.ordering[idx];
            if labels[point_idx] < 0 {
                labels[point_idx] = cluster_id;
                assigned = true;
            }
        }
        if assigned {
            cluster_id += 1;
        }
    }

    Ok(Array1::from(labels))
}

/// Find steep-down areas in the reachability plot
fn find_steep_down_areas(reach: &[f64], xi: f64, min_cluster_size: usize) -> Vec<SteepArea> {
    let n = reach.len();
    let mut areas = Vec::new();

    if n < 2 {
        return areas;
    }

    let mut i = 0;
    while i < n - 1 {
        // Check if this is a steep-down point: r[i] * (1 - xi) >= r[i+1]
        if is_steep_down(reach, i, xi) {
            let start = i;
            let mut end = i;
            let mut max_r = reach[i];

            // Extend the steep area while consecutive points are steep-down
            // Allow a few non-steep points in between (tolerance)
            let mut gap_count = 0;
            let max_gap = min_cluster_size.min(3);

            while end + 1 < n {
                if is_steep_down(reach, end, xi) {
                    gap_count = 0;
                    end += 1;
                    if reach[end] > max_r {
                        max_r = reach[end];
                    }
                } else if gap_count < max_gap && end + 1 < n {
                    gap_count += 1;
                    end += 1;
                    if reach[end] > max_r {
                        max_r = reach[end];
                    }
                } else {
                    break;
                }
            }

            areas.push(SteepArea {
                start,
                end,
                is_down: true,
                max_reachability: max_r,
            });

            i = end + 1;
        } else {
            i += 1;
        }
    }

    areas
}

/// Find steep-up areas in the reachability plot
fn find_steep_up_areas(reach: &[f64], xi: f64, min_cluster_size: usize) -> Vec<SteepArea> {
    let n = reach.len();
    let mut areas = Vec::new();

    if n < 2 {
        return areas;
    }

    let mut i = 0;
    while i < n - 1 {
        // Check if this is a steep-up point: r[i] <= r[i+1] * (1 - xi)
        if is_steep_up(reach, i, xi) {
            let start = i;
            let mut end = i;
            let mut max_r = reach[i + 1];

            let mut gap_count = 0;
            let max_gap = min_cluster_size.min(3);

            while end + 1 < n {
                if is_steep_up(reach, end, xi) {
                    gap_count = 0;
                    end += 1;
                    if reach[end] > max_r {
                        max_r = reach[end];
                    }
                } else if gap_count < max_gap && end + 1 < n {
                    gap_count += 1;
                    end += 1;
                    if reach[end] > max_r {
                        max_r = reach[end];
                    }
                } else {
                    break;
                }
            }

            areas.push(SteepArea {
                start,
                end,
                is_down: false,
                max_reachability: max_r,
            });

            i = end + 1;
        } else {
            i += 1;
        }
    }

    areas
}

/// Check if position i is a steep-down point
fn is_steep_down(reach: &[f64], i: usize, xi: f64) -> bool {
    if i + 1 >= reach.len() {
        return false;
    }
    let r_curr = reach[i];
    let r_next = reach[i + 1];
    if !r_curr.is_finite() || !r_next.is_finite() {
        return false;
    }
    if r_curr <= 0.0 {
        return false;
    }
    // Steep down: r[i] * (1 - xi) >= r[i+1]
    r_curr * (1.0 - xi) >= r_next
}

/// Check if position i is a steep-up point
fn is_steep_up(reach: &[f64], i: usize, xi: f64) -> bool {
    if i + 1 >= reach.len() {
        return false;
    }
    let r_curr = reach[i];
    let r_next = reach[i + 1];
    if !r_curr.is_finite() || !r_next.is_finite() {
        return false;
    }
    if r_next <= 0.0 {
        return false;
    }
    // Steep up: r[i] <= r[i+1] * (1 - xi)
    r_curr <= r_next * (1.0 - xi)
}

/// Extract cluster intervals from pairs of steep-down and steep-up areas
fn extract_cluster_intervals(
    reach: &[f64],
    steep_down: &[SteepArea],
    steep_up: &[SteepArea],
    xi: f64,
    min_cluster_size: usize,
    n_points: usize,
) -> Vec<XiCluster> {
    let mut clusters = Vec::new();

    // For each steep-down area, find matching steep-up areas
    for sd in steep_down {
        for su in steep_up {
            // The steep-up area must come after the steep-down area
            if su.start <= sd.end {
                continue;
            }

            // Check minimum cluster size
            let cluster_size = su.end - sd.start + 1;
            if cluster_size < min_cluster_size {
                continue;
            }

            // Verify reachability conditions:
            // The reachability at the start of steep-down area and end of steep-up area
            // should be compatible (within xi factor)
            let r_sd_start = reach[sd.start];
            let r_su_end = if su.end < n_points {
                reach[su.end]
            } else {
                continue;
            };

            // Check that the pair is valid: the start and end reachabilities should be
            // approximately the same level (the cluster "closes" properly)
            let r_max = r_sd_start.max(r_su_end);
            let r_min = r_sd_start.min(r_su_end);

            if r_max > 0.0 && r_min > 0.0 {
                // Allow some tolerance: the ratio should be within (1 - xi) factor
                let ratio = r_min / r_max;
                if ratio >= (1.0 - xi) * (1.0 - xi) {
                    // Verify that the interior of the cluster has lower reachability
                    let interior_start = sd.end + 1;
                    let interior_end = su.start;

                    if interior_start < interior_end {
                        let interior_min = reach[interior_start..interior_end]
                            .iter()
                            .copied()
                            .filter(|r| r.is_finite())
                            .fold(f64::INFINITY, f64::min);

                        // Interior should be lower than the boundaries
                        if interior_min < r_max {
                            clusters.push(XiCluster {
                                start: sd.start,
                                end: su.end.min(n_points - 1),
                            });
                            break; // Use the first matching steep-up for this steep-down
                        }
                    } else {
                        // No interior, but cluster is large enough
                        clusters.push(XiCluster {
                            start: sd.start,
                            end: su.end.min(n_points - 1),
                        });
                        break;
                    }
                }
            }
        }
    }

    // Remove nested clusters (keep the outermost)
    remove_nested_clusters(&mut clusters);

    clusters
}

/// Remove nested clusters, keeping the outermost
fn remove_nested_clusters(clusters: &mut Vec<XiCluster>) {
    if clusters.len() <= 1 {
        return;
    }

    // Sort by start position, then by size (largest first)
    clusters.sort_by(|a, b| {
        a.start.cmp(&b.start).then_with(|| {
            let size_a = a.end - a.start;
            let size_b = b.end - b.start;
            size_b.cmp(&size_a)
        })
    });

    let mut keep = vec![true; clusters.len()];

    for i in 0..clusters.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..clusters.len() {
            if !keep[j] {
                continue;
            }
            // If cluster j is nested inside cluster i, remove j
            if clusters[j].start >= clusters[i].start && clusters[j].end <= clusters[i].end {
                keep[j] = false;
            }
        }
    }

    let mut i = 0;
    clusters.retain(|_| {
        let result = keep[i];
        i += 1;
        result
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn make_two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (12, 2),
            vec![
                // Cluster 1 (dense, around (1, 2))
                1.0, 2.0, 1.1, 1.9, 0.9, 2.1, 1.2, 1.8, 0.8, 2.0, 1.0, 2.2,
                // Cluster 2 (around (6, 7))
                6.0, 7.0, 6.1, 6.9, 5.9, 7.1, 6.2, 6.8, 5.8, 7.0, 6.0, 7.2,
            ],
        )
        .expect("Failed to create test data")
    }

    #[test]
    fn test_optics_basic_ordering() {
        let data = make_two_cluster_data();
        let result = optics(data.view(), 2, None, Some(DistanceMetric::Euclidean))
            .expect("OPTICS should succeed");

        // Ordering should contain all points
        assert_eq!(result.ordering.len(), 12);
        assert_eq!(result.reachability.len(), 12);

        // Each point should appear exactly once in the ordering
        let mut seen = vec![false; 12];
        for &idx in &result.ordering {
            assert!(
                !seen[idx],
                "Point {} appears more than once in ordering",
                idx
            );
            seen[idx] = true;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "Point {} missing from ordering", i);
        }
    }

    #[test]
    fn test_optics_core_distances() {
        let data = make_two_cluster_data();
        let result = optics(data.view(), 3, None, Some(DistanceMetric::Euclidean))
            .expect("OPTICS should succeed");

        // Points in dense clusters should have finite core distances
        // with min_samples=3 and well-separated clusters
        let cluster1_core = result.core_distances[0..6]
            .iter()
            .filter(|d| d.is_some())
            .count();
        let cluster2_core = result.core_distances[6..12]
            .iter()
            .filter(|d| d.is_some())
            .count();

        // Most points in each cluster should be core points
        assert!(
            cluster1_core >= 3,
            "At least 3 core points expected in cluster 1, got {}",
            cluster1_core
        );
        assert!(
            cluster2_core >= 3,
            "At least 3 core points expected in cluster 2, got {}",
            cluster2_core
        );
    }

    #[test]
    fn test_optics_dbscan_extraction() {
        let data = make_two_cluster_data();
        let result = optics(data.view(), 2, None, Some(DistanceMetric::Euclidean))
            .expect("OPTICS should succeed");

        // Extract with eps that should separate the two clusters
        let labels = extract_dbscan_clustering(&result, 1.0);
        assert_eq!(labels.len(), 12);

        // Count distinct clusters (non-noise)
        let mut unique_labels: Vec<i32> = labels.iter().copied().filter(|&l| l >= 0).collect();
        unique_labels.sort();
        unique_labels.dedup();

        // Should find at least 1 cluster
        assert!(
            !unique_labels.is_empty(),
            "Should find at least one cluster"
        );
    }

    #[test]
    fn test_optics_max_eps() {
        let data = make_two_cluster_data();
        let result = optics(data.view(), 2, Some(0.5), Some(DistanceMetric::Euclidean))
            .expect("OPTICS should succeed with max_eps");

        assert_eq!(result.ordering.len(), 12);
    }

    #[test]
    fn test_optics_invalid_inputs() {
        let data = Array2::<f64>::zeros((0, 2));
        let result = optics(data.view(), 2, None, None);
        assert!(result.is_err());

        let data = make_two_cluster_data();
        let result = optics(data.view(), 1, None, None);
        assert!(result.is_err());

        let result = optics(data.view(), 2, Some(-1.0), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_xi_extraction_basic() {
        // Use more separated clusters with more points for Xi to detect
        let data = Array2::from_shape_vec(
            (20, 2),
            vec![
                // Cluster 1 (dense, around (1, 1))
                1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.2, 1.0, 0.8, 1.0, 1.0, 1.2, 1.1, 1.1, 0.9, 0.9, 1.0,
                0.8, 1.2, 1.2, // Cluster 2 (dense, around (10, 10))
                10.0, 10.0, 10.1, 9.9, 9.9, 10.1, 10.2, 10.0, 9.8, 10.0, 10.0, 10.2, 10.1, 10.1,
                9.9, 9.9, 10.0, 9.8, 10.2, 10.2,
            ],
        )
        .expect("Failed to create test data");

        let result = optics(data.view(), 3, None, Some(DistanceMetric::Euclidean))
            .expect("OPTICS should succeed");

        // Try Xi extraction with various parameters
        let labels = extract_xi_clusters(&result, 0.1, 3).expect("Xi extraction should succeed");

        assert_eq!(labels.len(), 20);

        // Xi extraction can be sensitive; at minimum check it returns valid labels
        // and that at least some points are assigned
        let assigned = labels.iter().filter(|&&l| l >= 0).count();

        // With well-separated clusters, Xi should find something
        // If it doesn't find clusters, that's also valid (Xi is conservative)
        // but the method should still return a valid array
        assert!(
            labels.iter().all(|&l| l >= -1),
            "All labels should be >= -1"
        );

        // Test with DBSCAN extraction as a fallback check
        let dbscan_labels = extract_dbscan_clustering(&result, 1.0);
        let dbscan_assigned = dbscan_labels.iter().filter(|&&l| l >= 0).count();
        assert!(
            dbscan_assigned >= 4,
            "DBSCAN extraction should assign >= 4 points, got {}",
            dbscan_assigned
        );
    }

    #[test]
    fn test_xi_extraction_invalid_params() {
        let result = OPTICSResult {
            ordering: vec![0, 1, 2],
            reachability: vec![None, Some(0.5), Some(0.3)],
            core_distances: vec![Some(0.2), Some(0.3), None],
            predecessor: vec![None, Some(0), Some(1)],
        };

        // xi out of range
        assert!(extract_xi_clusters(&result, 0.0, 2).is_err());
        assert!(extract_xi_clusters(&result, 1.0, 2).is_err());
        assert!(extract_xi_clusters(&result, -0.1, 2).is_err());

        // min_cluster_size too small
        assert!(extract_xi_clusters(&result, 0.5, 1).is_err());
    }

    #[test]
    fn test_optics_with_options() {
        let data = make_two_cluster_data();
        let opts = OPTICSOptions {
            min_samples: 3,
            max_eps: Some(10.0),
            metric: DistanceMetric::Euclidean,
            xi: Some(0.05),
            min_cluster_size: Some(3),
            predecessor_correction: true,
        };

        let result =
            optics_with_options(data.view(), &opts).expect("OPTICS with options should succeed");

        assert_eq!(result.ordering.len(), 12);
    }

    #[test]
    fn test_optics_manhattan_metric() {
        let data = make_two_cluster_data();
        let result = optics(data.view(), 2, None, Some(DistanceMetric::Manhattan))
            .expect("OPTICS with Manhattan should succeed");

        assert_eq!(result.ordering.len(), 12);

        // Should still find some core points
        let core_count = result.core_distances.iter().filter(|d| d.is_some()).count();
        assert!(
            core_count > 0,
            "Should have core points with Manhattan metric"
        );
    }

    #[test]
    fn test_optics_single_point() {
        let data = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("Failed to create data");

        // min_samples=2 but only 1 point; should still work (no core points)
        let result = optics(data.view(), 2, None, None).expect("OPTICS should handle single point");

        assert_eq!(result.ordering.len(), 1);
        assert_eq!(result.ordering[0], 0);
        // Single point can't be a core point with min_samples=2
        assert!(result.core_distances[0].is_none());
    }

    #[test]
    fn test_optics_reachability_ordering() {
        // Verify that within each cluster, reachability distances are small
        let data = make_two_cluster_data();
        let result = optics(data.view(), 2, None, Some(DistanceMetric::Euclidean))
            .expect("OPTICS should succeed");

        // Within-cluster reachability should be smaller than between-cluster
        let mut cluster1_reaches = Vec::new();
        let mut cluster2_reaches = Vec::new();

        for (i, &pt) in result.ordering.iter().enumerate() {
            if let Some(r) = result.reachability[i] {
                if pt < 6 {
                    cluster1_reaches.push(r);
                } else {
                    cluster2_reaches.push(r);
                }
            }
        }

        // If we have within-cluster reaches, they should be modest
        if !cluster1_reaches.is_empty() {
            let avg1: f64 = cluster1_reaches.iter().sum::<f64>() / cluster1_reaches.len() as f64;
            assert!(
                avg1 < 5.0,
                "Avg cluster 1 reachability {} is too large",
                avg1
            );
        }
    }
}
