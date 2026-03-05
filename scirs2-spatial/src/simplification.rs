//! Curve and polyline simplification algorithms
//!
//! This module provides several algorithms for reducing the number of points
//! in a polyline (or polygon boundary) while preserving its essential shape:
//!
//! - **Ramer-Douglas-Peucker**: Recursive perpendicular-distance simplification
//! - **Visvalingam-Whyatt**: Area-based elimination of least-significant vertices
//! - **Radial distance**: Remove points within a given distance of the previous kept point
//! - **Perpendicular distance**: Remove points close to the line between their neighbors
//! - **Topology-preserving**: Simplification that guarantees no self-intersections
//!
//! All algorithms operate on polylines represented as (n x d) arrays where
//! each row is a point.  Most are designed for 2D but work in arbitrary
//! dimensions where noted.
//!
//! # References
//!
//! * Ramer (1972) "An iterative procedure for the polygonal approximation of plane curves"
//! * Douglas & Peucker (1973) "Algorithms for the reduction of points required to
//!   represent a digitized line or its caricature"
//! * Visvalingam & Whyatt (1993) "Line generalisation by repeated elimination of points"

use crate::error::{SpatialError, SpatialResult};
use crate::safe_conversions::*;
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::Float;
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of a simplification operation.
#[derive(Clone, Debug)]
pub struct SimplificationResult<T: Float> {
    /// Simplified polyline (m x d) where m <= n
    pub points: Array2<T>,
    /// Indices of the kept points in the original polyline
    pub kept_indices: Vec<usize>,
    /// Compression ratio: kept / original
    pub compression_ratio: f64,
}

// ---------------------------------------------------------------------------
// Ramer-Douglas-Peucker
// ---------------------------------------------------------------------------

/// Simplify a polyline using the Ramer-Douglas-Peucker algorithm.
///
/// The algorithm recursively finds the point farthest from the line segment
/// connecting the endpoints.  If that distance exceeds `tolerance`, the point
/// is kept and the algorithm recurses on both halves; otherwise, all
/// intermediate points are discarded.
///
/// Works in arbitrary dimensions.
///
/// # Arguments
///
/// * `polyline` - Input polyline (n x d)
/// * `tolerance` - Maximum perpendicular distance for a point to be removed
///
/// # Complexity
///
/// O(n log n) average, O(n^2) worst case.
pub fn ramer_douglas_peucker<T: Float>(
    polyline: &ArrayView2<T>,
    tolerance: T,
) -> SpatialResult<SimplificationResult<T>> {
    let n = polyline.nrows();
    let d = polyline.ncols();

    if n == 0 {
        return Ok(SimplificationResult {
            points: Array2::zeros((0, d)),
            kept_indices: vec![],
            compression_ratio: 1.0,
        });
    }
    if n <= 2 {
        return Ok(SimplificationResult {
            points: polyline.to_owned(),
            kept_indices: (0..n).collect(),
            compression_ratio: 1.0,
        });
    }
    if tolerance < T::zero() {
        return Err(SpatialError::ValueError(
            "Tolerance must be non-negative".to_string(),
        ));
    }

    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    rdp_recurse(polyline, 0, n - 1, tolerance, &mut keep);

    build_result(polyline, &keep)
}

/// Iterative (stack-based) version of RDP to avoid stack overflow on large inputs.
pub fn ramer_douglas_peucker_iterative<T: Float>(
    polyline: &ArrayView2<T>,
    tolerance: T,
) -> SpatialResult<SimplificationResult<T>> {
    let n = polyline.nrows();
    let d = polyline.ncols();

    if n == 0 {
        return Ok(SimplificationResult {
            points: Array2::zeros((0, d)),
            kept_indices: vec![],
            compression_ratio: 1.0,
        });
    }
    if n <= 2 {
        return Ok(SimplificationResult {
            points: polyline.to_owned(),
            kept_indices: (0..n).collect(),
            compression_ratio: 1.0,
        });
    }
    if tolerance < T::zero() {
        return Err(SpatialError::ValueError(
            "Tolerance must be non-negative".to_string(),
        ));
    }

    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];

    while let Some((start, end)) = stack.pop() {
        if end <= start + 1 {
            continue;
        }

        let (max_idx, max_dist) = find_farthest(polyline, start, end);

        if max_dist > tolerance {
            keep[max_idx] = true;
            stack.push((start, max_idx));
            stack.push((max_idx, end));
        }
    }

    build_result(polyline, &keep)
}

fn rdp_recurse<T: Float>(
    polyline: &ArrayView2<T>,
    start: usize,
    end: usize,
    tolerance: T,
    keep: &mut [bool],
) {
    if end <= start + 1 {
        return;
    }

    let (max_idx, max_dist) = find_farthest(polyline, start, end);

    if max_dist > tolerance {
        keep[max_idx] = true;
        rdp_recurse(polyline, start, max_idx, tolerance, keep);
        rdp_recurse(polyline, max_idx, end, tolerance, keep);
    }
}

/// Find the point between `start` and `end` that is farthest from the
/// line through `polyline[start]` and `polyline[end]`.
fn find_farthest<T: Float>(polyline: &ArrayView2<T>, start: usize, end: usize) -> (usize, T) {
    let d = polyline.ncols();
    let mut max_dist = T::zero();
    let mut max_idx = start;

    for i in (start + 1)..end {
        let dist = perp_distance_nd(polyline, i, start, end, d);
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    (max_idx, max_dist)
}

/// Perpendicular distance from point `pi` to the line through `pa` and `pb`
/// in arbitrary dimensions.
fn perp_distance_nd<T: Float>(pts: &ArrayView2<T>, pi: usize, pa: usize, pb: usize, d: usize) -> T {
    // Vector AB and AP
    let mut ab_sq = T::zero();
    let mut ap_dot_ab = T::zero();
    for k in 0..d {
        let ab = pts[[pb, k]] - pts[[pa, k]];
        let ap = pts[[pi, k]] - pts[[pa, k]];
        ab_sq = ab_sq + ab * ab;
        ap_dot_ab = ap_dot_ab + ap * ab;
    }

    if ab_sq.is_zero() {
        // Degenerate segment: just return distance to pa
        let mut dist_sq = T::zero();
        for k in 0..d {
            let diff = pts[[pi, k]] - pts[[pa, k]];
            dist_sq = dist_sq + diff * diff;
        }
        return dist_sq.sqrt();
    }

    // Project P onto line AB
    let t = ap_dot_ab / ab_sq;
    let mut dist_sq = T::zero();
    for k in 0..d {
        let proj = pts[[pa, k]] + t * (pts[[pb, k]] - pts[[pa, k]]);
        let diff = pts[[pi, k]] - proj;
        dist_sq = dist_sq + diff * diff;
    }

    dist_sq.sqrt()
}

// ---------------------------------------------------------------------------
// Visvalingam-Whyatt
// ---------------------------------------------------------------------------

/// Simplify a polyline using the Visvalingam-Whyatt area-based algorithm.
///
/// Iteratively removes the vertex that forms the smallest triangle area
/// with its immediate neighbors, recalculating neighbor areas after each
/// removal, until all remaining areas exceed `min_area` or only 2 points
/// remain.
///
/// Designed for 2D polylines. For higher dimensions, uses the generalized
/// triangle area formula.
///
/// # Arguments
///
/// * `polyline` - Input polyline (n x 2, or n x d for generalized)
/// * `min_area`  - Minimum effective area to keep a vertex
pub fn visvalingam_whyatt<T: Float>(
    polyline: &ArrayView2<T>,
    min_area: T,
) -> SpatialResult<SimplificationResult<T>> {
    let n = polyline.nrows();
    let d = polyline.ncols();

    if n <= 2 {
        return Ok(SimplificationResult {
            points: polyline.to_owned(),
            kept_indices: (0..n).collect(),
            compression_ratio: 1.0,
        });
    }
    if min_area < T::zero() {
        return Err(SpatialError::ValueError(
            "min_area must be non-negative".to_string(),
        ));
    }

    // Active list: linked list via prev/next arrays
    let mut active = vec![true; n];
    let mut prev_idx: Vec<Option<usize>> = (0..n)
        .map(|i| if i == 0 { None } else { Some(i - 1) })
        .collect();
    let mut next_idx: Vec<Option<usize>> = (0..n)
        .map(|i| if i == n - 1 { None } else { Some(i + 1) })
        .collect();

    // Compute initial areas
    let mut areas: Vec<T> = vec![T::infinity(); n];
    for i in 1..n - 1 {
        areas[i] = triangle_area_nd(polyline, i - 1, i, i + 1, d);
    }

    // Repeatedly remove the vertex with smallest area
    loop {
        // Find active vertex with smallest area (excluding endpoints)
        let mut min_val = T::infinity();
        let mut min_i = None;
        for i in 0..n {
            if active[i] && areas[i] < min_val {
                min_val = areas[i];
                min_i = Some(i);
            }
        }

        match min_i {
            Some(idx) if min_val < min_area => {
                active[idx] = false;

                // Update neighbors' links
                let p = prev_idx[idx];
                let nx = next_idx[idx];
                if let Some(pi) = p {
                    next_idx[pi] = nx;
                }
                if let Some(ni) = nx {
                    prev_idx[ni] = p;
                }

                // Recalculate area for previous neighbor
                if let Some(pi) = p {
                    if let (Some(pp), Some(pn)) = (prev_idx[pi], next_idx[pi]) {
                        areas[pi] = triangle_area_nd(polyline, pp, pi, pn, d);
                        // Enforce monotonicity: effective area >= removed area
                        if areas[pi] < min_val {
                            areas[pi] = min_val;
                        }
                    } else {
                        areas[pi] = T::infinity();
                    }
                }
                // Recalculate area for next neighbor
                if let Some(ni) = nx {
                    if let (Some(np), Some(nn)) = (prev_idx[ni], next_idx[ni]) {
                        areas[ni] = triangle_area_nd(polyline, np, ni, nn, d);
                        if areas[ni] < min_val {
                            areas[ni] = min_val;
                        }
                    } else {
                        areas[ni] = T::infinity();
                    }
                }
            }
            _ => break,
        }

        // Stop if fewer than 3 points remain
        let count = active.iter().filter(|&&a| a).count();
        if count <= 2 {
            break;
        }
    }

    build_result(polyline, &active)
}

/// Visvalingam-Whyatt with a target number of output points instead of an
/// area threshold.
pub fn visvalingam_whyatt_n<T: Float>(
    polyline: &ArrayView2<T>,
    target_n: usize,
) -> SpatialResult<SimplificationResult<T>> {
    let n = polyline.nrows();
    let d = polyline.ncols();

    if n <= target_n || n <= 2 {
        return Ok(SimplificationResult {
            points: polyline.to_owned(),
            kept_indices: (0..n).collect(),
            compression_ratio: 1.0,
        });
    }
    if target_n < 2 {
        return Err(SpatialError::ValueError(
            "target_n must be at least 2".to_string(),
        ));
    }

    let mut active = vec![true; n];
    let mut prev_idx: Vec<Option<usize>> = (0..n)
        .map(|i| if i == 0 { None } else { Some(i - 1) })
        .collect();
    let mut next_idx: Vec<Option<usize>> = (0..n)
        .map(|i| if i == n - 1 { None } else { Some(i + 1) })
        .collect();

    let mut areas: Vec<T> = vec![T::infinity(); n];
    for i in 1..n - 1 {
        areas[i] = triangle_area_nd(polyline, i - 1, i, i + 1, d);
    }

    let mut remaining = n;
    while remaining > target_n {
        let mut min_val = T::infinity();
        let mut min_i = None;
        for i in 0..n {
            if active[i] && areas[i] < min_val {
                min_val = areas[i];
                min_i = Some(i);
            }
        }

        match min_i {
            Some(idx) => {
                active[idx] = false;
                remaining -= 1;

                let p = prev_idx[idx];
                let nx = next_idx[idx];
                if let Some(pi) = p {
                    next_idx[pi] = nx;
                }
                if let Some(ni) = nx {
                    prev_idx[ni] = p;
                }

                if let Some(pi) = p {
                    if let (Some(pp), Some(pn)) = (prev_idx[pi], next_idx[pi]) {
                        areas[pi] = triangle_area_nd(polyline, pp, pi, pn, d);
                        if areas[pi] < min_val {
                            areas[pi] = min_val;
                        }
                    } else {
                        areas[pi] = T::infinity();
                    }
                }
                if let Some(ni) = nx {
                    if let (Some(np), Some(nn)) = (prev_idx[ni], next_idx[ni]) {
                        areas[ni] = triangle_area_nd(polyline, np, ni, nn, d);
                        if areas[ni] < min_val {
                            areas[ni] = min_val;
                        }
                    } else {
                        areas[ni] = T::infinity();
                    }
                }
            }
            None => break,
        }
    }

    build_result(polyline, &active)
}

/// Generalized triangle area for n-dimensional points using the
/// cross-product magnitude formula.
fn triangle_area_nd<T: Float>(pts: &ArrayView2<T>, a: usize, b: usize, c: usize, d: usize) -> T {
    if d == 2 {
        // Shoelace for 2D
        let area = (pts[[a, 0]] * (pts[[b, 1]] - pts[[c, 1]])
            + pts[[b, 0]] * (pts[[c, 1]] - pts[[a, 1]])
            + pts[[c, 0]] * (pts[[a, 1]] - pts[[b, 1]]))
            / (T::one() + T::one());
        return area.abs();
    }

    // General: 0.5 * |AB x AC| using the sum-of-squares of pairwise cross products
    let mut cross_sq = T::zero();
    for i in 0..d {
        for j in (i + 1)..d {
            let ab_i = pts[[b, i]] - pts[[a, i]];
            let ab_j = pts[[b, j]] - pts[[a, j]];
            let ac_i = pts[[c, i]] - pts[[a, i]];
            let ac_j = pts[[c, j]] - pts[[a, j]];
            let cp = ab_i * ac_j - ab_j * ac_i;
            cross_sq = cross_sq + cp * cp;
        }
    }
    let two = T::one() + T::one();
    cross_sq.sqrt() / two
}

// ---------------------------------------------------------------------------
// Radial distance simplification
// ---------------------------------------------------------------------------

/// Simplify a polyline by removing points that are within `tolerance`
/// of the last kept point.
///
/// This is a simple one-pass O(n) algorithm. It always keeps the first
/// and last points.
///
/// Works in arbitrary dimensions.
pub fn radial_distance<T: Float>(
    polyline: &ArrayView2<T>,
    tolerance: T,
) -> SpatialResult<SimplificationResult<T>> {
    let n = polyline.nrows();
    let d = polyline.ncols();

    if n <= 2 {
        return Ok(SimplificationResult {
            points: polyline.to_owned(),
            kept_indices: (0..n).collect(),
            compression_ratio: 1.0,
        });
    }
    if tolerance < T::zero() {
        return Err(SpatialError::ValueError(
            "Tolerance must be non-negative".to_string(),
        ));
    }

    let tol_sq = tolerance * tolerance;
    let mut keep = vec![false; n];
    keep[0] = true;

    let mut last_kept = 0;
    for i in 1..n - 1 {
        let mut dist_sq = T::zero();
        for k in 0..d {
            let diff = polyline[[i, k]] - polyline[[last_kept, k]];
            dist_sq = dist_sq + diff * diff;
        }
        if dist_sq > tol_sq {
            keep[i] = true;
            last_kept = i;
        }
    }
    keep[n - 1] = true;

    build_result(polyline, &keep)
}

// ---------------------------------------------------------------------------
// Perpendicular distance simplification
// ---------------------------------------------------------------------------

/// Simplify a polyline by removing points whose perpendicular distance
/// to the line through their immediate neighbors is less than `tolerance`.
///
/// This is a single-pass O(n) algorithm. It always keeps the first and
/// last points.
///
/// Works in arbitrary dimensions.
pub fn perpendicular_distance<T: Float>(
    polyline: &ArrayView2<T>,
    tolerance: T,
) -> SpatialResult<SimplificationResult<T>> {
    let n = polyline.nrows();
    let d = polyline.ncols();

    if n <= 2 {
        return Ok(SimplificationResult {
            points: polyline.to_owned(),
            kept_indices: (0..n).collect(),
            compression_ratio: 1.0,
        });
    }
    if tolerance < T::zero() {
        return Err(SpatialError::ValueError(
            "Tolerance must be non-negative".to_string(),
        ));
    }

    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    // First pass: mark points based on their local perpendicular distance
    let mut prev_kept = 0;
    for i in 1..n - 1 {
        // Find next kept point (initially the end)
        let next_kept = n - 1;
        let dist = perp_distance_nd(polyline, i, prev_kept, next_kept, d);
        if dist >= tolerance {
            keep[i] = true;
            prev_kept = i;
        }
    }

    build_result(polyline, &keep)
}

/// Multi-pass perpendicular distance simplification that iterates until
/// convergence (no more points removed).
pub fn perpendicular_distance_iterative<T: Float>(
    polyline: &ArrayView2<T>,
    tolerance: T,
) -> SpatialResult<SimplificationResult<T>> {
    let n = polyline.nrows();
    let d = polyline.ncols();

    if n <= 2 {
        return Ok(SimplificationResult {
            points: polyline.to_owned(),
            kept_indices: (0..n).collect(),
            compression_ratio: 1.0,
        });
    }
    if tolerance < T::zero() {
        return Err(SpatialError::ValueError(
            "Tolerance must be non-negative".to_string(),
        ));
    }

    let mut keep = vec![true; n];

    loop {
        let mut changed = false;

        // Build list of active indices
        let active: Vec<usize> = keep
            .iter()
            .enumerate()
            .filter_map(|(i, &k)| if k { Some(i) } else { None })
            .collect();

        if active.len() <= 2 {
            break;
        }

        for ai in 1..active.len() - 1 {
            let prev = active[ai - 1];
            let curr = active[ai];
            let next = active[ai + 1];
            let dist = perp_distance_nd(polyline, curr, prev, next, d);
            if dist < tolerance {
                keep[curr] = false;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    build_result(polyline, &keep)
}

// ---------------------------------------------------------------------------
// Topology-preserving simplification
// ---------------------------------------------------------------------------

/// Simplify a 2D polyline using Ramer-Douglas-Peucker while ensuring
/// that the simplified polyline does not self-intersect.
///
/// After initial RDP simplification, segments are checked for self-intersection.
/// If any are found, additional points from the original polyline are
/// reinserted until no self-intersections remain.
///
/// Only works for 2D (ncols == 2).
pub fn topology_preserving<T: Float>(
    polyline: &ArrayView2<T>,
    tolerance: T,
) -> SpatialResult<SimplificationResult<T>> {
    if polyline.ncols() != 2 {
        return Err(SpatialError::DimensionError(
            "Topology-preserving simplification only supports 2D polylines".to_string(),
        ));
    }
    let n = polyline.nrows();
    if n <= 2 {
        return Ok(SimplificationResult {
            points: polyline.to_owned(),
            kept_indices: (0..n).collect(),
            compression_ratio: 1.0,
        });
    }

    // Start with RDP
    let initial = ramer_douglas_peucker(polyline, tolerance)?;
    let mut kept: Vec<usize> = initial.kept_indices;

    // Iteratively fix self-intersections
    let max_iters = n; // worst case: we add back all points
    for _ in 0..max_iters {
        let intersecting = find_self_intersection_2d(polyline, &kept);
        match intersecting {
            Some((seg_a, seg_b)) => {
                // Insert the original point with the largest deviation
                // between the two intersecting segments
                let start = kept[seg_a];
                let end = kept[seg_b + 1];
                let mut best_idx = None;
                let mut best_dist = T::zero();
                for orig in (start + 1)..end {
                    if !kept.contains(&orig) {
                        let d = perp_distance_nd(polyline, orig, start, end, 2);
                        if d > best_dist {
                            best_dist = d;
                            best_idx = Some(orig);
                        }
                    }
                }
                if let Some(idx) = best_idx {
                    // Insert in sorted order
                    let pos = kept.iter().position(|&k| k > idx).unwrap_or(kept.len());
                    kept.insert(pos, idx);
                } else {
                    break; // No more points to add
                }
            }
            None => break, // No intersections
        }
    }

    build_result_from_indices(polyline, &kept)
}

/// Find the first pair of non-adjacent segments that intersect.
/// Returns `Some((i, j))` where i < j are segment indices in `kept`.
fn find_self_intersection_2d<T: Float>(
    polyline: &ArrayView2<T>,
    kept: &[usize],
) -> Option<(usize, usize)> {
    let m = kept.len();
    if m < 4 {
        return None;
    }

    for i in 0..m - 1 {
        let a1 = (polyline[[kept[i], 0]], polyline[[kept[i], 1]]);
        let a2 = (polyline[[kept[i + 1], 0]], polyline[[kept[i + 1], 1]]);

        for j in (i + 2)..m - 1 {
            // Skip adjacent segments
            if j == i + 1 {
                continue;
            }
            let b1 = (polyline[[kept[j], 0]], polyline[[kept[j], 1]]);
            let b2 = (polyline[[kept[j + 1], 0]], polyline[[kept[j + 1], 1]]);

            if proper_intersection(a1, a2, b1, b2) {
                return Some((i, j));
            }
        }
    }

    None
}

/// Test for proper intersection (segments cross, not just touch).
fn proper_intersection<T: Float>(a1: (T, T), a2: (T, T), b1: (T, T), b2: (T, T)) -> bool {
    let d1 = cross_2d(a1, a2, b1);
    let d2 = cross_2d(a1, a2, b2);
    let d3 = cross_2d(b1, b2, a1);
    let d4 = cross_2d(b1, b2, a2);

    // Proper crossing: different signs on both pairs
    ((d1 > T::zero() && d2 < T::zero()) || (d1 < T::zero() && d2 > T::zero()))
        && ((d3 > T::zero() && d4 < T::zero()) || (d3 < T::zero() && d4 > T::zero()))
}

fn cross_2d<T: Float>(o: (T, T), a: (T, T), b: (T, T)) -> T {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_result<T: Float>(
    polyline: &ArrayView2<T>,
    keep: &[bool],
) -> SpatialResult<SimplificationResult<T>> {
    let d = polyline.ncols();
    let n = polyline.nrows();
    let kept_indices: Vec<usize> = keep
        .iter()
        .enumerate()
        .filter_map(|(i, &k)| if k { Some(i) } else { None })
        .collect();

    let m = kept_indices.len();
    let mut out = Array2::zeros((m, d));
    for (row, &idx) in kept_indices.iter().enumerate() {
        for k in 0..d {
            out[[row, k]] = polyline[[idx, k]];
        }
    }

    let ratio = if n > 0 { m as f64 / n as f64 } else { 1.0 };

    Ok(SimplificationResult {
        points: out,
        kept_indices,
        compression_ratio: ratio,
    })
}

fn build_result_from_indices<T: Float>(
    polyline: &ArrayView2<T>,
    indices: &[usize],
) -> SpatialResult<SimplificationResult<T>> {
    let d = polyline.ncols();
    let n = polyline.nrows();
    let m = indices.len();
    let mut out = Array2::zeros((m, d));
    for (row, &idx) in indices.iter().enumerate() {
        for k in 0..d {
            out[[row, k]] = polyline[[idx, k]];
        }
    }

    let ratio = if n > 0 { m as f64 / n as f64 } else { 1.0 };

    Ok(SimplificationResult {
        points: out,
        kept_indices: indices.to_vec(),
        compression_ratio: ratio,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    fn make_line_with_bump() -> Array2<f64> {
        // Straight line with a bump in the middle
        array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.5], // bump
            [3.0, 0.0],
            [4.0, 0.0],
        ]
    }

    fn make_noisy_line() -> Array2<f64> {
        array![
            [0.0, 0.0],
            [1.0, 0.01],
            [2.0, -0.01],
            [3.0, 0.02],
            [4.0, -0.02],
            [5.0, 0.01],
            [6.0, 0.0],
        ]
    }

    // --- RDP ---

    #[test]
    fn test_rdp_keep_all() {
        let pts = make_line_with_bump();
        let result = ramer_douglas_peucker(&pts.view(), 0.0).expect("rdp");
        // All points are kept with tolerance 0
        assert_eq!(result.kept_indices.len(), pts.nrows());
    }

    #[test]
    fn test_rdp_remove_bump() {
        let pts = make_line_with_bump();
        // Tolerance > 0.5 should remove the bump
        let result = ramer_douglas_peucker(&pts.view(), 0.6).expect("rdp");
        // Should keep first, last, possibly one middle
        assert!(result.points.nrows() < pts.nrows());
        // Must keep endpoints
        assert_eq!(result.kept_indices[0], 0);
        assert_eq!(*result.kept_indices.last().expect("last"), pts.nrows() - 1);
    }

    #[test]
    fn test_rdp_noisy_line() {
        let pts = make_noisy_line();
        let result = ramer_douglas_peucker(&pts.view(), 0.1).expect("rdp");
        // All noise is within 0.1, so only endpoints should remain
        assert_eq!(result.points.nrows(), 2);
        assert_relative_eq!(result.points[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.points[[1, 0]], 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rdp_empty() {
        let pts = Array2::<f64>::zeros((0, 2));
        let result = ramer_douglas_peucker(&pts.view(), 1.0).expect("rdp");
        assert_eq!(result.points.nrows(), 0);
    }

    #[test]
    fn test_rdp_two_points() {
        let pts = array![[0.0, 0.0], [1.0, 1.0]];
        let result = ramer_douglas_peucker(&pts.view(), 0.1).expect("rdp");
        assert_eq!(result.points.nrows(), 2);
    }

    #[test]
    fn test_rdp_negative_tolerance() {
        let pts = make_line_with_bump();
        let result = ramer_douglas_peucker(&pts.view(), -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rdp_iterative_matches_recursive() {
        let pts = make_line_with_bump();
        let r1 = ramer_douglas_peucker(&pts.view(), 0.3).expect("rdp");
        let r2 = ramer_douglas_peucker_iterative(&pts.view(), 0.3).expect("rdp iter");
        assert_eq!(r1.kept_indices, r2.kept_indices);
    }

    // --- Visvalingam-Whyatt ---

    #[test]
    fn test_vw_keep_all() {
        let pts = make_line_with_bump();
        let result = visvalingam_whyatt(&pts.view(), 0.0).expect("vw");
        assert_eq!(result.points.nrows(), pts.nrows());
    }

    #[test]
    fn test_vw_reduce() {
        let pts = make_noisy_line();
        let result = visvalingam_whyatt(&pts.view(), 0.1).expect("vw");
        assert!(result.points.nrows() < pts.nrows());
        // Endpoints are always kept
        assert_eq!(result.kept_indices[0], 0);
        assert_eq!(*result.kept_indices.last().expect("last"), pts.nrows() - 1);
    }

    #[test]
    fn test_vw_two_points() {
        let pts = array![[0.0, 0.0], [1.0, 0.0]];
        let result = visvalingam_whyatt(&pts.view(), 1.0).expect("vw");
        assert_eq!(result.points.nrows(), 2);
    }

    #[test]
    fn test_vw_n_target() {
        let pts = make_noisy_line();
        let result = visvalingam_whyatt_n(&pts.view(), 3).expect("vw_n");
        assert_eq!(result.points.nrows(), 3);
        // Endpoints are kept
        assert_eq!(result.kept_indices[0], 0);
        assert_eq!(*result.kept_indices.last().expect("last"), pts.nrows() - 1);
    }

    #[test]
    fn test_vw_n_target_exceeds() {
        let pts = make_line_with_bump();
        let result = visvalingam_whyatt_n(&pts.view(), 100).expect("vw_n");
        assert_eq!(result.points.nrows(), pts.nrows());
    }

    #[test]
    fn test_vw_negative_area() {
        let pts = make_line_with_bump();
        let result = visvalingam_whyatt(&pts.view(), -1.0);
        assert!(result.is_err());
    }

    // --- Radial distance ---

    #[test]
    fn test_radial_basic() {
        let pts = array![
            [0.0, 0.0],
            [0.1, 0.0], // within tolerance of [0,0]
            [0.2, 0.0], // within tolerance of [0,0]
            [1.0, 0.0], // far enough
            [1.1, 0.0], // within tolerance of [1,0]
            [2.0, 0.0],
        ];
        let result = radial_distance(&pts.view(), 0.5).expect("radial");
        // Should keep: 0, 3, 5 (last always kept)
        assert!(result.points.nrows() <= 4);
        assert_eq!(result.kept_indices[0], 0);
        assert_eq!(*result.kept_indices.last().expect("last"), pts.nrows() - 1);
    }

    #[test]
    fn test_radial_zero_tolerance() {
        let pts = make_line_with_bump();
        let result = radial_distance(&pts.view(), 0.0).expect("radial");
        assert_eq!(result.points.nrows(), pts.nrows());
    }

    #[test]
    fn test_radial_two_points() {
        let pts = array![[0.0, 0.0], [1.0, 0.0]];
        let result = radial_distance(&pts.view(), 0.5).expect("radial");
        assert_eq!(result.points.nrows(), 2);
    }

    // --- Perpendicular distance ---

    #[test]
    fn test_perp_basic() {
        let pts = make_noisy_line();
        let result = perpendicular_distance(&pts.view(), 0.05).expect("perp");
        assert!(result.points.nrows() <= pts.nrows());
    }

    #[test]
    fn test_perp_iterative() {
        let pts = make_noisy_line();
        let result = perpendicular_distance_iterative(&pts.view(), 0.05).expect("perp iter");
        assert!(result.points.nrows() <= pts.nrows());
        assert_eq!(result.kept_indices[0], 0);
    }

    #[test]
    fn test_perp_two_points() {
        let pts = array![[0.0, 0.0], [1.0, 0.0]];
        let result = perpendicular_distance(&pts.view(), 0.5).expect("perp");
        assert_eq!(result.points.nrows(), 2);
    }

    // --- Topology-preserving ---

    #[test]
    fn test_topology_preserving_basic() {
        let pts = make_line_with_bump();
        let result = topology_preserving(&pts.view(), 0.3).expect("topo");
        assert!(result.points.nrows() <= pts.nrows());
        assert_eq!(result.kept_indices[0], 0);
    }

    #[test]
    fn test_topology_preserving_no_self_intersect() {
        // A polyline that could self-intersect when simplified naively
        let pts = array![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [1.0, -0.1], [0.0, 0.0],];
        let result = topology_preserving(&pts.view(), 0.5).expect("topo");
        // Verify no self-intersections in the result
        let m = result.points.nrows();
        for i in 0..m.saturating_sub(1) {
            let a1 = (result.points[[i, 0]], result.points[[i, 1]]);
            let a2 = (result.points[[i + 1, 0]], result.points[[i + 1, 1]]);
            for j in (i + 2)..m.saturating_sub(1) {
                let b1 = (result.points[[j, 0]], result.points[[j, 1]]);
                let b2 = (result.points[[j + 1, 0]], result.points[[j + 1, 1]]);
                assert!(
                    !proper_intersection(a1, a2, b1, b2),
                    "Self-intersection found between segments {} and {}",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_topology_preserving_3d_fails() {
        let pts = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let result = topology_preserving(&pts.view(), 0.1);
        assert!(result.is_err());
    }

    // --- Compression ratio ---

    #[test]
    fn test_compression_ratio() {
        let pts = make_noisy_line();
        let result = ramer_douglas_peucker(&pts.view(), 0.1).expect("rdp");
        assert!(result.compression_ratio > 0.0);
        assert!(result.compression_ratio <= 1.0);
        let expected = result.points.nrows() as f64 / pts.nrows() as f64;
        assert_relative_eq!(result.compression_ratio, expected, epsilon = 1e-10);
    }

    // --- 3D support for non-topology algorithms ---

    #[test]
    fn test_rdp_3d() {
        let pts = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.5, 0.0], // bump in y
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        let result = ramer_douglas_peucker(&pts.view(), 0.6).expect("rdp 3d");
        assert!(result.points.nrows() < pts.nrows());
    }

    #[test]
    fn test_radial_3d() {
        let pts = array![[0.0, 0.0, 0.0], [0.1, 0.1, 0.1], [1.0, 1.0, 1.0],];
        let result = radial_distance(&pts.view(), 0.5).expect("radial 3d");
        assert!(result.points.nrows() <= 3);
    }

    // --- Edge cases ---

    #[test]
    fn test_single_point() {
        let pts = array![[5.0, 5.0]];
        let result = ramer_douglas_peucker(&pts.view(), 1.0).expect("rdp");
        assert_eq!(result.points.nrows(), 1);
    }

    #[test]
    fn test_all_same_points() {
        let pts = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0],];
        let result = ramer_douglas_peucker(&pts.view(), 0.1).expect("rdp");
        assert_eq!(result.points.nrows(), 2); // first and last
    }

    #[test]
    fn test_large_polyline() {
        let n = 500;
        let mut pts = Array2::zeros((n, 2));
        for i in 0..n {
            let x = i as f64;
            // Sine wave with noise
            let y = (x * 0.1).sin() * 0.01;
            pts[[i, 0]] = x;
            pts[[i, 1]] = y;
        }
        let result = ramer_douglas_peucker(&pts.view(), 0.05).expect("rdp large");
        assert!(result.points.nrows() < n);
        assert!(result.compression_ratio < 1.0);
    }
}
