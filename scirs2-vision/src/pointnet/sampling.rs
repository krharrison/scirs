//! Sampling and neighborhood query algorithms for PointNet++.
//!
//! Provides:
//! - Farthest Point Sampling (FPS) — greedily selects well-distributed points.
//! - Ball Query — finds neighbors within a fixed radius.
//! - K-Nearest Neighbors (KNN) — finds the k closest points.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::Array2;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two slices of equal length.
///
/// # Panics (debug)
/// Panics in debug builds if `a.len() != b.len()`.
#[inline]
pub(crate) fn euclidean_dist_sq(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "Dimension mismatch in euclidean_dist_sq");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Farthest Point Sampling
// ─────────────────────────────────────────────────────────────────────────────

/// Select `n_points` indices from `points` using Farthest Point Sampling.
///
/// The algorithm greedily picks the point farthest from the already-selected
/// set, which produces a well-spread-out subset suitable for PointNet++
/// centroids.
///
/// * Deterministic: always starts from index 0.
/// * Time complexity: O(N × n_points).
///
/// # Errors
/// Returns [`VisionError::InvalidParameter`] when:
/// - `points` is empty.
/// - `n_points > points.nrows()`.
/// - `points.ncols() != 3`.
pub fn farthest_point_sampling(points: &Array2<f64>, n_points: usize) -> Result<Vec<usize>> {
    let n = points.nrows();
    let dim = points.ncols();

    if n == 0 {
        return Err(VisionError::InvalidParameter(
            "farthest_point_sampling: empty point cloud".to_string(),
        ));
    }
    if n_points == 0 {
        return Ok(Vec::new());
    }
    if n_points > n {
        return Err(VisionError::InvalidParameter(format!(
            "farthest_point_sampling: n_points ({n_points}) > number of points ({n})"
        )));
    }
    if dim == 0 {
        return Err(VisionError::InvalidParameter(
            "farthest_point_sampling: points has zero columns".to_string(),
        ));
    }

    let mut selected = Vec::with_capacity(n_points);
    // dist[i] = minimum squared distance from point i to the selected set.
    let mut dist: Vec<f64> = vec![f64::INFINITY; n];

    // Start from index 0 for determinism.
    let first = 0usize;
    selected.push(first);

    // Update distances from the first centroid.
    let first_row: Vec<f64> = (0..dim).map(|d| points[[first, d]]).collect();
    for i in 0..n {
        let row: Vec<f64> = (0..dim).map(|d| points[[i, d]]).collect();
        dist[i] = euclidean_dist_sq(&first_row, &row);
    }

    for _ in 1..n_points {
        // Pick the point with the largest minimum distance to the selected set.
        let next = dist
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .ok_or_else(|| {
                VisionError::OperationError(
                    "farthest_point_sampling: failed to find farthest point".to_string(),
                )
            })?;

        selected.push(next);

        // Update distances with the new centroid.
        let next_row: Vec<f64> = (0..dim).map(|d| points[[next, d]]).collect();
        for i in 0..n {
            let row: Vec<f64> = (0..dim).map(|d| points[[i, d]]).collect();
            let d = euclidean_dist_sq(&next_row, &row);
            if d < dist[i] {
                dist[i] = d;
            }
        }
    }

    Ok(selected)
}

// ─────────────────────────────────────────────────────────────────────────────
// Ball Query
// ─────────────────────────────────────────────────────────────────────────────

/// For each center in `centers`, find all points in `points` within `radius`.
///
/// Returns a `Vec<Vec<usize>>` of length `M` (number of centers).  Each inner
/// `Vec` contains at most `max_points` indices into `points`, ordered by
/// insertion (the first up-to-`max_points` neighbors encountered).
///
/// When a ball contains fewer than `max_points` neighbors the last found index
/// is repeated to pad up to `max_points`, matching the PointNet++ padding
/// convention.
pub fn ball_query(
    points: &Array2<f64>,
    centers: &Array2<f64>,
    radius: f64,
    max_points: usize,
) -> Vec<Vec<usize>> {
    let n = points.nrows();
    let m = centers.nrows();
    let dim = points.ncols().min(centers.ncols());
    let r_sq = radius * radius;

    let mut result = Vec::with_capacity(m);

    for ci in 0..m {
        let cx: Vec<f64> = (0..dim).map(|d| centers[[ci, d]]).collect();
        let mut neighbors: Vec<usize> = Vec::with_capacity(max_points);

        for pi in 0..n {
            if neighbors.len() >= max_points {
                break;
            }
            let px: Vec<f64> = (0..dim).map(|d| points[[pi, d]]).collect();
            if euclidean_dist_sq(&cx, &px) <= r_sq {
                neighbors.push(pi);
            }
        }

        // Pad with last index if fewer than max_points (PointNet++ convention).
        if !neighbors.is_empty() && neighbors.len() < max_points {
            let last = *neighbors
                .last()
                .expect("neighbors is non-empty — checked above");
            while neighbors.len() < max_points {
                neighbors.push(last);
            }
        } else if neighbors.is_empty() {
            // No points in ball — use the center itself (index 0 fallback).
            let fallback = find_nearest(points, &cx).unwrap_or(0);
            for _ in 0..max_points {
                neighbors.push(fallback);
            }
        }

        result.push(neighbors);
    }

    result
}

/// Return the index of the point in `points` closest to `query`.
fn find_nearest(points: &Array2<f64>, query: &[f64]) -> Option<usize> {
    let n = points.nrows();
    let dim = points.ncols().min(query.len());
    if n == 0 {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_dist = f64::INFINITY;
    for pi in 0..n {
        let px: Vec<f64> = (0..dim).map(|d| points[[pi, d]]).collect();
        let d = euclidean_dist_sq(&px, &query[..dim]);
        if d < best_dist {
            best_dist = d;
            best_idx = pi;
        }
    }
    Some(best_idx)
}

// ─────────────────────────────────────────────────────────────────────────────
// K-Nearest Neighbors
// ─────────────────────────────────────────────────────────────────────────────

/// For each point in `queries`, find the `k` nearest points in `points`.
///
/// Returns a `Vec<Vec<usize>>` of length `queries.nrows()`.  Each inner `Vec`
/// has exactly `k` elements when `k ≤ points.nrows()`, otherwise it has
/// `points.nrows()` elements.
pub fn knn_query(points: &Array2<f64>, queries: &Array2<f64>, k: usize) -> Vec<Vec<usize>> {
    let n = points.nrows();
    let m = queries.nrows();
    let dim = points.ncols().min(queries.ncols());
    let k_clamped = k.min(n);

    let mut result = Vec::with_capacity(m);

    for qi in 0..m {
        let qx: Vec<f64> = (0..dim).map(|d| queries[[qi, d]]).collect();

        // Compute all distances and sort.
        let mut dists: Vec<(usize, f64)> = (0..n)
            .map(|pi| {
                let px: Vec<f64> = (0..dim).map(|d| points[[pi, d]]).collect();
                (pi, euclidean_dist_sq(&qx, &px))
            })
            .collect();

        dists.sort_unstable_by(|(_, a), (_, b)| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });

        let neighbors: Vec<usize> = dists[..k_clamped].iter().map(|(idx, _)| *idx).collect();
        result.push(neighbors);
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn simple_cloud() -> Array2<f64> {
        // 5 points on a line x = 0..4, y = z = 0.
        let mut pts = Array2::zeros((5, 3));
        for i in 0..5 {
            pts[[i, 0]] = i as f64;
        }
        pts
    }

    #[test]
    fn test_euclidean_dist_sq_zero() {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(euclidean_dist_sq(&a, &a), 0.0);
    }

    #[test]
    fn test_euclidean_dist_sq_known() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        assert!((euclidean_dist_sq(&a, &b) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_fps_selects_correct_count() {
        let pts = simple_cloud();
        let sel = farthest_point_sampling(&pts, 3).expect("fps failed");
        assert_eq!(sel.len(), 3);
    }

    #[test]
    fn test_fps_no_duplicates() {
        let pts = simple_cloud();
        let sel = farthest_point_sampling(&pts, 5).expect("fps failed");
        let mut sorted = sel.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 5, "FPS returned duplicate indices");
    }

    #[test]
    fn test_fps_diverse_selection() {
        // Points at 0, 1, 2, 3, 10 – FPS should pick 0 first then 10 (farthest).
        let mut pts = Array2::zeros((5, 3));
        let xs = [0.0f64, 1.0, 2.0, 3.0, 10.0];
        for (i, &x) in xs.iter().enumerate() {
            pts[[i, 0]] = x;
        }
        let sel = farthest_point_sampling(&pts, 2).expect("fps failed");
        assert_eq!(
            sel[0], 0,
            "first selected should be index 0 (deterministic)"
        );
        assert_eq!(
            sel[1], 4,
            "second selected should be the farthest point (index 4)"
        );
    }

    #[test]
    fn test_ball_query_radius_filter() {
        let pts = simple_cloud(); // x = 0,1,2,3,4
        let mut centers = Array2::zeros((1, 3));
        centers[[0, 0]] = 2.0; // center at x=2

        // radius = 1.5 → should include x=1,2,3 (indices 1,2,3)
        let neighbors = ball_query(&pts, &centers, 1.5, 10);
        assert_eq!(neighbors.len(), 1);
        let nb = &neighbors[0];
        // Indices 1, 2, 3 should all be in result
        assert!(nb.contains(&1), "index 1 should be in ball");
        assert!(nb.contains(&2), "index 2 should be in ball");
        assert!(nb.contains(&3), "index 3 should be in ball");
        // Index 0 (x=0) and 4 (x=4) are distance 2 — outside r=1.5
        assert!(
            !nb[..3].contains(&0) || nb.len() <= 3,
            "index 0 should not be in first 3"
        );
    }

    #[test]
    fn test_ball_query_max_points_limit() {
        let pts = simple_cloud();
        let mut centers = Array2::zeros((1, 3));
        centers[[0, 0]] = 2.0;

        // Even though 3 points are within r=1.5, limit to 2
        let neighbors = ball_query(&pts, &centers, 1.5, 2);
        assert_eq!(neighbors[0].len(), 2);
    }

    #[test]
    fn test_knn_returns_k_neighbors() {
        let pts = simple_cloud();
        let mut queries = Array2::zeros((1, 3));
        queries[[0, 0]] = 2.0;

        let knn = knn_query(&pts, &queries, 3);
        assert_eq!(knn.len(), 1);
        assert_eq!(knn[0].len(), 3);
    }
}
