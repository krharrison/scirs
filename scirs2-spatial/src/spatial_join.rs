//! Spatial Join operations for geometric queries and point-set relationships
//!
//! This module provides a collection of spatial join and geometric query
//! algorithms for 2D point and polygon data:
//!
//! - Point-in-polygon test (ray casting algorithm)
//! - Polygon intersection detection
//! - Nearest feature queries (point to line segment)
//! - Spatial join between two point sets
//! - Within-distance join
//! - Cross-distance matrix (all pairwise distances)
//!
//! # References
//!
//! * Preparata & Shamos (1985) "Computational Geometry: An Introduction"
//! * de Berg et al. (2008) "Computational Geometry: Algorithms and Applications"

use crate::error::{SpatialError, SpatialResult};
use crate::safe_conversions::*;
use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::Float;

// ---------------------------------------------------------------------------
// Point-in-polygon (ray casting)
// ---------------------------------------------------------------------------

/// Test whether a 2D point lies inside a simple polygon using ray casting.
///
/// Casts a horizontal ray from the point towards +x infinity and counts
/// edge crossings. An odd number of crossings means the point is inside.
///
/// # Arguments
///
/// * `px`, `py` - Coordinates of the query point
/// * `polygon` - Polygon vertices (n x 2), assumed in order (CW or CCW)
///
/// # Returns
///
/// `true` if the point is inside or on the boundary.
pub fn point_in_polygon_test<T: Float>(
    px: T,
    py: T,
    polygon: &ArrayView2<T>,
) -> SpatialResult<bool> {
    let n = polygon.nrows();
    if n < 3 {
        return Err(SpatialError::ValueError(
            "Polygon must have at least 3 vertices".to_string(),
        ));
    }
    if polygon.ncols() != 2 {
        return Err(SpatialError::DimensionError(
            "Polygon must have exactly 2 columns (x, y)".to_string(),
        ));
    }

    let eps = safe_from::<T>(1e-12, "pip epsilon")?;

    // Check boundary first
    for i in 0..n {
        let j = (i + 1) % n;
        let x1 = polygon[[i, 0]];
        let y1 = polygon[[i, 1]];
        let x2 = polygon[[j, 0]];
        let y2 = polygon[[j, 1]];
        if point_on_segment(px, py, x1, y1, x2, y2, eps) {
            return Ok(true);
        }
    }

    // Ray casting
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let yi = polygon[[i, 1]];
        let yj = polygon[[j, 1]];
        let xi = polygon[[i, 0]];
        let xj = polygon[[j, 0]];

        let crosses = ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
        if crosses {
            inside = !inside;
        }
        j = i;
    }

    Ok(inside)
}

/// Batch point-in-polygon test: test many points against a single polygon.
///
/// Returns a boolean vector with one entry per point.
pub fn batch_point_in_polygon<T: Float>(
    points: &ArrayView2<T>,
    polygon: &ArrayView2<T>,
) -> SpatialResult<Vec<bool>> {
    if points.ncols() != 2 {
        return Err(SpatialError::DimensionError(
            "Points must have 2 columns".to_string(),
        ));
    }
    let mut results = Vec::with_capacity(points.nrows());
    for i in 0..points.nrows() {
        let inside = point_in_polygon_test(points[[i, 0]], points[[i, 1]], polygon)?;
        results.push(inside);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Polygon intersection detection
// ---------------------------------------------------------------------------

/// Test whether two simple polygons intersect (share any area or boundary).
///
/// Two polygons intersect if:
/// 1. Any vertex of one polygon is inside the other, or
/// 2. Any pair of edges from the two polygons cross.
///
/// # Arguments
///
/// * `poly_a`, `poly_b` - Polygon vertices (n x 2 each)
pub fn polygons_intersect<T: Float>(
    poly_a: &ArrayView2<T>,
    poly_b: &ArrayView2<T>,
) -> SpatialResult<bool> {
    let na = poly_a.nrows();
    let nb = poly_b.nrows();
    if na < 3 || nb < 3 {
        return Err(SpatialError::ValueError(
            "Both polygons must have at least 3 vertices".to_string(),
        ));
    }
    if poly_a.ncols() != 2 || poly_b.ncols() != 2 {
        return Err(SpatialError::DimensionError(
            "Polygons must have 2 columns".to_string(),
        ));
    }

    // Check if any vertex of A is inside B
    for i in 0..na {
        if point_in_polygon_test(poly_a[[i, 0]], poly_a[[i, 1]], poly_b)? {
            return Ok(true);
        }
    }
    // Check if any vertex of B is inside A
    for i in 0..nb {
        if point_in_polygon_test(poly_b[[i, 0]], poly_b[[i, 1]], poly_a)? {
            return Ok(true);
        }
    }

    // Check edge-edge intersections
    for i in 0..na {
        let i2 = (i + 1) % na;
        let a1 = (poly_a[[i, 0]], poly_a[[i, 1]]);
        let a2 = (poly_a[[i2, 0]], poly_a[[i2, 1]]);

        for j in 0..nb {
            let j2 = (j + 1) % nb;
            let b1 = (poly_b[[j, 0]], poly_b[[j, 1]]);
            let b2 = (poly_b[[j2, 0]], poly_b[[j2, 1]]);

            if segments_cross(a1, a2, b1, b2) {
                return Ok(true);
            }
        }
    }

    Ok(false)
}

// ---------------------------------------------------------------------------
// Nearest feature: point to line segment
// ---------------------------------------------------------------------------

/// Result of a nearest-feature query from a point to a line segment.
#[derive(Clone, Debug)]
pub struct NearestFeatureResult<T: Float> {
    /// Distance from the query point to the nearest point on the segment
    pub distance: T,
    /// The nearest point on the segment (x, y)
    pub nearest_point: (T, T),
    /// Parameter t in [0, 1] indicating position along the segment
    /// (0.0 = at start, 1.0 = at end)
    pub t: T,
}

/// Find the nearest point on a line segment to a given query point.
///
/// # Arguments
///
/// * `px`, `py` - Query point
/// * `ax`, `ay` - Start of segment
/// * `bx`, `by` - End of segment
pub fn nearest_point_on_segment<T: Float>(
    px: T,
    py: T,
    ax: T,
    ay: T,
    bx: T,
    by: T,
) -> NearestFeatureResult<T> {
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;

    let t = if len_sq.is_zero() {
        T::zero()
    } else {
        let raw_t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
        clamp(raw_t, T::zero(), T::one())
    };

    let nx = ax + t * dx;
    let ny = ay + t * dy;
    let dist = ((px - nx) * (px - nx) + (py - ny) * (py - ny)).sqrt();

    NearestFeatureResult {
        distance: dist,
        nearest_point: (nx, ny),
        t,
    }
}

/// Find the nearest edge of a polygon to a query point.
///
/// Returns `(edge_index, NearestFeatureResult)` where `edge_index` is
/// the index of the starting vertex of the nearest edge.
pub fn nearest_polygon_edge<T: Float>(
    px: T,
    py: T,
    polygon: &ArrayView2<T>,
) -> SpatialResult<(usize, NearestFeatureResult<T>)> {
    let n = polygon.nrows();
    if n < 2 {
        return Err(SpatialError::ValueError(
            "Polygon must have at least 2 vertices".to_string(),
        ));
    }
    if polygon.ncols() != 2 {
        return Err(SpatialError::DimensionError(
            "Polygon must have 2 columns".to_string(),
        ));
    }

    let mut best_edge = 0;
    let mut best_result = nearest_point_on_segment(
        px,
        py,
        polygon[[0, 0]],
        polygon[[0, 1]],
        polygon[[1 % n, 0]],
        polygon[[1 % n, 1]],
    );

    for i in 1..n {
        let j = (i + 1) % n;
        let result = nearest_point_on_segment(
            px,
            py,
            polygon[[i, 0]],
            polygon[[i, 1]],
            polygon[[j, 0]],
            polygon[[j, 1]],
        );
        if result.distance < best_result.distance {
            best_result = result;
            best_edge = i;
        }
    }

    Ok((best_edge, best_result))
}

// ---------------------------------------------------------------------------
// Spatial join between two point sets
// ---------------------------------------------------------------------------

/// Result of a spatial join operation.
#[derive(Clone, Debug)]
pub struct JoinPair<T: Float> {
    /// Index in the first set
    pub idx_a: usize,
    /// Index in the second set
    pub idx_b: usize,
    /// Distance between the two points
    pub distance: T,
}

/// Spatial join: find all pairs of points (one from each set) within a
/// given distance threshold.
///
/// Uses a brute-force approach suitable for moderate-sized datasets.
/// For very large datasets, consider building a spatial index first.
///
/// # Arguments
///
/// * `points_a` - First point set (n x d)
/// * `points_b` - Second point set (m x d)
/// * `max_distance` - Maximum distance for a pair to be included
///
/// # Returns
///
/// A vector of `JoinPair`s sorted by distance.
pub fn within_distance_join<T: Float>(
    points_a: &ArrayView2<T>,
    points_b: &ArrayView2<T>,
    max_distance: T,
) -> SpatialResult<Vec<JoinPair<T>>> {
    if points_a.ncols() != points_b.ncols() {
        return Err(SpatialError::DimensionError(format!(
            "Dimension mismatch: A has {} cols, B has {} cols",
            points_a.ncols(),
            points_b.ncols()
        )));
    }
    if max_distance < T::zero() {
        return Err(SpatialError::ValueError(
            "max_distance must be non-negative".to_string(),
        ));
    }

    let d = points_a.ncols();
    let max_dist_sq = max_distance * max_distance;
    let mut pairs: Vec<JoinPair<T>> = Vec::new();

    for i in 0..points_a.nrows() {
        for j in 0..points_b.nrows() {
            let mut dist_sq = T::zero();
            for k in 0..d {
                let diff = points_a[[i, k]] - points_b[[j, k]];
                dist_sq = dist_sq + diff * diff;
            }
            if dist_sq <= max_dist_sq {
                pairs.push(JoinPair {
                    idx_a: i,
                    idx_b: j,
                    distance: dist_sq.sqrt(),
                });
            }
        }
    }

    pairs.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(pairs)
}

/// Nearest-neighbor join: for each point in A, find the nearest point in B.
///
/// Returns `(indices_in_b, distances)` with one entry per point in A.
pub fn nearest_neighbor_join<T: Float>(
    points_a: &ArrayView2<T>,
    points_b: &ArrayView2<T>,
) -> SpatialResult<(Vec<usize>, Vec<T>)> {
    if points_a.ncols() != points_b.ncols() {
        return Err(SpatialError::DimensionError(format!(
            "Dimension mismatch: A has {} cols, B has {} cols",
            points_a.ncols(),
            points_b.ncols()
        )));
    }
    if points_b.nrows() == 0 {
        return Err(SpatialError::ValueError(
            "Point set B must not be empty".to_string(),
        ));
    }

    let d = points_a.ncols();
    let na = points_a.nrows();
    let nb = points_b.nrows();

    let mut best_idx = vec![0usize; na];
    let mut best_dist = vec![T::infinity(); na];

    for i in 0..na {
        for j in 0..nb {
            let mut dist_sq = T::zero();
            for k in 0..d {
                let diff = points_a[[i, k]] - points_b[[j, k]];
                dist_sq = dist_sq + diff * diff;
            }
            let dist = dist_sq.sqrt();
            if dist < best_dist[i] {
                best_dist[i] = dist;
                best_idx[i] = j;
            }
        }
    }

    Ok((best_idx, best_dist))
}

// ---------------------------------------------------------------------------
// Cross-distance matrix
// ---------------------------------------------------------------------------

/// Compute the full pairwise distance matrix between two point sets.
///
/// Returns an (n x m) matrix where element (i, j) is the Euclidean
/// distance between point i in A and point j in B.
///
/// # Arguments
///
/// * `points_a` - First point set (n x d)
/// * `points_b` - Second point set (m x d)
pub fn cross_distance_matrix<T: Float>(
    points_a: &ArrayView2<T>,
    points_b: &ArrayView2<T>,
) -> SpatialResult<Array2<T>> {
    if points_a.ncols() != points_b.ncols() {
        return Err(SpatialError::DimensionError(format!(
            "Dimension mismatch: A has {} cols, B has {} cols",
            points_a.ncols(),
            points_b.ncols()
        )));
    }

    let na = points_a.nrows();
    let nb = points_b.nrows();
    let d = points_a.ncols();
    let mut result = Array2::zeros((na, nb));

    for i in 0..na {
        for j in 0..nb {
            let mut dist_sq = T::zero();
            for k in 0..d {
                let diff = points_a[[i, k]] - points_b[[j, k]];
                dist_sq = dist_sq + diff * diff;
            }
            result[[i, j]] = dist_sq.sqrt();
        }
    }

    Ok(result)
}

/// Compute the self-distance matrix (pairwise distances within a single set).
///
/// Returns an (n x n) symmetric matrix.
pub fn self_distance_matrix<T: Float>(points: &ArrayView2<T>) -> SpatialResult<Array2<T>> {
    let n = points.nrows();
    let d = points.ncols();
    let mut result = Array2::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let mut dist_sq = T::zero();
            for k in 0..d {
                let diff = points[[i, k]] - points[[j, k]];
                dist_sq = dist_sq + diff * diff;
            }
            let dist = dist_sq.sqrt();
            result[[i, j]] = dist;
            result[[j, i]] = dist;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Clamp value to [lo, hi]
fn clamp<T: Float>(val: T, lo: T, hi: T) -> T {
    if val < lo {
        lo
    } else if val > hi {
        hi
    } else {
        val
    }
}

/// Test whether point (px, py) lies on segment (x1,y1)-(x2,y2) within epsilon.
fn point_on_segment<T: Float>(px: T, py: T, x1: T, y1: T, x2: T, y2: T, eps: T) -> bool {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len_sq = dx * dx + dy * dy;
    if len_sq.is_zero() {
        let d = ((px - x1) * (px - x1) + (py - y1) * (py - y1)).sqrt();
        return d < eps;
    }
    let t = ((px - x1) * dx + (py - y1) * dy) / len_sq;
    if t < T::zero() - eps || t > T::one() + eps {
        return false;
    }
    let t_clamped = clamp(t, T::zero(), T::one());
    let proj_x = x1 + t_clamped * dx;
    let proj_y = y1 + t_clamped * dy;
    let d = ((px - proj_x) * (px - proj_x) + (py - proj_y) * (py - proj_y)).sqrt();
    d < eps
}

/// Test whether two line segments properly cross (not just touch at endpoints).
fn segments_cross<T: Float>(a1: (T, T), a2: (T, T), b1: (T, T), b2: (T, T)) -> bool {
    let d1 = cross_product_sign(b1, b2, a1);
    let d2 = cross_product_sign(b1, b2, a2);
    let d3 = cross_product_sign(a1, a2, b1);
    let d4 = cross_product_sign(a1, a2, b2);

    if ((d1 > T::zero() && d2 < T::zero()) || (d1 < T::zero() && d2 > T::zero()))
        && ((d3 > T::zero() && d4 < T::zero()) || (d3 < T::zero() && d4 > T::zero()))
    {
        return true;
    }

    // Collinear cases
    if d1.is_zero() && on_segment_1d(b1, b2, a1) {
        return true;
    }
    if d2.is_zero() && on_segment_1d(b1, b2, a2) {
        return true;
    }
    if d3.is_zero() && on_segment_1d(a1, a2, b1) {
        return true;
    }
    if d4.is_zero() && on_segment_1d(a1, a2, b2) {
        return true;
    }

    false
}

fn cross_product_sign<T: Float>(o: (T, T), a: (T, T), b: (T, T)) -> T {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

fn on_segment_1d<T: Float>(p: (T, T), q: (T, T), r: (T, T)) -> bool {
    r.0 <= p.0.max(q.0) && r.0 >= p.0.min(q.0) && r.1 <= p.1.max(q.1) && r.1 >= p.1.min(q.1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    // --- point_in_polygon_test ---

    #[test]
    fn test_pip_inside() {
        let poly = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(point_in_polygon_test(0.5, 0.5, &poly.view()).expect("pip"));
    }

    #[test]
    fn test_pip_outside() {
        let poly = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(!point_in_polygon_test(2.0, 2.0, &poly.view()).expect("pip"));
    }

    #[test]
    fn test_pip_on_edge() {
        let poly = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(point_in_polygon_test(0.5, 0.0, &poly.view()).expect("pip"));
    }

    #[test]
    fn test_pip_on_vertex() {
        let poly = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(point_in_polygon_test(0.0, 0.0, &poly.view()).expect("pip"));
    }

    #[test]
    fn test_pip_triangle() {
        let tri = array![[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]];
        assert!(point_in_polygon_test(2.0, 1.0, &tri.view()).expect("pip"));
        assert!(!point_in_polygon_test(0.0, 3.0, &tri.view()).expect("pip"));
    }

    #[test]
    fn test_pip_concave() {
        let poly = array![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0]];
        // Inside main body
        assert!(point_in_polygon_test(0.5, 0.5, &poly.view()).expect("pip"));
        // In the concavity
        assert!(!point_in_polygon_test(1.0, 1.5, &poly.view()).expect("pip"));
    }

    #[test]
    fn test_pip_degenerate() {
        let line = array![[0.0, 0.0], [1.0, 0.0]];
        assert!(point_in_polygon_test(0.0, 0.0, &line.view()).is_err());
    }

    // --- batch_point_in_polygon ---

    #[test]
    fn test_batch_pip() {
        let poly = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let pts = array![[0.5, 0.5], [2.0, 2.0], [0.5, 0.0],];
        let results = batch_point_in_polygon(&pts.view(), &poly.view()).expect("batch");
        assert_eq!(results, vec![true, false, true]);
    }

    // --- polygons_intersect ---

    #[test]
    fn test_polys_overlapping() {
        let a = array![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let b = array![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        assert!(polygons_intersect(&a.view(), &b.view()).expect("intersect"));
    }

    #[test]
    fn test_polys_disjoint() {
        let a = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let b = array![[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]];
        assert!(!polygons_intersect(&a.view(), &b.view()).expect("intersect"));
    }

    #[test]
    fn test_polys_contained() {
        let outer = array![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let inner = array![[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]];
        assert!(polygons_intersect(&outer.view(), &inner.view()).expect("intersect"));
    }

    #[test]
    fn test_polys_edge_touch() {
        let a = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let b = array![[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]];
        // They share an edge - point_in_polygon should detect the shared vertex
        assert!(polygons_intersect(&a.view(), &b.view()).expect("intersect"));
    }

    // --- nearest_point_on_segment ---

    #[test]
    fn test_nearest_on_segment_middle() {
        let result = nearest_point_on_segment(1.0, 1.0, 0.0, 0.0, 2.0, 0.0);
        assert_relative_eq!(result.distance, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.nearest_point.0, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.nearest_point.1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.t, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_nearest_on_segment_start() {
        let result = nearest_point_on_segment(-1.0, 0.0, 0.0, 0.0, 2.0, 0.0);
        assert_relative_eq!(result.nearest_point.0, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.t, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nearest_on_segment_end() {
        let result = nearest_point_on_segment(3.0, 0.0, 0.0, 0.0, 2.0, 0.0);
        assert_relative_eq!(result.nearest_point.0, 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.t, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_nearest_on_degenerate_segment() {
        let result = nearest_point_on_segment(1.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        assert_relative_eq!(result.distance, 2.0f64.sqrt(), epsilon = 1e-10);
    }

    // --- nearest_polygon_edge ---

    #[test]
    fn test_nearest_polygon_edge() {
        let poly = array![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let (edge, result) = nearest_polygon_edge(1.0, -0.5, &poly.view()).expect("nearest edge");
        assert_eq!(edge, 0); // bottom edge
        assert_relative_eq!(result.distance, 0.5, epsilon = 1e-10);
    }

    // --- within_distance_join ---

    #[test]
    fn test_within_distance_join_basic() {
        let a = array![[0.0, 0.0], [1.0, 0.0]];
        let b = array![[0.5, 0.0], [5.0, 5.0]];

        let pairs = within_distance_join(&a.view(), &b.view(), 0.6).expect("join");
        assert_eq!(pairs.len(), 2); // (0,0) with dist 0.5, (1,0) with dist 0.5
        for p in &pairs {
            assert_eq!(p.idx_b, 0);
            assert_relative_eq!(p.distance, 0.5, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_within_distance_join_empty() {
        let a = array![[0.0, 0.0]];
        let b = array![[10.0, 10.0]];

        let pairs = within_distance_join(&a.view(), &b.view(), 1.0).expect("join");
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_within_distance_dim_mismatch() {
        let a = array![[0.0, 0.0]];
        let b = array![[1.0, 1.0, 1.0]];
        let result = within_distance_join(&a.view(), &b.view(), 1.0);
        assert!(result.is_err());
    }

    // --- nearest_neighbor_join ---

    #[test]
    fn test_nn_join() {
        let a = array![[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]];
        let b = array![[0.1, 0.1], [4.9, 4.9]];

        let (idx, dist) = nearest_neighbor_join(&a.view(), &b.view()).expect("nn join");
        assert_eq!(idx.len(), 3);
        assert_eq!(idx[0], 0); // (0,0) -> (0.1,0.1)
        assert_eq!(idx[1], 0); // (1,1) -> (0.1,0.1)
        assert_eq!(idx[2], 1); // (5,5) -> (4.9,4.9)

        assert_relative_eq!(dist[0], (0.02f64).sqrt(), epsilon = 1e-10);
    }

    // --- cross_distance_matrix ---

    #[test]
    fn test_cross_distance_matrix() {
        let a = array![[0.0, 0.0], [1.0, 0.0]];
        let b = array![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let dm = cross_distance_matrix(&a.view(), &b.view()).expect("cdist");
        assert_eq!(dm.shape(), &[2, 3]);

        // a[0] to b[0] = 0
        assert_relative_eq!(dm[[0, 0]], 0.0, epsilon = 1e-10);
        // a[0] to b[1] = 1
        assert_relative_eq!(dm[[0, 1]], 1.0, epsilon = 1e-10);
        // a[0] to b[2] = sqrt(2)
        assert_relative_eq!(dm[[0, 2]], 2.0f64.sqrt(), epsilon = 1e-10);
        // a[1] to b[0] = 1
        assert_relative_eq!(dm[[1, 0]], 1.0, epsilon = 1e-10);
    }

    // --- self_distance_matrix ---

    #[test]
    fn test_self_distance_matrix() {
        let pts = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let dm = self_distance_matrix(&pts.view()).expect("pdist");
        assert_eq!(dm.shape(), &[3, 3]);

        // Diagonal is 0
        for i in 0..3 {
            assert_relative_eq!(dm[[i, i]], 0.0, epsilon = 1e-10);
        }
        // Symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(dm[[i, j]], dm[[j, i]], epsilon = 1e-10);
            }
        }
        // d(0,1) = 1
        assert_relative_eq!(dm[[0, 1]], 1.0, epsilon = 1e-10);
        // d(0,2) = 1
        assert_relative_eq!(dm[[0, 2]], 1.0, epsilon = 1e-10);
        // d(1,2) = sqrt(2)
        assert_relative_eq!(dm[[1, 2]], 2.0f64.sqrt(), epsilon = 1e-10);
    }

    // --- higher dimensional ---

    #[test]
    fn test_cross_distance_3d() {
        let a = array![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        let b = array![[1.0, 0.0, 0.0]];
        let dm = cross_distance_matrix(&a.view(), &b.view()).expect("cdist 3d");
        assert_relative_eq!(dm[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(dm[[1, 0]], (1.0 + 1.0f64).sqrt(), epsilon = 1e-10);
    }

    // --- join sorted order ---

    #[test]
    fn test_join_sorted_by_distance() {
        let a = array![[0.0, 0.0]];
        let b = array![[3.0, 0.0], [1.0, 0.0], [2.0, 0.0]];

        let pairs = within_distance_join(&a.view(), &b.view(), 5.0).expect("join");
        assert_eq!(pairs.len(), 3);
        assert!(pairs[0].distance <= pairs[1].distance);
        assert!(pairs[1].distance <= pairs[2].distance);
    }
}
