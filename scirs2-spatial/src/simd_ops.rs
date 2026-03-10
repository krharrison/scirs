//! SIMD-accelerated operations for spatial algorithms
//!
//! This module provides high-performance SIMD implementations for critical spatial operations:
//! - Distance computations (Euclidean, Manhattan, Chebyshev, Minkowski, Cosine)
//! - KD-Tree operations (bounding box tests, point-to-box distance)
//! - Nearest neighbor search (batch distance, priority queues, radius search)
//!
//! All operations use `scirs2_core::simd::SimdUnifiedOps` for optimal hardware utilization.
//!
//! # Architecture Support
//!
//! The SIMD operations are automatically optimized based on available hardware:
//! - AVX-512 (8x f64 vectors)
//! - AVX2 (4x f64 vectors)
//! - ARM NEON (2x f64 vectors)
//! - SSE (2x f64 vectors - fallback)
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::simd_ops::{simd_euclidean_distance, simd_batch_distances};
//! use scirs2_core::ndarray::array;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Single distance computation
//! let a = array![1.0, 2.0, 3.0];
//! let b = array![4.0, 5.0, 6.0];
//! let dist = simd_euclidean_distance(&a.view(), &b.view())?;
//!
//! // Batch distance computation
//! let points1 = array![[1.0, 2.0], [3.0, 4.0]];
//! let points2 = array![[2.0, 3.0], [4.0, 5.0]];
//! let distances = simd_batch_distances(&points1.view(), &points2.view())?;
//! # Ok(())
//! # }
//! ```

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::simd_ops::SimdUnifiedOps;

// ============================================================================
// Distance Computations (SIMD-accelerated)
// ============================================================================

/// SIMD-accelerated Euclidean distance between two points
///
/// Computes: `sqrt(sum((a[i] - b[i])^2))`
///
/// # Arguments
///
/// * `a` - First point
/// * `b` - Second point
///
/// # Returns
///
/// * Euclidean distance between the points
///
/// # Errors
///
/// Returns error if points have different dimensions
///
/// # Examples
///
/// ```
/// use scirs2_spatial::simd_ops::simd_euclidean_distance;
/// use scirs2_core::ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![4.0, 5.0, 6.0];
/// let dist = simd_euclidean_distance(&a.view(), &b.view()).unwrap();
/// assert!((dist - 5.196152422706632).abs() < 1e-10);
/// ```
pub fn simd_euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    let diff = f64::simd_sub(a, b);
    let squared = f64::simd_mul(&diff.view(), &diff.view());
    let sum = f64::simd_sum(&squared.view());
    Ok(sum.sqrt())
}

/// SIMD-accelerated squared Euclidean distance between two points
///
/// Computes: `sum((a[i] - b[i])^2)`
/// Faster than full Euclidean distance as it avoids the square root operation.
///
/// # Arguments
///
/// * `a` - First point
/// * `b` - Second point
///
/// # Returns
///
/// * Squared Euclidean distance between the points
///
/// # Errors
///
/// Returns error if points have different dimensions
pub fn simd_squared_euclidean_distance(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    let diff = f64::simd_sub(a, b);
    let squared = f64::simd_mul(&diff.view(), &diff.view());
    Ok(f64::simd_sum(&squared.view()))
}

/// SIMD-accelerated Manhattan distance between two points
///
/// Computes: `sum(|a[i] - b[i]|)`
///
/// # Arguments
///
/// * `a` - First point
/// * `b` - Second point
///
/// # Returns
///
/// * Manhattan (L1) distance between the points
///
/// # Errors
///
/// Returns error if points have different dimensions
///
/// # Examples
///
/// ```
/// use scirs2_spatial::simd_ops::simd_manhattan_distance;
/// use scirs2_core::ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![4.0, 5.0, 6.0];
/// let dist = simd_manhattan_distance(&a.view(), &b.view()).unwrap();
/// assert_eq!(dist, 9.0);
/// ```
pub fn simd_manhattan_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    let diff = f64::simd_sub(a, b);
    let abs_diff = f64::simd_abs(&diff.view());
    Ok(f64::simd_sum(&abs_diff.view()))
}

/// SIMD-accelerated Chebyshev distance between two points
///
/// Computes: `max(|a[i] - b[i]|)`
///
/// # Arguments
///
/// * `a` - First point
/// * `b` - Second point
///
/// # Returns
///
/// * Chebyshev (L∞) distance between the points
///
/// # Errors
///
/// Returns error if points have different dimensions
///
/// # Examples
///
/// ```
/// use scirs2_spatial::simd_ops::simd_chebyshev_distance;
/// use scirs2_core::ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![4.0, 6.0, 5.0];
/// let dist = simd_chebyshev_distance(&a.view(), &b.view()).unwrap();
/// assert_eq!(dist, 4.0); // max(|1-4|, |2-6|, |3-5|) = max(3, 4, 2) = 4
/// ```
pub fn simd_chebyshev_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    let diff = f64::simd_sub(a, b);
    let abs_diff = f64::simd_abs(&diff.view());
    Ok(f64::simd_max_element(&abs_diff.view()))
}

/// SIMD-accelerated Minkowski distance between two points
///
/// Computes: `(sum(|a[i] - b[i]|^p))^(1/p)`
///
/// # Arguments
///
/// * `a` - First point
/// * `b` - Second point
/// * `p` - Order of the norm (p >= 1.0)
///
/// # Returns
///
/// * Minkowski distance of order p
///
/// # Errors
///
/// Returns error if:
/// - Points have different dimensions
/// - p < 1.0
///
/// # Special Cases
///
/// - p = 1.0: Manhattan distance
/// - p = 2.0: Euclidean distance
/// - p → ∞: Chebyshev distance
///
/// # Examples
///
/// ```
/// use scirs2_spatial::simd_ops::simd_minkowski_distance;
/// use scirs2_core::ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![4.0, 5.0, 6.0];
/// let dist = simd_minkowski_distance(&a.view(), &b.view(), 3.0).unwrap();
/// assert!((dist - 4.3267487109222245).abs() < 1e-10);
/// ```
pub fn simd_minkowski_distance(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    p: f64,
) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    if p < 1.0 {
        return Err(SpatialError::ValueError(
            "Minkowski p must be >= 1.0".to_string(),
        ));
    }

    // Special cases for efficiency
    if (p - 1.0).abs() < 1e-10 {
        return simd_manhattan_distance(a, b);
    }
    if (p - 2.0).abs() < 1e-10 {
        return simd_euclidean_distance(a, b);
    }

    let diff = f64::simd_sub(a, b);
    let abs_diff = f64::simd_abs(&diff.view());
    let powered = f64::simd_powf(&abs_diff.view(), p);
    let sum = f64::simd_sum(&powered.view());
    Ok(sum.powf(1.0 / p))
}

/// SIMD-accelerated Cosine distance between two points
///
/// Computes: 1 - (a · b) / (||a|| * ||b||)
/// where · is dot product and ||·|| is L2 norm
///
/// # Arguments
///
/// * `a` - First point
/// * `b` - Second point
///
/// # Returns
///
/// * Cosine distance (1 - cosine similarity)
///
/// # Errors
///
/// Returns error if:
/// - Points have different dimensions
/// - Either point is zero vector
///
/// # Examples
///
/// ```
/// use scirs2_spatial::simd_ops::simd_cosine_distance;
/// use scirs2_core::ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![4.0, 5.0, 6.0];
/// let dist = simd_cosine_distance(&a.view(), &b.view()).unwrap();
/// assert!(dist < 0.03); // Very similar direction
/// ```
pub fn simd_cosine_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SpatialResult<f64> {
    if a.len() != b.len() {
        return Err(SpatialError::ValueError(
            "Points must have the same dimension".to_string(),
        ));
    }

    let dot_product = f64::simd_dot(a, b);
    let norm_a = f64::simd_norm(a);
    let norm_b = f64::simd_norm(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(SpatialError::ValueError(
            "Cannot compute cosine distance for zero vectors".to_string(),
        ));
    }

    let cosine_similarity = dot_product / (norm_a * norm_b);
    Ok(1.0 - cosine_similarity)
}

// ============================================================================
// KD-Tree Operations (SIMD-accelerated)
// ============================================================================

/// SIMD-accelerated point-to-axis-aligned-box minimum distance
///
/// Computes the minimum distance from a point to an axis-aligned bounding box.
/// Used for efficient KD-Tree traversal.
///
/// # Arguments
///
/// * `point` - Query point
/// * `box_min` - Minimum corner of the bounding box
/// * `box_max` - Maximum corner of the bounding box
///
/// # Returns
///
/// * Squared minimum distance from point to box
///
/// # Errors
///
/// Returns error if dimensions don't match
pub fn simd_point_to_box_min_distance_squared(
    point: &ArrayView1<f64>,
    box_min: &ArrayView1<f64>,
    box_max: &ArrayView1<f64>,
) -> SpatialResult<f64> {
    if point.len() != box_min.len() || point.len() != box_max.len() {
        return Err(SpatialError::ValueError(
            "Point and box dimensions must match".to_string(),
        ));
    }

    // For each dimension, compute the distance to the box
    // If point is inside box in that dimension, distance is 0
    // Otherwise, distance is to the nearest face

    // Clamp point to box: closest_point = clamp(point, box_min, box_max)
    let clamped = f64::simd_clamp(
        point,
        *box_min
            .first()
            .ok_or_else(|| SpatialError::ValueError("Empty array".to_string()))?,
        *box_max
            .first()
            .ok_or_else(|| SpatialError::ValueError("Empty array".to_string()))?,
    );

    // Compute element-wise clamping manually for each dimension
    let mut closest_point = Array1::zeros(point.len());
    for i in 0..point.len() {
        closest_point[i] = point[i].clamp(box_min[i], box_max[i]);
    }

    // Compute squared distance from point to closest point on box
    let diff = f64::simd_sub(point, &closest_point.view());
    let squared = f64::simd_mul(&diff.view(), &diff.view());
    Ok(f64::simd_sum(&squared.view()))
}

/// SIMD-accelerated axis-aligned bounding box intersection test
///
/// Tests if two axis-aligned bounding boxes intersect.
///
/// # Arguments
///
/// * `box1_min` - Minimum corner of first box
/// * `box1_max` - Maximum corner of first box
/// * `box2_min` - Minimum corner of second box
/// * `box2_max` - Maximum corner of second box
///
/// # Returns
///
/// * true if boxes intersect, false otherwise
///
/// # Errors
///
/// Returns error if dimensions don't match
pub fn simd_box_box_intersection(
    box1_min: &ArrayView1<f64>,
    box1_max: &ArrayView1<f64>,
    box2_min: &ArrayView1<f64>,
    box2_max: &ArrayView1<f64>,
) -> SpatialResult<bool> {
    if box1_min.len() != box1_max.len()
        || box1_min.len() != box2_min.len()
        || box1_min.len() != box2_max.len()
    {
        return Err(SpatialError::ValueError(
            "All box dimensions must match".to_string(),
        ));
    }

    // Boxes intersect if they overlap in all dimensions
    // They overlap in dimension i if: box1_max[i] >= box2_min[i] && box1_min[i] <= box2_max[i]

    for i in 0..box1_min.len() {
        if box1_max[i] < box2_min[i] || box1_min[i] > box2_max[i] {
            return Ok(false);
        }
    }

    Ok(true)
}

/// SIMD-accelerated batch distance computation for KD-Tree queries
///
/// Computes squared distances from a query point to multiple data points.
/// Used for efficient k-NN search in KD-Trees.
///
/// # Arguments
///
/// * `query_point` - Query point
/// * `data_points` - Matrix of data points (n_points x n_dims)
///
/// # Returns
///
/// * Array of squared distances
///
/// # Errors
///
/// Returns error if dimensions don't match
pub fn simd_batch_squared_distances(
    query_point: &ArrayView1<f64>,
    data_points: &ArrayView2<f64>,
) -> SpatialResult<Array1<f64>> {
    if query_point.len() != data_points.ncols() {
        return Err(SpatialError::ValueError(
            "Query point dimension must match data points".to_string(),
        ));
    }

    let n_points = data_points.nrows();
    let mut distances = Array1::zeros(n_points);

    for i in 0..n_points {
        let data_point = data_points.row(i);
        let diff = f64::simd_sub(query_point, &data_point);
        let squared = f64::simd_mul(&diff.view(), &diff.view());
        distances[i] = f64::simd_sum(&squared.view());
    }

    Ok(distances)
}

// ============================================================================
// Nearest Neighbor Search Operations (SIMD-accelerated)
// ============================================================================

/// SIMD-accelerated batch distance computation between point sets
///
/// Computes distances between corresponding points in two arrays.
///
/// # Arguments
///
/// * `points1` - First set of points (n_points x n_dims)
/// * `points2` - Second set of points (n_points x n_dims)
///
/// # Returns
///
/// * Array of distances (n_points)
///
/// # Errors
///
/// Returns error if shapes don't match
pub fn simd_batch_distances(
    points1: &ArrayView2<f64>,
    points2: &ArrayView2<f64>,
) -> SpatialResult<Array1<f64>> {
    if points1.shape() != points2.shape() {
        return Err(SpatialError::ValueError(
            "Point arrays must have the same shape".to_string(),
        ));
    }

    let n_points = points1.nrows();
    let mut distances = Array1::zeros(n_points);

    for i in 0..n_points {
        let p1 = points1.row(i);
        let p2 = points2.row(i);
        let diff = f64::simd_sub(&p1, &p2);
        let squared = f64::simd_mul(&diff.view(), &diff.view());
        let sum = f64::simd_sum(&squared.view());
        distances[i] = sum.sqrt();
    }

    Ok(distances)
}

/// SIMD-accelerated k-nearest neighbors distance computation
///
/// Finds k nearest neighbors and their distances using SIMD operations.
///
/// # Arguments
///
/// * `query_point` - Query point
/// * `data_points` - Matrix of data points (n_points x n_dims)
/// * `k` - Number of nearest neighbors to find
///
/// # Returns
///
/// * Tuple of (indices, distances) for k nearest neighbors
///
/// # Errors
///
/// Returns error if:
/// - Dimensions don't match
/// - k > number of data points
/// - k == 0
pub fn simd_knn_search(
    query_point: &ArrayView1<f64>,
    data_points: &ArrayView2<f64>,
    k: usize,
) -> SpatialResult<(Array1<usize>, Array1<f64>)> {
    if query_point.len() != data_points.ncols() {
        return Err(SpatialError::ValueError(
            "Query point dimension must match data points".to_string(),
        ));
    }

    let n_points = data_points.nrows();

    if k == 0 {
        return Err(SpatialError::ValueError(
            "k must be greater than 0".to_string(),
        ));
    }

    if k > n_points {
        return Err(SpatialError::ValueError(format!(
            "k ({}) cannot be larger than number of data points ({})",
            k, n_points
        )));
    }

    // Compute all distances using SIMD
    let squared_distances = simd_batch_squared_distances(query_point, data_points)?;

    // Convert to (distance, index) pairs and partial sort
    let mut indexed_distances: Vec<(f64, usize)> = squared_distances
        .iter()
        .enumerate()
        .map(|(idx, &dist)| (dist, idx))
        .collect();

    // Partial sort to get k smallest elements
    indexed_distances.select_nth_unstable_by(k - 1, |a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sort the k smallest for consistent ordering
    indexed_distances[..k]
        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Extract indices and compute full distances
    let mut indices = Array1::zeros(k);
    let mut distances = Array1::zeros(k);

    for (i, (dist_sq, idx)) in indexed_distances[..k].iter().enumerate() {
        indices[i] = *idx;
        distances[i] = dist_sq.sqrt();
    }

    Ok((indices, distances))
}

/// SIMD-accelerated radius search
///
/// Finds all points within a given radius of a query point.
///
/// # Arguments
///
/// * `query_point` - Query point
/// * `data_points` - Matrix of data points (n_points x n_dims)
/// * `radius` - Search radius
///
/// # Returns
///
/// * Tuple of (indices, distances) for points within radius
///
/// # Errors
///
/// Returns error if:
/// - Dimensions don't match
/// - radius < 0
pub fn simd_radius_search(
    query_point: &ArrayView1<f64>,
    data_points: &ArrayView2<f64>,
    radius: f64,
) -> SpatialResult<(Array1<usize>, Array1<f64>)> {
    if query_point.len() != data_points.ncols() {
        return Err(SpatialError::ValueError(
            "Query point dimension must match data points".to_string(),
        ));
    }

    if radius < 0.0 {
        return Err(SpatialError::ValueError(
            "Radius must be non-negative".to_string(),
        ));
    }

    // Compute all squared distances using SIMD
    let squared_distances = simd_batch_squared_distances(query_point, data_points)?;
    let radius_squared = radius * radius;

    // Filter points within radius
    let mut indices = Vec::new();
    let mut distances = Vec::new();

    for (idx, &dist_sq) in squared_distances.iter().enumerate() {
        if dist_sq <= radius_squared {
            indices.push(idx);
            distances.push(dist_sq.sqrt());
        }
    }

    Ok((Array1::from(indices), Array1::from(distances)))
}

/// SIMD-accelerated pairwise distance matrix computation
///
/// Computes all pairwise distances between points in a dataset.
///
/// # Arguments
///
/// * `points` - Matrix of points (n_points x n_dims)
///
/// # Returns
///
/// * Symmetric distance matrix (n_points x n_points)
pub fn simd_pairwise_distance_matrix(points: &ArrayView2<f64>) -> SpatialResult<Array2<f64>> {
    let n_points = points.nrows();
    let mut distances = Array2::zeros((n_points, n_points));

    // Only compute upper triangle (matrix is symmetric)
    for i in 0..n_points {
        let point_i = points.row(i);

        for j in (i + 1)..n_points {
            let point_j = points.row(j);
            let diff = f64::simd_sub(&point_i, &point_j);
            let squared = f64::simd_mul(&diff.view(), &diff.view());
            let sum = f64::simd_sum(&squared.view());
            let dist = sum.sqrt();

            distances[[i, j]] = dist;
            distances[[j, i]] = dist; // Symmetric
        }
    }

    Ok(distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_simd_euclidean_distance() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let dist =
            simd_euclidean_distance(&a.view(), &b.view()).expect("Distance computation failed");

        // Expected: sqrt(3^2 + 3^2 + 3^2) = sqrt(27) ≈ 5.196
        assert_relative_eq!(dist, 5.196152422706632, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_manhattan_distance() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let dist =
            simd_manhattan_distance(&a.view(), &b.view()).expect("Distance computation failed");

        // Expected: |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
        assert_eq!(dist, 9.0);
    }

    #[test]
    fn test_simd_chebyshev_distance() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 6.0, 5.0];

        let dist =
            simd_chebyshev_distance(&a.view(), &b.view()).expect("Distance computation failed");

        // Expected: max(|1-4|, |2-6|, |3-5|) = max(3, 4, 2) = 4
        assert_eq!(dist, 4.0);
    }

    #[test]
    fn test_simd_minkowski_distance() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        // Test p=1 (Manhattan)
        let dist_p1 = simd_minkowski_distance(&a.view(), &b.view(), 1.0)
            .expect("Distance computation failed");
        assert_eq!(dist_p1, 9.0);

        // Test p=2 (Euclidean)
        let dist_p2 = simd_minkowski_distance(&a.view(), &b.view(), 2.0)
            .expect("Distance computation failed");
        assert_relative_eq!(dist_p2, 5.196152422706632, epsilon = 1e-10);

        // Test p=3
        let dist_p3 = simd_minkowski_distance(&a.view(), &b.view(), 3.0)
            .expect("Distance computation failed");
        assert_relative_eq!(dist_p3, 4.3267487109222245, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_cosine_distance() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let dist = simd_cosine_distance(&a.view(), &b.view()).expect("Distance computation failed");

        // Vectors are in similar direction, distance should be small
        assert!(dist < 0.03);
        assert!(dist >= 0.0);
    }

    #[test]
    fn test_simd_batch_distances() {
        let points1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let points2 = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];

        let distances = simd_batch_distances(&points1.view(), &points2.view())
            .expect("Batch distance computation failed");

        assert_eq!(distances.len(), 3);

        // Each distance should be sqrt(2) ≈ 1.414
        for &dist in distances.iter() {
            assert_relative_eq!(dist, std::f64::consts::SQRT_2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_simd_knn_search() {
        let data_points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]];
        let query = array![0.5, 0.5];

        let (indices, distances) =
            simd_knn_search(&query.view(), &data_points.view(), 3).expect("k-NN search failed");

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);

        // Distances should be sorted
        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_simd_radius_search() {
        let data_points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [5.0, 5.0]];
        let query = array![0.5, 0.5];
        let radius = 1.0;

        let (indices, distances) = simd_radius_search(&query.view(), &data_points.view(), radius)
            .expect("Radius search failed");

        // Should find the 4 close points, not the far one at [5.0, 5.0]
        assert_eq!(indices.len(), 4);

        // All distances should be within radius
        for &dist in distances.iter() {
            assert!(dist <= radius);
        }
    }

    #[test]
    fn test_simd_point_to_box_distance() {
        let point = array![2.0, 2.0];
        let box_min = array![0.0, 0.0];
        let box_max = array![1.0, 1.0];

        let dist_sq =
            simd_point_to_box_min_distance_squared(&point.view(), &box_min.view(), &box_max.view())
                .expect("Point-to-box distance failed");

        // Point is at (2,2), box is [0,1] x [0,1]
        // Nearest point on box is (1,1)
        // Distance squared = (2-1)^2 + (2-1)^2 = 2
        assert_relative_eq!(dist_sq, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_simd_box_intersection() {
        let box1_min = array![0.0, 0.0];
        let box1_max = array![2.0, 2.0];
        let box2_min = array![1.0, 1.0];
        let box2_max = array![3.0, 3.0];

        let intersects = simd_box_box_intersection(
            &box1_min.view(),
            &box1_max.view(),
            &box2_min.view(),
            &box2_max.view(),
        )
        .expect("Box intersection test failed");

        assert!(intersects);

        // Test non-intersecting boxes
        let box3_min = array![10.0, 10.0];
        let box3_max = array![20.0, 20.0];

        let no_intersect = simd_box_box_intersection(
            &box1_min.view(),
            &box1_max.view(),
            &box3_min.view(),
            &box3_max.view(),
        )
        .expect("Box intersection test failed");

        assert!(!no_intersect);
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let a = array![1.0, 2.0];
        let b = array![1.0, 2.0, 3.0];

        assert!(simd_euclidean_distance(&a.view(), &b.view()).is_err());
        assert!(simd_manhattan_distance(&a.view(), &b.view()).is_err());
        assert!(simd_chebyshev_distance(&a.view(), &b.view()).is_err());
        assert!(simd_cosine_distance(&a.view(), &b.view()).is_err());
    }

    #[test]
    fn test_zero_vector_cosine() {
        let a = array![0.0, 0.0, 0.0];
        let b = array![1.0, 2.0, 3.0];

        assert!(simd_cosine_distance(&a.view(), &b.view()).is_err());
    }
}
