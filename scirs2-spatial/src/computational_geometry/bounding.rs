//! Minimum bounding rectangle and computational geometry utilities
//!
//! This module provides algorithms for computing:
//! - Axis-aligned bounding boxes (AABB)
//! - Minimum area bounding rectangles (using rotating calipers on convex hull)
//! - Oriented bounding boxes
//! - Polygon perimeter computation
//! - Convex hull area and perimeter convenience functions
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::computational_geometry::bounding::{
//!     axis_aligned_bounding_box, minimum_bounding_rectangle, polygon_perimeter,
//! };
//! use scirs2_core::ndarray::array;
//!
//! let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
//!
//! let aabb = axis_aligned_bounding_box(&points.view()).expect("Operation failed");
//! let mbr = minimum_bounding_rectangle(&points.view()).expect("Operation failed");
//! let perim = polygon_perimeter(&points.view());
//! ```

use crate::error::{SpatialError, SpatialResult};
use scirs2_core::ndarray::{Array2, ArrayView2};

/// An axis-aligned bounding box defined by its minimum and maximum coordinates
#[derive(Debug, Clone)]
pub struct AABB {
    /// Minimum corner (min_x, min_y)
    pub min: [f64; 2],
    /// Maximum corner (max_x, max_y)
    pub max: [f64; 2],
}

impl AABB {
    /// Width of the bounding box (x-extent)
    pub fn width(&self) -> f64 {
        self.max[0] - self.min[0]
    }

    /// Height of the bounding box (y-extent)
    pub fn height(&self) -> f64 {
        self.max[1] - self.min[1]
    }

    /// Area of the bounding box
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// Perimeter of the bounding box
    pub fn perimeter(&self) -> f64 {
        2.0 * (self.width() + self.height())
    }

    /// Center of the bounding box
    pub fn center(&self) -> [f64; 2] {
        [
            (self.min[0] + self.max[0]) / 2.0,
            (self.min[1] + self.max[1]) / 2.0,
        ]
    }

    /// Check if a point is inside the bounding box
    pub fn contains(&self, point: &[f64; 2]) -> bool {
        point[0] >= self.min[0]
            && point[0] <= self.max[0]
            && point[1] >= self.min[1]
            && point[1] <= self.max[1]
    }

    /// Check if this AABB intersects another AABB
    pub fn intersects(&self, other: &AABB) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
    }
}

/// An oriented bounding rectangle defined by center, axes, and half-extents
#[derive(Debug, Clone)]
pub struct OrientedBoundingRect {
    /// Center of the rectangle
    pub center: [f64; 2],
    /// First axis direction (unit vector)
    pub axis_u: [f64; 2],
    /// Second axis direction (unit vector, perpendicular to axis_u)
    pub axis_v: [f64; 2],
    /// Half-extent along axis_u
    pub half_width: f64,
    /// Half-extent along axis_v
    pub half_height: f64,
    /// Rotation angle in radians (angle of axis_u from positive x-axis)
    pub angle: f64,
    /// The four corner vertices of the rectangle
    pub corners: [[f64; 2]; 4],
}

impl OrientedBoundingRect {
    /// Area of the oriented bounding rectangle
    pub fn area(&self) -> f64 {
        4.0 * self.half_width * self.half_height
    }

    /// Perimeter of the oriented bounding rectangle
    pub fn perimeter(&self) -> f64 {
        4.0 * (self.half_width + self.half_height)
    }

    /// Check if a point is inside the oriented bounding rectangle
    pub fn contains(&self, point: &[f64; 2]) -> bool {
        // Project the point onto the rectangle's local coordinate system
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];

        let proj_u = dx * self.axis_u[0] + dy * self.axis_u[1];
        let proj_v = dx * self.axis_v[0] + dy * self.axis_v[1];

        proj_u.abs() <= self.half_width && proj_v.abs() <= self.half_height
    }
}

/// Compute the axis-aligned bounding box of a set of 2D points
///
/// # Arguments
///
/// * `points` - A 2D array of points (n x 2)
///
/// # Returns
///
/// * `SpatialResult<AABB>` - The axis-aligned bounding box
///
/// # Examples
///
/// ```
/// use scirs2_spatial::computational_geometry::bounding::axis_aligned_bounding_box;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 1.0], [2.0, 3.0], [1.0, 0.5]];
/// let aabb = axis_aligned_bounding_box(&points.view()).expect("Operation failed");
/// assert!((aabb.min[0] - 0.0).abs() < 1e-10);
/// assert!((aabb.max[0] - 2.0).abs() < 1e-10);
/// ```
pub fn axis_aligned_bounding_box(points: &ArrayView2<'_, f64>) -> SpatialResult<AABB> {
    if points.nrows() == 0 {
        return Err(SpatialError::ValueError(
            "Cannot compute bounding box of empty point set".to_string(),
        ));
    }
    if points.ncols() != 2 {
        return Err(SpatialError::DimensionError(
            "Points must be 2D for bounding box computation".to_string(),
        ));
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for i in 0..points.nrows() {
        let x = points[[i, 0]];
        let y = points[[i, 1]];
        if x < min_x {
            min_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if x > max_x {
            max_x = x;
        }
        if y > max_y {
            max_y = y;
        }
    }

    Ok(AABB {
        min: [min_x, min_y],
        max: [max_x, max_y],
    })
}

/// Compute the minimum area bounding rectangle of a set of 2D points
/// using the rotating calipers algorithm on the convex hull.
///
/// The minimum bounding rectangle is the rectangle with the smallest area
/// that contains all the input points. One edge of this rectangle is always
/// collinear with an edge of the convex hull.
///
/// # Algorithm
///
/// 1. Compute the convex hull of the points
/// 2. For each edge of the hull, compute the bounding rectangle aligned to that edge
/// 3. Return the rectangle with minimum area
///
/// Time complexity: O(n log n) for hull computation + O(h) for rotating calipers,
/// where h is the number of hull vertices.
///
/// # Arguments
///
/// * `points` - A 2D array of points (n x 2)
///
/// # Returns
///
/// * `SpatialResult<OrientedBoundingRect>` - The minimum area bounding rectangle
///
/// # Examples
///
/// ```
/// use scirs2_spatial::computational_geometry::bounding::minimum_bounding_rectangle;
/// use scirs2_core::ndarray::array;
///
/// // A rotated square
/// let points = array![
///     [1.0, 0.0],
///     [2.0, 1.0],
///     [1.0, 2.0],
///     [0.0, 1.0],
/// ];
///
/// let mbr = minimum_bounding_rectangle(&points.view()).expect("Operation failed");
/// // Area should be 2.0 (side = sqrt(2), area = 2)
/// assert!((mbr.area() - 2.0).abs() < 0.1);
/// ```
pub fn minimum_bounding_rectangle(
    points: &ArrayView2<'_, f64>,
) -> SpatialResult<OrientedBoundingRect> {
    if points.nrows() < 3 {
        return Err(SpatialError::ValueError(
            "Need at least 3 points for minimum bounding rectangle".to_string(),
        ));
    }
    if points.ncols() != 2 {
        return Err(SpatialError::DimensionError(
            "Points must be 2D for bounding rectangle computation".to_string(),
        ));
    }

    // Compute convex hull using Graham scan
    let hull = compute_convex_hull_ccw(points)?;
    let n = hull.len();

    if n < 3 {
        return Err(SpatialError::ComputationError(
            "Convex hull has fewer than 3 vertices (degenerate case)".to_string(),
        ));
    }

    let mut best_area = f64::INFINITY;
    let mut best_rect: Option<OrientedBoundingRect> = None;

    // For each edge of the convex hull, compute the aligned bounding rectangle
    for i in 0..n {
        let j = (i + 1) % n;

        // Edge direction
        let edge_dx = hull[j][0] - hull[i][0];
        let edge_dy = hull[j][1] - hull[i][1];
        let edge_len = (edge_dx * edge_dx + edge_dy * edge_dy).sqrt();

        if edge_len < 1e-15 {
            continue;
        }

        // Unit vectors along and perpendicular to the edge
        let ux = edge_dx / edge_len;
        let uy = edge_dy / edge_len;
        let vx = -uy;
        let vy = ux;

        // Project all hull points onto the edge coordinate system
        let mut min_u = f64::INFINITY;
        let mut max_u = f64::NEG_INFINITY;
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;

        for pt in &hull {
            let dx = pt[0] - hull[i][0];
            let dy = pt[1] - hull[i][1];

            let proj_u = dx * ux + dy * uy;
            let proj_v = dx * vx + dy * vy;

            if proj_u < min_u {
                min_u = proj_u;
            }
            if proj_u > max_u {
                max_u = proj_u;
            }
            if proj_v < min_v {
                min_v = proj_v;
            }
            if proj_v > max_v {
                max_v = proj_v;
            }
        }

        let width = max_u - min_u;
        let height = max_v - min_v;
        let area = width * height;

        if area < best_area {
            best_area = area;

            // Compute the center of the rectangle
            let center_u = (min_u + max_u) / 2.0;
            let center_v = (min_v + max_v) / 2.0;

            let center_x = hull[i][0] + center_u * ux + center_v * vx;
            let center_y = hull[i][1] + center_u * uy + center_v * vy;

            // Compute corners
            let hw = width / 2.0;
            let hh = height / 2.0;

            let corners = [
                [center_x - hw * ux - hh * vx, center_y - hw * uy - hh * vy],
                [center_x + hw * ux - hh * vx, center_y + hw * uy - hh * vy],
                [center_x + hw * ux + hh * vx, center_y + hw * uy + hh * vy],
                [center_x - hw * ux + hh * vx, center_y - hw * uy + hh * vy],
            ];

            let angle = uy.atan2(ux);

            best_rect = Some(OrientedBoundingRect {
                center: [center_x, center_y],
                axis_u: [ux, uy],
                axis_v: [vx, vy],
                half_width: hw,
                half_height: hh,
                angle,
                corners,
            });
        }
    }

    best_rect.ok_or_else(|| {
        SpatialError::ComputationError("Failed to compute minimum bounding rectangle".to_string())
    })
}

/// Compute the perimeter (circumference) of a polygon defined by ordered vertices
///
/// # Arguments
///
/// * `vertices` - Polygon vertices in order (n x 2)
///
/// # Returns
///
/// * The perimeter of the polygon
///
/// # Examples
///
/// ```
/// use scirs2_spatial::computational_geometry::bounding::polygon_perimeter;
/// use scirs2_core::ndarray::array;
///
/// let square = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let perim = polygon_perimeter(&square.view());
/// assert!((perim - 4.0).abs() < 1e-10);
/// ```
pub fn polygon_perimeter(vertices: &ArrayView2<'_, f64>) -> f64 {
    let n = vertices.nrows();
    if n < 2 {
        return 0.0;
    }

    let mut perimeter = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = vertices[[j, 0]] - vertices[[i, 0]];
        let dy = vertices[[j, 1]] - vertices[[i, 1]];
        perimeter += (dx * dx + dy * dy).sqrt();
    }

    perimeter
}

/// Compute the signed area of a polygon (positive for CCW, negative for CW)
///
/// Uses the Shoelace formula.
///
/// # Arguments
///
/// * `vertices` - Polygon vertices in order (n x 2)
///
/// # Returns
///
/// * The signed area (positive for counter-clockwise orientation)
pub fn signed_polygon_area(vertices: &ArrayView2<'_, f64>) -> f64 {
    let n = vertices.nrows();
    if n < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[[i, 0]] * vertices[[j, 1]] - vertices[[j, 0]] * vertices[[i, 1]];
    }

    area / 2.0
}

/// Compute the area and perimeter of a convex hull given as a set of points
///
/// First computes the convex hull, then returns its area and perimeter.
///
/// # Arguments
///
/// * `points` - Input points (n x 2)
///
/// # Returns
///
/// * `SpatialResult<(f64, f64)>` - (area, perimeter) of the convex hull
///
/// # Examples
///
/// ```
/// use scirs2_spatial::computational_geometry::bounding::convex_hull_area_perimeter;
/// use scirs2_core::ndarray::array;
///
/// let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]];
/// let (area, perim) = convex_hull_area_perimeter(&points.view()).expect("Operation failed");
/// assert!((area - 1.0).abs() < 1e-10);
/// assert!((perim - 4.0).abs() < 1e-10);
/// ```
pub fn convex_hull_area_perimeter(points: &ArrayView2<'_, f64>) -> SpatialResult<(f64, f64)> {
    if points.nrows() < 3 {
        return Err(SpatialError::ValueError(
            "Need at least 3 points".to_string(),
        ));
    }

    let hull = compute_convex_hull_ccw(points)?;
    let n = hull.len();

    // Compute area using Shoelace formula
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1];
    }
    area = area.abs() / 2.0;

    // Compute perimeter
    let mut perimeter = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        let dx = hull[j][0] - hull[i][0];
        let dy = hull[j][1] - hull[i][1];
        perimeter += (dx * dx + dy * dy).sqrt();
    }

    Ok((area, perimeter))
}

/// Compute just the convex hull area for a point set
///
/// # Arguments
///
/// * `points` - Input points (n x 2)
///
/// # Returns
///
/// * `SpatialResult<f64>` - Area of the convex hull
pub fn convex_hull_area(points: &ArrayView2<'_, f64>) -> SpatialResult<f64> {
    let (area, _) = convex_hull_area_perimeter(points)?;
    Ok(area)
}

/// Compute just the convex hull perimeter for a point set
///
/// # Arguments
///
/// * `points` - Input points (n x 2)
///
/// # Returns
///
/// * `SpatialResult<f64>` - Perimeter of the convex hull
pub fn convex_hull_perimeter(points: &ArrayView2<'_, f64>) -> SpatialResult<f64> {
    let (_, perim) = convex_hull_area_perimeter(points)?;
    Ok(perim)
}

/// Determine if a polygon (given as ordered vertices) is convex
///
/// # Arguments
///
/// * `vertices` - Polygon vertices in order (n x 2)
///
/// # Returns
///
/// * `bool` - True if the polygon is convex
pub fn is_convex(vertices: &ArrayView2<'_, f64>) -> bool {
    let n = vertices.nrows();
    if n < 3 {
        return false;
    }

    let mut sign = 0i32;

    for i in 0..n {
        let j = (i + 1) % n;
        let k = (i + 2) % n;

        let dx1 = vertices[[j, 0]] - vertices[[i, 0]];
        let dy1 = vertices[[j, 1]] - vertices[[i, 1]];
        let dx2 = vertices[[k, 0]] - vertices[[j, 0]];
        let dy2 = vertices[[k, 1]] - vertices[[j, 1]];

        let cross = dx1 * dy2 - dy1 * dx2;

        if cross.abs() > 1e-10 {
            let current_sign = if cross > 0.0 { 1 } else { -1 };
            if sign == 0 {
                sign = current_sign;
            } else if sign != current_sign {
                return false;
            }
        }
    }

    true
}

/// Compute the diameter of a point set (maximum distance between any two points)
///
/// Uses the rotating calipers method on the convex hull for efficiency.
///
/// # Arguments
///
/// * `points` - Input points (n x 2)
///
/// # Returns
///
/// * `SpatialResult<(f64, [f64; 2], [f64; 2])>` - (diameter, point_a, point_b)
pub fn point_set_diameter(
    points: &ArrayView2<'_, f64>,
) -> SpatialResult<(f64, [f64; 2], [f64; 2])> {
    if points.nrows() < 2 {
        return Err(SpatialError::ValueError(
            "Need at least 2 points to compute diameter".to_string(),
        ));
    }

    let hull = compute_convex_hull_ccw(points)?;
    let n = hull.len();

    if n < 2 {
        return Err(SpatialError::ComputationError(
            "Convex hull has fewer than 2 vertices".to_string(),
        ));
    }

    // Rotating calipers to find antipodal pairs
    let mut max_dist = 0.0;
    let mut best_a = hull[0];
    let mut best_b = hull[1];

    // Find initial antipodal pair: start at the bottom and top of the hull
    let mut j = 1;

    for i in 0..n {
        let i_next = (i + 1) % n;

        // Edge direction
        let edge_x = hull[i_next][0] - hull[i][0];
        let edge_y = hull[i_next][1] - hull[i][1];

        // Advance j until the cross product starts decreasing
        loop {
            let j_next = (j + 1) % n;
            let cross_curr =
                edge_x * (hull[j_next][1] - hull[j][1]) - edge_y * (hull[j_next][0] - hull[j][0]);

            if cross_curr > 0.0 {
                j = j_next;
            } else {
                break;
            }
        }

        // Check distance between i and j
        let dx = hull[j][0] - hull[i][0];
        let dy = hull[j][1] - hull[i][1];
        let dist = (dx * dx + dy * dy).sqrt();

        if dist > max_dist {
            max_dist = dist;
            best_a = hull[i];
            best_b = hull[j];
        }
    }

    Ok((max_dist, best_a, best_b))
}

/// Compute the width of a point set (minimum distance across)
///
/// The width is the minimum distance between two parallel lines that enclose all points.
///
/// # Arguments
///
/// * `points` - Input points (n x 2)
///
/// # Returns
///
/// * `SpatialResult<f64>` - The width of the point set
pub fn point_set_width(points: &ArrayView2<'_, f64>) -> SpatialResult<f64> {
    if points.nrows() < 3 {
        return Err(SpatialError::ValueError(
            "Need at least 3 points to compute width".to_string(),
        ));
    }

    let hull = compute_convex_hull_ccw(points)?;
    let n = hull.len();

    let mut min_width = f64::INFINITY;

    // For each edge of the hull, find the farthest point and compute the distance
    for i in 0..n {
        let j = (i + 1) % n;

        let edge_dx = hull[j][0] - hull[i][0];
        let edge_dy = hull[j][1] - hull[i][1];
        let edge_len = (edge_dx * edge_dx + edge_dy * edge_dy).sqrt();

        if edge_len < 1e-15 {
            continue;
        }

        // Normal direction (perpendicular to edge)
        let nx = -edge_dy / edge_len;
        let ny = edge_dx / edge_len;

        // Project all hull points onto the normal direction
        let mut min_proj = f64::INFINITY;
        let mut max_proj = f64::NEG_INFINITY;

        for pt in &hull {
            let proj = pt[0] * nx + pt[1] * ny;
            if proj < min_proj {
                min_proj = proj;
            }
            if proj > max_proj {
                max_proj = proj;
            }
        }

        let width = max_proj - min_proj;
        if width < min_width {
            min_width = width;
        }
    }

    Ok(min_width)
}

/// Internal: compute the convex hull of a point set in CCW order (Graham scan)
fn compute_convex_hull_ccw(points: &ArrayView2<'_, f64>) -> SpatialResult<Vec<[f64; 2]>> {
    let n = points.nrows();

    if n < 3 {
        let mut hull = Vec::with_capacity(n);
        for i in 0..n {
            hull.push([points[[i, 0]], points[[i, 1]]]);
        }
        return Ok(hull);
    }

    // Find the lowest (and leftmost) point
    let mut lowest = 0;
    for i in 1..n {
        if points[[i, 1]] < points[[lowest, 1]]
            || (points[[i, 1]] == points[[lowest, 1]] && points[[i, 0]] < points[[lowest, 0]])
        {
            lowest = i;
        }
    }

    let pivot_x = points[[lowest, 0]];
    let pivot_y = points[[lowest, 1]];

    // Sort points by polar angle relative to pivot
    let mut indexed: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let dx = points[[i, 0]] - pivot_x;
            let dy = points[[i, 1]] - pivot_y;
            let angle = dy.atan2(dx);
            (i, angle)
        })
        .collect();

    indexed.sort_by(|a, b| {
        let angle_cmp = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
        if angle_cmp == std::cmp::Ordering::Equal {
            let dist_a =
                (points[[a.0, 0]] - pivot_x).powi(2) + (points[[a.0, 1]] - pivot_y).powi(2);
            let dist_b =
                (points[[b.0, 0]] - pivot_x).powi(2) + (points[[b.0, 1]] - pivot_y).powi(2);
            dist_a
                .partial_cmp(&dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        } else {
            angle_cmp
        }
    });

    // Graham scan
    let mut hull_indices = vec![lowest];

    for &(idx, _) in &indexed {
        if idx == lowest {
            continue;
        }

        while hull_indices.len() >= 2 {
            let top = hull_indices.len() - 1;
            let a = hull_indices[top - 1];
            let b = hull_indices[top];
            let c = idx;

            let cross = (points[[b, 0]] - points[[a, 0]]) * (points[[c, 1]] - points[[a, 1]])
                - (points[[b, 1]] - points[[a, 1]]) * (points[[c, 0]] - points[[a, 0]]);

            if cross > 0.0 {
                break;
            }
            hull_indices.pop();
        }

        hull_indices.push(idx);
    }

    let hull: Vec<[f64; 2]> = hull_indices
        .iter()
        .map(|&i| [points[[i, 0]], points[[i, 1]]])
        .collect();

    Ok(hull)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_aabb_basic() {
        let points = array![[0.0, 0.0], [2.0, 3.0], [1.0, 1.0], [-1.0, 2.0]];
        let aabb = axis_aligned_bounding_box(&points.view()).expect("Operation failed");

        assert!((aabb.min[0] - (-1.0)).abs() < 1e-10);
        assert!((aabb.min[1] - 0.0).abs() < 1e-10);
        assert!((aabb.max[0] - 2.0).abs() < 1e-10);
        assert!((aabb.max[1] - 3.0).abs() < 1e-10);
        assert!((aabb.area() - 9.0).abs() < 1e-10);
        assert!((aabb.perimeter() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb_contains() {
        let points = array![[0.0, 0.0], [2.0, 2.0]];
        let aabb = axis_aligned_bounding_box(&points.view()).expect("Operation failed");

        assert!(aabb.contains(&[1.0, 1.0]));
        assert!(aabb.contains(&[0.0, 0.0]));
        assert!(aabb.contains(&[2.0, 2.0]));
        assert!(!aabb.contains(&[3.0, 1.0]));
        assert!(!aabb.contains(&[-1.0, 1.0]));
    }

    #[test]
    fn test_aabb_intersects() {
        let p1 = array![[0.0, 0.0], [2.0, 2.0]];
        let p2 = array![[1.0, 1.0], [3.0, 3.0]];
        let p3 = array![[5.0, 5.0], [6.0, 6.0]];

        let aabb1 = axis_aligned_bounding_box(&p1.view()).expect("Operation failed");
        let aabb2 = axis_aligned_bounding_box(&p2.view()).expect("Operation failed");
        let aabb3 = axis_aligned_bounding_box(&p3.view()).expect("Operation failed");

        assert!(aabb1.intersects(&aabb2));
        assert!(!aabb1.intersects(&aabb3));
    }

    #[test]
    fn test_aabb_center() {
        let points = array![[0.0, 0.0], [4.0, 6.0]];
        let aabb = axis_aligned_bounding_box(&points.view()).expect("Operation failed");
        let center = aabb.center();
        assert!((center[0] - 2.0).abs() < 1e-10);
        assert!((center[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_aabb_empty_input() {
        let points = Array2::<f64>::zeros((0, 2));
        let result = axis_aligned_bounding_box(&points.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_polygon_perimeter_square() {
        let square = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let perim = polygon_perimeter(&square.view());
        assert!((perim - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_perimeter_triangle() {
        let triangle = array![[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]];
        let perim = polygon_perimeter(&triangle.view());
        // 3 + 4 + 5 = 12
        assert!((perim - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_signed_area() {
        // CCW square -> positive area
        let ccw = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let area = signed_polygon_area(&ccw.view());
        assert!((area - 1.0).abs() < 1e-10);

        // CW square -> negative area
        let cw = array![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        let area = signed_polygon_area(&cw.view());
        assert!((area - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_convex_hull_area_perimeter_square() {
        let points = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5] // interior point
        ];

        let (area, perim) = convex_hull_area_perimeter(&points.view()).expect("Operation failed");
        assert!((area - 1.0).abs() < 1e-10);
        assert!((perim - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_convex_hull_area_function() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let area = convex_hull_area(&points.view()).expect("Operation failed");
        assert!((area - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_convex_hull_perimeter_function() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let perim = convex_hull_perimeter(&points.view()).expect("Operation failed");
        assert!((perim - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_minimum_bounding_rectangle_square() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mbr = minimum_bounding_rectangle(&points.view()).expect("Operation failed");

        // For an axis-aligned square, the MBR should have area 1.0
        assert!((mbr.area() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_minimum_bounding_rectangle_triangle() {
        let points = array![[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]];
        let mbr = minimum_bounding_rectangle(&points.view()).expect("Operation failed");

        // MBR area should be less than or equal to AABB area
        let aabb = axis_aligned_bounding_box(&points.view()).expect("Operation failed");
        assert!(mbr.area() <= aabb.area() + 1e-6);
    }

    #[test]
    fn test_minimum_bounding_rectangle_contains_all_points() {
        let points = array![[0.0, 0.0], [3.0, 1.0], [2.0, 4.0], [-1.0, 3.0], [1.0, 2.0]];

        let mbr = minimum_bounding_rectangle(&points.view()).expect("Operation failed");

        // All points should be inside the MBR (with tolerance)
        for i in 0..points.nrows() {
            let pt = [points[[i, 0]], points[[i, 1]]];
            // Expand the check tolerance slightly
            let dx = pt[0] - mbr.center[0];
            let dy = pt[1] - mbr.center[1];
            let proj_u = dx * mbr.axis_u[0] + dy * mbr.axis_u[1];
            let proj_v = dx * mbr.axis_v[0] + dy * mbr.axis_v[1];
            assert!(
                proj_u.abs() <= mbr.half_width + 1e-6,
                "Point {} outside MBR along u: {} > {}",
                i,
                proj_u.abs(),
                mbr.half_width
            );
            assert!(
                proj_v.abs() <= mbr.half_height + 1e-6,
                "Point {} outside MBR along v: {} > {}",
                i,
                proj_v.abs(),
                mbr.half_height
            );
        }
    }

    #[test]
    fn test_is_convex() {
        let convex = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        assert!(is_convex(&convex.view()));

        let concave = array![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [1.0, 0.5], // dent inward
            [0.0, 2.0]
        ];
        assert!(!is_convex(&concave.view()));
    }

    #[test]
    fn test_point_set_diameter() {
        let points = array![[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]];
        let (diameter, _, _) = point_set_diameter(&points.view()).expect("Operation failed");
        assert!((diameter - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_set_diameter_square() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let (diameter, _, _) = point_set_diameter(&points.view()).expect("Operation failed");
        // Diagonal of unit square = sqrt(2)
        assert!((diameter - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_point_set_width() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let width = point_set_width(&points.view()).expect("Operation failed");
        assert!((width - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_oriented_bounding_rect_contains() {
        let points = array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let mbr = minimum_bounding_rectangle(&points.view()).expect("Operation failed");

        assert!(mbr.contains(&[0.5, 0.5]));
        assert!(!mbr.contains(&[5.0, 5.0]));
    }
}
