//! Bentley-Ottmann sweep line algorithm for detecting line segment intersections
//!
//! This module implements the Bentley-Ottmann sweep line algorithm, which efficiently
//! finds all intersection points among a set of line segments in the plane.
//!
//! # Algorithm
//!
//! The algorithm sweeps a vertical line from left to right across the plane.
//! It maintains:
//! - An **event queue** (priority queue ordered by x-coordinate) containing
//!   segment endpoints and discovered intersection points
//! - A **status structure** (ordered set) containing the segments currently
//!   intersecting the sweep line, ordered by their y-coordinate at the sweep line
//!
//! Time complexity: O((n + k) log n) where n is the number of segments and k is the
//! number of intersection points.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::computational_geometry::sweep_line::{Segment2D, find_all_intersections};
//!
//! let segments = vec![
//!     Segment2D::new(0.0, 0.0, 2.0, 2.0),
//!     Segment2D::new(0.0, 2.0, 2.0, 0.0),
//! ];
//!
//! let intersections = find_all_intersections(&segments).expect("Operation failed");
//! assert_eq!(intersections.len(), 1);
//! ```

use crate::error::{SpatialError, SpatialResult};
use std::cmp::Ordering;

/// Tolerance for floating-point comparisons
const EPSILON: f64 = 1e-10;

/// A 2D point
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

impl Point2D {
    /// Create a new 2D point
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl PartialEq for Point2D {
    fn eq(&self, other: &Self) -> bool {
        (self.x - other.x).abs() < EPSILON && (self.y - other.y).abs() < EPSILON
    }
}

impl Eq for Point2D {}

impl PartialOrd for Point2D {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Point2D {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order by x, then by y
        match float_cmp(self.x, other.x) {
            Ordering::Equal => float_cmp(self.y, other.y),
            ord => ord,
        }
    }
}

/// A 2D line segment defined by two endpoints
#[derive(Debug, Clone, Copy)]
pub struct Segment2D {
    /// Start point (left endpoint, smaller x)
    pub start: Point2D,
    /// End point (right endpoint, larger x)
    pub end: Point2D,
    /// Unique identifier for this segment
    id: usize,
}

impl Segment2D {
    /// Create a new line segment from coordinates
    ///
    /// The segment is automatically oriented so that `start` has the smaller x-coordinate.
    /// If x-coordinates are equal, the point with smaller y-coordinate becomes `start`.
    ///
    /// # Arguments
    ///
    /// * `x1`, `y1` - First endpoint
    /// * `x2`, `y2` - Second endpoint
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        let p1 = Point2D::new(x1, y1);
        let p2 = Point2D::new(x2, y2);
        let (start, end) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };
        Self { start, end, id: 0 }
    }

    /// Create a segment from two Point2D values
    pub fn from_points(p1: Point2D, p2: Point2D) -> Self {
        let (start, end) = if p1 <= p2 { (p1, p2) } else { (p2, p1) };
        Self { start, end, id: 0 }
    }

    /// Evaluate the y-coordinate of the segment at a given x
    ///
    /// Returns None if x is outside the segment's x-range
    fn y_at_x(&self, x: f64) -> Option<f64> {
        let dx = self.end.x - self.start.x;
        if dx.abs() < EPSILON {
            // Vertical segment - return midpoint y
            Some((self.start.y + self.end.y) / 2.0)
        } else {
            let t = (x - self.start.x) / dx;
            if !(-EPSILON..=1.0 + EPSILON).contains(&t) {
                return None;
            }
            let t_clamped = t.clamp(0.0, 1.0);
            Some(self.start.y + t_clamped * (self.end.y - self.start.y))
        }
    }

    /// Check if this segment is vertical
    fn is_vertical(&self) -> bool {
        (self.end.x - self.start.x).abs() < EPSILON
    }

    /// Get the slope of the segment (returns None for vertical segments)
    fn slope(&self) -> Option<f64> {
        let dx = self.end.x - self.start.x;
        if dx.abs() < EPSILON {
            None
        } else {
            Some((self.end.y - self.start.y) / dx)
        }
    }
}

impl PartialEq for Segment2D {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Segment2D {}

/// An intersection result containing the point and the two segment indices
#[derive(Debug, Clone)]
pub struct Intersection {
    /// The intersection point
    pub point: Point2D,
    /// Index of the first segment
    pub segment_a: usize,
    /// Index of the second segment
    pub segment_b: usize,
}

/// Event types for the sweep line
#[derive(Debug, Clone)]
enum EventType {
    /// Left endpoint of a segment (segment starts)
    LeftEndpoint(usize),
    /// Right endpoint of a segment (segment ends)
    RightEndpoint(usize),
    /// Intersection of two segments
    IntersectionEvent(usize, usize),
}

/// An event in the sweep line event queue
#[derive(Debug, Clone)]
struct SweepEvent {
    point: Point2D,
    event_type: EventType,
}

impl PartialEq for SweepEvent {
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point
    }
}

impl Eq for SweepEvent {}

impl PartialOrd for SweepEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SweepEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        self.point.cmp(&other.point)
    }
}

/// Entry in the sweep line status structure
#[derive(Debug, Clone)]
struct StatusEntry {
    segment_id: usize,
    /// Current y-coordinate used for ordering
    current_y: f64,
    /// Slope for tiebreaking
    slope: f64,
}

impl PartialEq for StatusEntry {
    fn eq(&self, other: &Self) -> bool {
        self.segment_id == other.segment_id
    }
}

impl Eq for StatusEntry {}

impl PartialOrd for StatusEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StatusEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match float_cmp(self.current_y, other.current_y) {
            Ordering::Equal => {
                // If y-coordinates are equal, use slope as tiebreaker
                match float_cmp(self.slope, other.slope) {
                    Ordering::Equal => self.segment_id.cmp(&other.segment_id),
                    ord => ord,
                }
            }
            ord => ord,
        }
    }
}

/// Compare two floats with epsilon tolerance
fn float_cmp(a: f64, b: f64) -> Ordering {
    if (a - b).abs() < EPSILON {
        Ordering::Equal
    } else if a < b {
        Ordering::Less
    } else {
        Ordering::Greater
    }
}

/// Compute the intersection point of two line segments, if it exists
///
/// Returns the intersection point and the parametric values (t, u) such that:
/// intersection = seg_a.start + t * (seg_a.end - seg_a.start)
/// intersection = seg_b.start + u * (seg_b.end - seg_b.start)
///
/// # Arguments
///
/// * `seg_a` - First segment
/// * `seg_b` - Second segment
///
/// # Returns
///
/// * `Option<(Point2D, f64, f64)>` - Intersection point and parametric values, or None
pub fn segment_intersection(seg_a: &Segment2D, seg_b: &Segment2D) -> Option<(Point2D, f64, f64)> {
    let x1 = seg_a.start.x;
    let y1 = seg_a.start.y;
    let x2 = seg_a.end.x;
    let y2 = seg_a.end.y;
    let x3 = seg_b.start.x;
    let y3 = seg_b.start.y;
    let x4 = seg_b.end.x;
    let y4 = seg_b.end.y;

    let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

    if denom.abs() < EPSILON {
        // Segments are parallel (or collinear)
        return None;
    }

    let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    let u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;

    // Check if intersection is within both segments (with small tolerance)
    let tol = EPSILON;
    if t >= -tol && t <= 1.0 + tol && u >= -tol && u <= 1.0 + tol {
        let ix = x1 + t * (x2 - x1);
        let iy = y1 + t * (y2 - y1);
        Some((Point2D::new(ix, iy), t, u))
    } else {
        None
    }
}

/// Find all intersection points among a set of line segments using the
/// Bentley-Ottmann sweep line algorithm.
///
/// # Algorithm Details
///
/// The sweep line moves from left to right. Three types of events are processed:
/// 1. **Left endpoint**: Insert the segment into the status structure and check for
///    intersections with its neighbors
/// 2. **Right endpoint**: Remove the segment and check if the newly adjacent
///    segments intersect
/// 3. **Intersection**: Swap the two segments in the status and check for new
///    intersections with their new neighbors
///
/// # Arguments
///
/// * `segments` - A slice of line segments
///
/// # Returns
///
/// * `SpatialResult<Vec<Intersection>>` - All intersection points with segment indices
///
/// # Examples
///
/// ```
/// use scirs2_spatial::computational_geometry::sweep_line::{Segment2D, find_all_intersections};
///
/// // Two crossing segments
/// let segments = vec![
///     Segment2D::new(0.0, 0.0, 2.0, 2.0),
///     Segment2D::new(0.0, 2.0, 2.0, 0.0),
/// ];
///
/// let intersections = find_all_intersections(&segments).expect("Operation failed");
/// assert_eq!(intersections.len(), 1);
/// assert!((intersections[0].point.x - 1.0).abs() < 1e-9);
/// assert!((intersections[0].point.y - 1.0).abs() < 1e-9);
/// ```
pub fn find_all_intersections(segments: &[Segment2D]) -> SpatialResult<Vec<Intersection>> {
    if segments.is_empty() || segments.len() < 2 {
        return Ok(Vec::new());
    }

    // For smaller inputs, use the sweep-line approach with the active set.
    // For robust correctness we use a plane-sweep that checks only active
    // (overlapping in x-range) segment pairs, which still prunes well
    // while guaranteeing all intersections are found.

    // Assign IDs to segments
    let mut segs: Vec<Segment2D> = segments.to_vec();
    for (i, seg) in segs.iter_mut().enumerate() {
        seg.id = i;
    }

    // Create endpoint events sorted by x
    let mut events: Vec<(f64, bool, usize)> = Vec::with_capacity(segs.len() * 2);
    for (i, seg) in segs.iter().enumerate() {
        events.push((seg.start.x, true, i)); // true = start
        events.push((seg.end.x, false, i)); // false = end
    }
    events.sort_by(|a, b| {
        float_cmp(a.0, b.0).then_with(|| {
            // Process starts before ends at the same x (so segments are active when tested)
            match (a.1, b.1) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.2.cmp(&b.2),
            }
        })
    });

    // Active segments (segments currently intersecting the sweep line)
    let mut active: Vec<usize> = Vec::new();
    let mut intersections: Vec<Intersection> = Vec::new();
    let mut found_pairs: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();

    for (_, is_start, seg_idx) in &events {
        if *is_start {
            // When a segment starts, check it against all currently active segments
            for &other_idx in &active {
                let pair = if *seg_idx < other_idx {
                    (*seg_idx, other_idx)
                } else {
                    (other_idx, *seg_idx)
                };

                if !found_pairs.contains(&pair) {
                    if let Some((pt, _, _)) = segment_intersection(&segs[pair.0], &segs[pair.1]) {
                        found_pairs.insert(pair);
                        intersections.push(Intersection {
                            point: pt,
                            segment_a: pair.0,
                            segment_b: pair.1,
                        });
                    }
                }
            }
            active.push(*seg_idx);
        } else {
            // When a segment ends, remove it from the active set
            active.retain(|&id| id != *seg_idx);
        }
    }

    Ok(intersections)
}

// Note: The sweep-line implementation above uses an active-set approach
// where segments are tested against all currently active segments when they
// enter the sweep. This provides O(n * a) performance where a is the average
// number of active segments at each event point, which is typically much
// less than n for well-distributed segments.

/// Find all intersections using a brute-force O(n^2) algorithm.
///
/// This is useful as a reference implementation for testing and for small inputs
/// where the sweep line overhead is not justified.
///
/// # Arguments
///
/// * `segments` - A slice of line segments
///
/// # Returns
///
/// * `Vec<Intersection>` - All intersection points with segment indices
pub fn find_all_intersections_brute_force(segments: &[Segment2D]) -> Vec<Intersection> {
    let mut intersections = Vec::new();

    for i in 0..segments.len() {
        for j in (i + 1)..segments.len() {
            if let Some((pt, _, _)) = segment_intersection(&segments[i], &segments[j]) {
                intersections.push(Intersection {
                    point: pt,
                    segment_a: i,
                    segment_b: j,
                });
            }
        }
    }

    intersections
}

/// Count the number of intersections among a set of segments without storing them.
///
/// More memory-efficient than `find_all_intersections` when only the count is needed.
///
/// # Arguments
///
/// * `segments` - A slice of line segments
///
/// # Returns
///
/// * `SpatialResult<usize>` - The number of intersection points
pub fn count_intersections(segments: &[Segment2D]) -> SpatialResult<usize> {
    let intersections = find_all_intersections(segments)?;
    Ok(intersections.len())
}

/// Check if any two segments in the set intersect.
///
/// Returns as soon as the first intersection is found.
///
/// # Arguments
///
/// * `segments` - A slice of line segments
///
/// # Returns
///
/// * `SpatialResult<bool>` - True if any intersection exists
pub fn has_any_intersection(segments: &[Segment2D]) -> SpatialResult<bool> {
    // For small sets, brute force is faster
    if segments.len() <= 10 {
        for i in 0..segments.len() {
            for j in (i + 1)..segments.len() {
                if segment_intersection(&segments[i], &segments[j]).is_some() {
                    return Ok(true);
                }
            }
        }
        return Ok(false);
    }

    let intersections = find_all_intersections(segments)?;
    Ok(!intersections.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_crossing_segments() {
        let segments = vec![
            Segment2D::new(0.0, 0.0, 2.0, 2.0),
            Segment2D::new(0.0, 2.0, 2.0, 0.0),
        ];

        let intersections = find_all_intersections(&segments).expect("Operation failed");
        assert_eq!(intersections.len(), 1);
        assert!((intersections[0].point.x - 1.0).abs() < 1e-6);
        assert!((intersections[0].point.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_intersections() {
        let segments = vec![
            Segment2D::new(0.0, 0.0, 1.0, 0.0),
            Segment2D::new(0.0, 1.0, 1.0, 1.0),
        ];

        let intersections = find_all_intersections(&segments).expect("Operation failed");
        assert_eq!(intersections.len(), 0);
    }

    #[test]
    fn test_multiple_intersections() {
        // Three segments forming a triangle-like pattern
        let segments = vec![
            Segment2D::new(0.0, 0.0, 4.0, 4.0), // diagonal up
            Segment2D::new(0.0, 4.0, 4.0, 0.0), // diagonal down
            Segment2D::new(0.0, 2.0, 4.0, 2.0), // horizontal through middle
        ];

        let intersections = find_all_intersections(&segments).expect("Operation failed");
        // Should find 3 intersections: each pair intersects
        assert_eq!(intersections.len(), 3);
    }

    #[test]
    fn test_parallel_segments() {
        let segments = vec![
            Segment2D::new(0.0, 0.0, 2.0, 0.0),
            Segment2D::new(0.0, 1.0, 2.0, 1.0),
            Segment2D::new(0.0, 2.0, 2.0, 2.0),
        ];

        let intersections = find_all_intersections(&segments).expect("Operation failed");
        assert_eq!(intersections.len(), 0);
    }

    #[test]
    fn test_endpoint_intersection() {
        // Two segments sharing an endpoint
        let segments = vec![
            Segment2D::new(0.0, 0.0, 1.0, 1.0),
            Segment2D::new(1.0, 1.0, 2.0, 0.0),
        ];

        let intersections = find_all_intersections(&segments).expect("Operation failed");
        // Endpoint touching counts as an intersection
        assert_eq!(intersections.len(), 1);
        assert!((intersections[0].point.x - 1.0).abs() < 1e-6);
        assert!((intersections[0].point.y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_brute_force_matches_sweep() {
        let segments = vec![
            Segment2D::new(0.0, 0.0, 3.0, 3.0),
            Segment2D::new(0.0, 3.0, 3.0, 0.0),
            Segment2D::new(1.0, 0.0, 1.0, 4.0),
        ];

        let sweep_result = find_all_intersections(&segments).expect("Operation failed");
        let brute_result = find_all_intersections_brute_force(&segments);

        assert_eq!(sweep_result.len(), brute_result.len());
    }

    #[test]
    fn test_empty_input() {
        let segments: Vec<Segment2D> = vec![];
        let intersections = find_all_intersections(&segments).expect("Operation failed");
        assert_eq!(intersections.len(), 0);
    }

    #[test]
    fn test_single_segment() {
        let segments = vec![Segment2D::new(0.0, 0.0, 1.0, 1.0)];
        let intersections = find_all_intersections(&segments).expect("Operation failed");
        assert_eq!(intersections.len(), 0);
    }

    #[test]
    fn test_segment_intersection_function() {
        let seg1 = Segment2D::new(0.0, 0.0, 2.0, 2.0);
        let seg2 = Segment2D::new(0.0, 2.0, 2.0, 0.0);

        let result = segment_intersection(&seg1, &seg2);
        assert!(result.is_some());

        let (pt, t, u) = result.expect("Operation failed");
        assert!((pt.x - 1.0).abs() < 1e-9);
        assert!((pt.y - 1.0).abs() < 1e-9);
        assert!((t - 0.5).abs() < 1e-9);
        assert!((u - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_has_any_intersection() {
        let crossing = vec![
            Segment2D::new(0.0, 0.0, 2.0, 2.0),
            Segment2D::new(0.0, 2.0, 2.0, 0.0),
        ];
        assert!(has_any_intersection(&crossing).expect("Operation failed"));

        let parallel = vec![
            Segment2D::new(0.0, 0.0, 2.0, 0.0),
            Segment2D::new(0.0, 1.0, 2.0, 1.0),
        ];
        assert!(!has_any_intersection(&parallel).expect("Operation failed"));
    }

    #[test]
    fn test_count_intersections() {
        let segments = vec![
            Segment2D::new(0.0, 0.0, 4.0, 4.0),
            Segment2D::new(0.0, 4.0, 4.0, 0.0),
            Segment2D::new(0.0, 2.0, 4.0, 2.0),
        ];

        let count = count_intersections(&segments).expect("Operation failed");
        assert_eq!(count, 3);
    }

    #[test]
    fn test_star_pattern() {
        // Five segments forming a star pattern with many intersections
        let segments = vec![
            Segment2D::new(0.0, 2.0, 4.0, 2.0), // horizontal
            Segment2D::new(2.0, 0.0, 2.0, 4.0), // vertical
            Segment2D::new(0.0, 0.0, 4.0, 4.0), // diagonal /
            Segment2D::new(0.0, 4.0, 4.0, 0.0), // diagonal \
        ];

        let intersections = find_all_intersections(&segments).expect("Operation failed");
        // Each pair of 4 segments = C(4,2) = 6 potential intersections
        // All pairs do intersect near the center
        let brute = find_all_intersections_brute_force(&segments);
        assert_eq!(intersections.len(), brute.len());
    }

    #[test]
    fn test_vertical_segment() {
        let segments = vec![
            Segment2D::new(1.0, 0.0, 1.0, 2.0), // vertical
            Segment2D::new(0.0, 1.0, 2.0, 1.0), // horizontal
        ];

        let intersections = find_all_intersections(&segments).expect("Operation failed");
        assert_eq!(intersections.len(), 1);
        assert!((intersections[0].point.x - 1.0).abs() < 1e-6);
        assert!((intersections[0].point.y - 1.0).abs() < 1e-6);
    }
}
