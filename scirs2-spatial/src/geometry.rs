//! Computational geometry primitives
//!
//! This module provides fundamental geometric types and algorithms for 2D and 3D space,
//! including:
//! - Point types (`Point2D`, `Point3D`) with vector operations
//! - Line segments with intersection testing
//! - Polygons with area, perimeter, and point-in-polygon queries
//! - Circles with circumscribed/inscribed computations
//! - Minimum enclosing circle (Welzl's algorithm)
//! - Minimum bounding rectangle (rotating calipers)
//!
//! # Examples
//!
//! ```rust
//! use scirs2_spatial::geometry::{Point2D, Segment, Polygon};
//!
//! let p1 = Point2D::new(0.0, 0.0);
//! let p2 = Point2D::new(3.0, 4.0);
//! assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-10);
//!
//! let seg = Segment::new(p1, p2);
//! let query = Point2D::new(1.5, 2.0);
//! let dist = seg.distance_to_point(&query);
//! assert!(dist < 1.0);
//!
//! let vertices = vec![
//!     Point2D::new(0.0, 0.0),
//!     Point2D::new(4.0, 0.0),
//!     Point2D::new(4.0, 3.0),
//!     Point2D::new(0.0, 3.0),
//! ];
//! let poly = Polygon::new(vertices).unwrap();
//! assert!((poly.area() - 12.0).abs() < 1e-10);
//! assert!(poly.contains_point(&Point2D::new(2.0, 1.5)));
//! ```

use crate::error::{SpatialError, SpatialResult};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Point2D
// ---------------------------------------------------------------------------

/// A point in 2D Euclidean space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

impl Point2D {
    /// Create a new 2D point.
    #[inline]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to another point.
    #[inline]
    pub fn distance_to(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Squared Euclidean distance (avoids a `sqrt`).
    #[inline]
    pub fn distance_sq(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Dot product of the position vectors.
    #[inline]
    pub fn dot(&self, other: &Point2D) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 2D cross product (scalar z-component of the 3D cross product).
    #[inline]
    pub fn cross(&self, other: &Point2D) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Vector from `self` to `other`.
    #[inline]
    pub fn vector_to(&self, other: &Point2D) -> Point2D {
        Point2D::new(other.x - self.x, other.y - self.y)
    }

    /// Component-wise addition.
    #[inline]
    pub fn add(&self, other: &Point2D) -> Point2D {
        Point2D::new(self.x + other.x, self.y + other.y)
    }

    /// Scale by a scalar.
    #[inline]
    pub fn scale(&self, s: f64) -> Point2D {
        Point2D::new(self.x * s, self.y * s)
    }

    /// Euclidean norm (distance from origin).
    #[inline]
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Normalized unit vector; returns `None` if the point is at the origin.
    pub fn normalized(&self) -> Option<Point2D> {
        let n = self.norm();
        if n < f64::EPSILON {
            None
        } else {
            Some(Point2D::new(self.x / n, self.y / n))
        }
    }

    /// Midpoint between `self` and `other`.
    #[inline]
    pub fn midpoint(&self, other: &Point2D) -> Point2D {
        Point2D::new((self.x + other.x) * 0.5, (self.y + other.y) * 0.5)
    }
}

impl std::fmt::Display for Point2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

// ---------------------------------------------------------------------------
// Point3D
// ---------------------------------------------------------------------------

/// A point in 3D Euclidean space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3D {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
    /// Z coordinate
    pub z: f64,
}

impl Point3D {
    /// Create a new 3D point.
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Euclidean distance to another point.
    #[inline]
    pub fn distance_to(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Squared Euclidean distance.
    #[inline]
    pub fn distance_sq(&self, other: &Point3D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Dot product of the position vectors.
    #[inline]
    pub fn dot(&self, other: &Point3D) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// 3D cross product.
    #[inline]
    pub fn cross(&self, other: &Point3D) -> Point3D {
        Point3D::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    /// Euclidean norm.
    #[inline]
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalized unit vector; returns `None` if the point is at the origin.
    pub fn normalized(&self) -> Option<Point3D> {
        let n = self.norm();
        if n < f64::EPSILON {
            None
        } else {
            Some(Point3D::new(self.x / n, self.y / n, self.z / n))
        }
    }

    /// Midpoint between `self` and `other`.
    #[inline]
    pub fn midpoint(&self, other: &Point3D) -> Point3D {
        Point3D::new(
            (self.x + other.x) * 0.5,
            (self.y + other.y) * 0.5,
            (self.z + other.z) * 0.5,
        )
    }

    /// Component-wise addition.
    #[inline]
    pub fn add(&self, other: &Point3D) -> Point3D {
        Point3D::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    /// Scale by a scalar.
    #[inline]
    pub fn scale(&self, s: f64) -> Point3D {
        Point3D::new(self.x * s, self.y * s, self.z * s)
    }
}

impl std::fmt::Display for Point3D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

// ---------------------------------------------------------------------------
// Segment
// ---------------------------------------------------------------------------

/// A line segment in 2D space defined by two endpoints.
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    /// Start point
    pub start: Point2D,
    /// End point
    pub end: Point2D,
}

/// Result of a segment intersection test.
#[derive(Debug, Clone)]
pub enum IntersectionResult {
    /// Segments intersect at this point.
    Point(Point2D),
    /// Segments overlap along a sub-segment.
    Overlap(Segment),
    /// Segments do not intersect.
    None,
}

impl Segment {
    /// Create a new segment from two endpoints.
    pub fn new(start: Point2D, end: Point2D) -> Self {
        Self { start, end }
    }

    /// Length of the segment.
    pub fn length(&self) -> f64 {
        self.start.distance_to(&self.end)
    }

    /// Direction vector (not normalised).
    pub fn direction(&self) -> Point2D {
        self.start.vector_to(&self.end)
    }

    /// Midpoint of the segment.
    pub fn midpoint(&self) -> Point2D {
        self.start.midpoint(&self.end)
    }

    /// Minimum squared distance from a point to the segment.
    pub fn distance_sq_to_point(&self, p: &Point2D) -> f64 {
        let d = self.direction();
        let len_sq = d.x * d.x + d.y * d.y;
        if len_sq < f64::EPSILON {
            return self.start.distance_sq(p);
        }
        let t = ((p.x - self.start.x) * d.x + (p.y - self.start.y) * d.y) / len_sq;
        let t_clamped = t.clamp(0.0, 1.0);
        let closest = Point2D::new(
            self.start.x + t_clamped * d.x,
            self.start.y + t_clamped * d.y,
        );
        p.distance_sq(&closest)
    }

    /// Distance from a point to the segment.
    pub fn distance_to_point(&self, p: &Point2D) -> f64 {
        self.distance_sq_to_point(p).sqrt()
    }

    /// Test whether this segment intersects another, returning the intersection.
    ///
    /// Uses the parametric form and handles collinear / parallel cases.
    pub fn intersect(&self, other: &Segment) -> IntersectionResult {
        let p = self.start;
        let r = self.direction();
        let q = other.start;
        let s = other.direction();

        let r_cross_s = r.x * s.y - r.y * s.x;
        let qp = Point2D::new(q.x - p.x, q.y - p.y);
        let qp_cross_r = qp.x * r.y - qp.y * r.x;
        let qp_cross_s = qp.x * s.y - qp.y * s.x;

        if r_cross_s.abs() < f64::EPSILON {
            // Parallel
            if qp_cross_r.abs() < f64::EPSILON {
                // Collinear – check for overlap
                let r_len_sq = r.x * r.x + r.y * r.y;
                if r_len_sq < f64::EPSILON {
                    return IntersectionResult::None;
                }
                let t0 = (qp.x * r.x + qp.y * r.y) / r_len_sq;
                let s_dot_r = s.x * r.x + s.y * r.y;
                let t1 = t0 + s_dot_r / r_len_sq;
                let (lo, hi) = if t0 <= t1 { (t0, t1) } else { (t1, t0) };
                let overlap_start = lo.max(0.0);
                let overlap_end = hi.min(1.0);
                if overlap_start > overlap_end + f64::EPSILON {
                    return IntersectionResult::None;
                }
                if (overlap_end - overlap_start).abs() < f64::EPSILON {
                    // Single point overlap
                    let pt = Point2D::new(
                        p.x + overlap_start * r.x,
                        p.y + overlap_start * r.y,
                    );
                    return IntersectionResult::Point(pt);
                }
                let pt_start = Point2D::new(
                    p.x + overlap_start * r.x,
                    p.y + overlap_start * r.y,
                );
                let pt_end = Point2D::new(
                    p.x + overlap_end * r.x,
                    p.y + overlap_end * r.y,
                );
                return IntersectionResult::Overlap(Segment::new(pt_start, pt_end));
            }
            return IntersectionResult::None;
        }

        let t = qp_cross_s / r_cross_s;
        let u = qp_cross_r / r_cross_s;

        if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
            IntersectionResult::Point(Point2D::new(p.x + t * r.x, p.y + t * r.y))
        } else {
            IntersectionResult::None
        }
    }

    /// Check whether this segment strictly intersects another (returns `true` for a crossing point,
    /// `false` for collinear / no intersection).
    pub fn intersects(&self, other: &Segment) -> bool {
        matches!(self.intersect(other), IntersectionResult::Point(_) | IntersectionResult::Overlap(_))
    }
}

// ---------------------------------------------------------------------------
// Polygon
// ---------------------------------------------------------------------------

/// A simple (possibly non-convex) polygon in 2D space.
///
/// Vertices are stored in order (counter-clockwise convention is recommended,
/// but the implementation handles both orientations).
#[derive(Debug, Clone)]
pub struct Polygon {
    /// Ordered list of vertices (the last vertex connects back to the first).
    vertices: Vec<Point2D>,
}

impl Polygon {
    /// Create a new polygon from a list of vertices.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 3 vertices are provided.
    pub fn new(vertices: Vec<Point2D>) -> SpatialResult<Self> {
        if vertices.len() < 3 {
            return Err(SpatialError::ValueError(
                "A polygon requires at least 3 vertices".to_string(),
            ));
        }
        Ok(Self { vertices })
    }

    /// Reference to the vertex list.
    pub fn vertices(&self) -> &[Point2D] {
        &self.vertices
    }

    /// Number of vertices / edges.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Signed area computed via the shoelace formula.
    ///
    /// Positive for CCW orientation, negative for CW.
    pub fn signed_area(&self) -> f64 {
        let n = self.vertices.len();
        let mut area = 0.0_f64;
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.vertices[i].x * self.vertices[j].y;
            area -= self.vertices[j].x * self.vertices[i].y;
        }
        area * 0.5
    }

    /// Unsigned area.
    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    /// Perimeter (sum of edge lengths).
    pub fn perimeter(&self) -> f64 {
        let n = self.vertices.len();
        (0..n)
            .map(|i| self.vertices[i].distance_to(&self.vertices[(i + 1) % n]))
            .sum()
    }

    /// Centroid of the polygon (computed from the signed-area formula).
    pub fn centroid(&self) -> Point2D {
        let n = self.vertices.len();
        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        let area = self.signed_area();
        for i in 0..n {
            let j = (i + 1) % n;
            let cross = self.vertices[i].x * self.vertices[j].y
                - self.vertices[j].x * self.vertices[i].y;
            cx += (self.vertices[i].x + self.vertices[j].x) * cross;
            cy += (self.vertices[i].y + self.vertices[j].y) * cross;
        }
        let factor = 1.0 / (6.0 * area);
        Point2D::new(cx * factor, cy * factor)
    }

    /// Point-in-polygon test using the ray-casting (winding) algorithm.
    ///
    /// Returns `true` if the point is strictly inside (touching the boundary
    /// is treated as inside for robustness).
    pub fn contains_point(&self, p: &Point2D) -> bool {
        let n = self.vertices.len();
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let vi = &self.vertices[i];
            let vj = &self.vertices[j];
            if ((vi.y > p.y) != (vj.y > p.y))
                && (p.x < (vj.x - vi.x) * (p.y - vi.y) / (vj.y - vi.y) + vi.x)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    /// Returns `true` if the polygon is convex.
    pub fn is_convex(&self) -> bool {
        let n = self.vertices.len();
        if n < 3 {
            return false;
        }
        let mut sign: Option<bool> = None;
        for i in 0..n {
            let a = &self.vertices[i];
            let b = &self.vertices[(i + 1) % n];
            let c = &self.vertices[(i + 2) % n];
            let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
            if cross.abs() > f64::EPSILON {
                let positive = cross > 0.0;
                match sign {
                    None => sign = Some(positive),
                    Some(s) if s != positive => return false,
                    _ => {}
                }
            }
        }
        true
    }

    /// Edges of the polygon as `Segment`s.
    pub fn edges(&self) -> Vec<Segment> {
        let n = self.vertices.len();
        (0..n)
            .map(|i| Segment::new(self.vertices[i], self.vertices[(i + 1) % n]))
            .collect()
    }

    /// Test whether the polygon is simple (no self-intersections).
    pub fn is_simple(&self) -> bool {
        let edges = self.edges();
        let n = edges.len();
        for i in 0..n {
            for j in (i + 2)..n {
                // Adjacent edges share a vertex, skip trivial adjacency
                if i == 0 && j == n - 1 {
                    continue;
                }
                if edges[i].intersects(&edges[j]) {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Circle
// ---------------------------------------------------------------------------

/// A circle in 2D space.
#[derive(Debug, Clone, Copy)]
pub struct Circle {
    /// Center of the circle.
    pub center: Point2D,
    /// Radius of the circle (always non-negative).
    pub radius: f64,
}

impl Circle {
    /// Create a new circle.
    ///
    /// # Errors
    ///
    /// Returns an error if radius is negative.
    pub fn new(center: Point2D, radius: f64) -> SpatialResult<Self> {
        if radius < 0.0 {
            return Err(SpatialError::ValueError(
                "Circle radius must be non-negative".to_string(),
            ));
        }
        Ok(Self { center, radius })
    }

    /// Area of the circle.
    pub fn area(&self) -> f64 {
        PI * self.radius * self.radius
    }

    /// Circumference of the circle.
    pub fn circumference(&self) -> f64 {
        2.0 * PI * self.radius
    }

    /// Check whether a point is inside (or on) the circle.
    pub fn contains(&self, p: &Point2D) -> bool {
        let r2 = self.radius * self.radius;
        // Use a relative tolerance for numerical robustness
        let tol = 1e-10 * (r2 + 1.0);
        self.center.distance_sq(p) <= r2 + tol
    }

    /// Circumscribed circle of a triangle defined by three points.
    ///
    /// Returns `None` if the three points are collinear.
    pub fn circumscribed(a: &Point2D, b: &Point2D, c: &Point2D) -> Option<Circle> {
        let ax = a.x;
        let ay = a.y;
        let bx = b.x;
        let by = b.y;
        let cx = c.x;
        let cy = c.y;

        let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        if d.abs() < f64::EPSILON {
            return None;
        }

        let ux = ((ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by))
            / d;
        let uy = ((ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax))
            / d;

        let center = Point2D::new(ux, uy);
        let radius = center.distance_to(a);
        Some(Circle { center, radius })
    }

    /// Inscribed circle of a triangle (largest circle fitting inside).
    ///
    /// Returns `None` if the triangle is degenerate.
    pub fn inscribed(a: &Point2D, b: &Point2D, c: &Point2D) -> Option<Circle> {
        let side_a = b.distance_to(c); // opposite to vertex A
        let side_b = a.distance_to(c); // opposite to vertex B
        let side_c = a.distance_to(b); // opposite to vertex C
        let perimeter = side_a + side_b + side_c;
        if perimeter < f64::EPSILON {
            return None;
        }
        let cx = (side_a * a.x + side_b * b.x + side_c * c.x) / perimeter;
        let cy = (side_a * a.y + side_b * b.y + side_c * c.y) / perimeter;
        // Area via shoelace
        let area =
            0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs();
        let radius = 2.0 * area / perimeter;
        Some(Circle {
            center: Point2D::new(cx, cy),
            radius,
        })
    }
}

// ---------------------------------------------------------------------------
// MinEnclosingCircle (Welzl's algorithm)
// ---------------------------------------------------------------------------

/// Minimum enclosing circle of a point set.
///
/// Uses Welzl's randomised algorithm with linear expected time.
pub struct MinEnclosingCircle;

impl MinEnclosingCircle {
    /// Compute the smallest enclosing circle of the given points.
    ///
    /// # Errors
    ///
    /// Returns an error if the point list is empty.
    pub fn compute(points: &[Point2D]) -> SpatialResult<Circle> {
        if points.is_empty() {
            return Err(SpatialError::ValueError(
                "Cannot compute enclosing circle of empty point set".to_string(),
            ));
        }
        // Shuffle for expected linear time – we copy and shuffle with a simple
        // deterministic Fisher-Yates using a fixed seed (no external deps).
        let mut pts: Vec<Point2D> = points.to_vec();
        Self::shuffle(&mut pts);

        let result = Self::welzl(&pts, Vec::new(), pts.len());
        result.ok_or_else(|| {
            SpatialError::ComputationError(
                "Welzl algorithm failed to find enclosing circle".to_string(),
            )
        })
    }

    /// Simple deterministic shuffle (LCG-based, fast and dependency-free).
    fn shuffle(v: &mut [Point2D]) {
        let n = v.len();
        let mut state: u64 = 0x9e3779b97f4a7c15;
        for i in (1..n).rev() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            v.swap(i, j);
        }
    }

    /// Recursive Welzl step.
    fn welzl(pts: &[Point2D], boundary: Vec<Point2D>, n: usize) -> Option<Circle> {
        if n == 0 || boundary.len() == 3 {
            return Self::min_circle_trivial(&boundary);
        }
        let p = pts[n - 1];
        let d = Self::welzl(pts, boundary.clone(), n - 1)?;
        if d.contains(&p) {
            return Some(d);
        }
        let mut new_boundary = boundary;
        new_boundary.push(p);
        Self::welzl(pts, new_boundary, n - 1)
    }

    /// Solve for the minimum enclosing circle of ≤3 boundary points.
    fn min_circle_trivial(boundary: &[Point2D]) -> Option<Circle> {
        match boundary.len() {
            0 => Some(Circle {
                center: Point2D::new(0.0, 0.0),
                radius: 0.0,
            }),
            1 => Some(Circle {
                center: boundary[0],
                radius: 0.0,
            }),
            2 => {
                let center = boundary[0].midpoint(&boundary[1]);
                let radius = boundary[0].distance_to(&boundary[1]) * 0.5;
                Some(Circle { center, radius })
            }
            3 => Circle::circumscribed(&boundary[0], &boundary[1], &boundary[2]).or_else(|| {
                // Collinear: pick the widest pair.
                let d01 = boundary[0].distance_to(&boundary[1]);
                let d12 = boundary[1].distance_to(&boundary[2]);
                let d02 = boundary[0].distance_to(&boundary[2]);
                if d01 >= d12 && d01 >= d02 {
                    let center = boundary[0].midpoint(&boundary[1]);
                    Some(Circle { center, radius: d01 * 0.5 })
                } else if d12 >= d01 && d12 >= d02 {
                    let center = boundary[1].midpoint(&boundary[2]);
                    Some(Circle { center, radius: d12 * 0.5 })
                } else {
                    let center = boundary[0].midpoint(&boundary[2]);
                    Some(Circle { center, radius: d02 * 0.5 })
                }
            }),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// MinBoundingRect (rotating calipers)
// ---------------------------------------------------------------------------

/// Minimum-area bounding rectangle of a point set.
///
/// Computed via rotating calipers on the convex hull.
#[derive(Debug, Clone)]
pub struct MinBoundingRect {
    /// Center of the rectangle.
    pub center: Point2D,
    /// Half-width (along the principal axis).
    pub half_width: f64,
    /// Half-height (along the secondary axis).
    pub half_height: f64,
    /// Angle (radians) of the principal axis with the positive x-axis.
    pub angle: f64,
}

impl MinBoundingRect {
    /// Compute the minimum-area bounding rectangle of the given points.
    ///
    /// Uses Graham scan for the convex hull followed by rotating calipers.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 points are provided or the hull is degenerate.
    pub fn compute(points: &[Point2D]) -> SpatialResult<Self> {
        if points.len() < 2 {
            return Err(SpatialError::ValueError(
                "Need at least 2 points for bounding rectangle".to_string(),
            ));
        }
        let hull = graham_scan(points);
        if hull.len() < 2 {
            return Err(SpatialError::ComputationError(
                "Convex hull has fewer than 2 points".to_string(),
            ));
        }
        Self::from_hull(&hull)
    }

    fn from_hull(hull: &[Point2D]) -> SpatialResult<Self> {
        let n = hull.len();
        if n == 1 {
            return Ok(Self {
                center: hull[0],
                half_width: 0.0,
                half_height: 0.0,
                angle: 0.0,
            });
        }

        let mut min_area = f64::INFINITY;
        let mut best = Self {
            center: hull[0],
            half_width: 0.0,
            half_height: 0.0,
            angle: 0.0,
        };

        for i in 0..n {
            let edge = hull[i].vector_to(&hull[(i + 1) % n]);
            let len = edge.norm();
            if len < f64::EPSILON {
                continue;
            }
            // Unit vectors along and perpendicular to this edge
            let ux = edge.x / len;
            let uy = edge.y / len;
            let vx = -uy;
            let vy = ux;

            // Project all hull points onto (u, v)
            let mut min_u = f64::INFINITY;
            let mut max_u = f64::NEG_INFINITY;
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            for p in hull {
                let u = p.x * ux + p.y * uy;
                let v = p.x * vx + p.y * vy;
                min_u = min_u.min(u);
                max_u = max_u.max(u);
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }

            let w = max_u - min_u;
            let h = max_v - min_v;
            let area = w * h;

            if area < min_area {
                min_area = area;
                let mid_u = (min_u + max_u) * 0.5;
                let mid_v = (min_v + max_v) * 0.5;
                let cx = mid_u * ux + mid_v * vx;
                let cy = mid_u * uy + mid_v * vy;
                best = Self {
                    center: Point2D::new(cx, cy),
                    half_width: w * 0.5,
                    half_height: h * 0.5,
                    angle: uy.atan2(ux),
                };
            }
        }

        Ok(best)
    }

    /// Area of the bounding rectangle.
    pub fn area(&self) -> f64 {
        4.0 * self.half_width * self.half_height
    }

    /// The four corner vertices of the bounding rectangle.
    pub fn corners(&self) -> [Point2D; 4] {
        let cos_a = self.angle.cos();
        let sin_a = self.angle.sin();
        let w = self.half_width;
        let h = self.half_height;
        let cx = self.center.x;
        let cy = self.center.y;

        let offsets = [
            (w, h),
            (-w, h),
            (-w, -h),
            (w, -h),
        ];

        offsets.map(|(du, dv)| {
            Point2D::new(
                cx + du * cos_a - dv * sin_a,
                cy + du * sin_a + dv * cos_a,
            )
        })
    }
}

// ---------------------------------------------------------------------------
// Convex hull (Graham scan / Andrew's monotone chain)
// ---------------------------------------------------------------------------

/// Compute the convex hull of a set of 2D points using Andrew's monotone chain.
///
/// Returns the hull vertices in counter-clockwise order.
/// Returns an empty vector if the input is empty.
pub fn graham_scan(points: &[Point2D]) -> Vec<Point2D> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    if n <= 2 {
        // Deduplicate and return
        let mut result = vec![points[0]];
        if n == 2 && (points[0].x - points[1].x).abs() + (points[0].y - points[1].y).abs() > f64::EPSILON {
            result.push(points[1]);
        }
        return result;
    }

    // Sort lexicographically by (x, y)
    let mut sorted: Vec<Point2D> = points.to_vec();
    sorted.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal))
    });

    let cross = |o: &Point2D, a: &Point2D, b: &Point2D| -> f64 {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    };

    // Build lower hull
    let mut lower: Vec<Point2D> = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], &p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }

    // Build upper hull
    let mut upper: Vec<Point2D> = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], &p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }

    // Concatenate (remove last points as they are duplicated)
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d_distance() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(3.0, 4.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_point3d_cross() {
        let i = Point3D::new(1.0, 0.0, 0.0);
        let j = Point3D::new(0.0, 1.0, 0.0);
        let k = i.cross(&j);
        assert!((k.x).abs() < 1e-12);
        assert!((k.y).abs() < 1e-12);
        assert!((k.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_segment_intersection() {
        let s1 = Segment::new(Point2D::new(0.0, 0.0), Point2D::new(2.0, 2.0));
        let s2 = Segment::new(Point2D::new(0.0, 2.0), Point2D::new(2.0, 0.0));
        assert!(s1.intersects(&s2));
    }

    #[test]
    fn test_segment_no_intersection() {
        let s1 = Segment::new(Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0));
        let s2 = Segment::new(Point2D::new(0.0, 1.0), Point2D::new(1.0, 1.0));
        assert!(!s1.intersects(&s2));
    }

    #[test]
    fn test_polygon_area() {
        let verts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 3.0),
            Point2D::new(0.0, 3.0),
        ];
        let poly = Polygon::new(verts).unwrap();
        assert!((poly.area() - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_polygon_contains() {
        let verts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(4.0, 3.0),
            Point2D::new(0.0, 3.0),
        ];
        let poly = Polygon::new(verts).unwrap();
        assert!(poly.contains_point(&Point2D::new(2.0, 1.5)));
        assert!(!poly.contains_point(&Point2D::new(5.0, 1.5)));
    }

    #[test]
    fn test_circle_circumscribed() {
        let a = Point2D::new(0.0, 0.0);
        let b = Point2D::new(2.0, 0.0);
        let c = Point2D::new(1.0, 2.0);
        let circ = Circle::circumscribed(&a, &b, &c).unwrap();
        assert!(circ.contains(&a));
        assert!(circ.contains(&b));
        assert!(circ.contains(&c));
    }

    #[test]
    fn test_min_enclosing_circle() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(4.0, 0.0),
            Point2D::new(2.0, 3.0),
        ];
        let mec = MinEnclosingCircle::compute(&pts).unwrap();
        for p in &pts {
            assert!(mec.contains(p), "Point {} should be inside MEC", p);
        }
    }

    #[test]
    fn test_graham_scan_square() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(0.5, 0.5), // interior
        ];
        let hull = graham_scan(&pts);
        assert_eq!(hull.len(), 4);
    }

    #[test]
    fn test_min_bounding_rect() {
        let pts = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(3.0, 0.0),
            Point2D::new(3.0, 2.0),
            Point2D::new(0.0, 2.0),
        ];
        let mbr = MinBoundingRect::compute(&pts).unwrap();
        // Area should be 6.0
        assert!((mbr.area() - 6.0).abs() < 1e-6, "Expected area ~6, got {}", mbr.area());
    }
}
