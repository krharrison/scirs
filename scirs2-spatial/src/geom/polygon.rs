//! Polygon operations: area, centroid, point-in-polygon, perimeter.
//!
//! All area computations use the **shoelace formula** (signed).  A positive
//! signed area means the vertices are ordered counter-clockwise (CCW).
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::geom::polygon::{polygon_area, point_in_polygon, Polygon};
//!
//! let square = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
//! assert!((polygon_area(&square) - 1.0).abs() < 1e-12);
//! assert!(point_in_polygon([0.5, 0.5], &square));
//! assert!(!point_in_polygon([2.0, 2.0], &square));
//! ```

use crate::error::{SpatialError, SpatialResult};

// ── Shoelace formula ──────────────────────────────────────────────────────────

/// Compute the **signed area** of a simple polygon given its vertices.
///
/// - Positive → counter-clockwise (CCW) orientation.
/// - Negative → clockwise (CW) orientation.
///
/// Returns 0.0 for fewer than 3 vertices.
pub fn polygon_area(vertices: &[[f64; 2]]) -> f64 {
    let n = vertices.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        area += vertices[i][0] * vertices[j][1];
        area -= vertices[j][0] * vertices[i][1];
    }
    area * 0.5
}

// ── Point-in-polygon (ray casting) ───────────────────────────────────────────

/// Test whether point `p` is strictly **inside** a simple polygon using the
/// ray casting algorithm.
///
/// Points exactly on the boundary may return either `true` or `false`
/// (undefined behaviour for degenerate cases).
pub fn point_in_polygon(p: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let (px, py) = (p[0], p[1]);
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let xi = polygon[i][0];
        let yi = polygon[i][1];
        let xj = polygon[j][0];
        let yj = polygon[j][1];
        // Edge crosses ray?
        if (yi > py) != (yj > py) {
            let x_intersect = xj + (py - yj) * (xi - xj) / (yi - yj);
            if px < x_intersect {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

// ── Centroid ──────────────────────────────────────────────────────────────────

/// Compute the **centroid** of a simple polygon using the standard formula
/// derived from the shoelace decomposition.
///
/// Falls back to the arithmetic mean of vertices for degenerate (zero-area)
/// polygons.  Returns `[0.0, 0.0]` for empty input.
pub fn polygon_centroid(vertices: &[[f64; 2]]) -> [f64; 2] {
    let n = vertices.len();
    if n == 0 {
        return [0.0, 0.0];
    }
    let area = polygon_area(vertices);
    if area.abs() < 1e-15 {
        let cx = vertices.iter().map(|v| v[0]).sum::<f64>() / n as f64;
        let cy = vertices.iter().map(|v| v[1]).sum::<f64>() / n as f64;
        return [cx, cy];
    }
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        let cross = vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1];
        cx += (vertices[i][0] + vertices[j][0]) * cross;
        cy += (vertices[i][1] + vertices[j][1]) * cross;
    }
    let factor = 1.0 / (6.0 * area);
    [cx * factor, cy * factor]
}

// ── Perimeter ─────────────────────────────────────────────────────────────────

/// Perimeter of a polygon (sum of edge lengths, closing edge included).
pub fn polygon_perimeter(vertices: &[[f64; 2]]) -> f64 {
    let n = vertices.len();
    if n < 2 {
        return 0.0;
    }
    (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            let dx = vertices[j][0] - vertices[i][0];
            let dy = vertices[j][1] - vertices[i][1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum()
}

// ── Polygon struct ────────────────────────────────────────────────────────────

/// A simple polygon with optional holes.
///
/// The outer boundary (`vertices`) should be CCW; each hole should be CW.
/// All coordinates are in the same 2D coordinate system.
///
/// # Examples
///
/// ```
/// use scirs2_spatial::geom::polygon::Polygon;
///
/// let rect = Polygon::new(vec![
///     [0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0],
/// ]);
/// assert!((rect.area() - 12.0).abs() < 1e-12);
/// assert!(rect.contains([2.0, 1.5]));
/// ```
#[derive(Debug, Clone)]
pub struct Polygon {
    /// Outer boundary vertices.
    pub vertices: Vec<[f64; 2]>,
    /// Interior holes (each a list of vertices).
    pub holes: Vec<Vec<[f64; 2]>>,
}

impl Polygon {
    /// Create a polygon from its outer boundary vertices.
    pub fn new(vertices: Vec<[f64; 2]>) -> Self {
        Self { vertices, holes: Vec::new() }
    }

    /// Add a hole to the polygon.
    pub fn add_hole(&mut self, hole: Vec<[f64; 2]>) {
        self.holes.push(hole);
    }

    /// Absolute area of the polygon (outer boundary minus holes).
    pub fn area(&self) -> f64 {
        let outer = polygon_area(&self.vertices).abs();
        let holes: f64 = self.holes.iter().map(|h| polygon_area(h).abs()).sum();
        outer - holes
    }

    /// Centroid of the outer boundary (ignores holes for simplicity).
    pub fn centroid(&self) -> [f64; 2] {
        polygon_centroid(&self.vertices)
    }

    /// Return `true` if `p` is inside the polygon and not inside any hole.
    pub fn contains(&self, p: [f64; 2]) -> bool {
        if !point_in_polygon(p, &self.vertices) {
            return false;
        }
        !self.holes.iter().any(|h| point_in_polygon(p, h))
    }

    /// Perimeter of the outer boundary.
    pub fn perimeter(&self) -> f64 {
        polygon_perimeter(&self.vertices)
    }

    /// Winding direction: returns `true` if vertices are CCW (positive area).
    pub fn is_ccw(&self) -> bool {
        polygon_area(&self.vertices) > 0.0
    }

    /// Reverse vertex order (flips CCW ↔ CW).
    pub fn reverse(&mut self) {
        self.vertices.reverse();
    }

    /// Return a validated polygon, checking basic sanity.
    pub fn validate(&self) -> SpatialResult<()> {
        if self.vertices.len() < 3 {
            return Err(SpatialError::ValueError(
                "Polygon must have at least 3 vertices".into(),
            ));
        }
        Ok(())
    }

    /// Translate polygon by `(dx, dy)`.
    pub fn translate(&self, dx: f64, dy: f64) -> Self {
        Self {
            vertices: self.vertices.iter().map(|v| [v[0] + dx, v[1] + dy]).collect(),
            holes: self.holes.iter().map(|h| h.iter().map(|v| [v[0] + dx, v[1] + dy]).collect()).collect(),
        }
    }

    /// Scale polygon about the origin by `sx`, `sy`.
    pub fn scale(&self, sx: f64, sy: f64) -> Self {
        Self {
            vertices: self.vertices.iter().map(|v| [v[0] * sx, v[1] * sy]).collect(),
            holes: self.holes.iter().map(|h| h.iter().map(|v| [v[0] * sx, v[1] * sy]).collect()).collect(),
        }
    }

    /// Axis-aligned bounding box as `(min_x, min_y, max_x, max_y)`, or `None` for empty.
    pub fn bounding_box(&self) -> Option<(f64, f64, f64, f64)> {
        let v = &self.vertices;
        if v.is_empty() { return None; }
        let min_x = v.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        let min_y = v.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        let max_x = v.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
        let max_y = v.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
        Some((min_x, min_y, max_x, max_y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polygon_area_square_ccw() {
        let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let a = polygon_area(&sq);
        assert!((a - 1.0).abs() < 1e-12, "CCW square area should be +1, got {}", a);
    }

    #[test]
    fn test_polygon_area_square_cw_negative() {
        let sq = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        let a = polygon_area(&sq);
        assert!((a + 1.0).abs() < 1e-12, "CW square area should be -1, got {}", a);
    }

    #[test]
    fn test_polygon_area_triangle() {
        // Triangle with base 2, height 3 → area = 3
        let tri = [[0.0, 0.0], [2.0, 0.0], [1.0, 3.0]];
        let a = polygon_area(&tri).abs();
        assert!((a - 3.0).abs() < 1e-12, "Expected 3.0, got {}", a);
    }

    #[test]
    fn test_point_in_polygon_inside() {
        let sq = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        assert!(point_in_polygon([2.0, 2.0], &sq));
    }

    #[test]
    fn test_point_in_polygon_outside() {
        let sq = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        assert!(!point_in_polygon([5.0, 5.0], &sq));
        assert!(!point_in_polygon([-1.0, 2.0], &sq));
    }

    #[test]
    fn test_polygon_centroid_square() {
        let sq = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let c = polygon_centroid(&sq);
        assert!((c[0] - 2.0).abs() < 1e-10, "cx={}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-10, "cy={}", c[1]);
    }

    #[test]
    fn test_polygon_perimeter_square() {
        let sq = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let p = polygon_perimeter(&sq);
        assert!((p - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_polygon_struct_area_with_hole() {
        let mut p = Polygon::new(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
        p.add_hole(vec![[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]]);
        // Outer = 100, hole = 4 → net = 96
        assert!((p.area() - 96.0).abs() < 1e-10, "Expected 96, got {}", p.area());
    }

    #[test]
    fn test_polygon_contains_with_hole() {
        let mut p = Polygon::new(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
        p.add_hole(vec![[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]]);
        assert!(p.contains([1.0, 1.0]));     // outside hole
        assert!(!p.contains([5.0, 5.0]));    // inside hole
        assert!(!p.contains([11.0, 5.0]));   // outside polygon
    }

    #[test]
    fn test_polygon_validate() {
        let ok = Polygon::new(vec![[0.0,0.0],[1.0,0.0],[0.0,1.0]]);
        assert!(ok.validate().is_ok());
        let bad = Polygon::new(vec![[0.0,0.0],[1.0,0.0]]);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_polygon_bounding_box() {
        let p = Polygon::new(vec![[1.0,2.0],[5.0,2.0],[5.0,8.0],[1.0,8.0]]);
        let bb = p.bounding_box().expect("non-empty");
        assert!((bb.0 - 1.0).abs() < 1e-12);
        assert!((bb.1 - 2.0).abs() < 1e-12);
        assert!((bb.2 - 5.0).abs() < 1e-12);
        assert!((bb.3 - 8.0).abs() < 1e-12);
    }
}
