//! Polygon triangulation algorithms.
//!
//! Provides:
//! - **Ear clipping** (`ear_clipping`) — O(n²), works for simple polygons.
//! - **Fan triangulation** (`fan_triangulation`) — O(n), only correct for convex polygons.
//! - Triangle utilities: area, circumcircle, incircle.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::geom::triangulation::ear_clipping;
//!
//! let hexagon: Vec<[f64; 2]> = (0..6).map(|i| {
//!     let a = i as f64 * std::f64::consts::TAU / 6.0;
//!     [a.cos(), a.sin()]
//! }).collect();
//! let tris = ear_clipping(&hexagon);
//! assert_eq!(tris.len(), 4); // hexagon → 4 triangles
//! ```

// ── Cross product sign ────────────────────────────────────────────────────────

#[inline]
fn cross2(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

// ── Point inside triangle (sign method) ───────────────────────────────────────

fn point_in_triangle(p: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let d1 = cross2(a, b, p);
    let d2 = cross2(b, c, p);
    let d3 = cross2(c, a, p);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

// ── Ear clipping triangulation ────────────────────────────────────────────────

/// Triangulate a simple polygon by **ear clipping**.
///
/// The polygon must be a simple (non-self-intersecting) polygon given as a
/// slice of vertex coordinates.  Returns a list of triangles, each represented
/// as `[i, j, k]` — indices into the original `polygon` slice.
///
/// - Handles both CCW and CW orientations automatically.
/// - Collinear ears are accepted to avoid infinite loops on degenerate input.
/// - Returns `vec![[0, 1, 2]]` for exactly 3 vertices.
/// - Returns an empty vector for fewer than 3 vertices.
///
/// Time complexity: O(n²).
pub fn ear_clipping(polygon: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }
    if n == 3 {
        return vec![[0, 1, 2]];
    }

    // Determine winding; if area < 0 the polygon is CW → use negated cross test
    let area: f64 = (0..n)
        .map(|i| {
            let j = (i + 1) % n;
            polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
        })
        .sum::<f64>() * 0.5;
    let sign = if area >= 0.0 { 1.0_f64 } else { -1.0_f64 };

    // Working list of vertex indices
    let mut ring: Vec<usize> = (0..n).collect();
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(n - 2);
    let mut iterations_without_ear = 0usize;

    while ring.len() > 3 {
        let len = ring.len();
        let mut found_ear = false;

        for pos in 0..len {
            let prev = (pos + len - 1) % len;
            let next = (pos + 1) % len;

            let a = polygon[ring[prev]];
            let b = polygon[ring[pos]];
            let c = polygon[ring[next]];

            // Ear condition 1: must be convex (CCW turn in CCW polygon)
            if sign * cross2(a, b, c) <= 0.0 {
                continue;
            }

            // Ear condition 2: no other vertex of the ring inside triangle (a,b,c)
            let is_ear = !ring.iter().enumerate().any(|(k, &vi)| {
                k != prev && k != pos && k != next
                    && point_in_triangle(polygon[vi], a, b, c)
            });

            if is_ear {
                tris.push([ring[prev], ring[pos], ring[next]]);
                ring.remove(pos);
                found_ear = true;
                iterations_without_ear = 0;
                break;
            }
        }

        if !found_ear {
            iterations_without_ear += 1;
            // Safety valve: if we can't find any ear after a full pass,
            // emit a degenerate triangle and remove the middle vertex to make progress.
            if iterations_without_ear >= ring.len() {
                let a = ring[0];
                let b = ring[1];
                let c = ring[2 % ring.len()];
                tris.push([a, b, c]);
                ring.remove(1);
                iterations_without_ear = 0;
            }
        }
    }

    // Final triangle
    if ring.len() == 3 {
        tris.push([ring[0], ring[1], ring[2]]);
    }

    tris
}

// ── Fan triangulation (convex polygons only) ───────────────────────────────────

/// Triangulate a **convex** polygon by fan decomposition from vertex 0.
///
/// Returns `n-2` triangles for an n-gon.  This is O(n) but only correct for
/// convex polygons.
pub fn fan_triangulation(polygon: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = polygon.len();
    if n < 3 { return Vec::new(); }
    (1..n - 1).map(|i| [0, i, i + 1]).collect()
}

// ── Triangle utilities ─────────────────────────────────────────────────────────

/// Signed area of a triangle with vertices `a`, `b`, `c`.
/// Positive = CCW, negative = CW.
#[inline]
pub fn triangle_area(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    cross2(a, b, c) * 0.5
}

/// Circumcircle centre and radius of triangle `(a, b, c)`.
///
/// Returns `None` for degenerate (collinear) triangles.
pub fn circumcircle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<([f64; 2], f64)> {
    let ax = b[0] - a[0]; let ay = b[1] - a[1];
    let bx = c[0] - a[0]; let by = c[1] - a[1];
    let d = 2.0 * (ax * by - ay * bx);
    if d.abs() < 1e-15 { return None; }
    let ux = (by * (ax*ax + ay*ay) - ay * (bx*bx + by*by)) / d;
    let uy = (ax * (bx*bx + by*by) - bx * (ax*ax + ay*ay)) / d;
    let cx = a[0] + ux;
    let cy = a[1] + uy;
    let r = (ux*ux + uy*uy).sqrt();
    Some(([cx, cy], r))
}

/// Incircle radius of triangle `(a, b, c)`.
///
/// Returns `None` for degenerate triangles.
pub fn incircle_radius(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<f64> {
    let ab = ((b[0]-a[0]).powi(2) + (b[1]-a[1]).powi(2)).sqrt();
    let bc = ((c[0]-b[0]).powi(2) + (c[1]-b[1]).powi(2)).sqrt();
    let ca = ((a[0]-c[0]).powi(2) + (a[1]-c[1]).powi(2)).sqrt();
    let s = ab + bc + ca;
    if s < 1e-15 { return None; }
    let area = triangle_area(a, b, c).abs();
    Some(2.0 * area / s)
}

/// Quality metrics for a triangle: aspect ratio (circumradius / inradius).
/// Returns `None` for degenerate triangles.
pub fn triangle_quality(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<f64> {
    let (_, cr) = circumcircle(a, b, c)?;
    let ir = incircle_radius(a, b, c)?;
    if ir < 1e-15 { return None; }
    Some(cr / ir)
}

// ── Constrained triangulation (point cloud with constraints) ──────────────────

/// Triangulate a point cloud (not necessarily a polygon) using a simple
/// incremental method based on fan decomposition of the convex hull.
///
/// Returns triangles as triplets of indices into `points`.
/// This is an approximation: for proper Delaunay triangulation use
/// `scirs2_spatial::computational_geometry::fortune_voronoi`.
pub fn point_cloud_triangulation(points: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let n = points.len();
    if n < 3 { return Vec::new(); }

    // Sort by x to find a convex-hull-like ordering
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        points[a][0]
            .partial_cmp(&points[b][0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                points[a][1]
                    .partial_cmp(&points[b][1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    // Convex hull (monotone chain)
    let sorted_pts: Vec<[f64; 2]> = idx.iter().map(|&i| points[i]).collect();
    let hull = crate::geom::convex_hull::convex_hull_2d(sorted_pts);

    // Fan triangulate the hull (only correct for convex shapes)
    let hull_n = hull.len();
    if hull_n < 3 { return Vec::new(); }

    // Map hull vertices back to original indices
    let find_orig = |p: [f64; 2]| -> usize {
        points
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = (a[0]-p[0]).powi(2) + (a[1]-p[1]).powi(2);
                let db = (b[0]-p[0]).powi(2) + (b[1]-p[1]).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    };

    (1..hull_n - 1)
        .map(|i| {
            [
                find_orig(hull[0]),
                find_orig(hull[i]),
                find_orig(hull[i + 1]),
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ear_clip_triangle() {
        let pts = [[0.0_f64, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let t = ear_clipping(&pts);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0], [0, 1, 2]);
    }

    #[test]
    fn test_ear_clip_square() {
        let sq = [[0.0_f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let t = ear_clipping(&sq);
        assert_eq!(t.len(), 2, "Square should give 2 triangles");
    }

    #[test]
    fn test_ear_clip_hexagon() {
        let hex: Vec<[f64; 2]> = (0..6)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 6.0;
                [a.cos(), a.sin()]
            })
            .collect();
        let t = ear_clipping(&hex);
        assert_eq!(t.len(), 4, "Hexagon should give 4 triangles");
    }

    #[test]
    fn test_ear_clip_total_area() {
        let sq = [[0.0_f64, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        let tris = ear_clipping(&sq);
        let total: f64 = tris
            .iter()
            .map(|&[i, j, k]| triangle_area(sq[i], sq[j], sq[k]).abs())
            .sum();
        assert!((total - 4.0).abs() < 1e-10, "Expected area 4, got {total}");
    }

    #[test]
    fn test_fan_triangulation() {
        let pentagon: Vec<[f64; 2]> = (0..5)
            .map(|i| {
                let a = i as f64 * std::f64::consts::TAU / 5.0;
                [a.cos(), a.sin()]
            })
            .collect();
        let t = fan_triangulation(&pentagon);
        assert_eq!(t.len(), 3, "Pentagon → 3 triangles");
    }

    #[test]
    fn test_triangle_area() {
        let a = [0.0_f64, 0.0];
        let b = [4.0, 0.0];
        let c = [0.0, 3.0];
        assert!((triangle_area(a, b, c).abs() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_circumcircle_right_triangle() {
        // Right angle at origin; circumcentre = midpoint of hypotenuse
        let a = [0.0_f64, 0.0];
        let b = [2.0, 0.0];
        let c = [0.0, 2.0];
        let (centre, r) = circumcircle(a, b, c).expect("valid triangle");
        assert!((centre[0] - 1.0).abs() < 1e-10, "cx={}", centre[0]);
        assert!((centre[1] - 1.0).abs() < 1e-10, "cy={}", centre[1]);
        assert!((r - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_circumcircle_degenerate() {
        let a = [0.0_f64, 0.0];
        let b = [1.0, 0.0];
        let c = [2.0, 0.0]; // collinear
        assert!(circumcircle(a, b, c).is_none());
    }

    #[test]
    fn test_incircle_equilateral() {
        let s = 1.0_f64;
        let a = [0.0, 0.0];
        let b = [s, 0.0];
        let c = [s / 2.0, (3.0_f64.sqrt() / 2.0) * s];
        let r = incircle_radius(a, b, c).expect("valid triangle");
        // Inradius of equilateral triangle with side 1 is 1/(2√3)
        let expected = s / (2.0 * 3.0_f64.sqrt());
        assert!((r - expected).abs() < 1e-10, "Expected {expected}, got {r}");
    }
}
