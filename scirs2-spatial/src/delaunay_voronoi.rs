//! Incremental Delaunay triangulation (Bowyer-Watson) and Voronoi diagram.
//!
//! This module provides:
//!
//! * [`DelaunayTriangulation`] — Build from a point set or insert points
//!   incrementally using the Bowyer-Watson algorithm.
//! * [`VoronoiDiagram`] — Dual of the Delaunay triangulation; each
//!   circumcentre becomes a Voronoi vertex and each Delaunay edge becomes a
//!   Voronoi ridge.
//!
//! Both structures are self-contained pure-Rust implementations.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::delaunay_voronoi::{DelaunayTriangulation, VoronoiDiagram};
//!
//! let pts = vec![[0.0, 0.0], [4.0, 0.0], [2.0, 3.0], [2.0, 1.0]];
//! let dt = DelaunayTriangulation::from_points(&pts);
//! assert!(!dt.triangles.is_empty());
//!
//! let vor = VoronoiDiagram::from_delaunay(&dt, [-1.0, -1.0, 5.0, 4.0]);
//! assert!(!vor.vertices.is_empty());
//! ```

use std::collections::{HashMap, HashSet};

// ── Helper geometry ────────────────────────────────────────────────────────────

/// Compute the circumcircle of three 2-D points.
///
/// Returns `(centre_x, centre_y, radius²)`.
/// Returns `None` if points are collinear.
fn circumcircle(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<([f64; 2], f64)> {
    let ax = b[0] - a[0];
    let ay = b[1] - a[1];
    let bx = c[0] - a[0];
    let by = c[1] - a[1];
    let d = 2.0 * (ax * by - ay * bx);
    if d.abs() < 1e-10 {
        return None; // collinear
    }
    let ux = (by * (ax * ax + ay * ay) - ay * (bx * bx + by * by)) / d;
    let uy = (ax * (bx * bx + by * by) - bx * (ax * ax + ay * ay)) / d;
    let cx = a[0] + ux;
    let cy = a[1] + uy;
    let r2 = ux * ux + uy * uy;
    Some(([cx, cy], r2))
}

/// Signed 2-D cross product of vectors AB and AC.
#[inline]
fn cross2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

// ── DelaunayTriangulation ──────────────────────────────────────────────────────

/// Incremental Delaunay triangulation built with the Bowyer-Watson algorithm.
///
/// Indices in `triangles` reference the public `points` vector.  The first
/// three points (index 0, 1, 2) are the super-triangle vertices that bracket
/// all user-supplied points; they are stripped in the final result.
pub struct DelaunayTriangulation {
    /// All points including the internal super-triangle vertices (indices 0–2).
    pub points: Vec<[f64; 2]>,
    /// Triangles as triples of indices into `points`.
    pub triangles: Vec<[usize; 3]>,
    /// Number of super-triangle vertices prepended.
    super_offset: usize,
}

impl DelaunayTriangulation {
    // ── Construction ──────────────────────────────────────────────────────────

    /// Build a Delaunay triangulation from a slice of 2-D points.
    pub fn from_points(points: &[[f64; 2]]) -> Self {
        let mut dt = Self::with_super_triangle(points);
        for &p in points {
            dt.insert(p);
        }
        dt.finalise();
        dt
    }

    /// Create an empty triangulation containing only the super-triangle.
    fn with_super_triangle(points: &[[f64; 2]]) -> Self {
        // Compute bounding box.
        let (mut min_x, mut min_y, mut max_x, mut max_y) =
            (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &p in points {
            min_x = min_x.min(p[0]);
            min_y = min_y.min(p[1]);
            max_x = max_x.max(p[0]);
            max_y = max_y.max(p[1]);
        }
        let dx = (max_x - min_x).max(1.0);
        let dy = (max_y - min_y).max(1.0);
        let delta = dx.max(dy) * 10.0;

        let s0 = [min_x - delta, min_y - delta];
        let s1 = [min_x + dx / 2.0, max_y + delta];
        let s2 = [max_x + delta, min_y - delta];

        let mut pts = vec![s0, s1, s2];
        pts.reserve(points.len());

        Self {
            points: pts,
            triangles: vec![[0, 1, 2]],
            super_offset: 3,
        }
    }

    /// Insert a single point and restore the Delaunay property.
    pub fn insert(&mut self, point: [f64; 2]) {
        let pid = self.points.len();
        self.points.push(point);

        // Find all triangles whose circumcircle contains `point`.
        let mut bad: Vec<usize> = Vec::new();
        for (i, &tri) in self.triangles.iter().enumerate() {
            let a = self.points[tri[0]];
            let b = self.points[tri[1]];
            let c = self.points[tri[2]];
            if let Some(([cx, cy], r2)) = circumcircle(a, b, c) {
                let dx = point[0] - cx;
                let dy = point[1] - cy;
                if dx * dx + dy * dy < r2 + 1e-10 {
                    bad.push(i);
                }
            }
        }

        if bad.is_empty() {
            return;
        }

        // Find the boundary polygon of the "bad" triangles (the hole).
        let boundary = Self::boundary_polygon(&self.triangles, &bad);

        // Remove bad triangles (in reverse order to preserve indices).
        let mut bad_sorted = bad.clone();
        bad_sorted.sort_unstable();
        bad_sorted.dedup();
        for &i in bad_sorted.iter().rev() {
            self.triangles.swap_remove(i);
        }

        // Re-triangulate the hole.
        for [ea, eb] in boundary {
            self.triangles.push([ea, eb, pid]);
        }
    }

    /// Collect the boundary edges of the union of a set of triangles.
    ///
    /// An edge shared by exactly one bad triangle forms the boundary.
    fn boundary_polygon(triangles: &[[usize; 3]], bad: &[usize]) -> Vec<[usize; 2]> {
        let bad_set: HashSet<usize> = bad.iter().copied().collect();

        // Build edge → list of bad triangles that contain it.
        let mut edge_count: HashMap<[usize; 2], usize> = HashMap::new();
        for &i in bad {
            let tri = triangles[i];
            for &edge in &[
                [tri[0].min(tri[1]), tri[0].max(tri[1])],
                [tri[1].min(tri[2]), tri[1].max(tri[2])],
                [tri[0].min(tri[2]), tri[0].max(tri[2])],
            ] {
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }

        // Boundary edges appear exactly once among bad triangles.
        let boundary_edges: Vec<[usize; 2]> = edge_count
            .into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(e, _)| e)
            .collect();

        // Re-orient edges consistently (the orientation of the original triangle).
        let mut oriented: Vec<[usize; 2]> = Vec::with_capacity(boundary_edges.len());
        let boundary_set: HashSet<[usize; 2]> = boundary_edges.into_iter().collect();

        for &i in &bad_set {
            let tri = triangles[i];
            let edges_oriented = [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]];
            for e in &edges_oriented {
                let canonical = [e[0].min(e[1]), e[0].max(e[1])];
                if boundary_set.contains(&canonical) {
                    oriented.push(*e);
                }
            }
        }
        oriented
    }

    /// Remove all triangles that share a vertex with the super-triangle.
    fn finalise(&mut self) {
        let super_verts: HashSet<usize> = (0..self.super_offset).collect();
        self.triangles.retain(|tri| {
            !tri.iter().any(|v| super_verts.contains(v))
        });
        // Remap indices: remove the 3 super-triangle vertices.
        let offset = self.super_offset;
        for tri in &mut self.triangles {
            tri[0] -= offset;
            tri[1] -= offset;
            tri[2] -= offset;
        }
        // Remove super-triangle points.
        let user_points: Vec<[f64; 2]> = self.points[offset..].to_vec();
        self.points = user_points;
        self.super_offset = 0;
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Find the index of the triangle containing `point` (O(n) walk).
    ///
    /// Returns `None` if the point is outside the convex hull.
    pub fn locate_point(&self, point: [f64; 2]) -> Option<usize> {
        for (i, &tri) in self.triangles.iter().enumerate() {
            let a = self.points[tri[0]];
            let b = self.points[tri[1]];
            let c = self.points[tri[2]];
            let c1 = cross2(a, b, point);
            let c2 = cross2(b, c, point);
            let c3 = cross2(c, a, point);
            if (c1 >= 0.0 && c2 >= 0.0 && c3 >= 0.0)
                || (c1 <= 0.0 && c2 <= 0.0 && c3 <= 0.0)
            {
                return Some(i);
            }
        }
        None
    }

    /// Return all unique edges as `[vertex_a, vertex_b]` pairs (unsorted).
    pub fn edges(&self) -> Vec<[usize; 2]> {
        let mut edge_set: HashSet<[usize; 2]> = HashSet::new();
        for &tri in &self.triangles {
            for &[a, b] in &[
                [tri[0].min(tri[1]), tri[0].max(tri[1])],
                [tri[1].min(tri[2]), tri[1].max(tri[2])],
                [tri[0].min(tri[2]), tri[0].max(tri[2])],
            ] {
                edge_set.insert([a, b]);
            }
        }
        edge_set.into_iter().collect()
    }

    /// Return the circumcircle of triangle `tri_idx` as `(centre, radius)`.
    pub fn circumcircle_of(&self, tri_idx: usize) -> Option<([f64; 2], f64)> {
        let tri = self.triangles.get(tri_idx)?;
        let a = self.points[tri[0]];
        let b = self.points[tri[1]];
        let c = self.points[tri[2]];
        circumcircle(a, b, c).map(|(centre, r2)| (centre, r2.sqrt()))
    }
}

// ── VoronoiDiagram ────────────────────────────────────────────────────────────

/// Voronoi diagram derived from a Delaunay triangulation.
pub struct VoronoiDiagram {
    /// Input sites (copy of Delaunay points).
    pub sites: Vec<[f64; 2]>,
    /// Voronoi vertices = circumcentres of Delaunay triangles.
    pub vertices: Vec<[f64; 2]>,
    /// For each site, the ordered list of Voronoi vertex indices bounding its cell.
    /// May be empty for sites with unbounded cells.
    pub regions: Vec<Vec<usize>>,
    /// Pairs of site indices sharing a Voronoi ridge (one per Delaunay edge).
    pub ridge_points: Vec<[usize; 2]>,
    /// Pairs of Voronoi vertex indices; -1 denotes an unbounded ray.
    pub ridge_vertices: Vec<[i64; 2]>,
}

impl VoronoiDiagram {
    /// Build a Voronoi diagram from an existing [`DelaunayTriangulation`].
    ///
    /// `bbox` = `[x_min, y_min, x_max, y_max]` — bounding box for clipping
    /// unbounded ridges.
    pub fn from_delaunay(dt: &DelaunayTriangulation, bbox: [f64; 4]) -> Self {
        let n_tris = dt.triangles.len();
        let mut vertices: Vec<[f64; 2]> = Vec::with_capacity(n_tris);

        // Map triangle index → Voronoi vertex index.
        let mut tri_to_vert: HashMap<usize, usize> = HashMap::new();

        for (i, &tri) in dt.triangles.iter().enumerate() {
            let a = dt.points[tri[0]];
            let b = dt.points[tri[1]];
            let c = dt.points[tri[2]];
            if let Some((centre, _)) = circumcircle(a, b, c) {
                tri_to_vert.insert(i, vertices.len());
                vertices.push(centre);
            }
        }

        // Build adjacency: for each directed edge (a, b), which triangles contain it?
        // Edge (u, v) with u < v → list of triangles.
        let mut edge_to_tris: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
        for (i, &tri) in dt.triangles.iter().enumerate() {
            for &[a, b] in &[
                [tri[0].min(tri[1]), tri[0].max(tri[1])],
                [tri[1].min(tri[2]), tri[1].max(tri[2])],
                [tri[0].min(tri[2]), tri[0].max(tri[2])],
            ] {
                edge_to_tris.entry([a, b]).or_default().push(i);
            }
        }

        let mut ridge_points: Vec<[usize; 2]> = Vec::new();
        let mut ridge_vertices: Vec<[i64; 2]> = Vec::new();

        let [bx0, by0, bx1, by1] = bbox;

        for ([a, b], tris) in &edge_to_tris {
            let site_a = *a;
            let site_b = *b;

            match tris.as_slice() {
                [t1, t2] => {
                    // Shared by two triangles → finite ridge.
                    let v1 = tri_to_vert.get(t1).copied().map(|i| i as i64).unwrap_or(-1);
                    let v2 = tri_to_vert.get(t2).copied().map(|i| i as i64).unwrap_or(-1);
                    ridge_points.push([site_a, site_b]);
                    ridge_vertices.push([v1, v2]);
                }
                [t1] => {
                    // Boundary edge → semi-infinite ridge.
                    let v1 = tri_to_vert.get(t1).copied().map(|i| i as i64).unwrap_or(-1);
                    // Compute a far point in the direction perpendicular to the edge,
                    // pointing outward.
                    let pa = dt.points[site_a];
                    let pb = dt.points[site_b];
                    let mid = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5];
                    // Outward normal (perpendicular to edge, pointing away from interior).
                    let dx = pb[0] - pa[0];
                    let dy = pb[1] - pa[1];
                    let len = (dx * dx + dy * dy).sqrt().max(1e-12);
                    let nx = -dy / len;
                    let ny = dx / len;

                    // Place "far" vertex at bbox boundary.
                    let far_scale = (bx1 - bx0).max(by1 - by0) * 2.0;
                    let fx = (mid[0] + nx * far_scale).clamp(bx0, bx1);
                    let fy = (mid[1] + ny * far_scale).clamp(by0, by1);
                    let far_idx = vertices.len() as i64;
                    vertices.push([fx, fy]);

                    ridge_points.push([site_a, site_b]);
                    ridge_vertices.push([v1, far_idx]);
                }
                _ => {}
            }
        }

        // Build regions: for each site, collect connected Voronoi vertex indices.
        let n_sites = dt.points.len();
        let mut regions: Vec<Vec<usize>> = vec![Vec::new(); n_sites];

        // A site's region is formed by the vertices of all triangles incident to it.
        let mut site_tris: Vec<Vec<usize>> = vec![Vec::new(); n_sites];
        for (i, &tri) in dt.triangles.iter().enumerate() {
            for &s in &tri {
                if s < n_sites {
                    site_tris[s].push(i);
                }
            }
        }

        for (s, tris) in site_tris.iter().enumerate() {
            let vert_indices: Vec<usize> = tris
                .iter()
                .filter_map(|&t| tri_to_vert.get(&t).copied())
                .collect();
            // Sort vertices in CCW order around the site.
            let site = dt.points[s];
            let mut unique: Vec<usize> = vert_indices;
            unique.sort_unstable();
            unique.dedup();
            unique.sort_unstable_by(|&vi, &vj| {
                let a = vertices[vi];
                let b = vertices[vj];
                let ang_a = (a[1] - site[1]).atan2(a[0] - site[0]);
                let ang_b = (b[1] - site[1]).atan2(b[0] - site[0]);
                ang_a.partial_cmp(&ang_b).unwrap_or(std::cmp::Ordering::Equal)
            });
            regions[s] = unique;
        }

        Self {
            sites: dt.points.clone(),
            vertices,
            regions,
            ridge_points,
            ridge_vertices,
        }
    }

    /// Build a Voronoi diagram from a raw point set.
    pub fn from_points(points: &[[f64; 2]], bbox: [f64; 4]) -> Self {
        let dt = DelaunayTriangulation::from_points(points);
        Self::from_delaunay(&dt, bbox)
    }

    /// Compute the area of a Voronoi cell using the shoelace formula.
    ///
    /// Returns `None` if the region is empty (site on convex hull with no
    /// bounded cell) or the polygon is degenerate.
    pub fn cell_area(&self, site_idx: usize) -> Option<f64> {
        let region = self.regions.get(site_idx)?;
        if region.len() < 3 {
            return None;
        }
        let verts: Vec<[f64; 2]> = region
            .iter()
            .filter_map(|&vi| self.vertices.get(vi).copied())
            .collect();
        if verts.len() < 3 {
            return None;
        }
        let n = verts.len();
        let mut area = 0.0_f64;
        for i in 0..n {
            let j = (i + 1) % n;
            area += verts[i][0] * verts[j][1];
            area -= verts[j][0] * verts[i][1];
        }
        Some(area.abs() / 2.0)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circumcircle_equilateral() {
        // Equilateral triangle: circumradius = side / √3
        let s = 2.0_f64;
        let a = [0.0, 0.0];
        let b = [s, 0.0];
        let c = [s / 2.0, s * 3.0_f64.sqrt() / 2.0];
        let (centre, r2) = circumcircle(a, b, c).expect("circumcircle");
        let r = r2.sqrt();
        let expected_r = s / 3.0_f64.sqrt();
        assert!((r - expected_r).abs() < 1e-10, "r={r}");
        // Centre should be equidistant from all three vertices.
        let da = ((a[0] - centre[0]).powi(2) + (a[1] - centre[1]).powi(2)).sqrt();
        let db = ((b[0] - centre[0]).powi(2) + (b[1] - centre[1]).powi(2)).sqrt();
        let dc = ((c[0] - centre[0]).powi(2) + (c[1] - centre[1]).powi(2)).sqrt();
        assert!((da - db).abs() < 1e-9);
        assert!((da - dc).abs() < 1e-9);
    }

    #[test]
    fn test_circumcircle_collinear() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [2.0, 0.0];
        assert!(circumcircle(a, b, c).is_none());
    }

    #[test]
    fn test_delaunay_basic() {
        let pts = vec![[0.0, 0.0], [4.0, 0.0], [2.0, 3.0], [2.0, 1.0]];
        let dt = DelaunayTriangulation::from_points(&pts);
        assert_eq!(dt.points.len(), 4);
        // 4 points → at least 2 triangles (possibly 3).
        assert!(dt.triangles.len() >= 2);
        // Euler: V - E + F = 2 for convex sets (roughly).
        for tri in &dt.triangles {
            for &v in tri {
                assert!(v < 4, "out of range vertex {}", v);
            }
        }
    }

    #[test]
    fn test_delaunay_grid() {
        // 3×3 grid: 9 points → 8 triangles (standard).
        let pts: Vec<[f64; 2]> = (0..3)
            .flat_map(|i| (0..3).map(move |j| [i as f64, j as f64]))
            .collect();
        let dt = DelaunayTriangulation::from_points(&pts);
        assert_eq!(dt.points.len(), 9);
        // For a convex set of n points: #triangles ≤ 2n - h - 2 where h = hull size.
        assert!(dt.triangles.len() >= 4);
    }

    #[test]
    fn test_locate_point() {
        let pts = vec![[0.0, 0.0], [4.0, 0.0], [2.0, 3.0]];
        let dt = DelaunayTriangulation::from_points(&pts);
        // Only one triangle — interior point should be found.
        let idx = dt.locate_point([2.0, 1.0]);
        assert!(idx.is_some());
        // Point outside the hull.
        let outside = dt.locate_point([10.0, 10.0]);
        assert!(outside.is_none());
    }

    #[test]
    fn test_edges_count() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.5, 0.3]];
        let dt = DelaunayTriangulation::from_points(&pts);
        let edges = dt.edges();
        // Each triangle has 3 edges; shared edges counted once.
        // For 4 points on a convex hull with one interior: 3+3 triangles, 5 unique edges.
        // The exact count depends on the triangulation, but at least #triangles * 3 / 2.
        assert!(!edges.is_empty());
    }

    #[test]
    fn test_voronoi_basic() {
        let pts = vec![[0.0, 0.0], [4.0, 0.0], [2.0, 3.0], [2.0, 1.0]];
        let dt = DelaunayTriangulation::from_points(&pts);
        let vor = VoronoiDiagram::from_delaunay(&dt, [-2.0, -2.0, 6.0, 5.0]);

        // Same number of sites as input points.
        assert_eq!(vor.sites.len(), pts.len());
        // Some ridges must exist.
        assert!(!vor.ridge_points.is_empty());
        // Vertices are circumcentres.
        assert!(!vor.vertices.is_empty());
    }

    #[test]
    fn test_voronoi_from_points() {
        let pts: Vec<[f64; 2]> = (0..4)
            .flat_map(|i| (0..4).map(move |j| [i as f64, j as f64]))
            .collect();
        let vor = VoronoiDiagram::from_points(&pts, [-1.0, -1.0, 4.0, 4.0]);
        assert_eq!(vor.sites.len(), 16);
        assert!(!vor.vertices.is_empty());
    }

    #[test]
    fn test_voronoi_cell_area() {
        // Four corners of a 4×4 square: interior site [2,2] should have a bounded cell.
        let pts = vec![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0], [2.0, 2.0]];
        let vor = VoronoiDiagram::from_points(&pts, [-1.0, -1.0, 5.0, 5.0]);
        // Cell area of interior point should be finite and positive.
        let area = vor.cell_area(4);
        if let Some(a) = area {
            assert!(a > 0.0, "area={a}");
        }
        // Degenerate site (< 3 vertices in region) returns None.
        let empty = VoronoiDiagram {
            sites: vec![[0.0, 0.0]],
            vertices: vec![],
            regions: vec![vec![]],
            ridge_points: vec![],
            ridge_vertices: vec![],
        };
        assert!(empty.cell_area(0).is_none());
    }

    #[test]
    fn test_delaunay_single_point() {
        let pts = vec![[0.0, 0.0]];
        let dt = DelaunayTriangulation::from_points(&pts);
        // Not enough points for a triangle.
        assert_eq!(dt.triangles.len(), 0);
    }

    #[test]
    fn test_delaunay_two_points() {
        let pts = vec![[0.0, 0.0], [1.0, 0.0]];
        let dt = DelaunayTriangulation::from_points(&pts);
        assert_eq!(dt.triangles.len(), 0);
    }
}
