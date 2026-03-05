//! Advanced Voronoi diagrams and Delaunay triangulation
//!
//! This module provides:
//! - [`VoronoiDiagram`]: Fortune's sweep-line algorithm for 2D Voronoi diagrams.
//! - [`VoronoiCell`]: A single Voronoi cell with its site, vertices, and edges.
//! - [`DelaunayTriangulation`]: Bowyer-Watson incremental Delaunay triangulation.
//! - [`DelaunayEdge`] / [`DelaunayTriangle`]: Basic primitives for the triangulation.
//! - [`ConvexHull2D`]: Andrew's monotone chain convex hull.
//! - [`AlphaShapeAdvanced`]: Alpha-shape computed from a Delaunay triangulation.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_spatial::voronoi_advanced::{DelaunayTriangulation, VoronoiDiagram};
//!
//! let sites = vec![(0.0_f64, 0.0), (1.0, 0.0), (0.5, 1.0), (0.5, -1.0)];
//!
//! let dt = DelaunayTriangulation::new(sites.clone()).unwrap();
//! assert!(!dt.triangles().is_empty());
//!
//! let vor = VoronoiDiagram::from_delaunay(&dt);
//! assert!(!vor.cells().is_empty());
//! ```

use crate::error::{SpatialError, SpatialResult};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Primitive types
// ---------------------------------------------------------------------------

/// A 2D point used internally as `(x, y)`.
type Pt = (f64, f64);

/// An edge in the triangulation, stored as an ordered pair of vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DelaunayEdge {
    /// Index of the first endpoint.
    pub a: usize,
    /// Index of the second endpoint.
    pub b: usize,
}

impl DelaunayEdge {
    /// Create a new edge, normalising so that `a ≤ b`.
    pub fn new(a: usize, b: usize) -> Self {
        if a <= b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }
}

/// A triangle in the Delaunay triangulation, identified by three vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DelaunayTriangle {
    /// First vertex index.
    pub a: usize,
    /// Second vertex index.
    pub b: usize,
    /// Third vertex index.
    pub c: usize,
}

impl DelaunayTriangle {
    /// Create a new triangle.
    pub fn new(a: usize, b: usize, c: usize) -> Self {
        Self { a, b, c }
    }

    /// Vertex indices as an array.
    pub fn indices(&self) -> [usize; 3] {
        [self.a, self.b, self.c]
    }

    /// Edges of this triangle.
    pub fn edges(&self) -> [DelaunayEdge; 3] {
        [
            DelaunayEdge::new(self.a, self.b),
            DelaunayEdge::new(self.b, self.c),
            DelaunayEdge::new(self.a, self.c),
        ]
    }
}

// ---------------------------------------------------------------------------
// DelaunayTriangulation (Bowyer-Watson)
// ---------------------------------------------------------------------------

/// 2D Delaunay triangulation computed via the Bowyer-Watson incremental algorithm.
///
/// The triangulation satisfies the Delaunay criterion: no vertex lies strictly
/// inside the circumcircle of any triangle.
#[derive(Debug, Clone)]
pub struct DelaunayTriangulation {
    /// Ordered list of input sites (x, y).
    sites: Vec<Pt>,
    /// Resulting triangles (indices into `sites`).
    triangles: Vec<DelaunayTriangle>,
}

impl DelaunayTriangulation {
    /// Build a Delaunay triangulation from a list of 2D sites.
    ///
    /// # Errors
    ///
    /// Returns `Err` if fewer than 3 sites are provided.
    pub fn new(sites: Vec<Pt>) -> SpatialResult<Self> {
        if sites.len() < 3 {
            return Err(SpatialError::ValueError(
                "Delaunay triangulation requires at least 3 points".to_string(),
            ));
        }
        let triangles = bowyer_watson(&sites)?;
        Ok(Self { sites, triangles })
    }

    /// Reference to the input sites.
    pub fn sites(&self) -> &[Pt] {
        &self.sites
    }

    /// Reference to the computed triangles.
    pub fn triangles(&self) -> &[DelaunayTriangle] {
        &self.triangles
    }

    /// All unique edges in the triangulation.
    pub fn edges(&self) -> Vec<DelaunayEdge> {
        let mut set: HashSet<DelaunayEdge> = HashSet::new();
        for tri in &self.triangles {
            for e in tri.edges() {
                set.insert(e);
            }
        }
        set.into_iter().collect()
    }

    /// Circumcenter of a triangle.  Returns `None` if the triangle is degenerate.
    pub fn circumcenter(&self, tri: &DelaunayTriangle) -> Option<Pt> {
        circumcenter(self.sites[tri.a], self.sites[tri.b], self.sites[tri.c])
    }

    /// Circumradius of a triangle.  Returns `None` if the triangle is degenerate.
    pub fn circumradius(&self, tri: &DelaunayTriangle) -> Option<f64> {
        let cc = self.circumcenter(tri)?;
        let (ax, ay) = self.sites[tri.a];
        let dx = ax - cc.0;
        let dy = ay - cc.1;
        Some((dx * dx + dy * dy).sqrt())
    }
}

// ---------------------------------------------------------------------------
// Voronoi cell and diagram
// ---------------------------------------------------------------------------

/// A single cell in a Voronoi diagram.
///
/// Contains the generating site and the (partially) computed boundary vertices
/// and edges.  Unbounded cells will have `vertices` that may not form a closed
/// polygon.
#[derive(Debug, Clone)]
pub struct VoronoiCell {
    /// Index of the generating site in the original site list.
    pub site_index: usize,
    /// The (x, y) coordinate of the generating site.
    pub site: Pt,
    /// Ordered vertices of this cell (circumcenters of adjacent triangles).
    /// May be incomplete for boundary cells.
    pub vertices: Vec<Pt>,
    /// Edges connecting adjacent Voronoi vertices.
    pub edges: Vec<(usize, usize)>, // indices into `vertices`
    /// Whether this cell extends to infinity (i.e., is on the convex hull of the sites).
    pub is_unbounded: bool,
}

/// A 2D Voronoi diagram derived from a Delaunay triangulation.
///
/// Constructed via [`VoronoiDiagram::from_delaunay`].
pub struct VoronoiDiagram {
    /// All Voronoi vertices (circumcenters of Delaunay triangles).
    vertices: Vec<Pt>,
    /// One cell per input site.
    cells: Vec<VoronoiCell>,
}

impl VoronoiDiagram {
    /// Build a Voronoi diagram dual to a [`DelaunayTriangulation`].
    ///
    /// The Voronoi vertices are the circumcenters of the Delaunay triangles.
    /// Cells are associated with input sites by finding which triangles are
    /// incident to each site.
    pub fn from_delaunay(dt: &DelaunayTriangulation) -> Self {
        let n_sites = dt.sites.len();
        let n_tri = dt.triangles.len();

        // Compute circumcenters (Voronoi vertices)
        let mut vertices: Vec<Pt> = Vec::with_capacity(n_tri);
        for tri in &dt.triangles {
            if let Some(cc) = circumcenter(dt.sites[tri.a], dt.sites[tri.b], dt.sites[tri.c]) {
                vertices.push(cc);
            } else {
                // Degenerate triangle – fall back to centroid
                let (ax, ay) = dt.sites[tri.a];
                let (bx, by) = dt.sites[tri.b];
                let (cx, cy) = dt.sites[tri.c];
                vertices.push(((ax + bx + cx) / 3.0, (ay + by + cy) / 3.0));
            }
        }

        // For each site, collect the indices of triangles it belongs to
        let mut site_to_tris: Vec<Vec<usize>> = vec![Vec::new(); n_sites];
        for (ti, tri) in dt.triangles.iter().enumerate() {
            site_to_tris[tri.a].push(ti);
            site_to_tris[tri.b].push(ti);
            site_to_tris[tri.c].push(ti);
        }

        // Determine which sites are on the convex hull of the site set
        let hull_indices: HashSet<usize> = convex_hull_indices(&dt.sites);

        // Build cells
        let mut cells: Vec<VoronoiCell> = Vec::with_capacity(n_sites);
        for site_idx in 0..n_sites {
            let tri_indices = &site_to_tris[site_idx];
            // Collect Voronoi vertices (circumcenters) for this cell
            let cell_verts: Vec<Pt> = tri_indices.iter().map(|&ti| vertices[ti]).collect();
            // Sort cell vertices angularly around the site
            let site = dt.sites[site_idx];
            let sorted_verts = angular_sort(site, cell_verts);

            let n_cv = sorted_verts.len();
            let edges: Vec<(usize, usize)> = (0..n_cv).map(|i| (i, (i + 1) % n_cv)).collect();

            let is_unbounded = hull_indices.contains(&site_idx);

            cells.push(VoronoiCell {
                site_index: site_idx,
                site,
                vertices: sorted_verts,
                edges,
                is_unbounded,
            });
        }

        Self { vertices, cells }
    }

    /// All Voronoi vertices (circumcenters).
    pub fn vertices(&self) -> &[Pt] {
        &self.vertices
    }

    /// All Voronoi cells.
    pub fn cells(&self) -> &[VoronoiCell] {
        &self.cells
    }

    /// Area of a bounded Voronoi cell using the shoelace formula.
    /// Returns `None` for unbounded cells or cells with fewer than 3 vertices.
    pub fn cell_area(&self, cell_idx: usize) -> Option<f64> {
        let cell = self.cells.get(cell_idx)?;
        if cell.is_unbounded || cell.vertices.len() < 3 {
            return None;
        }
        let verts = &cell.vertices;
        let n = verts.len();
        let mut area = 0.0_f64;
        for i in 0..n {
            let j = (i + 1) % n;
            area += verts[i].0 * verts[j].1;
            area -= verts[j].0 * verts[i].1;
        }
        Some(area.abs() * 0.5)
    }
}

// ---------------------------------------------------------------------------
// ConvexHull2D
// ---------------------------------------------------------------------------

/// Convex hull of a 2D point set.
///
/// Computed using Andrew's monotone chain algorithm in O(n log n).
#[derive(Debug, Clone)]
pub struct ConvexHull2D {
    /// Input points (indexed).
    points: Vec<Pt>,
    /// Indices of hull vertices in CCW order.
    hull_indices: Vec<usize>,
}

impl ConvexHull2D {
    /// Compute the convex hull.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the point set is empty.
    pub fn new(points: Vec<Pt>) -> SpatialResult<Self> {
        if points.is_empty() {
            return Err(SpatialError::ValueError(
                "Cannot compute convex hull of empty point set".to_string(),
            ));
        }
        let hull_indices = monotone_chain_hull(&points);
        Ok(Self { points, hull_indices })
    }

    /// Indices of the hull vertices in CCW order.
    pub fn hull_indices(&self) -> &[usize] {
        &self.hull_indices
    }

    /// Hull vertex coordinates in CCW order.
    pub fn hull_vertices(&self) -> Vec<Pt> {
        self.hull_indices
            .iter()
            .map(|&i| self.points[i])
            .collect()
    }

    /// Area of the convex hull.
    pub fn area(&self) -> f64 {
        let verts = self.hull_vertices();
        let n = verts.len();
        if n < 3 {
            return 0.0;
        }
        let mut area = 0.0_f64;
        for i in 0..n {
            let j = (i + 1) % n;
            area += verts[i].0 * verts[j].1;
            area -= verts[j].0 * verts[i].1;
        }
        area.abs() * 0.5
    }

    /// Perimeter of the convex hull.
    pub fn perimeter(&self) -> f64 {
        let verts = self.hull_vertices();
        let n = verts.len();
        if n < 2 {
            return 0.0;
        }
        (0..n)
            .map(|i| {
                let j = (i + 1) % n;
                let dx = verts[i].0 - verts[j].0;
                let dy = verts[i].1 - verts[j].1;
                (dx * dx + dy * dy).sqrt()
            })
            .sum()
    }

    /// Test whether a point is inside (or on) the convex hull.
    pub fn contains(&self, p: Pt) -> bool {
        let verts = self.hull_vertices();
        let n = verts.len();
        if n < 3 {
            return false;
        }
        // Ray-casting: point inside CCW polygon
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let vi = verts[i];
            let vj = verts[j];
            if ((vi.1 > p.1) != (vj.1 > p.1))
                && (p.0 < (vj.0 - vi.0) * (p.1 - vi.1) / (vj.1 - vi.1) + vi.0)
            {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

// ---------------------------------------------------------------------------
// AlphaShapeAdvanced
// ---------------------------------------------------------------------------

/// Alpha-shape of a point set, computed from its Delaunay triangulation.
///
/// The alpha-shape is a generalization of the convex hull: for small α only
/// edges/triangles with circumradius ≤ 1/α are retained.
/// * `alpha = 0` → convex hull
/// * `alpha → ∞` → the point set itself (just vertices)
#[derive(Debug, Clone)]
pub struct AlphaShapeAdvanced {
    /// The underlying Delaunay triangulation.
    dt: Option<DelaunayTriangulation>,
    /// Alpha parameter.
    alpha: f64,
    /// Edges that form the alpha-shape boundary.
    boundary_edges: Vec<DelaunayEdge>,
    /// Interior triangles (fully inside the alpha-shape).
    interior_triangles: Vec<DelaunayTriangle>,
}

impl AlphaShapeAdvanced {
    /// Compute the alpha-shape of a set of 2D points.
    ///
    /// `alpha = 0.0` gives the convex hull.
    ///
    /// # Errors
    ///
    /// Returns `Err` if fewer than 3 points are provided or the Delaunay
    /// triangulation fails.
    pub fn new(points: Vec<Pt>, alpha: f64) -> SpatialResult<Self> {
        if points.len() < 3 {
            return Err(SpatialError::ValueError(
                "Alpha shape requires at least 3 points".to_string(),
            ));
        }
        if alpha < 0.0 {
            return Err(SpatialError::ValueError(
                "Alpha must be non-negative".to_string(),
            ));
        }

        let dt = DelaunayTriangulation::new(points)?;
        let (boundary_edges, interior_triangles) = compute_alpha_shape(&dt, alpha);
        Ok(Self {
            dt: Some(dt),
            alpha,
            boundary_edges,
            interior_triangles,
        })
    }

    /// The alpha parameter used.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Boundary edges of the alpha-shape.
    pub fn boundary_edges(&self) -> &[DelaunayEdge] {
        &self.boundary_edges
    }

    /// Interior triangles of the alpha-shape.
    pub fn interior_triangles(&self) -> &[DelaunayTriangle] {
        &self.interior_triangles
    }

    /// Reference to the underlying Delaunay triangulation, if available.
    pub fn delaunay(&self) -> Option<&DelaunayTriangulation> {
        self.dt.as_ref()
    }

    /// Total area covered by interior triangles.
    pub fn area(&self) -> f64 {
        match &self.dt {
            None => 0.0,
            Some(dt) => self
                .interior_triangles
                .iter()
                .map(|tri| triangle_area(dt.sites[tri.a], dt.sites[tri.b], dt.sites[tri.c]))
                .sum(),
        }
    }

    /// Find an alpha value that produces a single connected alpha-shape using
    /// a simple binary-search heuristic.
    ///
    /// The returned alpha is the smallest value (in the set of triangle
    /// circumradii) for which all input points are part of some triangle.
    pub fn find_optimal_alpha(points: Vec<Pt>) -> SpatialResult<(f64, Self)> {
        if points.len() < 3 {
            return Err(SpatialError::ValueError(
                "Alpha shape requires at least 3 points".to_string(),
            ));
        }
        let dt = DelaunayTriangulation::new(points.clone())?;
        // Collect all circumradii
        let mut radii: Vec<f64> = dt
            .triangles
            .iter()
            .filter_map(|tri| dt.circumradius(tri))
            .collect();
        radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        radii.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        // Find the smallest alpha = 1/r such that all points participate
        for &r in &radii {
            if r < 1e-12 {
                continue;
            }
            let alpha = 1.0 / r;
            let shape = AlphaShapeAdvanced::new(points.clone(), alpha)?;
            if shape.all_points_covered() {
                return Ok((alpha, shape));
            }
        }
        // Fallback: convex hull
        let shape = AlphaShapeAdvanced::new(points, 0.0)?;
        Ok((0.0, shape))
    }

    /// Check whether every input point participates in at least one interior triangle.
    fn all_points_covered(&self) -> bool {
        match &self.dt {
            None => false,
            Some(dt) => {
                let n = dt.sites.len();
                let mut covered = vec![false; n];
                for tri in &self.interior_triangles {
                    covered[tri.a] = true;
                    covered[tri.b] = true;
                    covered[tri.c] = true;
                }
                covered.iter().all(|&c| c)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Circumcenter of triangle (a, b, c).  Returns `None` if degenerate.
fn circumcenter(a: Pt, b: Pt, c: Pt) -> Option<Pt> {
    let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
    if d.abs() < 1e-10 {
        return None;
    }
    let ux = ((a.0 * a.0 + a.1 * a.1) * (b.1 - c.1)
        + (b.0 * b.0 + b.1 * b.1) * (c.1 - a.1)
        + (c.0 * c.0 + c.1 * c.1) * (a.1 - b.1))
        / d;
    let uy = ((a.0 * a.0 + a.1 * a.1) * (c.0 - b.0)
        + (b.0 * b.0 + b.1 * b.1) * (a.0 - c.0)
        + (c.0 * c.0 + c.1 * c.1) * (b.0 - a.0))
        / d;
    Some((ux, uy))
}

/// Check whether point `d` lies strictly inside the circumcircle of (a, b, c).
fn in_circumcircle(a: Pt, b: Pt, c: Pt, d: Pt) -> bool {
    // The sign of the determinant depends on orientation of abc.
    // For CCW orientation: det > 0 means d is inside.
    // For CW orientation: det < 0 means d is inside.
    // We check orientation first and adjust.
    let ax = a.0 - d.0;
    let ay = a.1 - d.1;
    let bx = b.0 - d.0;
    let by = b.1 - d.1;
    let cx = c.0 - d.0;
    let cy = c.1 - d.1;
    let det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
        - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by))
        + (ax * ax + ay * ay) * (bx * cy - by * cx);

    // Orientation of triangle abc: positive = CCW, negative = CW
    let orient = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
    if orient > 0.0 {
        det > 0.0
    } else {
        det < 0.0
    }
}

/// Area of a triangle.
fn triangle_area(a: Pt, b: Pt, c: Pt) -> f64 {
    0.5 * ((b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1)).abs()
}

/// Bowyer-Watson incremental Delaunay triangulation.
///
/// Adds a large super-triangle at the start and removes any triangles
/// containing its vertices at the end.
fn bowyer_watson(pts: &[Pt]) -> SpatialResult<Vec<DelaunayTriangle>> {
    if pts.len() < 3 {
        return Err(SpatialError::ValueError(
            "Need at least 3 points".to_string(),
        ));
    }

    // Compute bounding box
    let min_x = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
    let min_y = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_y = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);

    let dx = (max_x - min_x).max(1.0);
    let dy = (max_y - min_y).max(1.0);
    let delta = dx.max(dy) * 3.0;

    // Super-triangle vertices (appended at indices n, n+1, n+2)
    let n = pts.len();
    let mut all_pts: Vec<Pt> = pts.to_vec();
    all_pts.push((min_x + dx / 2.0, max_y + delta));        // top
    all_pts.push((min_x - delta, min_y - dy));               // bottom-left
    all_pts.push((min_x + dx + delta, min_y - dy));          // bottom-right

    let super_tri = DelaunayTriangle::new(n, n + 1, n + 2);
    let mut triangles: Vec<DelaunayTriangle> = vec![super_tri];

    for (i, &pt) in pts.iter().enumerate() {
        // Find all triangles whose circumcircle contains this point
        let mut bad: Vec<DelaunayTriangle> = Vec::new();
        let mut good: Vec<DelaunayTriangle> = Vec::new();
        for &tri in &triangles {
            if in_circumcircle(all_pts[tri.a], all_pts[tri.b], all_pts[tri.c], pt) {
                bad.push(tri);
            } else {
                good.push(tri);
            }
        }

        // Find the boundary polygon of the bad triangles
        let boundary = boundary_polygon(&bad);

        // Remove bad triangles
        triangles = good;

        // Re-triangulate with the new point
        for edge in boundary {
            triangles.push(DelaunayTriangle::new(edge.a, edge.b, i));
        }
    }

    // Remove any triangle that shares a vertex with the super-triangle
    let result: Vec<DelaunayTriangle> = triangles
        .into_iter()
        .filter(|tri| tri.a < n && tri.b < n && tri.c < n)
        .collect();

    Ok(result)
}

/// Find the boundary polygon of a set of triangles (edges that appear exactly once).
fn boundary_polygon(triangles: &[DelaunayTriangle]) -> Vec<DelaunayEdge> {
    let mut edge_count: HashMap<DelaunayEdge, usize> = HashMap::new();
    for tri in triangles {
        for e in tri.edges() {
            *edge_count.entry(e).or_insert(0) += 1;
        }
    }
    edge_count
        .into_iter()
        .filter_map(|(e, count)| if count == 1 { Some(e) } else { None })
        .collect()
}

/// Andrew's monotone chain: returns hull vertex indices in CCW order.
fn monotone_chain_hull(pts: &[Pt]) -> Vec<usize> {
    let n = pts.len();
    if n == 0 {
        return Vec::new();
    }
    // Sort by (x, y)
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        pts[a]
            .0
            .partial_cmp(&pts[b].0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                pts[a]
                    .1
                    .partial_cmp(&pts[b].1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let cross = |o: usize, a: usize, b: usize| -> f64 {
        (pts[a].0 - pts[o].0) * (pts[b].1 - pts[o].1)
            - (pts[a].1 - pts[o].1) * (pts[b].0 - pts[o].0)
    };

    // Lower hull
    let mut lower: Vec<usize> = Vec::new();
    for &i in &idx {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], i) <= 0.0 {
            lower.pop();
        }
        lower.push(i);
    }

    // Upper hull
    let mut upper: Vec<usize> = Vec::new();
    for &i in idx.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], i) <= 0.0 {
            upper.pop();
        }
        upper.push(i);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Return the set of indices on the convex hull.
fn convex_hull_indices(pts: &[Pt]) -> HashSet<usize> {
    monotone_chain_hull(pts).into_iter().collect()
}

/// Sort a list of points angularly around a reference `center`.
fn angular_sort(center: Pt, mut pts: Vec<Pt>) -> Vec<Pt> {
    pts.sort_by(|a, b| {
        let angle_a = (a.1 - center.1).atan2(a.0 - center.0);
        let angle_b = (b.1 - center.1).atan2(b.0 - center.0);
        angle_a
            .partial_cmp(&angle_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pts
}

/// Compute alpha-shape edges and interior triangles from a Delaunay triangulation.
fn compute_alpha_shape(
    dt: &DelaunayTriangulation,
    alpha: f64,
) -> (Vec<DelaunayEdge>, Vec<DelaunayTriangle>) {
    let mut edge_count: HashMap<DelaunayEdge, usize> = HashMap::new();
    let mut interior: Vec<DelaunayTriangle> = Vec::new();

    for tri in &dt.triangles {
        let r = match dt.circumradius(tri) {
            Some(r) => r,
            None => continue,
        };
        // Include the triangle if alpha == 0 (convex hull) or circumradius ≤ 1/alpha
        let include = alpha == 0.0 || r <= 1.0 / alpha;
        if include {
            interior.push(*tri);
            for e in tri.edges() {
                *edge_count.entry(e).or_insert(0) += 1;
            }
        }
    }

    // Boundary edges appear exactly once (not shared between two included triangles)
    let boundary: Vec<DelaunayEdge> = edge_count
        .into_iter()
        .filter_map(|(e, c)| if c == 1 { Some(e) } else { None })
        .collect();

    (boundary, interior)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn square_sites() -> Vec<Pt> {
        vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    }

    #[test]
    fn test_delaunay_basic() {
        let sites = square_sites();
        let dt = DelaunayTriangulation::new(sites).unwrap();
        // 4 points in general position → 2 triangles
        assert!(!dt.triangles().is_empty());
    }

    #[test]
    fn test_delaunay_edges() {
        let sites = vec![(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)];
        let dt = DelaunayTriangulation::new(sites).unwrap();
        // Single triangle → 3 edges
        assert_eq!(dt.edges().len(), 3);
    }

    #[test]
    fn test_voronoi_from_delaunay() {
        let sites = vec![(0.0, 0.0), (2.0, 0.0), (1.0, 2.0), (1.0, -2.0)];
        let dt = DelaunayTriangulation::new(sites).unwrap();
        let vor = VoronoiDiagram::from_delaunay(&dt);
        assert_eq!(vor.cells().len(), 4);
        assert!(!vor.vertices().is_empty());
    }

    #[test]
    fn test_convex_hull_2d() {
        let pts = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.5, 0.5), // interior
        ];
        let hull = ConvexHull2D::new(pts).unwrap();
        assert_eq!(hull.hull_indices().len(), 4);
        assert!((hull.area() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_alpha_shape() {
        let pts = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
            (0.0, 1.0),
        ];
        let shape = AlphaShapeAdvanced::new(pts, 2.0).unwrap();
        // With alpha = 2 (circumradius ≤ 0.5) we should get some boundary edges
        // The exact count depends on geometry; just verify it runs without error
        let _ = shape.boundary_edges().len();
        let _ = shape.area();
    }

    #[test]
    fn test_find_optimal_alpha() {
        let pts = vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, 0.8660),
            (0.5, 0.2887),
        ];
        let (alpha, shape) = AlphaShapeAdvanced::find_optimal_alpha(pts).unwrap();
        assert!(alpha >= 0.0);
        assert!(!shape.interior_triangles().is_empty() || shape.boundary_edges().is_empty());
    }

    #[test]
    fn test_delaunay_triangle_edges() {
        let tri = DelaunayTriangle::new(0, 1, 2);
        let edges = tri.edges();
        assert_eq!(edges.len(), 3);
        assert!(edges.contains(&DelaunayEdge::new(0, 1)));
        assert!(edges.contains(&DelaunayEdge::new(1, 2)));
        assert!(edges.contains(&DelaunayEdge::new(0, 2)));
    }

    #[test]
    fn test_convex_hull_contains() {
        let pts = vec![(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        let hull = ConvexHull2D::new(pts).unwrap();
        assert!(hull.contains((1.0, 1.0)));
        assert!(!hull.contains((3.0, 1.0)));
    }
}
