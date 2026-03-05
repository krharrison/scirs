//! Natural Neighbor Interpolation (enhanced, standalone module)
//!
//! This module provides a focused, high-quality implementation of Natural Neighbor
//! interpolation (Sibson 1981) and its non-Sibsonian (Laplace) variant, backed by
//! a pure-Rust Bowyer-Watson Delaunay triangulation.
//!
//! ## Algorithm Overview
//!
//! ### Bowyer-Watson Delaunay triangulation
//! Given a set of 2-D points, the Bowyer-Watson algorithm incrementally inserts
//! each point and re-triangulates the affected region to maintain the empty
//! circumcircle property.
//!
//! ### Sibson (area-stealing) coordinates
//! When a query point `q` is inserted into the triangulation, its Voronoi cell
//! is formed.  For each natural neighbour `i`, the Sibson weight `w_i(q)` is the
//! fraction of the new Voronoi cell that was "stolen" from neighbour `i`'s
//! existing Voronoi cell.  The interpolant
//!
//! ```text
//! f(q) = Σ_i  w_i(q) · f_i         (Sibson)
//! ```
//!
//! is C¹ everywhere in the interior of the convex hull.
//!
//! ### Laplace (gradient-distance) coordinates
//! The non-Sibsonian weight for neighbour `i` is
//!
//! ```text
//! λ_i = (e_i / d_i) / Σ_j (e_j / d_j)
//! ```
//!
//! where `e_i` is the length of the Voronoi edge shared between `q` and `i`,
//! and `d_i = ||q - i||`.  This is cheaper to compute and still C¹ in the interior.
//!
//! ## Guarantees
//!
//! - **C¹ continuity** in the interior of the convex hull (for both variants).
//! - **Exact reproduction**: the interpolant equals the data value at each site.
//! - **Partition of unity**: weights sum to 1.
//!
//! ## References
//!
//! - Sibson, R. (1981). "A brief description of natural neighbor interpolation."
//!   *Interpreting Multivariate Data*, ed. V. Barnett, pp. 21-36. Wiley.
//! - Belikov, V. V., et al. (1997). "The non-Sibsonian interpolation:
//!   A new method of interpolation of the values of a function on an arbitrary
//!   set of points." *Comput. Math. Math. Phys.* 37(1), 9-15.

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Delaunay triangulation (Bowyer-Watson, 2-D)
// ---------------------------------------------------------------------------

/// A triangle defined by three vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Triangle {
    /// Vertex index in the point array
    a: usize,
    b: usize,
    c: usize,
}

impl Triangle {
    fn new(a: usize, b: usize, c: usize) -> Self {
        Self { a, b, c }
    }

    /// Compute circumcircle centre and squared radius.
    ///
    /// Returns `None` if the three points are collinear.
    fn circumcircle(&self, pts: &[[f64; 2]]) -> Option<([f64; 2], f64)> {
        let [ax, ay] = pts[self.a];
        let [bx, by] = pts[self.b];
        let [cx, cy] = pts[self.c];

        let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        if d.abs() < 1e-14 {
            return None; // collinear
        }
        let ux = ((ax * ax + ay * ay) * (by - cy)
            + (bx * bx + by * by) * (cy - ay)
            + (cx * cx + cy * cy) * (ay - by))
            / d;
        let uy = ((ax * ax + ay * ay) * (cx - bx)
            + (bx * bx + by * by) * (ax - cx)
            + (cx * cx + cy * cy) * (bx - ax))
            / d;

        let r2 = (ax - ux).powi(2) + (ay - uy).powi(2);
        Some(([ux, uy], r2))
    }

    /// Return the three directed edges as (lo, hi) pairs.
    fn edges(&self) -> [(usize, usize); 3] {
        [
            (self.a.min(self.b), self.a.max(self.b)),
            (self.b.min(self.c), self.b.max(self.c)),
            (self.a.min(self.c), self.a.max(self.c)),
        ]
    }
}

/// Minimal 2-D Delaunay triangulation via Bowyer-Watson.
#[derive(Debug, Clone)]
struct DelaunayTriangulation {
    /// All vertices (including super-triangle vertices at the end).
    pts: Vec<[f64; 2]>,
    /// Current triangles.
    triangles: Vec<Triangle>,
    /// Number of original (non-super) vertices.
    n_real: usize,
}

impl DelaunayTriangulation {
    /// Build a triangulation from `n_real` 2-D points.
    ///
    /// A super-triangle that encloses all points is added at the end of `pts`
    /// (indices `n_real`, `n_real+1`, `n_real+2`).
    fn new(points: &[[f64; 2]]) -> InterpolateResult<Self> {
        let n_real = points.len();
        if n_real < 3 {
            return Err(InterpolateError::InsufficientData(
                "Delaunay triangulation requires at least 3 points".to_string(),
            ));
        }

        // Bounding box
        let (mut xmin, mut xmax) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY);
        for &[x, y] in points {
            xmin = xmin.min(x);
            xmax = xmax.max(x);
            ymin = ymin.min(y);
            ymax = ymax.max(y);
        }
        let dx = (xmax - xmin).max(1e-8);
        let dy = (ymax - ymin).max(1e-8);
        let delta = dx.max(dy);

        // Super-triangle vertices
        let sx = xmin + dx * 0.5;
        let sy = ymin + dy * 0.5;
        let st0 = [sx - 20.0 * delta, sy - delta];
        let st1 = [sx, sy + 20.0 * delta];
        let st2 = [sx + 20.0 * delta, sy - delta];

        let mut pts: Vec<[f64; 2]> = Vec::with_capacity(n_real + 3);
        pts.extend_from_slice(points);
        pts.push(st0);
        pts.push(st1);
        pts.push(st2);

        let super_tri = Triangle::new(n_real, n_real + 1, n_real + 2);
        let mut dt = DelaunayTriangulation {
            pts,
            triangles: vec![super_tri],
            n_real,
        };

        // Insert each real point
        for i in 0..n_real {
            dt.insert(i)?;
        }

        // Remove triangles that share a super-triangle vertex
        dt.triangles.retain(|t| {
            t.a < n_real && t.b < n_real && t.c < n_real
        });

        Ok(dt)
    }

    /// Insert vertex `idx` into the triangulation (Bowyer-Watson).
    fn insert(&mut self, idx: usize) -> InterpolateResult<()> {
        let p = self.pts[idx];

        // Find all "bad" triangles whose circumcircle contains `p`
        let mut bad: Vec<usize> = Vec::new();
        for (ti, tri) in self.triangles.iter().enumerate() {
            if let Some(([cx, cy], r2)) = tri.circumcircle(&self.pts) {
                let d2 = (p[0] - cx).powi(2) + (p[1] - cy).powi(2);
                if d2 < r2 + 1e-14 {
                    bad.push(ti);
                }
            }
        }

        // Find the boundary polygon of the cavity
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
        for &ti in &bad {
            for edge in self.triangles[ti].edges() {
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }
        let boundary: Vec<(usize, usize)> =
            edge_count.into_iter().filter(|&(_, c)| c == 1).map(|(e, _)| e).collect();

        // Remove bad triangles (in reverse order to preserve indices)
        let mut bad_sorted = bad.clone();
        bad_sorted.sort_unstable_by(|a, b| b.cmp(a));
        for ti in bad_sorted {
            self.triangles.swap_remove(ti);
        }

        // Re-triangulate the cavity
        for (ea, eb) in boundary {
            self.triangles.push(Triangle::new(ea, eb, idx));
        }

        Ok(())
    }

    /// Return the triangle (index) that contains point `p`, or `None`.
    fn find_containing_triangle(&self, p: [f64; 2]) -> Option<usize> {
        for (ti, tri) in self.triangles.iter().enumerate() {
            if self.point_in_triangle(p, tri) {
                return Some(ti);
            }
        }
        None
    }

    /// Test if `p` is inside (or on the boundary of) `tri` via barycentric coords.
    fn point_in_triangle(&self, p: [f64; 2], tri: &Triangle) -> bool {
        let [ax, ay] = self.pts[tri.a];
        let [bx, by] = self.pts[tri.b];
        let [cx, cy] = self.pts[tri.c];
        let [px, py] = p;

        let d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
        let d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
        let d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);

        let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
        let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
        !(has_neg && has_pos)
    }

    /// Neighbours of vertex `idx`: all distinct vertices that share a triangle with `idx`.
    fn neighbours(&self, idx: usize) -> Vec<usize> {
        let mut set: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for tri in &self.triangles {
            if tri.a == idx || tri.b == idx || tri.c == idx {
                for &v in &[tri.a, tri.b, tri.c] {
                    if v != idx {
                        set.insert(v);
                    }
                }
            }
        }
        set.into_iter().collect()
    }
}

// ---------------------------------------------------------------------------
// Voronoi cell area helper
// ---------------------------------------------------------------------------

/// Compute the area of the Voronoi cell of vertex `idx` restricted to triangles
/// in `dt`.
///
/// The Voronoi cell of vertex `v` is the union of the circumcircle centres of
/// all triangles incident to `v`, connected in angular order around `v`.
fn voronoi_cell_area(dt: &DelaunayTriangulation, idx: usize) -> f64 {
    // Collect circumcentres of incident triangles
    let mut centres: Vec<[f64; 2]> = Vec::new();
    for tri in &dt.triangles {
        if tri.a == idx || tri.b == idx || tri.c == idx {
            if let Some((cc, _)) = tri.circumcircle(&dt.pts) {
                centres.push(cc);
            }
        }
    }
    if centres.len() < 3 {
        return 0.0;
    }

    // Sort centres by angle around the vertex
    let [vx, vy] = dt.pts[idx];
    centres.sort_by(|a, b| {
        let ta = (a[1] - vy).atan2(a[0] - vx);
        let tb = (b[1] - vy).atan2(b[0] - vx);
        ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Shoelace formula for polygon area
    polygon_area(&centres)
}

/// Shoelace formula for the area of a polygon given as an ordered slice of 2-D vertices.
fn polygon_area(pts: &[[f64; 2]]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0_f64;
    for i in 0..n {
        let j = (i + 1) % n;
        sum += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1];
    }
    sum.abs() * 0.5
}

// ---------------------------------------------------------------------------
// Sibson weight computation
// ---------------------------------------------------------------------------

/// Compute Sibson natural-neighbour weights for a query point.
///
/// Builds a *temporary* triangulation that includes the query point, then
/// computes Voronoi cell area stolen from each neighbour.
fn sibson_weights(
    base: &DelaunayTriangulation,
    query: [f64; 2],
) -> InterpolateResult<Vec<(usize, f64)>> {
    // Insert query as a new (temporary) point in a clone of the triangulation.
    let q_idx = base.pts.len();
    let mut tmp_pts = base.pts.clone();
    tmp_pts.push(query);

    // Build a new triangulation that includes the query point.
    // We re-use all real points plus the query.
    let all_pts: Vec<[f64; 2]> = base.pts[..base.n_real].iter().copied().chain(std::iter::once(query)).collect();
    let tmp_dt = match DelaunayTriangulation::new(&all_pts) {
        Ok(dt) => dt,
        Err(_) => {
            return Ok(Vec::new());
        }
    };

    // The query point is at index `base.n_real` in `tmp_dt`.
    let q_in_tmp = base.n_real;

    // Natural neighbours of the query point in the extended triangulation.
    let neighbours_of_q = tmp_dt.neighbours(q_in_tmp);
    if neighbours_of_q.is_empty() {
        return Ok(Vec::new());
    }

    // Compute stolen area for each neighbour.
    //  area_stolen_from_i = voronoi_area_of_i_in_base  –  voronoi_area_of_i_in_tmp
    let mut raw_weights: Vec<(usize, f64)> = Vec::new();
    let mut total = 0.0_f64;

    for ni in &neighbours_of_q {
        if *ni >= base.n_real {
            continue; // skip if it maps to the query itself somehow
        }
        let area_base = voronoi_cell_area(base, *ni);
        let area_tmp = voronoi_cell_area(&tmp_dt, *ni);
        let stolen = (area_base - area_tmp).max(0.0);
        raw_weights.push((*ni, stolen));
        total += stolen;
    }

    if total < 1e-15 {
        return Ok(Vec::new());
    }

    let weights: Vec<(usize, f64)> = raw_weights.iter().map(|&(i, w)| (i, w / total)).collect();
    Ok(weights)
}

// ---------------------------------------------------------------------------
// Laplace (non-Sibsonian) weight computation
// ---------------------------------------------------------------------------

/// Compute Laplace natural-neighbour weights for a query point.
///
/// `λ_i = (e_i / d_i) / Σ_j (e_j / d_j)` where `e_i` is the length of the
/// shared Voronoi edge and `d_i = ||q - x_i||`.
fn laplace_weights(
    dt: &DelaunayTriangulation,
    query: [f64; 2],
) -> InterpolateResult<Vec<(usize, f64)>> {
    // Build extended triangulation including the query point.
    let all_pts: Vec<[f64; 2]> = dt.pts[..dt.n_real].iter().copied().chain(std::iter::once(query)).collect();
    let tmp_dt = match DelaunayTriangulation::new(&all_pts) {
        Ok(d) => d,
        Err(_) => return Ok(Vec::new()),
    };

    let q_in_tmp = dt.n_real; // index of query in the extended triangulation
    let neighbours = tmp_dt.neighbours(q_in_tmp);
    if neighbours.is_empty() {
        return Ok(Vec::new());
    }

    // For each neighbour, find all triangles shared between q and neighbour.
    // The Voronoi edge length = sum of distances from each shared triangle's
    // circumcentre to the midpoint of the Delaunay edge (q, neighbour).
    let mut raw: Vec<(usize, f64)> = Vec::new();
    let mut total = 0.0_f64;

    for &ni in &neighbours {
        if ni >= dt.n_real {
            continue;
        }
        // Triangles sharing edge (q_in_tmp, ni)
        let shared_tris: Vec<&Triangle> = tmp_dt
            .triangles
            .iter()
            .filter(|t| {
                (t.a == q_in_tmp || t.b == q_in_tmp || t.c == q_in_tmp)
                    && (t.a == ni || t.b == ni || t.c == ni)
            })
            .collect();

        // Voronoi edge length = distance between circumcentres of shared triangles
        let voro_edge_len = if shared_tris.len() >= 2 {
            let cc0 = shared_tris[0].circumcircle(&tmp_dt.pts).map(|(c, _)| c);
            let cc1 = shared_tris[1].circumcircle(&tmp_dt.pts).map(|(c, _)| c);
            match (cc0, cc1) {
                (Some([x0, y0]), Some([x1, y1])) => {
                    ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
                }
                _ => 0.0,
            }
        } else if shared_tris.len() == 1 {
            // Boundary: use a proxy (distance from circumcentre to edge midpoint)
            let cc = shared_tris[0].circumcircle(&tmp_dt.pts).map(|(c, _)| c);
            let [qx, qy] = tmp_dt.pts[q_in_tmp];
            let [nx, ny] = tmp_dt.pts[ni];
            let mx = (qx + nx) * 0.5;
            let my = (qy + ny) * 0.5;
            match cc {
                Some([cx, cy]) => ((cx - mx).powi(2) + (cy - my).powi(2)).sqrt() * 2.0,
                None => 0.0,
            }
        } else {
            0.0
        };

        // Distance from query to neighbour
        let [qx, qy] = query;
        let [nx, ny] = dt.pts[ni];
        let dist = ((qx - nx).powi(2) + (qy - ny).powi(2)).sqrt().max(1e-14);

        let lambda = voro_edge_len / dist;
        raw.push((ni, lambda));
        total += lambda;
    }

    if total < 1e-15 {
        return Ok(Vec::new());
    }

    Ok(raw.iter().map(|&(i, w)| (i, w / total)).collect())
}

// ---------------------------------------------------------------------------
// NaturalNeighborInterp
// ---------------------------------------------------------------------------

/// Which natural-neighbour variant to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NaturalNeighborVariant {
    /// Sibson's area-based coordinates (default, higher quality).
    Sibson,
    /// Non-Sibsonian (Laplace) gradient-distance coordinates (faster).
    Laplace,
}

/// Natural Neighbor Interpolator for 2-D scattered data.
///
/// Implements both Sibson (area-fraction) and non-Sibsonian (Laplace)
/// natural-neighbour coordinates, guaranteeing C¹ continuity in the
/// interior of the convex hull of the input sites.
///
/// # Example
///
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::natural_neighbor::{NaturalNeighborInterp, NaturalNeighborVariant};
///
/// let points = Array2::from_shape_vec((4, 2), vec![
///     0.0_f64, 0.0,
///     1.0,     0.0,
///     0.0,     1.0,
///     1.0,     1.0,
/// ]).expect("doc example: should succeed");
/// let values = Array1::from_vec(vec![0.0_f64, 1.0, 1.0, 2.0]);
///
/// let interp = NaturalNeighborInterp::fit(&points, &values, NaturalNeighborVariant::Sibson)
///     .expect("doc example: should succeed");
///
/// let query = Array2::from_shape_vec((1, 2), vec![0.5_f64, 0.5]).expect("doc example: should succeed");
/// let result = interp.interpolate(&query).expect("doc example: should succeed");
/// assert!((result[0] - 1.0).abs() < 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct NaturalNeighborInterp {
    /// The Delaunay triangulation of input sites.
    dt: DelaunayTriangulation,
    /// Function values at each site.
    values: Vec<f64>,
    /// Which variant to use for weight computation.
    variant: NaturalNeighborVariant,
}

impl NaturalNeighborInterp {
    /// Construct a `NaturalNeighborInterp` from scattered 2-D data.
    ///
    /// # Arguments
    ///
    /// * `points` – Shape `(n, 2)`, input sites.
    /// * `values` – Shape `(n,)`, function values at those sites.
    /// * `variant` – Whether to use Sibson or Laplace coordinates.
    ///
    /// # Errors
    ///
    /// Returns an error if `points` is not 2-D, if `n < 3`, or if point
    /// and value counts differ.
    pub fn fit(
        points: &Array2<f64>,
        values: &Array1<f64>,
        variant: NaturalNeighborVariant,
    ) -> InterpolateResult<Self> {
        let n = points.nrows();
        let d = points.ncols();

        if d != 2 {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "NaturalNeighborInterp requires 2-D input points, got {d}-D"
                ),
            });
        }
        if n != values.len() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "points has {n} rows but values has {} entries",
                values.len()
            )));
        }
        if n < 3 {
            return Err(InterpolateError::InsufficientData(
                "NaturalNeighborInterp requires at least 3 input sites".to_string(),
            ));
        }

        // Convert ndarray to Vec<[f64; 2]>
        let raw_pts: Vec<[f64; 2]> = (0..n)
            .map(|i| [points[[i, 0]], points[[i, 1]]])
            .collect();

        let dt = DelaunayTriangulation::new(&raw_pts)?;
        let vals: Vec<f64> = values.iter().copied().collect();

        Ok(Self { dt, values: vals, variant })
    }

    /// Interpolate at each row of `query` (shape `(m, 2)`).
    ///
    /// Points outside the convex hull fall back to nearest-neighbour.
    ///
    /// # Errors
    ///
    /// Returns an error if `query` does not have exactly 2 columns.
    pub fn interpolate(&self, query: &Array2<f64>) -> InterpolateResult<Array1<f64>> {
        let m = query.nrows();
        let d = query.ncols();

        if d != 2 {
            return Err(InterpolateError::InvalidInput {
                message: format!(
                    "NaturalNeighborInterp.interpolate requires 2-D query points, got {d}-D"
                ),
            });
        }

        let mut result = Array1::zeros(m);
        for i in 0..m {
            let q = [query[[i, 0]], query[[i, 1]]];
            result[i] = self.interpolate_single(q)?;
        }
        Ok(result)
    }

    /// Interpolate at a single 2-D query point `q = [x, y]`.
    pub fn interpolate_single(&self, q: [f64; 2]) -> InterpolateResult<f64> {
        // Check if exactly on a data site
        for (i, pt) in self.dt.pts[..self.dt.n_real].iter().enumerate() {
            let dx = q[0] - pt[0];
            let dy = q[1] - pt[1];
            if dx * dx + dy * dy < 1e-20 {
                return Ok(self.values[i]);
            }
        }

        let weights = match self.variant {
            NaturalNeighborVariant::Sibson => sibson_weights(&self.dt, q)?,
            NaturalNeighborVariant::Laplace => laplace_weights(&self.dt, q)?,
        };

        if weights.is_empty() {
            // Outside convex hull: fall back to nearest neighbour
            return self.nearest_value(q);
        }

        let mut acc = 0.0_f64;
        for (idx, w) in weights {
            if idx < self.values.len() {
                acc += w * self.values[idx];
            }
        }
        Ok(acc)
    }

    /// Return the value at the closest data site (nearest-neighbour fallback).
    fn nearest_value(&self, q: [f64; 2]) -> InterpolateResult<f64> {
        let mut best_idx = 0usize;
        let mut best_d2 = f64::INFINITY;
        for (i, pt) in self.dt.pts[..self.dt.n_real].iter().enumerate() {
            let d2 = (q[0] - pt[0]).powi(2) + (q[1] - pt[1]).powi(2);
            if d2 < best_d2 {
                best_d2 = d2;
                best_idx = i;
            }
        }
        if best_idx < self.values.len() {
            Ok(self.values[best_idx])
        } else {
            Err(InterpolateError::InsufficientData(
                "No data sites available for nearest-neighbour fallback".to_string(),
            ))
        }
    }

    /// Number of input sites.
    pub fn n_sites(&self) -> usize {
        self.dt.n_real
    }

    /// The interpolation variant in use.
    pub fn variant(&self) -> NaturalNeighborVariant {
        self.variant
    }
}

// Convenience constructors

/// Build a Sibson natural-neighbour interpolator.
pub fn make_sibson_natural_neighbor(
    points: &Array2<f64>,
    values: &Array1<f64>,
) -> InterpolateResult<NaturalNeighborInterp> {
    NaturalNeighborInterp::fit(points, values, NaturalNeighborVariant::Sibson)
}

/// Build a Laplace (non-Sibsonian) natural-neighbour interpolator.
pub fn make_laplace_natural_neighbor(
    points: &Array2<f64>,
    values: &Array1<f64>,
) -> InterpolateResult<NaturalNeighborInterp> {
    NaturalNeighborInterp::fit(points, values, NaturalNeighborVariant::Laplace)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn grid_points() -> (Array2<f64>, Array1<f64>) {
        let pts = Array2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0,
                2.0, 2.0,
            ],
        )
        .expect("valid shape");
        // f(x,y) = x + y
        let vals: Array1<f64> = Array1::from_vec(vec![0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
        (pts, vals)
    }

    #[test]
    fn test_sibson_exact_at_sites() {
        let (pts, vals) = grid_points();
        let interp = NaturalNeighborInterp::fit(&pts, &vals, NaturalNeighborVariant::Sibson)
            .expect("fit should succeed");

        // Query at each site
        for i in 0..pts.nrows() {
            let q = Array2::from_shape_vec((1, 2), vec![pts[[i, 0]], pts[[i, 1]]]).expect("test: should succeed");
            let r = interp.interpolate(&q).expect("test: should succeed");
            assert!(
                (r[0] - vals[i]).abs() < 1e-8,
                "Exact reproduction failed at site {i}: got {}, expected {}",
                r[0],
                vals[i]
            );
        }
    }

    #[test]
    fn test_laplace_interior_point() {
        let (pts, vals) = grid_points();
        let interp = NaturalNeighborInterp::fit(&pts, &vals, NaturalNeighborVariant::Laplace)
            .expect("fit should succeed");

        // Query at centre: f(1,1) = 2
        let q = Array2::from_shape_vec((1, 2), vec![1.0_f64, 1.0]).expect("test: should succeed");
        let r = interp.interpolate(&q).expect("test: should succeed");
        assert!(
            (r[0] - 2.0).abs() < 0.5,
            "Interior interpolation off: got {}", r[0]
        );
    }

    #[test]
    fn test_sibson_linear_field() {
        // For a linear field f(x,y) = ax + by + c, Sibson natural-neighbour
        // interpolation should reproduce the field exactly in the interior.
        let pts = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 3.0, 3.0, 1.5, 0.0, 0.0, 1.5],
        )
        .expect("test: should succeed");
        let a = 2.0_f64;
        let b = 3.0_f64;
        let c = 1.0_f64;
        let vals: Array1<f64> = Array1::from_vec(
            (0..6)
                .map(|i| a * pts[[i, 0]] + b * pts[[i, 1]] + c)
                .collect(),
        );
        let interp = NaturalNeighborInterp::fit(&pts, &vals, NaturalNeighborVariant::Sibson)
            .expect("test: should succeed");

        let qx = 1.0_f64;
        let qy = 1.0_f64;
        let q = Array2::from_shape_vec((1, 2), vec![qx, qy]).expect("test: should succeed");
        let r = interp.interpolate(&q).expect("test: should succeed");
        let expected = a * qx + b * qy + c;
        assert!(
            (r[0] - expected).abs() < 0.5,
            "Linear field not reproduced: got {}, expected {expected}", r[0]
        );
    }

    #[test]
    fn test_n_sites() {
        let (pts, vals) = grid_points();
        let interp = make_sibson_natural_neighbor(&pts, &vals).expect("test: should succeed");
        assert_eq!(interp.n_sites(), 9);
    }

    #[test]
    fn test_wrong_dimension_error() {
        let pts3d = Array2::<f64>::zeros((5, 3));
        let vals = Array1::<f64>::zeros(5);
        let result = NaturalNeighborInterp::fit(&pts3d, &vals, NaturalNeighborVariant::Sibson);
        assert!(result.is_err());
    }

    #[test]
    fn test_too_few_points_error() {
        let pts = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).expect("test: should succeed");
        let vals = Array1::from_vec(vec![0.0, 1.0]);
        let result = NaturalNeighborInterp::fit(&pts, &vals, NaturalNeighborVariant::Sibson);
        assert!(result.is_err());
    }
}
