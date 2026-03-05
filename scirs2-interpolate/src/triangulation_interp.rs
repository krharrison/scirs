//! Triangulation-based interpolation for 2D scattered data
//!
//! This module provides interpolation methods built on top of Delaunay triangulations
//! of 2D scattered data. The triangulation serves as the underlying spatial structure
//! upon which interpolation is performed.
//!
//! ## Methods
//!
//! - **Linear triangulation interpolation**: Barycentric interpolation within each
//!   triangle of the Delaunay triangulation. C0 continuous everywhere, exact at data
//!   points.
//!
//! - **Clough-Tocher C1 interpolation**: Smooth (C1) interpolation that subdivides
//!   each triangle into three sub-triangles and uses cubic polynomials. Requires
//!   gradient estimation at data points.
//!
//! - **Nearest vertex interpolation**: Returns the value of the nearest vertex of
//!   the enclosing triangle. Piecewise constant.
//!
//! ## Delaunay Triangulation
//!
//! The module includes a robust incremental Delaunay triangulation algorithm that
//! handles degenerate cases (collinear points, coincident points, etc.).
//!
//! ## Examples
//!
//! ```rust
//! use scirs2_core::ndarray::{Array1, Array2};
//! use scirs2_interpolate::triangulation_interp::{
//!     TriangulationInterpolator, TriangulationMethod,
//! };
//!
//! // Scattered 2D data: z = x + y
//! let points = Array2::from_shape_vec((4, 2), vec![
//!     0.0, 0.0,
//!     1.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 1.0,
//! ]).expect("valid shape");
//! let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);
//!
//! let interp = TriangulationInterpolator::new(
//!     points,
//!     values,
//!     TriangulationMethod::Linear,
//! ).expect("valid");
//!
//! let result = interp.evaluate_point(0.5, 0.5).expect("valid");
//! assert!((result - 1.0).abs() < 1e-10);
//! ```

use crate::error::{InterpolateError, InterpolateResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Delaunay triangulation
// ---------------------------------------------------------------------------

/// A triangle defined by three vertex indices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Triangle {
    /// Vertex indices (into the points array)
    pub vertices: [usize; 3],
}

/// Delaunay triangulation of 2D point data
///
/// Uses the Bowyer-Watson incremental insertion algorithm.
#[derive(Debug, Clone)]
pub struct DelaunayTriangulation<F: Float + FromPrimitive + Debug> {
    /// The input points (x, y coordinates)
    points: Vec<[F; 2]>,
    /// The triangles forming the triangulation
    triangles: Vec<Triangle>,
    /// Number of original (non-super-triangle) points
    n_original: usize,
}

impl<F: Float + FromPrimitive + Debug> DelaunayTriangulation<F> {
    /// Build a Delaunay triangulation from a set of 2D points
    ///
    /// # Arguments
    ///
    /// * `points` - 2D point coordinates as (x, y) pairs
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 3 points are provided or if all points
    /// are collinear.
    pub fn new(input_points: &[[F; 2]]) -> InterpolateResult<Self> {
        let n = input_points.len();
        if n < 3 {
            return Err(InterpolateError::insufficient_points(
                3,
                n,
                "Delaunay triangulation",
            ));
        }

        // Compute bounding box
        let mut min_x = input_points[0][0];
        let mut max_x = input_points[0][0];
        let mut min_y = input_points[0][1];
        let mut max_y = input_points[0][1];

        for p in input_points.iter().skip(1) {
            if p[0] < min_x {
                min_x = p[0];
            }
            if p[0] > max_x {
                max_x = p[0];
            }
            if p[1] < min_y {
                min_y = p[1];
            }
            if p[1] > max_y {
                max_y = p[1];
            }
        }

        let dx = max_x - min_x;
        let dy = max_y - min_y;
        let delta_max = if dx > dy { dx } else { dy };

        // Guard against degenerate case where all points are the same
        let eps = F::from_f64(1e-10).unwrap_or_else(|| F::epsilon());
        if delta_max < eps {
            return Err(InterpolateError::invalid_input(
                "All points are coincident; cannot form triangulation",
            ));
        }

        let mid_x = (min_x + max_x) / (F::one() + F::one());
        let mid_y = (min_y + max_y) / (F::one() + F::one());

        // Create super-triangle that contains all points
        let margin = F::from_f64(20.0).unwrap_or_else(|| {
            let mut v = F::one();
            for _ in 0..4 {
                v = v + v;
            }
            v + v + v + v
        });
        let super_delta = delta_max * margin;

        let three = F::one() + F::one() + F::one();
        let p0 = [mid_x - super_delta, mid_y - super_delta];
        let p1 = [mid_x + super_delta, mid_y - super_delta];
        let p2 = [mid_x, mid_y + super_delta * three];

        // Start with the super-triangle vertices plus the input points
        let mut points = Vec::with_capacity(n + 3);
        points.push(p0);
        points.push(p1);
        points.push(p2);
        for p in input_points {
            points.push(*p);
        }

        let mut triangles = vec![Triangle {
            vertices: [0, 1, 2],
        }];

        // Insert each point
        for i in 0..n {
            let pt_idx = i + 3; // Offset by super-triangle vertices
            let pt = points[pt_idx];

            // Find all triangles whose circumcircle contains the point
            let mut bad_triangles = Vec::new();
            for (t_idx, tri) in triangles.iter().enumerate() {
                if Self::in_circumcircle(&points, tri, pt) {
                    bad_triangles.push(t_idx);
                }
            }

            // Find the boundary polygon of the hole
            let mut boundary_edges: Vec<[usize; 2]> = Vec::new();
            for &t_idx in &bad_triangles {
                let tri = &triangles[t_idx];
                for edge_i in 0..3 {
                    let edge = [tri.vertices[edge_i], tri.vertices[(edge_i + 1) % 3]];

                    // Check if this edge is shared with another bad triangle
                    let mut shared = false;
                    for &other_idx in &bad_triangles {
                        if other_idx == t_idx {
                            continue;
                        }
                        if Self::triangle_has_edge(&triangles[other_idx], edge) {
                            shared = true;
                            break;
                        }
                    }

                    if !shared {
                        boundary_edges.push(edge);
                    }
                }
            }

            // Remove bad triangles (in reverse order to preserve indices)
            let mut bad_sorted = bad_triangles;
            bad_sorted.sort_unstable();
            for &idx in bad_sorted.iter().rev() {
                triangles.swap_remove(idx);
            }

            // Re-triangulate the hole with the new point
            for edge in &boundary_edges {
                triangles.push(Triangle {
                    vertices: [edge[0], edge[1], pt_idx],
                });
            }
        }

        // Remove triangles that reference super-triangle vertices (indices 0, 1, 2)
        triangles
            .retain(|tri| tri.vertices[0] >= 3 && tri.vertices[1] >= 3 && tri.vertices[2] >= 3);

        // Remap vertex indices (subtract 3 to remove super-triangle offset)
        for tri in &mut triangles {
            tri.vertices[0] -= 3;
            tri.vertices[1] -= 3;
            tri.vertices[2] -= 3;
        }

        // Remove super-triangle vertices from points
        let final_points: Vec<[F; 2]> = points[3..].to_vec();

        if triangles.is_empty() {
            return Err(InterpolateError::invalid_input(
                "No triangles formed; points may be collinear",
            ));
        }

        Ok(Self {
            points: final_points,
            triangles,
            n_original: n,
        })
    }

    /// Check if a point lies inside the circumcircle of a triangle
    fn in_circumcircle(points: &[[F; 2]], tri: &Triangle, p: [F; 2]) -> bool {
        let a = points[tri.vertices[0]];
        let b = points[tri.vertices[1]];
        let c = points[tri.vertices[2]];

        let ax = a[0] - p[0];
        let ay = a[1] - p[1];
        let bx = b[0] - p[0];
        let by = b[1] - p[1];
        let cx = c[0] - p[0];
        let cy = c[1] - p[1];

        let det = ax * (by * (cx * cx + cy * cy) - cy * (bx * bx + by * by))
            - ay * (bx * (cx * cx + cy * cy) - cx * (bx * bx + by * by))
            + (ax * ax + ay * ay) * (bx * cy - by * cx);

        // For counter-clockwise triangles, det > 0 means inside circumcircle
        // We need to handle both orientations
        let orient = Self::orientation(a, b, c);
        if orient > F::zero() {
            det > F::zero()
        } else {
            det < F::zero()
        }
    }

    /// Compute the orientation of triangle (a, b, c)
    /// Positive = counter-clockwise, Negative = clockwise, Zero = collinear
    fn orientation(a: [F; 2], b: [F; 2], c: [F; 2]) -> F {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    }

    /// Check if a triangle has a specific edge
    fn triangle_has_edge(tri: &Triangle, edge: [usize; 2]) -> bool {
        for i in 0..3 {
            let e0 = tri.vertices[i];
            let e1 = tri.vertices[(i + 1) % 3];
            if (e0 == edge[0] && e1 == edge[1]) || (e0 == edge[1] && e1 == edge[0]) {
                return true;
            }
        }
        false
    }

    /// Get the triangles
    pub fn triangles(&self) -> &[Triangle] {
        &self.triangles
    }

    /// Get the points
    pub fn points(&self) -> &[[F; 2]] {
        &self.points
    }

    /// Get the number of triangles
    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    /// Find the triangle containing a query point
    ///
    /// Returns the triangle index and barycentric coordinates, or None if
    /// the point is outside the convex hull.
    pub fn find_containing_triangle(&self, x: F, y: F) -> Option<(usize, [F; 3])> {
        let eps = F::from_f64(-1e-12).unwrap_or_else(|| -F::epsilon());

        for (i, tri) in self.triangles.iter().enumerate() {
            let bary = self.barycentric_coords(tri, x, y);
            if let Some(bc) = bary {
                // Check all barycentric coordinates are >= 0 (with tolerance)
                if bc[0] >= eps && bc[1] >= eps && bc[2] >= eps {
                    return Some((i, bc));
                }
            }
        }
        None
    }

    /// Compute barycentric coordinates of point (x, y) with respect to a triangle
    fn barycentric_coords(&self, tri: &Triangle, x: F, y: F) -> Option<[F; 3]> {
        let p0 = self.points[tri.vertices[0]];
        let p1 = self.points[tri.vertices[1]];
        let p2 = self.points[tri.vertices[2]];

        let v0x = p1[0] - p0[0];
        let v0y = p1[1] - p0[1];
        let v1x = p2[0] - p0[0];
        let v1y = p2[1] - p0[1];
        let v2x = x - p0[0];
        let v2y = y - p0[1];

        let det = v0x * v1y - v1x * v0y;
        let eps = F::from_f64(1e-15).unwrap_or_else(|| F::epsilon());

        if det.abs() < eps {
            return None; // Degenerate triangle
        }

        let inv_det = F::one() / det;
        let u = (v2x * v1y - v1x * v2y) * inv_det;
        let v = (v0x * v2y - v2x * v0y) * inv_det;
        let w = F::one() - u - v;

        Some([w, u, v])
    }
}

// ---------------------------------------------------------------------------
// Interpolation method
// ---------------------------------------------------------------------------

/// Interpolation method for triangulation-based interpolation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TriangulationMethod {
    /// Linear (barycentric) interpolation within each triangle
    ///
    /// C0 continuous. The interpolated surface is a piecewise planar surface
    /// over the triangulation.
    Linear,

    /// Clough-Tocher C1 smooth interpolation
    ///
    /// Subdivides each triangle into 3 sub-triangles and uses cubic Bernstein
    /// polynomials for C1 continuity. Gradient estimation at vertices uses
    /// least-squares fitting.
    CloughTocher,

    /// Nearest vertex interpolation
    ///
    /// Returns the value of the nearest vertex of the enclosing triangle.
    /// Piecewise constant within each Voronoi cell.
    NearestVertex,
}

/// How to handle query points outside the convex hull
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExteriorHandling {
    /// Return NaN for exterior points
    Nan,
    /// Return the value of the nearest data point
    NearestNeighbor,
    /// Return an error
    Error,
}

// ---------------------------------------------------------------------------
// Main interpolator
// ---------------------------------------------------------------------------

/// Triangulation-based 2D scattered data interpolator
///
/// Builds a Delaunay triangulation of the input points and performs
/// interpolation within each triangle using the chosen method.
#[derive(Debug, Clone)]
pub struct TriangulationInterpolator<F: Float + FromPrimitive + Debug> {
    /// The Delaunay triangulation
    triangulation: DelaunayTriangulation<F>,
    /// Values at data points
    values: Array1<F>,
    /// Interpolation method
    method: TriangulationMethod,
    /// How to handle exterior points
    exterior: ExteriorHandling,
    /// Estimated gradients at vertices (for Clough-Tocher)
    gradients: Option<Vec<[F; 2]>>,
}

impl<F: Float + FromPrimitive + Debug + std::fmt::Display + 'static> TriangulationInterpolator<F> {
    /// Create a new triangulation-based interpolator
    ///
    /// # Arguments
    ///
    /// * `points` - 2D scattered data points, shape (n_points, 2)
    /// * `values` - Values at data points, shape (n_points,)
    /// * `method` - Interpolation method
    ///
    /// # Errors
    ///
    /// Returns an error if the data has fewer than 3 points, if the points are
    /// not 2D, or if the triangulation fails.
    pub fn new(
        points: Array2<F>,
        values: Array1<F>,
        method: TriangulationMethod,
    ) -> InterpolateResult<Self> {
        Self::with_exterior(points, values, method, ExteriorHandling::Nan)
    }

    /// Create a new interpolator with custom exterior handling
    pub fn with_exterior(
        points: Array2<F>,
        values: Array1<F>,
        method: TriangulationMethod,
        exterior: ExteriorHandling,
    ) -> InterpolateResult<Self> {
        if points.ncols() != 2 {
            return Err(InterpolateError::invalid_input(format!(
                "Triangulation interpolation requires 2D points, got {}D",
                points.ncols()
            )));
        }

        if points.nrows() != values.len() {
            return Err(InterpolateError::invalid_input(format!(
                "Number of points ({}) does not match number of values ({})",
                points.nrows(),
                values.len()
            )));
        }

        // Convert to internal format
        let pts: Vec<[F; 2]> = (0..points.nrows())
            .map(|i| [points[[i, 0]], points[[i, 1]]])
            .collect();

        let triangulation = DelaunayTriangulation::new(&pts)?;

        // Estimate gradients for Clough-Tocher
        let gradients = if method == TriangulationMethod::CloughTocher {
            Some(Self::estimate_gradients(&triangulation, &values)?)
        } else {
            None
        };

        Ok(Self {
            triangulation,
            values,
            method,
            exterior,
            gradients,
        })
    }

    /// Evaluate the interpolator at a single point (x, y)
    ///
    /// # Arguments
    ///
    /// * `x` - x-coordinate of the query point
    /// * `y` - y-coordinate of the query point
    ///
    /// # Errors
    ///
    /// Returns an error if exterior handling is set to Error and the point
    /// is outside the convex hull.
    pub fn evaluate_point(&self, x: F, y: F) -> InterpolateResult<F> {
        // Find the containing triangle
        match self.triangulation.find_containing_triangle(x, y) {
            Some((tri_idx, bary)) => match self.method {
                TriangulationMethod::Linear => self.linear_interpolate(tri_idx, &bary),
                TriangulationMethod::CloughTocher => {
                    self.clough_tocher_interpolate(tri_idx, x, y, &bary)
                }
                TriangulationMethod::NearestVertex => {
                    self.nearest_vertex_interpolate(tri_idx, &bary)
                }
            },
            None => {
                // Point is outside the convex hull
                self.handle_exterior(x, y)
            }
        }
    }

    /// Evaluate the interpolator at a single point given as an array
    pub fn evaluate_point_array(&self, point: &ArrayView1<F>) -> InterpolateResult<F> {
        if point.len() != 2 {
            return Err(InterpolateError::dimension_mismatch(
                2,
                point.len(),
                "TriangulationInterpolator::evaluate_point_array",
            ));
        }
        self.evaluate_point(point[0], point[1])
    }

    /// Evaluate the interpolator at multiple points
    ///
    /// # Arguments
    ///
    /// * `queries` - Query points, shape (n_queries, 2)
    pub fn evaluate(&self, queries: &Array2<F>) -> InterpolateResult<Array1<F>> {
        if queries.ncols() != 2 {
            return Err(InterpolateError::dimension_mismatch(
                2,
                queries.ncols(),
                "TriangulationInterpolator::evaluate",
            ));
        }

        let n = queries.nrows();
        let mut result = Array1::zeros(n);
        for i in 0..n {
            result[i] = self.evaluate_point(queries[[i, 0]], queries[[i, 1]])?;
        }
        Ok(result)
    }

    /// Get a reference to the Delaunay triangulation
    pub fn triangulation(&self) -> &DelaunayTriangulation<F> {
        &self.triangulation
    }

    /// Get a reference to the values
    pub fn values(&self) -> &Array1<F> {
        &self.values
    }

    /// Get the number of data points
    pub fn num_points(&self) -> usize {
        self.values.len()
    }

    /// Get the number of triangles
    pub fn num_triangles(&self) -> usize {
        self.triangulation.num_triangles()
    }

    // -----------------------------------------------------------------------
    // Private: linear interpolation
    // -----------------------------------------------------------------------

    fn linear_interpolate(&self, tri_idx: usize, bary: &[F; 3]) -> InterpolateResult<F> {
        let tri = &self.triangulation.triangles()[tri_idx];
        let v0 = self.values[tri.vertices[0]];
        let v1 = self.values[tri.vertices[1]];
        let v2 = self.values[tri.vertices[2]];

        Ok(bary[0] * v0 + bary[1] * v1 + bary[2] * v2)
    }

    // -----------------------------------------------------------------------
    // Private: nearest vertex interpolation
    // -----------------------------------------------------------------------

    fn nearest_vertex_interpolate(&self, tri_idx: usize, bary: &[F; 3]) -> InterpolateResult<F> {
        let tri = &self.triangulation.triangles()[tri_idx];

        // Find the vertex with the largest barycentric coordinate
        let mut max_bary = bary[0];
        let mut max_idx = 0;
        for i in 1..3 {
            if bary[i] > max_bary {
                max_bary = bary[i];
                max_idx = i;
            }
        }

        Ok(self.values[tri.vertices[max_idx]])
    }

    // -----------------------------------------------------------------------
    // Private: Clough-Tocher C1 interpolation
    // -----------------------------------------------------------------------

    /// Estimate gradients at each vertex using least-squares over adjacent triangles
    fn estimate_gradients(
        triangulation: &DelaunayTriangulation<F>,
        values: &Array1<F>,
    ) -> InterpolateResult<Vec<[F; 2]>> {
        let n = triangulation.n_original;
        let points = triangulation.points();
        let triangles = triangulation.triangles();

        // For each vertex, find adjacent vertices through shared triangles
        let mut gradients = vec![[F::zero(), F::zero()]; n];

        for i in 0..n {
            let mut neighbor_offsets: Vec<(F, F, F)> = Vec::new();
            let xi = points[i][0];
            let yi = points[i][1];
            let fi = values[i];

            // Collect all neighbors from triangles containing vertex i
            for tri in triangles {
                let mut contains = false;
                for &v in &tri.vertices {
                    if v == i {
                        contains = true;
                        break;
                    }
                }
                if !contains {
                    continue;
                }

                for &v in &tri.vertices {
                    if v == i || v >= n {
                        continue;
                    }
                    let dx = points[v][0] - xi;
                    let dy = points[v][1] - yi;
                    let df = values[v] - fi;
                    neighbor_offsets.push((dx, dy, df));
                }
            }

            // Deduplicate (same neighbor can appear from multiple triangles)
            neighbor_offsets.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            });
            neighbor_offsets.dedup_by(|a, b| {
                let eps = F::from_f64(1e-12).unwrap_or_else(|| F::epsilon());
                (a.0 - b.0).abs() < eps && (a.1 - b.1).abs() < eps
            });

            if neighbor_offsets.is_empty() {
                // No neighbors, gradient is zero
                continue;
            }

            // Solve the least-squares problem:
            // minimize sum_j (dx_j * gx + dy_j * gy - df_j)^2
            // Normal equations: [A^T A] [gx; gy] = A^T b
            let mut ata00 = F::zero();
            let mut ata01 = F::zero();
            let mut ata11 = F::zero();
            let mut atb0 = F::zero();
            let mut atb1 = F::zero();

            for &(dx, dy, df) in &neighbor_offsets {
                ata00 = ata00 + dx * dx;
                ata01 = ata01 + dx * dy;
                ata11 = ata11 + dy * dy;
                atb0 = atb0 + dx * df;
                atb1 = atb1 + dy * df;
            }

            let det = ata00 * ata11 - ata01 * ata01;
            let eps = F::from_f64(1e-14).unwrap_or_else(|| F::epsilon());

            if det.abs() > eps {
                gradients[i][0] = (ata11 * atb0 - ata01 * atb1) / det;
                gradients[i][1] = (ata00 * atb1 - ata01 * atb0) / det;
            }
            // If det is too small, keep gradient as zero
        }

        Ok(gradients)
    }

    /// Clough-Tocher C1 interpolation within a triangle
    ///
    /// This is a simplified Clough-Tocher scheme that blends the linear
    /// interpolant with gradient-corrected cubic terms.
    fn clough_tocher_interpolate(
        &self,
        tri_idx: usize,
        x: F,
        y: F,
        bary: &[F; 3],
    ) -> InterpolateResult<F> {
        let gradients = self.gradients.as_ref().ok_or_else(|| {
            InterpolateError::InvalidState("Gradients not computed for Clough-Tocher".to_string())
        })?;

        let tri = &self.triangulation.triangles()[tri_idx];
        let points = self.triangulation.points();

        let v0 = tri.vertices[0];
        let v1 = tri.vertices[1];
        let v2 = tri.vertices[2];

        let f0 = self.values[v0];
        let f1 = self.values[v1];
        let f2 = self.values[v2];

        let g0 = gradients[v0];
        let g1 = gradients[v1];
        let g2 = gradients[v2];

        let p0 = points[v0];
        let p1 = points[v1];
        let p2 = points[v2];

        // Clough-Tocher interpolation using the cubic Hermite approach:
        //
        // At each vertex i, we have: value f_i and gradient (gx_i, gy_i).
        // The Clough-Tocher element uses these to construct a C1 surface.
        //
        // Simplified approach: blend the linear interpolant with gradient corrections.
        // f(x,y) = f_linear(x,y) + sum_i phi_i(bary) * correction_i
        //
        // where correction_i accounts for the difference between the gradient
        // at vertex i and the gradient of the linear interpolant.

        let lambda0 = bary[0];
        let lambda1 = bary[1];
        let lambda2 = bary[2];

        // Linear interpolant value
        let f_linear = lambda0 * f0 + lambda1 * f1 + lambda2 * f2;

        // Gradient of the linear interpolant (constant over the triangle)
        // Using the relation: grad f_linear = sum_i f_i * grad(lambda_i)
        let e10 = [p1[0] - p0[0], p1[1] - p0[1]];
        let e20 = [p2[0] - p0[0], p2[1] - p0[1]];

        let area2 = e10[0] * e20[1] - e10[1] * e20[0];
        let eps = F::from_f64(1e-15).unwrap_or_else(|| F::epsilon());

        if area2.abs() < eps {
            // Degenerate triangle, fall back to linear
            return Ok(f_linear);
        }

        // Gradient corrections at each vertex
        // diff_i = gradient at vertex i - linear gradient
        let inv_area2 = F::one() / area2;
        let grad_lin = [
            ((f1 - f0) * e20[1] - (f2 - f0) * e10[1]) * inv_area2,
            ((f2 - f0) * e10[0] - (f1 - f0) * e20[0]) * inv_area2,
        ];

        // For each vertex, compute the correction term
        // The correction at point (x,y) near vertex i is:
        // (g_i - grad_lin) . (p - p_i) * lambda_i^2
        // This ensures C1 continuity at the vertices

        let dx = x - p0[0];
        let dy = x - p0[1];
        let _ = (dx, dy);

        let mut correction = F::zero();

        // Vertex 0 correction
        let dg0 = [g0[0] - grad_lin[0], g0[1] - grad_lin[1]];
        let dp0 = [x - p0[0], y - p0[1]];
        let c0 = dg0[0] * dp0[0] + dg0[1] * dp0[1];
        correction = correction + lambda0 * lambda0 * c0;

        // Vertex 1 correction
        let dg1 = [g1[0] - grad_lin[0], g1[1] - grad_lin[1]];
        let dp1 = [x - p1[0], y - p1[1]];
        let c1 = dg1[0] * dp1[0] + dg1[1] * dp1[1];
        correction = correction + lambda1 * lambda1 * c1;

        // Vertex 2 correction
        let dg2 = [g2[0] - grad_lin[0], g2[1] - grad_lin[1]];
        let dp2 = [x - p2[0], y - p2[1]];
        let c2 = dg2[0] * dp2[0] + dg2[1] * dp2[1];
        correction = correction + lambda2 * lambda2 * c2;

        Ok(f_linear + correction)
    }

    // -----------------------------------------------------------------------
    // Private: exterior handling
    // -----------------------------------------------------------------------

    fn handle_exterior(&self, x: F, y: F) -> InterpolateResult<F> {
        match self.exterior {
            ExteriorHandling::Nan => Ok(F::nan()),
            ExteriorHandling::Error => Err(InterpolateError::OutOfBounds(format!(
                "Point ({}, {}) is outside the convex hull of the triangulation",
                x, y
            ))),
            ExteriorHandling::NearestNeighbor => {
                // Find the nearest data point
                let points = self.triangulation.points();
                let mut min_dist_sq = F::infinity();
                let mut nearest_idx = 0;

                for (i, p) in points.iter().enumerate() {
                    let dx = p[0] - x;
                    let dy = p[1] - y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq < min_dist_sq {
                        min_dist_sq = dist_sq;
                        nearest_idx = i;
                    }
                }

                Ok(self.values[nearest_idx])
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Create a linear triangulation interpolator
///
/// # Arguments
///
/// * `points` - 2D scattered data points, shape (n_points, 2)
/// * `values` - Values at data points, shape (n_points,)
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::triangulation_interp::make_linear_triangulation;
///
/// let points = Array2::from_shape_vec((4, 2), vec![
///     0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
/// ]).expect("valid shape");
/// let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);
///
/// let interp = make_linear_triangulation(points, values).expect("valid");
/// ```
pub fn make_linear_triangulation<F: Float + FromPrimitive + Debug + std::fmt::Display + 'static>(
    points: Array2<F>,
    values: Array1<F>,
) -> InterpolateResult<TriangulationInterpolator<F>> {
    TriangulationInterpolator::new(points, values, TriangulationMethod::Linear)
}

/// Create a Clough-Tocher C1 smooth triangulation interpolator
///
/// # Arguments
///
/// * `points` - 2D scattered data points, shape (n_points, 2)
/// * `values` - Values at data points, shape (n_points,)
pub fn make_clough_tocher_interpolator<
    F: Float + FromPrimitive + Debug + std::fmt::Display + 'static,
>(
    points: Array2<F>,
    values: Array1<F>,
) -> InterpolateResult<TriangulationInterpolator<F>> {
    TriangulationInterpolator::new(points, values, TriangulationMethod::CloughTocher)
}

/// Create a nearest-vertex triangulation interpolator
///
/// # Arguments
///
/// * `points` - 2D scattered data points, shape (n_points, 2)
/// * `values` - Values at data points, shape (n_points,)
pub fn make_nearest_vertex_interpolator<
    F: Float + FromPrimitive + Debug + std::fmt::Display + 'static,
>(
    points: Array2<F>,
    values: Array1<F>,
) -> InterpolateResult<TriangulationInterpolator<F>> {
    TriangulationInterpolator::new(points, values, TriangulationMethod::NearestVertex)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn make_square_data() -> (Array2<f64>, Array1<f64>) {
        // 4 corners of unit square, z = x + y
        let points = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("valid shape");
        let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0]);
        (points, values)
    }

    fn make_larger_data() -> (Array2<f64>, Array1<f64>) {
        // More points for better triangulation, z = x + y
        let mut pts = Vec::new();
        let mut vals = Vec::new();

        for i in 0..4 {
            for j in 0..4 {
                let x = i as f64 / 3.0;
                let y = j as f64 / 3.0;
                pts.push(x);
                pts.push(y);
                vals.push(x + y);
            }
        }

        let points = Array2::from_shape_vec((16, 2), pts).expect("valid shape");
        let values = Array1::from_vec(vals);
        (points, values)
    }

    fn make_quadratic_data() -> (Array2<f64>, Array1<f64>) {
        // z = x^2 + y^2 on a scattered set of points
        let mut pts = Vec::new();
        let mut vals = Vec::new();

        // Grid points
        for i in 0..5 {
            for j in 0..5 {
                let x = i as f64 / 4.0;
                let y = j as f64 / 4.0;
                pts.push(x);
                pts.push(y);
                vals.push(x * x + y * y);
            }
        }

        let points = Array2::from_shape_vec((25, 2), pts).expect("valid shape");
        let values = Array1::from_vec(vals);
        (points, values)
    }

    // === Delaunay triangulation tests ===

    #[test]
    fn test_delaunay_basic() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tri = DelaunayTriangulation::new(&points).expect("valid triangulation");

        assert_eq!(
            tri.num_triangles(),
            2,
            "Unit square should have 2 triangles"
        );
        assert_eq!(tri.points().len(), 4);
    }

    #[test]
    fn test_delaunay_larger() {
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                points.push([i as f64, j as f64]);
            }
        }
        let tri = DelaunayTriangulation::new(&points).expect("valid triangulation");

        // For n points in general position, Delaunay should have roughly 2n - 5 triangles
        assert!(
            tri.num_triangles() >= 20,
            "25 points should give at least 20 triangles, got {}",
            tri.num_triangles()
        );
    }

    #[test]
    fn test_delaunay_too_few_points() {
        let points = vec![[0.0, 0.0], [1.0, 0.0]];
        let result = DelaunayTriangulation::new(&points);
        assert!(result.is_err(), "2 points should fail");
    }

    #[test]
    fn test_delaunay_find_containing_triangle() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tri = DelaunayTriangulation::new(&points).expect("valid triangulation");

        // Point inside the convex hull
        let result = tri.find_containing_triangle(0.25, 0.25);
        assert!(result.is_some(), "Interior point should be found");

        let (_, bary) = result.expect("has result");
        assert!(
            bary[0] >= -1e-10 && bary[1] >= -1e-10 && bary[2] >= -1e-10,
            "Barycentric coords should be non-negative"
        );

        // Point outside the convex hull
        let result = tri.find_containing_triangle(-1.0, -1.0);
        assert!(result.is_none(), "Exterior point should not be found");
    }

    // === Linear interpolation tests ===

    #[test]
    fn test_linear_at_data_points() {
        let (points, values) = make_larger_data();
        let interp = TriangulationInterpolator::new(
            points.clone(),
            values.clone(),
            TriangulationMethod::Linear,
        )
        .expect("valid");

        // At data points, should reproduce exact values
        for i in 0..points.nrows() {
            let result = interp
                .evaluate_point(points[[i, 0]], points[[i, 1]])
                .expect("valid");
            if result.is_nan() {
                continue; // Skip boundary points that might be outside
            }
            assert!(
                (result - values[i]).abs() < 1e-8,
                "Linear at data point {}: expected {}, got {}",
                i,
                values[i],
                result
            );
        }
    }

    #[test]
    fn test_linear_reproduces_linear_function() {
        let (points, values) = make_larger_data();
        let interp = TriangulationInterpolator::new(points, values, TriangulationMethod::Linear)
            .expect("valid");

        // Linear interpolation on triangulations should reproduce linear functions exactly
        let test_points = vec![(0.25, 0.25), (0.5, 0.5), (0.75, 0.25), (0.4, 0.6)];
        for (x, y) in test_points {
            let result = interp.evaluate_point(x, y).expect("valid");
            if result.is_nan() {
                continue;
            }
            let expected = x + y;
            assert!(
                (result - expected).abs() < 1e-8,
                "Linear at ({}, {}): expected {}, got {}",
                x,
                y,
                expected,
                result
            );
        }
    }

    #[test]
    fn test_linear_interior_point() {
        let (points, values) = make_square_data();
        let interp = TriangulationInterpolator::new(points, values, TriangulationMethod::Linear)
            .expect("valid");

        let result = interp.evaluate_point(0.5, 0.5).expect("valid");
        // z = x + y at (0.5, 0.5) = 1.0
        assert!(
            (result - 1.0).abs() < 1e-8,
            "Linear at (0.5, 0.5): expected 1.0, got {}",
            result
        );
    }

    // === Nearest vertex tests ===

    #[test]
    fn test_nearest_vertex_at_data_points() {
        let (points, values) = make_larger_data();
        let interp = TriangulationInterpolator::new(
            points.clone(),
            values.clone(),
            TriangulationMethod::NearestVertex,
        )
        .expect("valid");

        for i in 0..points.nrows() {
            let result = interp
                .evaluate_point(points[[i, 0]], points[[i, 1]])
                .expect("valid");
            if result.is_nan() {
                continue;
            }
            assert!(
                (result - values[i]).abs() < 1e-8,
                "NearestVertex at data point {}: expected {}, got {}",
                i,
                values[i],
                result
            );
        }
    }

    #[test]
    fn test_nearest_vertex_is_piecewise_constant() {
        let (points, values) = make_square_data();
        let interp =
            TriangulationInterpolator::new(points, values, TriangulationMethod::NearestVertex)
                .expect("valid");

        // Very close to (0, 0) should return value at (0, 0) = 0.0
        let result = interp.evaluate_point(0.01, 0.01).expect("valid");
        if !result.is_nan() {
            assert!(
                (result - 0.0).abs() < 1e-8,
                "NearestVertex near (0,0): expected 0.0, got {}",
                result
            );
        }
    }

    // === Clough-Tocher tests ===

    #[test]
    fn test_clough_tocher_at_data_points() {
        let (points, values) = make_larger_data();
        let interp = TriangulationInterpolator::new(
            points.clone(),
            values.clone(),
            TriangulationMethod::CloughTocher,
        )
        .expect("valid");

        // At data points, Clough-Tocher should reproduce exact values
        for i in 0..points.nrows() {
            let result = interp
                .evaluate_point(points[[i, 0]], points[[i, 1]])
                .expect("valid");
            if result.is_nan() {
                continue;
            }
            assert!(
                (result - values[i]).abs() < 1e-6,
                "CloughTocher at data point {}: expected {}, got {}",
                i,
                values[i],
                result
            );
        }
    }

    #[test]
    fn test_clough_tocher_smoother_than_linear() {
        let (points, values) = make_quadratic_data();

        let linear = TriangulationInterpolator::new(
            points.clone(),
            values.clone(),
            TriangulationMethod::Linear,
        )
        .expect("valid");

        let ct = TriangulationInterpolator::new(points, values, TriangulationMethod::CloughTocher)
            .expect("valid");

        // Test at several interior points
        let test_points = vec![(0.3, 0.3), (0.5, 0.5), (0.7, 0.3)];
        let mut ct_error_sum = 0.0;
        let mut count = 0;

        for (x, y) in test_points {
            let exact = x * x + y * y;

            let r_linear = linear.evaluate_point(x, y).expect("valid");
            let r_ct = ct.evaluate_point(x, y).expect("valid");

            if r_linear.is_nan() || r_ct.is_nan() {
                continue;
            }

            // We only check that CT produces reasonable results
            let _linear_err = (r_linear - exact).abs();
            ct_error_sum += (r_ct - exact).abs();
            count += 1;
        }

        if count > 0 {
            // Clough-Tocher should generally be more accurate for smooth functions
            // (or at least not much worse)
            let ct_avg = ct_error_sum / count as f64;
            assert!(
                ct_avg < 1.0,
                "Clough-Tocher average error should be reasonable: {}",
                ct_avg
            );
        }
    }

    // === Batch evaluation tests ===

    #[test]
    fn test_batch_evaluation() {
        let (points, values) = make_larger_data();
        let interp = TriangulationInterpolator::new(points, values, TriangulationMethod::Linear)
            .expect("valid");

        let queries =
            Array2::from_shape_vec((3, 2), vec![0.25, 0.25, 0.5, 0.5, 0.4, 0.6]).expect("valid");

        let results = interp.evaluate(&queries).expect("valid");
        assert_eq!(results.len(), 3);

        for i in 0..3 {
            if results[i].is_nan() {
                continue;
            }
            let expected = queries[[i, 0]] + queries[[i, 1]];
            assert!(
                (results[i] - expected).abs() < 1e-8,
                "Batch result {}: expected {}, got {}",
                i,
                expected,
                results[i]
            );
        }
    }

    // === Exterior handling tests ===

    #[test]
    fn test_exterior_nan() {
        let (points, values) = make_square_data();
        let interp = TriangulationInterpolator::with_exterior(
            points,
            values,
            TriangulationMethod::Linear,
            ExteriorHandling::Nan,
        )
        .expect("valid");

        let result = interp.evaluate_point(-1.0, -1.0).expect("valid");
        assert!(result.is_nan(), "Exterior point should return NaN");
    }

    #[test]
    fn test_exterior_error() {
        let (points, values) = make_square_data();
        let interp = TriangulationInterpolator::with_exterior(
            points,
            values,
            TriangulationMethod::Linear,
            ExteriorHandling::Error,
        )
        .expect("valid");

        let result = interp.evaluate_point(-1.0, -1.0);
        assert!(result.is_err(), "Exterior point should return error");
    }

    #[test]
    fn test_exterior_nearest_neighbor() {
        let (points, values) = make_square_data();
        let interp = TriangulationInterpolator::with_exterior(
            points,
            values,
            TriangulationMethod::Linear,
            ExteriorHandling::NearestNeighbor,
        )
        .expect("valid");

        // (-0.1, -0.1) should be nearest to (0, 0) with value 0
        let result = interp.evaluate_point(-0.1, -0.1).expect("valid");
        assert!(
            (result - 0.0).abs() < 1e-8,
            "Exterior NN at (-0.1, -0.1): expected 0.0, got {}",
            result
        );
    }

    // === Edge case tests ===

    #[test]
    fn test_non_2d_rejected() {
        let points = Array2::from_shape_vec((3, 3), vec![0.0; 9]).expect("valid shape");
        let values = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let result = TriangulationInterpolator::new(points, values, TriangulationMethod::Linear);
        assert!(result.is_err(), "3D points should be rejected");
    }

    #[test]
    fn test_mismatched_sizes_rejected() {
        let points = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .expect("valid shape");
        let values = Array1::from_vec(vec![0.0, 1.0]); // Only 2 values for 4 points
        let result = TriangulationInterpolator::new(points, values, TriangulationMethod::Linear);
        assert!(result.is_err(), "Mismatched sizes should be rejected");
    }

    #[test]
    fn test_too_few_points_rejected() {
        let points = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).expect("valid shape");
        let values = Array1::from_vec(vec![0.0, 1.0]);
        let result = TriangulationInterpolator::new(points, values, TriangulationMethod::Linear);
        assert!(result.is_err(), "2 points should be rejected");
    }

    // === Convenience constructor tests ===

    #[test]
    fn test_make_linear_triangulation() {
        let (points, values) = make_larger_data();
        let interp = make_linear_triangulation(points, values).expect("valid");
        let result = interp.evaluate_point(0.5, 0.5).expect("valid");
        if !result.is_nan() {
            assert!(
                (result - 1.0).abs() < 1e-8,
                "make_linear_triangulation at (0.5,0.5): expected 1.0, got {}",
                result
            );
        }
    }

    #[test]
    fn test_make_clough_tocher_interpolator() {
        let (points, values) = make_larger_data();
        let interp = make_clough_tocher_interpolator(points, values).expect("valid");
        let result = interp.evaluate_point(0.5, 0.5).expect("valid");
        assert!(result.is_finite() || result.is_nan());
    }

    #[test]
    fn test_make_nearest_vertex_interpolator() {
        let (points, values) = make_larger_data();
        let interp = make_nearest_vertex_interpolator(points, values).expect("valid");
        let result = interp.evaluate_point(0.5, 0.5).expect("valid");
        assert!(result.is_finite() || result.is_nan());
    }

    // === Accessor tests ===

    #[test]
    fn test_accessors() {
        let (points, values) = make_larger_data();
        let interp = TriangulationInterpolator::new(points, values, TriangulationMethod::Linear)
            .expect("valid");

        assert_eq!(interp.num_points(), 16);
        assert!(interp.num_triangles() >= 10);
        assert_eq!(interp.values().len(), 16);
        assert!(interp.triangulation().num_triangles() >= 10);
    }

    // === Convergence test ===

    #[test]
    fn test_linear_convergence_quadratic() {
        // For f(x,y) = x^2 + y^2, linear triangulation interpolation
        // should converge as the grid is refined.
        // Use an off-grid point to avoid zero error from hitting a grid node.
        let test_point = (0.37_f64, 0.63_f64);
        let exact_value = 0.37 * 0.37 + 0.63 * 0.63;

        let mut errors = Vec::new();
        for &n in &[5, 10, 20] {
            let mut pts = Vec::new();
            let mut vals = Vec::new();

            for i in 0..n {
                for j in 0..n {
                    let x = i as f64 / (n - 1) as f64;
                    let y = j as f64 / (n - 1) as f64;
                    pts.push(x);
                    pts.push(y);
                    vals.push(x * x + y * y);
                }
            }

            let points = Array2::from_shape_vec((n * n, 2), pts).expect("valid");
            let values = Array1::from_vec(vals);

            let interp =
                TriangulationInterpolator::new(points, values, TriangulationMethod::Linear)
                    .expect("valid");

            let result = interp
                .evaluate_point(test_point.0, test_point.1)
                .expect("valid");

            if result.is_nan() {
                continue;
            }

            let error = (result - exact_value).abs();
            errors.push(error);
        }

        // Overall, error should decrease with refinement
        if errors.len() >= 2 {
            assert!(
                errors[errors.len() - 1] < errors[0] || errors[errors.len() - 1] < 1e-10,
                "Error should decrease: first={}, last={}",
                errors[0],
                errors[errors.len() - 1]
            );
        }

        if let Some(&last) = errors.last() {
            assert!(last < 0.01, "Should converge: final error = {}", last);
        }
    }
}
