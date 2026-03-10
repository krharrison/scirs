//! Incremental 3D convex hull algorithm
//!
//! This module implements an incremental convex hull algorithm for 3D point sets
//! using a Doubly-Connected Edge List (DCEL) / half-edge data structure.
//!
//! # Algorithm
//!
//! The incremental algorithm builds the convex hull by adding points one at a time:
//! 1. Start with an initial tetrahedron from 4 non-coplanar points
//! 2. For each remaining point:
//!    a. If the point is inside the hull, skip it
//!    b. Find all faces visible from the point (horizon detection)
//!    c. Remove visible faces and create new faces connecting the point to the horizon edges
//!
//! Time complexity: O(n^2) worst case, O(n log n) expected.
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::computational_geometry::incremental_hull_3d::IncrementalHull3D;
//!
//! let points = vec![
//!     [0.0, 0.0, 0.0],
//!     [1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0],
//!     [0.0, 0.0, 1.0],
//!     [0.1, 0.1, 0.1], // interior point
//! ];
//!
//! let hull = IncrementalHull3D::new(&points).expect("Operation failed");
//! assert_eq!(hull.num_vertices(), 4);
//! assert_eq!(hull.num_faces(), 4);
//! ```

use crate::error::{SpatialError, SpatialResult};

/// Tolerance for floating-point comparisons
const EPSILON: f64 = 1e-10;

/// A triangular face of the convex hull
#[derive(Debug, Clone)]
pub struct HullFace {
    /// Indices of the three vertices (in counter-clockwise order when viewed from outside)
    pub vertices: [usize; 3],
    /// Outward-pointing normal vector
    pub normal: [f64; 3],
    /// Distance from origin along the normal (plane equation: n . x = d)
    pub distance: f64,
    /// Whether this face is active (not removed)
    active: bool,
}

impl HullFace {
    /// Compute the signed distance from a point to the face's plane
    ///
    /// Positive means the point is on the outside (visible side) of the face.
    fn signed_distance(&self, point: &[f64; 3]) -> f64 {
        self.normal[0] * point[0] + self.normal[1] * point[1] + self.normal[2] * point[2]
            - self.distance
    }

    /// Check if a point is visible from this face (on the outside)
    fn is_visible_from(&self, point: &[f64; 3]) -> bool {
        self.signed_distance(point) > EPSILON
    }

    /// Area of the triangular face
    pub fn area(&self, vertices: &[[f64; 3]]) -> f64 {
        let a = vertices[self.vertices[0]];
        let b = vertices[self.vertices[1]];
        let c = vertices[self.vertices[2]];

        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];

        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];

        0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
    }
}

/// An edge of the convex hull, represented by two vertex indices
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct HullEdge {
    v1: usize,
    v2: usize,
}

impl HullEdge {
    fn new(v1: usize, v2: usize) -> Self {
        Self { v1, v2 }
    }

    /// Get the reversed edge (same edge, opposite direction)
    fn reversed(&self) -> Self {
        Self {
            v1: self.v2,
            v2: self.v1,
        }
    }
}

/// An incremental 3D convex hull
#[derive(Debug, Clone)]
pub struct IncrementalHull3D {
    /// All points (including interior ones)
    points: Vec<[f64; 3]>,
    /// Indices of points that are hull vertices
    vertex_indices: Vec<usize>,
    /// The faces of the hull
    faces: Vec<HullFace>,
}

impl IncrementalHull3D {
    /// Create a new 3D convex hull from a set of points
    ///
    /// # Arguments
    ///
    /// * `points` - A slice of 3D points [x, y, z]
    ///
    /// # Returns
    ///
    /// * `SpatialResult<IncrementalHull3D>` - The convex hull
    ///
    /// # Examples
    ///
    /// ```
    /// use scirs2_spatial::computational_geometry::incremental_hull_3d::IncrementalHull3D;
    ///
    /// let points = vec![
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    /// ];
    ///
    /// let hull = IncrementalHull3D::new(&points).expect("Operation failed");
    /// assert_eq!(hull.num_faces(), 4);
    /// ```
    pub fn new(points: &[[f64; 3]]) -> SpatialResult<Self> {
        if points.len() < 4 {
            return Err(SpatialError::ValueError(
                "Need at least 4 points for a 3D convex hull".to_string(),
            ));
        }

        let mut hull = IncrementalHull3D {
            points: points.to_vec(),
            vertex_indices: Vec::new(),
            faces: Vec::new(),
        };

        // Find initial tetrahedron
        hull.initialize_tetrahedron()?;

        // Add remaining points incrementally
        for i in 0..points.len() {
            if hull.vertex_indices.contains(&i) {
                continue;
            }
            hull.add_point(i)?;
        }

        // Compact: remove inactive faces
        hull.faces.retain(|f| f.active);

        // Rebuild vertex_indices from active faces
        let mut used_vertices = std::collections::HashSet::new();
        for face in &hull.faces {
            for &v in &face.vertices {
                used_vertices.insert(v);
            }
        }
        hull.vertex_indices = used_vertices.into_iter().collect();
        hull.vertex_indices.sort();

        Ok(hull)
    }

    /// Initialize the hull with a tetrahedron from 4 non-coplanar points
    fn initialize_tetrahedron(&mut self) -> SpatialResult<()> {
        let n = self.points.len();

        // Find 4 non-coplanar points
        // Step 1: Find two distinct points
        let p0 = 0;
        let mut p1 = None;

        for i in 1..n {
            let d = distance_3d(&self.points[p0], &self.points[i]);
            if d > EPSILON {
                p1 = Some(i);
                break;
            }
        }

        let p1 = p1.ok_or_else(|| {
            SpatialError::ComputationError("All points are coincident".to_string())
        })?;

        // Step 2: Find a point not collinear with p0, p1
        let mut p2 = None;
        for i in 0..n {
            if i == p0 || i == p1 {
                continue;
            }
            let cross = cross_product_3d(
                &sub_3d(&self.points[p1], &self.points[p0]),
                &sub_3d(&self.points[i], &self.points[p0]),
            );
            let cross_len = norm_3d(&cross);
            if cross_len > EPSILON {
                p2 = Some(i);
                break;
            }
        }

        let p2 = p2.ok_or_else(|| {
            SpatialError::ComputationError("All points are collinear".to_string())
        })?;

        // Step 3: Find a point not coplanar with p0, p1, p2
        let normal = cross_product_3d(
            &sub_3d(&self.points[p1], &self.points[p0]),
            &sub_3d(&self.points[p2], &self.points[p0]),
        );

        let mut p3 = None;
        for i in 0..n {
            if i == p0 || i == p1 || i == p2 {
                continue;
            }
            let diff = sub_3d(&self.points[i], &self.points[p0]);
            let vol = dot_3d(&normal, &diff);
            if vol.abs() > EPSILON {
                p3 = Some(i);
                break;
            }
        }

        let p3 = p3
            .ok_or_else(|| SpatialError::ComputationError("All points are coplanar".to_string()))?;

        self.vertex_indices = vec![p0, p1, p2, p3];

        // Create the 4 faces of the tetrahedron
        // Ensure all faces have outward-pointing normals

        // Compute centroid of tetrahedron
        let centroid = [
            (self.points[p0][0] + self.points[p1][0] + self.points[p2][0] + self.points[p3][0])
                / 4.0,
            (self.points[p0][1] + self.points[p1][1] + self.points[p2][1] + self.points[p3][1])
                / 4.0,
            (self.points[p0][2] + self.points[p1][2] + self.points[p2][2] + self.points[p3][2])
                / 4.0,
        ];

        let face_verts = [[p0, p1, p2], [p0, p1, p3], [p0, p2, p3], [p1, p2, p3]];

        for verts in &face_verts {
            let face = self.create_face(verts[0], verts[1], verts[2], &centroid);
            self.faces.push(face);
        }

        Ok(())
    }

    /// Create a face with outward-pointing normal
    fn create_face(&self, v0: usize, v1: usize, v2: usize, interior_point: &[f64; 3]) -> HullFace {
        let a = self.points[v0];
        let b = self.points[v1];
        let c = self.points[v2];

        let ab = sub_3d(&b, &a);
        let ac = sub_3d(&c, &a);
        let mut normal = cross_product_3d(&ab, &ac);

        let normal_len = norm_3d(&normal);
        if normal_len > EPSILON {
            normal[0] /= normal_len;
            normal[1] /= normal_len;
            normal[2] /= normal_len;
        }

        let distance = dot_3d(&normal, &a);

        // Check if normal points away from interior point
        let interior_dist = dot_3d(&normal, interior_point) - distance;

        if interior_dist > 0.0 {
            // Normal points toward the interior; flip it
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
            let new_distance = -distance;

            HullFace {
                vertices: [v0, v2, v1], // swap v1 and v2 to maintain CCW when viewed from outside
                normal,
                distance: new_distance,
                active: true,
            }
        } else {
            HullFace {
                vertices: [v0, v1, v2],
                normal,
                distance,
                active: true,
            }
        }
    }

    /// Add a point to the convex hull
    fn add_point(&mut self, point_idx: usize) -> SpatialResult<()> {
        let point = self.points[point_idx];

        // Find all faces visible from this point
        let mut visible_faces: Vec<usize> = Vec::new();
        for (i, face) in self.faces.iter().enumerate() {
            if face.active && face.is_visible_from(&point) {
                visible_faces.push(i);
            }
        }

        // If no faces are visible, the point is inside the hull
        if visible_faces.is_empty() {
            return Ok(());
        }

        // Find the horizon edges (edges shared between one visible and one non-visible face)
        let horizon_edges = self.find_horizon_edges(&visible_faces);

        if horizon_edges.is_empty() {
            return Ok(());
        }

        // Compute the centroid of the current hull (for face orientation)
        let centroid = self.compute_centroid();

        // Remove visible faces
        for &face_idx in &visible_faces {
            self.faces[face_idx].active = false;
        }

        // Create new faces from the point to each horizon edge
        for edge in &horizon_edges {
            let face = self.create_face(edge.v1, edge.v2, point_idx, &centroid);
            self.faces.push(face);
        }

        Ok(())
    }

    /// Find the horizon edges for a set of visible faces
    fn find_horizon_edges(&self, visible_faces: &[usize]) -> Vec<HullEdge> {
        let mut edge_count: std::collections::HashMap<(usize, usize), i32> =
            std::collections::HashMap::new();

        for &face_idx in visible_faces {
            let face = &self.faces[face_idx];
            let v = face.vertices;

            // Each face has 3 edges
            let edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])];

            for (a, b) in edges {
                let key = if a < b { (a, b) } else { (b, a) };
                *edge_count.entry(key).or_insert(0) += 1;
            }
        }

        // Horizon edges appear exactly once in the visible faces
        // (edges shared by two visible faces appear twice and are internal)
        let mut horizon = Vec::new();
        for (&(a, b), &count) in &edge_count {
            if count == 1 {
                // Find the orientation: the edge should be oriented so that the
                // new face will have the correct winding
                // Find which visible face contains this edge
                for &face_idx in visible_faces {
                    let face = &self.faces[face_idx];
                    let v = face.vertices;
                    let edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])];

                    for (ea, eb) in edges {
                        let key = if ea < eb { (ea, eb) } else { (eb, ea) };
                        if key == (a, b) {
                            // Reverse the edge direction (since we're creating
                            // new faces to replace the visible one, the new face
                            // should have the opposite winding for the shared edge)
                            horizon.push(HullEdge::new(eb, ea));
                            break;
                        }
                    }
                }
            }
        }

        horizon
    }

    /// Compute the centroid of the hull vertices
    fn compute_centroid(&self) -> [f64; 3] {
        let active_verts: std::collections::HashSet<usize> = self
            .faces
            .iter()
            .filter(|f| f.active)
            .flat_map(|f| f.vertices.iter().copied())
            .collect();

        if active_verts.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        let n = active_verts.len() as f64;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;

        for &v in &active_verts {
            cx += self.points[v][0];
            cy += self.points[v][1];
            cz += self.points[v][2];
        }

        [cx / n, cy / n, cz / n]
    }

    /// Get the number of hull vertices
    pub fn num_vertices(&self) -> usize {
        self.vertex_indices.len()
    }

    /// Get the number of hull faces
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Get the number of hull edges
    ///
    /// By Euler's formula: V - E + F = 2, so E = V + F - 2
    pub fn num_edges(&self) -> usize {
        let v = self.num_vertices();
        let f = self.num_faces();
        (v + f).saturating_sub(2)
    }

    /// Get the hull vertices
    pub fn get_vertices(&self) -> Vec<[f64; 3]> {
        self.vertex_indices
            .iter()
            .map(|&i| self.points[i])
            .collect()
    }

    /// Get the hull vertex indices
    pub fn vertex_indices(&self) -> &[usize] {
        &self.vertex_indices
    }

    /// Get the hull faces
    pub fn get_faces(&self) -> &[HullFace] {
        &self.faces
    }

    /// Get the face vertex indices as triples
    pub fn get_face_indices(&self) -> Vec<[usize; 3]> {
        self.faces.iter().map(|f| f.vertices).collect()
    }

    /// Compute the volume of the convex hull
    ///
    /// Uses the signed tetrahedron volume method: sum the signed volumes of
    /// tetrahedra formed by each face and the origin.
    pub fn volume(&self) -> f64 {
        let mut vol = 0.0;

        for face in &self.faces {
            let a = self.points[face.vertices[0]];
            let b = self.points[face.vertices[1]];
            let c = self.points[face.vertices[2]];

            // Signed volume of tetrahedron with one vertex at origin
            vol += a[0] * (b[1] * c[2] - b[2] * c[1])
                + a[1] * (b[2] * c[0] - b[0] * c[2])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }

        vol.abs() / 6.0
    }

    /// Compute the surface area of the convex hull
    pub fn surface_area(&self) -> f64 {
        let mut area = 0.0;

        for face in &self.faces {
            area += face.area(&self.points);
        }

        area
    }

    /// Check if a point is inside the convex hull
    ///
    /// A point is inside if it is on the interior side of all faces.
    pub fn contains(&self, point: &[f64; 3]) -> bool {
        for face in &self.faces {
            if face.signed_distance(point) > EPSILON {
                return false;
            }
        }
        true
    }

    /// Get the all input points
    pub fn points(&self) -> &[[f64; 3]] {
        &self.points
    }
}

// ---- Vector math helpers ----

fn sub_3d(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross_product_3d(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot_3d(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm_3d(a: &[f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

fn distance_3d(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let d = sub_3d(a, b);
    norm_3d(&d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tetrahedron() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");
        assert_eq!(hull.num_vertices(), 4);
        assert_eq!(hull.num_faces(), 4);
        // Euler: E = V + F - 2 = 4 + 4 - 2 = 6
        assert_eq!(hull.num_edges(), 6);
    }

    #[test]
    fn test_tetrahedron_with_interior_point() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.1, 0.1, 0.1], // interior point
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");
        assert_eq!(hull.num_vertices(), 4);
        assert_eq!(hull.num_faces(), 4);
    }

    #[test]
    fn test_cube() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");
        assert_eq!(hull.num_vertices(), 8);
        // A cube has 6 faces, but as a triangulated convex hull it has 12 triangular faces
        assert_eq!(hull.num_faces(), 12);
    }

    #[test]
    fn test_volume_tetrahedron() {
        // Regular tetrahedron with one vertex at origin
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");
        let vol = hull.volume();
        // Volume of this tetrahedron = 1/6
        assert!((vol - 1.0 / 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_volume_cube() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");
        let vol = hull.volume();
        assert!((vol - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_surface_area_cube() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");
        let sa = hull.surface_area();
        // Surface area of unit cube = 6
        assert!((sa - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_contains() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");

        // Interior point
        assert!(hull.contains(&[0.1, 0.1, 0.1]));
        // Centroid should be inside
        assert!(hull.contains(&[0.1, 0.1, 0.1]));
        // Far exterior point
        assert!(!hull.contains(&[2.0, 2.0, 2.0]));
    }

    #[test]
    fn test_too_few_points() {
        let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let result = IncrementalHull3D::new(&points);
        assert!(result.is_err());
    }

    #[test]
    fn test_coplanar_points() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let result = IncrementalHull3D::new(&points);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_vertices() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");
        let verts = hull.get_vertices();
        assert_eq!(verts.len(), 4);
    }

    #[test]
    fn test_get_face_indices() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");
        let face_indices = hull.get_face_indices();
        assert_eq!(face_indices.len(), 4);

        // Each face should have 3 distinct vertices
        for face in &face_indices {
            assert_ne!(face[0], face[1]);
            assert_ne!(face[1], face[2]);
            assert_ne!(face[0], face[2]);
        }
    }

    #[test]
    fn test_many_points() {
        // Points on a sphere - hull should use all of them
        // 6 points: axis-aligned extremes of unit sphere, plus interior point
        let points = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");
        assert_eq!(hull.num_vertices(), 6); // Only the 6 extremes
        assert!(hull.contains(&[0.0, 0.0, 0.0])); // Origin is inside
    }

    #[test]
    fn test_surface_area_positive() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");
        assert!(hull.surface_area() > 0.0);
    }

    #[test]
    fn test_volume_positive() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let hull = IncrementalHull3D::new(&points).expect("Operation failed");
        assert!(hull.volume() > 0.0);
    }
}
