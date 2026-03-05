//! Triangle mesh operations for spatial computing
//!
//! This module provides a comprehensive triangle mesh representation and operations
//! including simplification, smoothing, normal computation, quality metrics, and I/O.
//!
//! # Features
//!
//! * **Mesh Representation**: Indexed face set with vertex/face adjacency
//! * **Simplification**: Edge collapse with quadric error metrics (QEM)
//! * **Smoothing**: Laplacian and Taubin smoothing
//! * **Normals**: Vertex and face normal computation
//! * **Quality Metrics**: Aspect ratio, minimum angle per triangle
//! * **I/O**: ASCII STL import/export

mod quality;
mod simplification;
mod smoothing;

use crate::error::{SpatialError, SpatialResult};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// A 3D vertex
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vertex {
    /// Create a new vertex
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Compute the squared distance to another vertex
    pub fn distance_sq(&self, other: &Vertex) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }

    /// Compute the distance to another vertex
    pub fn distance(&self, other: &Vertex) -> f64 {
        self.distance_sq(other).sqrt()
    }

    /// Subtract another vertex, returning a vector
    pub fn sub(&self, other: &Vertex) -> [f64; 3] {
        [self.x - other.x, self.y - other.y, self.z - other.z]
    }

    /// Midpoint with another vertex
    pub fn midpoint(&self, other: &Vertex) -> Vertex {
        Vertex {
            x: (self.x + other.x) * 0.5,
            y: (self.y + other.y) * 0.5,
            z: (self.z + other.z) * 0.5,
        }
    }
}

impl fmt::Display for Vertex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

/// A triangular face referencing 3 vertex indices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Face {
    pub v0: usize,
    pub v1: usize,
    pub v2: usize,
}

impl Face {
    /// Create a new face
    pub fn new(v0: usize, v1: usize, v2: usize) -> Self {
        Self { v0, v1, v2 }
    }

    /// Get the indices as an array
    pub fn indices(&self) -> [usize; 3] {
        [self.v0, self.v1, self.v2]
    }

    /// Check if this face contains a vertex index
    pub fn contains_vertex(&self, v: usize) -> bool {
        self.v0 == v || self.v1 == v || self.v2 == v
    }

    /// Replace a vertex index in this face
    pub fn replace_vertex(&mut self, old: usize, new: usize) {
        if self.v0 == old {
            self.v0 = new;
        }
        if self.v1 == old {
            self.v1 = new;
        }
        if self.v2 == old {
            self.v2 = new;
        }
    }

    /// Check if this face is degenerate (has duplicate vertices)
    pub fn is_degenerate(&self) -> bool {
        self.v0 == self.v1 || self.v1 == self.v2 || self.v0 == self.v2
    }
}

/// An edge represented as a pair of vertex indices (ordered so that a < b)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
}

impl Edge {
    /// Create a new edge (automatically ordered)
    pub fn new(v0: usize, v1: usize) -> Self {
        if v0 <= v1 {
            Self { a: v0, b: v1 }
        } else {
            Self { a: v1, b: v0 }
        }
    }
}

/// A triangle mesh with indexed face set representation
///
/// # Examples
///
/// ```
/// use scirs2_spatial::mesh::{TriangleMesh, Vertex, Face};
///
/// let vertices = vec![
///     Vertex::new(0.0, 0.0, 0.0),
///     Vertex::new(1.0, 0.0, 0.0),
///     Vertex::new(0.5, 1.0, 0.0),
///     Vertex::new(0.5, 0.5, 1.0),
/// ];
///
/// let faces = vec![
///     Face::new(0, 1, 2),
///     Face::new(0, 1, 3),
///     Face::new(1, 2, 3),
///     Face::new(0, 2, 3),
/// ];
///
/// let mesh = TriangleMesh::new(vertices, faces).expect("Operation failed");
/// assert_eq!(mesh.num_vertices(), 4);
/// assert_eq!(mesh.num_faces(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct TriangleMesh {
    /// Vertex positions
    pub vertices: Vec<Vertex>,
    /// Triangle faces (indices into vertices)
    pub faces: Vec<Face>,
}

impl TriangleMesh {
    /// Create a new triangle mesh
    ///
    /// # Arguments
    ///
    /// * `vertices` - List of vertex positions
    /// * `faces` - List of triangular faces
    ///
    /// # Returns
    ///
    /// A validated triangle mesh
    pub fn new(vertices: Vec<Vertex>, faces: Vec<Face>) -> SpatialResult<Self> {
        // Validate face indices
        let n = vertices.len();
        for (i, face) in faces.iter().enumerate() {
            for &idx in &[face.v0, face.v1, face.v2] {
                if idx >= n {
                    return Err(SpatialError::ValueError(format!(
                        "Face {} references vertex index {} but only {} vertices exist",
                        i, idx, n
                    )));
                }
            }
            if face.is_degenerate() {
                return Err(SpatialError::ValueError(format!(
                    "Face {} is degenerate (has duplicate vertices)",
                    i
                )));
            }
        }

        Ok(Self { vertices, faces })
    }

    /// Create an empty mesh
    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Number of vertices
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Number of faces
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Number of edges (computed from face connectivity)
    pub fn num_edges(&self) -> usize {
        self.edges().len()
    }

    /// Get all unique edges in the mesh
    pub fn edges(&self) -> HashSet<Edge> {
        let mut edges = HashSet::new();
        for face in &self.faces {
            edges.insert(Edge::new(face.v0, face.v1));
            edges.insert(Edge::new(face.v1, face.v2));
            edges.insert(Edge::new(face.v0, face.v2));
        }
        edges
    }

    /// Build vertex-to-face adjacency map
    pub fn vertex_faces(&self) -> HashMap<usize, Vec<usize>> {
        let mut vf: HashMap<usize, Vec<usize>> = HashMap::new();
        for (fi, face) in self.faces.iter().enumerate() {
            for &vi in &[face.v0, face.v1, face.v2] {
                vf.entry(vi).or_default().push(fi);
            }
        }
        vf
    }

    /// Build vertex-to-vertex adjacency (neighbors)
    pub fn vertex_neighbors(&self) -> HashMap<usize, HashSet<usize>> {
        let mut vn: HashMap<usize, HashSet<usize>> = HashMap::new();
        for face in &self.faces {
            let indices = face.indices();
            for i in 0..3 {
                let vi = indices[i];
                for j in 0..3 {
                    if i != j {
                        vn.entry(vi).or_default().insert(indices[j]);
                    }
                }
            }
        }
        vn
    }

    /// Compute face normal for a given face index (unnormalized)
    pub fn face_normal_raw(&self, face_idx: usize) -> SpatialResult<[f64; 3]> {
        if face_idx >= self.faces.len() {
            return Err(SpatialError::ValueError(format!(
                "Face index {} out of range (num faces = {})",
                face_idx,
                self.faces.len()
            )));
        }

        let face = &self.faces[face_idx];
        let v0 = &self.vertices[face.v0];
        let v1 = &self.vertices[face.v1];
        let v2 = &self.vertices[face.v2];

        let e1 = v1.sub(v0);
        let e2 = v2.sub(v0);

        // Cross product e1 x e2
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];

        Ok([nx, ny, nz])
    }

    /// Compute normalized face normal for a given face index
    pub fn face_normal(&self, face_idx: usize) -> SpatialResult<[f64; 3]> {
        let n = self.face_normal_raw(face_idx)?;
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len < 1e-15 {
            return Ok([0.0, 0.0, 0.0]);
        }
        Ok([n[0] / len, n[1] / len, n[2] / len])
    }

    /// Compute all face normals (normalized)
    pub fn compute_face_normals(&self) -> SpatialResult<Vec<[f64; 3]>> {
        let mut normals = Vec::with_capacity(self.faces.len());
        for i in 0..self.faces.len() {
            normals.push(self.face_normal(i)?);
        }
        Ok(normals)
    }

    /// Compute vertex normals by averaging adjacent face normals
    /// weighted by face area
    pub fn compute_vertex_normals(&self) -> SpatialResult<Vec<[f64; 3]>> {
        let n = self.vertices.len();
        let mut normals = vec![[0.0_f64; 3]; n];

        for (fi, face) in self.faces.iter().enumerate() {
            let fn_raw = self.face_normal_raw(fi)?;
            // The magnitude of the raw normal = 2 * face area, so weighting
            // by face area is automatic with unnormalized normals
            for &vi in &[face.v0, face.v1, face.v2] {
                normals[vi][0] += fn_raw[0];
                normals[vi][1] += fn_raw[1];
                normals[vi][2] += fn_raw[2];
            }
        }

        // Normalize
        for normal in &mut normals {
            let len =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if len > 1e-15 {
                normal[0] /= len;
                normal[1] /= len;
                normal[2] /= len;
            }
        }

        Ok(normals)
    }

    /// Compute the area of a face
    pub fn face_area(&self, face_idx: usize) -> SpatialResult<f64> {
        let n = self.face_normal_raw(face_idx)?;
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        Ok(len * 0.5)
    }

    /// Compute the total surface area of the mesh
    pub fn surface_area(&self) -> SpatialResult<f64> {
        let mut total = 0.0;
        for i in 0..self.faces.len() {
            total += self.face_area(i)?;
        }
        Ok(total)
    }

    /// Compute the bounding box of the mesh
    pub fn bounding_box(&self) -> Option<(Vertex, Vertex)> {
        if self.vertices.is_empty() {
            return None;
        }

        let mut min_v = self.vertices[0];
        let mut max_v = self.vertices[0];

        for v in &self.vertices {
            min_v.x = min_v.x.min(v.x);
            min_v.y = min_v.y.min(v.y);
            min_v.z = min_v.z.min(v.z);
            max_v.x = max_v.x.max(v.x);
            max_v.y = max_v.y.max(v.y);
            max_v.z = max_v.z.max(v.z);
        }

        Some((min_v, max_v))
    }

    /// Export to ASCII STL format string
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the solid in the STL file
    pub fn to_stl_ascii(&self, name: &str) -> SpatialResult<String> {
        let mut out = format!("solid {}\n", name);

        for (fi, face) in self.faces.iter().enumerate() {
            let n = self.face_normal(fi)?;
            out.push_str(&format!("  facet normal {} {} {}\n", n[0], n[1], n[2]));
            out.push_str("    outer loop\n");

            let v0 = &self.vertices[face.v0];
            let v1 = &self.vertices[face.v1];
            let v2 = &self.vertices[face.v2];

            out.push_str(&format!("      vertex {} {} {}\n", v0.x, v0.y, v0.z));
            out.push_str(&format!("      vertex {} {} {}\n", v1.x, v1.y, v1.z));
            out.push_str(&format!("      vertex {} {} {}\n", v2.x, v2.y, v2.z));

            out.push_str("    endloop\n");
            out.push_str("  endfacet\n");
        }

        out.push_str(&format!("endsolid {}\n", name));
        Ok(out)
    }

    /// Import from ASCII STL format string
    ///
    /// # Arguments
    ///
    /// * `stl_data` - The ASCII STL content as a string
    pub fn from_stl_ascii(stl_data: &str) -> SpatialResult<Self> {
        let mut vertices: Vec<Vertex> = Vec::new();
        let mut faces: Vec<Face> = Vec::new();
        let mut vertex_map: HashMap<[u64; 3], usize> = HashMap::new();
        let mut current_face_verts: Vec<usize> = Vec::new();

        for line in stl_data.lines() {
            let trimmed = line.trim();

            if let Some(rest) = trimmed.strip_prefix("vertex") {
                let parts: Vec<&str> = rest.split_whitespace().collect();
                if parts.len() < 3 {
                    return Err(SpatialError::ValueError(
                        "Invalid vertex line in STL".to_string(),
                    ));
                }

                let x: f64 = parts[0].parse().map_err(|e| {
                    SpatialError::ValueError(format!("Failed to parse vertex x: {}", e))
                })?;
                let y: f64 = parts[1].parse().map_err(|e| {
                    SpatialError::ValueError(format!("Failed to parse vertex y: {}", e))
                })?;
                let z: f64 = parts[2].parse().map_err(|e| {
                    SpatialError::ValueError(format!("Failed to parse vertex z: {}", e))
                })?;

                // Use bit representation for hashing (exact vertex matching)
                let key = [x.to_bits(), y.to_bits(), z.to_bits()];

                let idx = if let Some(&existing) = vertex_map.get(&key) {
                    existing
                } else {
                    let idx = vertices.len();
                    vertices.push(Vertex::new(x, y, z));
                    vertex_map.insert(key, idx);
                    idx
                };

                current_face_verts.push(idx);
            } else if trimmed == "endloop" {
                if current_face_verts.len() == 3 {
                    let v0 = current_face_verts[0];
                    let v1 = current_face_verts[1];
                    let v2 = current_face_verts[2];

                    // Skip degenerate faces
                    if v0 != v1 && v1 != v2 && v0 != v2 {
                        faces.push(Face::new(v0, v1, v2));
                    }
                }
                current_face_verts.clear();
            }
        }

        Ok(Self { vertices, faces })
    }

    /// Check if the mesh is manifold (every edge has exactly 1 or 2 adjacent faces)
    pub fn is_manifold(&self) -> bool {
        let mut edge_count: HashMap<Edge, usize> = HashMap::new();
        for face in &self.faces {
            *edge_count.entry(Edge::new(face.v0, face.v1)).or_insert(0) += 1;
            *edge_count.entry(Edge::new(face.v1, face.v2)).or_insert(0) += 1;
            *edge_count.entry(Edge::new(face.v0, face.v2)).or_insert(0) += 1;
        }
        edge_count.values().all(|&c| c == 1 || c == 2)
    }

    /// Check if the mesh is closed (every edge has exactly 2 adjacent faces)
    pub fn is_closed(&self) -> bool {
        let mut edge_count: HashMap<Edge, usize> = HashMap::new();
        for face in &self.faces {
            *edge_count.entry(Edge::new(face.v0, face.v1)).or_insert(0) += 1;
            *edge_count.entry(Edge::new(face.v1, face.v2)).or_insert(0) += 1;
            *edge_count.entry(Edge::new(face.v0, face.v2)).or_insert(0) += 1;
        }
        edge_count.values().all(|&c| c == 2)
    }

    /// Get boundary edges (edges with only 1 adjacent face)
    pub fn boundary_edges(&self) -> Vec<Edge> {
        let mut edge_count: HashMap<Edge, usize> = HashMap::new();
        for face in &self.faces {
            *edge_count.entry(Edge::new(face.v0, face.v1)).or_insert(0) += 1;
            *edge_count.entry(Edge::new(face.v1, face.v2)).or_insert(0) += 1;
            *edge_count.entry(Edge::new(face.v0, face.v2)).or_insert(0) += 1;
        }
        edge_count
            .into_iter()
            .filter(|&(_, c)| c == 1)
            .map(|(e, _)| e)
            .collect()
    }

    /// Compute the Euler characteristic: V - E + F
    pub fn euler_characteristic(&self) -> i64 {
        let v = self.num_vertices() as i64;
        let e = self.num_edges() as i64;
        let f = self.num_faces() as i64;
        v - e + f
    }
}

// Re-exports from submodules
pub use quality::{face_aspect_ratio, face_min_angle, mesh_quality_stats, QualityStats};
pub use simplification::simplify_mesh;
pub use smoothing::{laplacian_smooth, taubin_smooth};

#[cfg(test)]
mod tests {
    use super::*;

    fn tetrahedron() -> TriangleMesh {
        let vertices = vec![
            Vertex::new(0.0, 0.0, 0.0),
            Vertex::new(1.0, 0.0, 0.0),
            Vertex::new(0.5, 1.0, 0.0),
            Vertex::new(0.5, 0.5, 1.0),
        ];
        let faces = vec![
            Face::new(0, 1, 2),
            Face::new(0, 1, 3),
            Face::new(1, 2, 3),
            Face::new(0, 2, 3),
        ];
        TriangleMesh::new(vertices, faces).expect("tetrahedron should be valid")
    }

    #[test]
    fn test_mesh_creation() {
        let mesh = tetrahedron();
        assert_eq!(mesh.num_vertices(), 4);
        assert_eq!(mesh.num_faces(), 4);
        assert_eq!(mesh.num_edges(), 6);
    }

    #[test]
    fn test_mesh_invalid_face() {
        let vertices = vec![Vertex::new(0.0, 0.0, 0.0), Vertex::new(1.0, 0.0, 0.0)];
        let faces = vec![Face::new(0, 1, 5)]; // index 5 out of range
        let result = TriangleMesh::new(vertices, faces);
        assert!(result.is_err());
    }

    #[test]
    fn test_degenerate_face() {
        let vertices = vec![Vertex::new(0.0, 0.0, 0.0), Vertex::new(1.0, 0.0, 0.0)];
        let faces = vec![Face::new(0, 0, 1)]; // degenerate
        let result = TriangleMesh::new(vertices, faces);
        assert!(result.is_err());
    }

    #[test]
    fn test_face_normals() {
        let mesh = tetrahedron();
        let normals = mesh.compute_face_normals().expect("compute normals failed");
        assert_eq!(normals.len(), 4);

        // Face 0 (0,1,2) lies in XY plane, normal should be (0,0,+/-1)
        let n0 = normals[0];
        assert!((n0[0].abs()) < 1e-10);
        assert!((n0[1].abs()) < 1e-10);
        assert!((n0[2].abs() - 1.0) < 1e-10);
    }

    #[test]
    fn test_vertex_normals() {
        let mesh = tetrahedron();
        let normals = mesh
            .compute_vertex_normals()
            .expect("compute normals failed");
        assert_eq!(normals.len(), 4);

        // Each vertex normal should be unit length
        for n in &normals {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-10, "Normal length: {}", len);
        }
    }

    #[test]
    fn test_surface_area() {
        let mesh = tetrahedron();
        let area = mesh.surface_area().expect("area failed");
        assert!(area > 0.0);
    }

    #[test]
    fn test_bounding_box() {
        let mesh = tetrahedron();
        let (min_v, max_v) = mesh.bounding_box().expect("non-empty mesh");
        assert!((min_v.x - 0.0).abs() < 1e-10);
        assert!((min_v.y - 0.0).abs() < 1e-10);
        assert!((min_v.z - 0.0).abs() < 1e-10);
        assert!((max_v.x - 1.0).abs() < 1e-10);
        assert!((max_v.y - 1.0).abs() < 1e-10);
        assert!((max_v.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_manifold_closed() {
        let mesh = tetrahedron();
        assert!(mesh.is_manifold());
        assert!(mesh.is_closed());
        assert!(mesh.boundary_edges().is_empty());
    }

    #[test]
    fn test_euler_characteristic() {
        let mesh = tetrahedron();
        // V - E + F = 4 - 6 + 4 = 2 (sphere topology)
        assert_eq!(mesh.euler_characteristic(), 2);
    }

    #[test]
    fn test_stl_roundtrip() {
        let mesh = tetrahedron();
        let stl_str = mesh.to_stl_ascii("test").expect("STL export failed");
        assert!(stl_str.starts_with("solid test"));
        assert!(stl_str.contains("facet normal"));
        assert!(stl_str.contains("vertex"));

        let mesh2 = TriangleMesh::from_stl_ascii(&stl_str).expect("STL import failed");
        // Vertex count may differ due to deduplication, but face count should match
        assert_eq!(mesh2.num_faces(), mesh.num_faces());
    }

    #[test]
    fn test_vertex_neighbors() {
        let mesh = tetrahedron();
        let neighbors = mesh.vertex_neighbors();

        // In a tetrahedron, every vertex connects to every other vertex
        for i in 0..4 {
            let nbs = neighbors.get(&i).expect("vertex should have neighbors");
            assert_eq!(nbs.len(), 3);
        }
    }

    #[test]
    fn test_open_mesh() {
        // Single triangle - open mesh
        let vertices = vec![
            Vertex::new(0.0, 0.0, 0.0),
            Vertex::new(1.0, 0.0, 0.0),
            Vertex::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![Face::new(0, 1, 2)];
        let mesh = TriangleMesh::new(vertices, faces).expect("valid");

        assert!(mesh.is_manifold());
        assert!(!mesh.is_closed());
        assert_eq!(mesh.boundary_edges().len(), 3);
    }
}
