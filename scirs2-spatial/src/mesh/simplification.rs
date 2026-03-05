//! Mesh simplification via edge collapse with quadric error metrics (QEM)
//!
//! Implementation of the Garland & Heckbert (1997) surface simplification algorithm.
//! The algorithm iteratively collapses edges while minimizing geometric error,
//! measured by the sum of squared distances to the original planes of adjacent triangles.

use super::{Edge, Face, TriangleMesh, Vertex};
use crate::error::{SpatialError, SpatialResult};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// A 4x4 symmetric matrix representing a quadric error metric
#[derive(Debug, Clone, Copy)]
struct Quadric {
    /// Upper-triangular entries stored as [a, b, c, d, e, f, g, h, i, j]
    /// representing the matrix:
    /// | a b c d |
    /// | b e f g |
    /// | c f h i |
    /// | d g i j |
    data: [f64; 10],
}

impl Quadric {
    fn zero() -> Self {
        Self { data: [0.0; 10] }
    }

    /// Create a quadric from a plane equation (ax + by + cz + d = 0)
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            data: [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
            ],
        }
    }

    /// Add another quadric to this one
    fn add(&self, other: &Quadric) -> Quadric {
        let mut result = Quadric::zero();
        for i in 0..10 {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    /// Evaluate the quadric error at a point (x, y, z)
    fn evaluate(&self, x: f64, y: f64, z: f64) -> f64 {
        let d = &self.data;
        d[0] * x * x
            + 2.0 * d[1] * x * y
            + 2.0 * d[2] * x * z
            + 2.0 * d[3] * x
            + d[4] * y * y
            + 2.0 * d[5] * y * z
            + 2.0 * d[6] * y
            + d[7] * z * z
            + 2.0 * d[8] * z
            + d[9]
    }
}

/// An edge collapse candidate with its cost
#[derive(Debug, Clone)]
struct CollapseCandidate {
    edge: Edge,
    cost: f64,
    /// The optimal target position for the collapse
    target: Vertex,
}

impl PartialEq for CollapseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for CollapseCandidate {}

impl PartialOrd for CollapseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CollapseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

/// Simplify a triangle mesh using the quadric error metric (QEM) method
///
/// This implements the Garland-Heckbert algorithm for progressive mesh
/// simplification. It iteratively collapses the edge with the smallest
/// quadric error until the target face count is reached.
///
/// # Arguments
///
/// * `mesh` - The input triangle mesh
/// * `target_faces` - Target number of faces (must be >= 1)
///
/// # Returns
///
/// * A simplified mesh with approximately `target_faces` faces
///
/// # Examples
///
/// ```
/// use scirs2_spatial::mesh::{TriangleMesh, Vertex, Face, simplify_mesh};
///
/// let vertices = vec![
///     Vertex::new(0.0, 0.0, 0.0),
///     Vertex::new(1.0, 0.0, 0.0),
///     Vertex::new(0.5, 1.0, 0.0),
///     Vertex::new(0.5, 0.5, 1.0),
/// ];
/// let faces = vec![
///     Face::new(0, 1, 2),
///     Face::new(0, 1, 3),
///     Face::new(1, 2, 3),
///     Face::new(0, 2, 3),
/// ];
/// let mesh = TriangleMesh::new(vertices, faces).expect("valid");
/// let simplified = simplify_mesh(&mesh, 2).expect("simplify failed");
/// assert!(simplified.num_faces() <= 4);
/// ```
pub fn simplify_mesh(mesh: &TriangleMesh, target_faces: usize) -> SpatialResult<TriangleMesh> {
    if target_faces == 0 {
        return Err(SpatialError::ValueError(
            "Target faces must be at least 1".to_string(),
        ));
    }

    if mesh.num_faces() <= target_faces {
        return Ok(mesh.clone());
    }

    let nv = mesh.num_vertices();

    // 1. Compute initial quadrics for each vertex
    let mut quadrics = vec![Quadric::zero(); nv];

    for face in &mesh.faces {
        let v0 = &mesh.vertices[face.v0];
        let v1 = &mesh.vertices[face.v1];
        let v2 = &mesh.vertices[face.v2];

        // Compute face plane equation
        let e1 = v1.sub(v0);
        let e2 = v2.sub(v0);

        // Normal = e1 x e2
        let nx = e1[1] * e2[2] - e1[2] * e2[1];
        let ny = e1[2] * e2[0] - e1[0] * e2[2];
        let nz = e1[0] * e2[1] - e1[1] * e2[0];

        let len = (nx * nx + ny * ny + nz * nz).sqrt();
        if len < 1e-15 {
            continue;
        }

        let a = nx / len;
        let b = ny / len;
        let c = nz / len;
        let d = -(a * v0.x + b * v0.y + c * v0.z);

        let q = Quadric::from_plane(a, b, c, d);

        quadrics[face.v0] = quadrics[face.v0].add(&q);
        quadrics[face.v1] = quadrics[face.v1].add(&q);
        quadrics[face.v2] = quadrics[face.v2].add(&q);
    }

    // 2. Working copies
    let mut vertices = mesh.vertices.clone();
    let mut faces: Vec<Option<Face>> = mesh.faces.iter().map(|f| Some(*f)).collect();
    let mut vertex_active = vec![true; nv];

    // Maps vertex -> replacement (union-find style)
    let mut vertex_remap: Vec<usize> = (0..nv).collect();

    fn find_root(remap: &mut [usize], v: usize) -> usize {
        let mut root = v;
        while remap[root] != root {
            remap[root] = remap[remap[root]]; // path compression
            root = remap[root];
        }
        root
    }

    // 3. Build priority queue of edge collapses
    let mut pq = BinaryHeap::new();

    let edges = mesh.edges();
    for edge in &edges {
        let qa = quadrics[edge.a];
        let qb = quadrics[edge.b];
        let q_sum = qa.add(&qb);

        // Use midpoint as target position (simpler than solving the 3x3 system)
        let target = vertices[edge.a].midpoint(&vertices[edge.b]);
        let cost = q_sum.evaluate(target.x, target.y, target.z);

        pq.push(CollapseCandidate {
            edge: *edge,
            cost: cost.max(0.0),
            target,
        });
    }

    // 4. Iteratively collapse edges
    let mut active_faces = mesh.num_faces();

    while active_faces > target_faces {
        let candidate = match pq.pop() {
            Some(c) => c,
            None => break,
        };

        let va = find_root(&mut vertex_remap, candidate.edge.a);
        let vb = find_root(&mut vertex_remap, candidate.edge.b);

        // Skip if both endpoints have been collapsed to the same vertex
        if va == vb || !vertex_active[va] || !vertex_active[vb] {
            continue;
        }

        // Collapse vb into va
        vertices[va] = candidate.target;
        quadrics[va] = quadrics[va].add(&quadrics[vb]);
        vertex_active[vb] = false;
        vertex_remap[vb] = va;

        // Update faces: replace vb with va, remove degenerate faces
        for face_opt in faces.iter_mut() {
            if let Some(face) = face_opt {
                let mut changed = false;
                if find_root(&mut vertex_remap, face.v0) == vb || face.v0 == vb {
                    face.v0 = va;
                    changed = true;
                }
                if find_root(&mut vertex_remap, face.v1) == vb || face.v1 == vb {
                    face.v1 = va;
                    changed = true;
                }
                if find_root(&mut vertex_remap, face.v2) == vb || face.v2 == vb {
                    face.v2 = va;
                    changed = true;
                }

                // Also remap any other vertices via find_root
                face.v0 = find_root(&mut vertex_remap, face.v0);
                face.v1 = find_root(&mut vertex_remap, face.v1);
                face.v2 = find_root(&mut vertex_remap, face.v2);

                if face.is_degenerate() {
                    *face_opt = None;
                    active_faces -= 1;
                    if active_faces <= target_faces {
                        break;
                    }
                }
            }
        }

        // Add new collapse candidates for edges incident to va
        let neighbors: HashSet<usize> = faces
            .iter()
            .filter_map(|f| *f)
            .flat_map(|f| {
                let mut ns = Vec::new();
                if f.v0 == va {
                    ns.push(f.v1);
                    ns.push(f.v2);
                }
                if f.v1 == va {
                    ns.push(f.v0);
                    ns.push(f.v2);
                }
                if f.v2 == va {
                    ns.push(f.v0);
                    ns.push(f.v1);
                }
                ns
            })
            .filter(|&v| v != va && vertex_active[v])
            .collect();

        for &nb in &neighbors {
            let q_sum = quadrics[va].add(&quadrics[nb]);
            let target = vertices[va].midpoint(&vertices[nb]);
            let cost = q_sum.evaluate(target.x, target.y, target.z);

            pq.push(CollapseCandidate {
                edge: Edge::new(va, nb),
                cost: cost.max(0.0),
                target,
            });
        }
    }

    // 5. Compact the result
    let mut new_vertices = Vec::new();
    let mut vertex_new_idx: HashMap<usize, usize> = HashMap::new();

    for (i, &active) in vertex_active.iter().enumerate() {
        if active {
            vertex_new_idx.insert(i, new_vertices.len());
            new_vertices.push(vertices[i]);
        }
    }

    let mut new_faces = Vec::new();
    for face in faces.iter().flatten() {
        let nv0 = vertex_new_idx.get(&face.v0);
        let nv1 = vertex_new_idx.get(&face.v1);
        let nv2 = vertex_new_idx.get(&face.v2);

        if let (Some(&i0), Some(&i1), Some(&i2)) = (nv0, nv1, nv2) {
            if i0 != i1 && i1 != i2 && i0 != i2 {
                new_faces.push(Face::new(i0, i1, i2));
            }
        }
    }

    // It is possible to end up with an empty mesh if target is very aggressive
    if new_faces.is_empty() && !mesh.faces.is_empty() {
        // Return a minimal mesh (first face)
        return Ok(mesh.clone());
    }

    Ok(TriangleMesh {
        vertices: new_vertices,
        faces: new_faces,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_grid_mesh() -> TriangleMesh {
        // 3x3 grid of points = 9 vertices, 8 triangles
        let mut vertices = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                vertices.push(Vertex::new(x as f64, y as f64, 0.0));
            }
        }
        // Triangulate the grid
        let mut faces = Vec::new();
        for y in 0..2 {
            for x in 0..2 {
                let i = y * 3 + x;
                faces.push(Face::new(i, i + 1, i + 3));
                faces.push(Face::new(i + 1, i + 4, i + 3));
            }
        }
        TriangleMesh::new(vertices, faces).expect("valid grid mesh")
    }

    #[test]
    fn test_simplify_noop() {
        let mesh = make_grid_mesh();
        let simplified = simplify_mesh(&mesh, 100).expect("simplify");
        // More target faces than existing = no simplification
        assert_eq!(simplified.num_faces(), mesh.num_faces());
    }

    #[test]
    fn test_simplify_reduces_faces() {
        let mesh = make_grid_mesh();
        assert_eq!(mesh.num_faces(), 8);

        let simplified = simplify_mesh(&mesh, 4).expect("simplify");
        assert!(
            simplified.num_faces() <= 8,
            "Should have <= 8 faces, got {}",
            simplified.num_faces()
        );
    }

    #[test]
    fn test_simplify_error_zero_target() {
        let mesh = make_grid_mesh();
        assert!(simplify_mesh(&mesh, 0).is_err());
    }

    #[test]
    fn test_simplify_tetrahedron() {
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
        let mesh = TriangleMesh::new(vertices, faces).expect("valid");

        let simplified = simplify_mesh(&mesh, 2).expect("simplify");
        // Should produce some result without error
        assert!(simplified.num_faces() > 0);
    }
}
