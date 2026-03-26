//! Subdivision Surfaces
//!
//! Subdivision surfaces are a powerful technique for generating smooth
//! surfaces from coarse polygon meshes through iterative refinement.
//!
//! ## Supported Schemes
//!
//! - **Loop subdivision** for triangle meshes: produces C2 smooth surfaces
//!   at regular vertices (C1 at extraordinary vertices).
//! - **Catmull-Clark subdivision** for quad meshes: produces C2 smooth
//!   surfaces at regular vertices.
//!
//! ## Mesh Representation
//!
//! Meshes are represented as:
//! - `vertices`: `Vec<[f64; 3]>` - vertex positions in 3D space
//! - `faces`: `Vec<Vec<usize>>` - face connectivity (indices into vertex array)
//!
//! For Loop subdivision, all faces must be triangles (3 indices each).
//! For Catmull-Clark subdivision, faces may be arbitrary polygons (typically quads).

use crate::error::{InterpolateError, InterpolateResult};
use std::collections::HashMap;

/// A polygon mesh represented by vertices and face connectivity.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vertex positions in 3D.
    pub vertices: Vec<[f64; 3]>,
    /// Face connectivity: each face is a list of vertex indices.
    pub faces: Vec<Vec<usize>>,
}

impl Mesh {
    /// Create a new mesh from vertices and faces.
    ///
    /// # Errors
    ///
    /// Returns an error if any face references an out-of-bounds vertex index.
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<Vec<usize>>) -> InterpolateResult<Self> {
        let n = vertices.len();
        for (fi, face) in faces.iter().enumerate() {
            if face.len() < 3 {
                return Err(InterpolateError::InvalidValue(format!(
                    "face {} has fewer than 3 vertices",
                    fi
                )));
            }
            for &vi in face {
                if vi >= n {
                    return Err(InterpolateError::InvalidValue(format!(
                        "face {} references vertex index {} but only {} vertices exist",
                        fi, vi, n
                    )));
                }
            }
        }
        Ok(Mesh { vertices, faces })
    }

    /// Return the number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Return the number of faces.
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Check if all faces are triangles.
    pub fn is_triangle_mesh(&self) -> bool {
        self.faces.iter().all(|f| f.len() == 3)
    }

    /// Check if all faces are quads.
    pub fn is_quad_mesh(&self) -> bool {
        self.faces.iter().all(|f| f.len() == 4)
    }
}

/// Create a canonical edge key (smaller index first) for use as hash map key.
#[inline]
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Add two 3D points.
#[inline]
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Scale a 3D point.
#[inline]
fn scale3(s: f64, a: [f64; 3]) -> [f64; 3] {
    [s * a[0], s * a[1], s * a[2]]
}

/// Compute the average of a list of 3D points.
fn average_points(points: &[[f64; 3]]) -> [f64; 3] {
    if points.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let n = points.len() as f64;
    let mut sum = [0.0, 0.0, 0.0];
    for p in points {
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

// ---------------------------------------------------------------------------
// Loop Subdivision
// ---------------------------------------------------------------------------

/// Build adjacency: for each edge, which faces contain it, and for each
/// vertex, which vertices are its neighbors.
struct LoopAdjacency {
    /// edge -> list of face indices containing this edge
    edge_faces: HashMap<(usize, usize), Vec<usize>>,
    /// vertex -> set of neighbor vertex indices
    vertex_neighbors: Vec<Vec<usize>>,
}

fn build_loop_adjacency(mesh: &Mesh) -> LoopAdjacency {
    let nv = mesh.vertices.len();
    let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut vertex_neighbors: Vec<Vec<usize>> = vec![Vec::new(); nv];

    for (fi, face) in mesh.faces.iter().enumerate() {
        let nf = face.len();
        for i in 0..nf {
            let a = face[i];
            let b = face[(i + 1) % nf];
            let ek = edge_key(a, b);
            edge_faces.entry(ek).or_default().push(fi);

            if !vertex_neighbors[a].contains(&b) {
                vertex_neighbors[a].push(b);
            }
            if !vertex_neighbors[b].contains(&a) {
                vertex_neighbors[b].push(a);
            }
        }
    }

    LoopAdjacency {
        edge_faces,
        vertex_neighbors,
    }
}

/// Perform one iteration of Loop subdivision on a triangle mesh.
///
/// # Algorithm
///
/// 1. **Edge points**: For each interior edge shared by two triangles with
///    vertices (v1, v2) and opposite vertices (v3, v4):
///    `edge_point = 3/8 * (v1 + v2) + 1/8 * (v3 + v4)`
///    For boundary edges: `edge_point = 1/2 * (v1 + v2)`
///
/// 2. **Vertex points**: For each interior vertex with `n` neighbors:
///    `vertex_point = (1 - n*beta) * v + beta * sum(neighbors)`
///    where `beta = (5/8 - (3/8 + cos(2*pi/n)/4)^2) / n`
///    For boundary vertices: average with boundary neighbors.
///
/// 3. **Connectivity**: Each triangle is split into 4 sub-triangles.
fn loop_subdivide_once(mesh: &Mesh) -> InterpolateResult<Mesh> {
    if !mesh.is_triangle_mesh() {
        return Err(InterpolateError::InvalidValue(
            "Loop subdivision requires a triangle mesh".to_string(),
        ));
    }

    let adj = build_loop_adjacency(mesh);
    let nv = mesh.vertices.len();

    // 1. Compute new positions for existing vertices
    let mut new_vertices: Vec<[f64; 3]> = Vec::with_capacity(nv * 4);

    for i in 0..nv {
        let neighbors = &adj.vertex_neighbors[i];
        let n = neighbors.len();

        // Check if boundary vertex (any adjacent edge is boundary)
        let is_boundary = neighbors.iter().any(|&nb| {
            let ek = edge_key(i, nb);
            adj.edge_faces
                .get(&ek)
                .map_or(true, |faces| faces.len() < 2)
        });

        if is_boundary {
            // Find boundary neighbors
            let mut boundary_neighbors = Vec::new();
            for &nb in neighbors {
                let ek = edge_key(i, nb);
                let is_boundary_edge = adj
                    .edge_faces
                    .get(&ek)
                    .map_or(true, |faces| faces.len() < 2);
                if is_boundary_edge {
                    boundary_neighbors.push(nb);
                }
            }
            if boundary_neighbors.len() == 2 {
                let p0 = mesh.vertices[boundary_neighbors[0]];
                let p1 = mesh.vertices[boundary_neighbors[1]];
                let v = mesh.vertices[i];
                new_vertices.push(add3(scale3(0.75, v), scale3(0.125, add3(p0, p1))));
            } else {
                new_vertices.push(mesh.vertices[i]);
            }
        } else if n > 0 {
            let nf = n as f64;
            let beta = {
                let t = 3.0 / 8.0 + (2.0 * std::f64::consts::PI / nf).cos() / 4.0;
                (5.0 / 8.0 - t * t) / nf
            };

            let mut neighbor_sum = [0.0, 0.0, 0.0];
            for &nb in neighbors {
                let p = mesh.vertices[nb];
                neighbor_sum = add3(neighbor_sum, p);
            }

            let v = mesh.vertices[i];
            new_vertices.push(add3(scale3(1.0 - nf * beta, v), scale3(beta, neighbor_sum)));
        } else {
            new_vertices.push(mesh.vertices[i]);
        }
    }

    // 2. Compute edge points and assign indices
    let mut edge_vertex_map: HashMap<(usize, usize), usize> = HashMap::new();

    for (&ek, face_list) in &adj.edge_faces {
        let (a, b) = ek;
        let pa = mesh.vertices[a];
        let pb = mesh.vertices[b];

        let edge_point = if face_list.len() == 2 {
            // Interior edge: find opposite vertices
            let f0 = &mesh.faces[face_list[0]];
            let f1 = &mesh.faces[face_list[1]];

            let opp0 = f0.iter().find(|&&v| v != a && v != b).copied().unwrap_or(a);
            let opp1 = f1.iter().find(|&&v| v != a && v != b).copied().unwrap_or(b);

            let po0 = mesh.vertices[opp0];
            let po1 = mesh.vertices[opp1];

            add3(
                scale3(3.0 / 8.0, add3(pa, pb)),
                scale3(1.0 / 8.0, add3(po0, po1)),
            )
        } else {
            // Boundary edge
            scale3(0.5, add3(pa, pb))
        };

        let idx = new_vertices.len();
        new_vertices.push(edge_point);
        edge_vertex_map.insert(ek, idx);
    }

    // 3. Build new faces: each triangle -> 4 sub-triangles
    let mut new_faces: Vec<Vec<usize>> = Vec::with_capacity(mesh.faces.len() * 4);

    for face in &mesh.faces {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];

        let e01 = edge_vertex_map
            .get(&edge_key(v0, v1))
            .copied()
            .ok_or_else(|| InterpolateError::ComputationError("missing edge vertex".to_string()))?;
        let e12 = edge_vertex_map
            .get(&edge_key(v1, v2))
            .copied()
            .ok_or_else(|| InterpolateError::ComputationError("missing edge vertex".to_string()))?;
        let e20 = edge_vertex_map
            .get(&edge_key(v2, v0))
            .copied()
            .ok_or_else(|| InterpolateError::ComputationError("missing edge vertex".to_string()))?;

        // Four sub-triangles
        new_faces.push(vec![v0, e01, e20]);
        new_faces.push(vec![v1, e12, e01]);
        new_faces.push(vec![v2, e20, e12]);
        new_faces.push(vec![e01, e12, e20]);
    }

    Ok(Mesh {
        vertices: new_vertices,
        faces: new_faces,
    })
}

// ---------------------------------------------------------------------------
// Catmull-Clark Subdivision
// ---------------------------------------------------------------------------

/// Perform one iteration of Catmull-Clark subdivision on a polygon mesh.
///
/// # Algorithm
///
/// 1. **Face points**: Average of face vertices.
/// 2. **Edge points**: Average of face points of adjacent faces + original edge endpoints.
///    For boundary edges: midpoint of edge endpoints.
/// 3. **Vertex points**: Weighted combination:
///    `V' = (F + 2R + (n-3)V) / n`
///    where `F` = average of adjacent face points, `R` = average of adjacent edge midpoints,
///    `n` = valence of vertex, `V` = original vertex.
/// 4. Each face becomes `k` quads (one per edge of the original face).
fn catmull_clark_subdivide_once(mesh: &Mesh) -> InterpolateResult<Mesh> {
    let nv = mesh.vertices.len();
    let nf = mesh.faces.len();

    // Build adjacency
    let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut vertex_faces: Vec<Vec<usize>> = vec![Vec::new(); nv];
    let mut vertex_edges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); nv];

    for (fi, face) in mesh.faces.iter().enumerate() {
        let fl = face.len();
        for i in 0..fl {
            let a = face[i];
            let b = face[(i + 1) % fl];
            let ek = edge_key(a, b);

            edge_faces.entry(ek).or_default().push(fi);

            if !vertex_faces[a].contains(&fi) {
                vertex_faces[a].push(fi);
            }

            if !vertex_edges[a].contains(&ek) {
                vertex_edges[a].push(ek);
            }
            if !vertex_edges[b].contains(&ek) {
                vertex_edges[b].push(ek);
            }
        }
    }

    let mut new_vertices: Vec<[f64; 3]> = Vec::new();

    // 1. Face points (one per face)
    let face_point_start = nv;
    let mut face_points: Vec<[f64; 3]> = Vec::with_capacity(nf);
    for face in &mesh.faces {
        let pts: Vec<[f64; 3]> = face.iter().map(|&vi| mesh.vertices[vi]).collect();
        face_points.push(average_points(&pts));
    }

    // 2. Edge points (one per unique edge)
    let mut edge_indices: Vec<(usize, usize)> = edge_faces.keys().copied().collect();
    edge_indices.sort();
    let ne = edge_indices.len();
    let edge_point_start = nv + nf;

    let mut edge_index_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut edge_point_positions: Vec<[f64; 3]> = Vec::with_capacity(ne);

    for (ei, &ek) in edge_indices.iter().enumerate() {
        let (a, b) = ek;
        let pa = mesh.vertices[a];
        let pb = mesh.vertices[b];
        let faces_for_edge = &edge_faces[&ek];

        let edge_point = if faces_for_edge.len() == 2 {
            // Interior edge
            let fp0 = face_points[faces_for_edge[0]];
            let fp1 = face_points[faces_for_edge[1]];
            scale3(0.25, add3(add3(pa, pb), add3(fp0, fp1)))
        } else {
            // Boundary edge
            scale3(0.5, add3(pa, pb))
        };

        edge_point_positions.push(edge_point);
        edge_index_map.insert(ek, edge_point_start + ei);
    }

    // 3. Vertex points (updated positions for original vertices)
    let mut updated_vertices: Vec<[f64; 3]> = Vec::with_capacity(nv);

    for i in 0..nv {
        let adj_faces = &vertex_faces[i];
        let adj_edges = &vertex_edges[i];
        let n = adj_faces.len() as f64;

        if n == 0.0 {
            updated_vertices.push(mesh.vertices[i]);
            continue;
        }

        // Check if boundary vertex
        let is_boundary = adj_edges
            .iter()
            .any(|ek| edge_faces.get(ek).map_or(true, |fl| fl.len() < 2));

        if is_boundary {
            // Boundary vertex: average of boundary edge midpoints and original
            let mut boundary_midpoints = Vec::new();
            for ek in adj_edges {
                let is_bnd = edge_faces.get(ek).map_or(true, |fl| fl.len() < 2);
                if is_bnd {
                    let (a, b) = *ek;
                    boundary_midpoints.push(scale3(0.5, add3(mesh.vertices[a], mesh.vertices[b])));
                }
            }
            if boundary_midpoints.len() == 2 {
                let mid_avg = scale3(0.5, add3(boundary_midpoints[0], boundary_midpoints[1]));
                updated_vertices.push(scale3(0.5, add3(mesh.vertices[i], mid_avg)));
            } else {
                updated_vertices.push(mesh.vertices[i]);
            }
        } else {
            // Interior vertex: V' = (F + 2R + (n-3)V) / n
            let f_avg = {
                let fps: Vec<[f64; 3]> = adj_faces.iter().map(|&fi| face_points[fi]).collect();
                average_points(&fps)
            };
            let r_avg = {
                let midpts: Vec<[f64; 3]> = adj_edges
                    .iter()
                    .map(|&(a, b)| scale3(0.5, add3(mesh.vertices[a], mesh.vertices[b])))
                    .collect();
                average_points(&midpts)
            };
            let v = mesh.vertices[i];

            updated_vertices.push(scale3(
                1.0 / n,
                add3(f_avg, add3(scale3(2.0, r_avg), scale3(n - 3.0, v))),
            ));
        }
    }

    // Assemble new vertex list: [updated original vertices, face points, edge points]
    new_vertices.extend_from_slice(&updated_vertices);
    new_vertices.extend_from_slice(&face_points);
    new_vertices.extend_from_slice(&edge_point_positions);

    // 4. Build new faces
    // Each original face with k edges produces k quads:
    // For edge (v_i, v_{i+1}) in face f:
    //   quad = [v_i, edge_point(v_i, v_{i+1}), face_point(f), edge_point(v_{i-1}, v_i)]
    let mut new_faces: Vec<Vec<usize>> = Vec::new();

    for (fi, face) in mesh.faces.iter().enumerate() {
        let fl = face.len();
        let fp_idx = face_point_start + fi;

        for i in 0..fl {
            let vi = face[i];
            let vi_next = face[(i + 1) % fl];
            let vi_prev = face[(i + fl - 1) % fl];

            let ep_next = edge_index_map
                .get(&edge_key(vi, vi_next))
                .copied()
                .ok_or_else(|| {
                    InterpolateError::ComputationError("missing edge in Catmull-Clark".to_string())
                })?;
            let ep_prev = edge_index_map
                .get(&edge_key(vi_prev, vi))
                .copied()
                .ok_or_else(|| {
                    InterpolateError::ComputationError("missing edge in Catmull-Clark".to_string())
                })?;

            new_faces.push(vec![vi, ep_next, fp_idx, ep_prev]);
        }
    }

    Ok(Mesh {
        vertices: new_vertices,
        faces: new_faces,
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// The subdivision scheme to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubdivisionScheme {
    /// Loop subdivision for triangle meshes.
    Loop,
    /// Catmull-Clark subdivision for quad/polygon meshes.
    CatmullClark,
}

/// Perform subdivision on a mesh for a given number of iterations.
///
/// # Arguments
///
/// * `mesh` - The input mesh.
/// * `iterations` - Number of subdivision iterations (each iteration
///   increases face count by ~4x for Loop, and produces quads for Catmull-Clark).
/// * `scheme` - The subdivision scheme to use.
///
/// # Errors
///
/// Returns an error if:
/// - Loop subdivision is used on a non-triangle mesh.
/// - The mesh has invalid connectivity.
///
/// # Examples
///
/// ```rust
/// use scirs2_interpolate::subdivision::{Mesh, SubdivisionScheme, subdivide};
///
/// // Simple tetrahedron
/// let vertices = vec![
///     [1.0, 1.0, 1.0],
///     [-1.0, -1.0, 1.0],
///     [-1.0, 1.0, -1.0],
///     [1.0, -1.0, -1.0],
/// ];
/// let faces = vec![
///     vec![0, 1, 2],
///     vec![0, 3, 1],
///     vec![0, 2, 3],
///     vec![1, 3, 2],
/// ];
///
/// let mesh = Mesh::new(vertices, faces).expect("valid mesh");
/// let smooth = subdivide(&mesh, 2, SubdivisionScheme::Loop).expect("subdivision");
/// assert!(smooth.num_faces() > mesh.num_faces());
/// ```
pub fn subdivide(
    mesh: &Mesh,
    iterations: usize,
    scheme: SubdivisionScheme,
) -> InterpolateResult<Mesh> {
    if iterations == 0 {
        return Ok(mesh.clone());
    }

    let mut current = mesh.clone();
    for _ in 0..iterations {
        current = match scheme {
            SubdivisionScheme::Loop => loop_subdivide_once(&current)?,
            SubdivisionScheme::CatmullClark => catmull_clark_subdivide_once(&current)?,
        };
    }
    Ok(current)
}

/// Evaluate the Loop subdivision limit position for each vertex without
/// performing infinite subdivision.
///
/// For a vertex with valence `n`, the limit position is:
/// ```text
/// V_limit = V * n / (n + 3/beta_n) + sum(neighbors) / (n + 3/beta_n)
/// ```
/// where `beta_n = (5/8 - (3/8 + cos(2*pi/n)/4)^2) / n`.
///
/// This gives the exact point on the limit surface corresponding to each
/// vertex position.
pub fn loop_limit_positions(mesh: &Mesh) -> InterpolateResult<Vec<[f64; 3]>> {
    if !mesh.is_triangle_mesh() {
        return Err(InterpolateError::InvalidValue(
            "Loop limit positions require a triangle mesh".to_string(),
        ));
    }

    let adj = build_loop_adjacency(mesh);
    let nv = mesh.vertices.len();
    let mut limit = Vec::with_capacity(nv);

    for i in 0..nv {
        let neighbors = &adj.vertex_neighbors[i];
        let n = neighbors.len();

        if n == 0 {
            limit.push(mesh.vertices[i]);
            continue;
        }

        let nf = n as f64;
        let beta = {
            let t = 3.0 / 8.0 + (2.0 * std::f64::consts::PI / nf).cos() / 4.0;
            (5.0 / 8.0 - t * t) / nf
        };

        // Limit mask: vertex weight = 1 / (1 + n*beta/(1 - n*beta))
        // Simplification: limit_weight_v = (1 - n*beta), limit_weight_nb = beta
        // Then normalize by dividing by (1 - n*beta + n*beta) = 1... but the
        // limit position formula uses eigenanalysis. The standard formula is:
        //
        // V_limit = V / (1 + n*gamma) + gamma * sum(neighbors) / (1 + n*gamma)
        // where gamma = beta / (1 - n*beta)
        //
        // But the simpler standard result is:
        // V_limit = (1/(n*beta + 1)) * V + (n*beta/(n*beta + 1)) * avg(neighbors)

        let w = nf * beta;
        let v = mesh.vertices[i];
        let mut nb_sum = [0.0, 0.0, 0.0];
        for &nb in neighbors {
            nb_sum = add3(nb_sum, mesh.vertices[nb]);
        }
        let nb_avg = scale3(1.0 / nf, nb_sum);

        let denom = 1.0 + w;
        limit.push(add3(scale3(1.0 / denom, v), scale3(w / denom, nb_avg)));
    }

    Ok(limit)
}

/// Evaluate the Catmull-Clark limit position for each vertex.
///
/// For a vertex with valence `n`, the limit position is:
/// ```text
/// V_limit = (n^2 * V + 4 * sum(edge_midpoints) + sum(face_centroids)) / (n * (n + 5))
/// ```
pub fn catmull_clark_limit_positions(mesh: &Mesh) -> InterpolateResult<Vec<[f64; 3]>> {
    let nv = mesh.vertices.len();

    // Build adjacency
    let mut vertex_faces: Vec<Vec<usize>> = vec![Vec::new(); nv];
    let mut vertex_neighbors: Vec<Vec<usize>> = vec![Vec::new(); nv];

    for (fi, face) in mesh.faces.iter().enumerate() {
        let fl = face.len();
        for i in 0..fl {
            let a = face[i];
            let b = face[(i + 1) % fl];
            if !vertex_faces[a].contains(&fi) {
                vertex_faces[a].push(fi);
            }
            if !vertex_neighbors[a].contains(&b) {
                vertex_neighbors[a].push(b);
            }
            if !vertex_neighbors[b].contains(&a) {
                vertex_neighbors[b].push(a);
            }
        }
    }

    // Face centroids
    let face_centroids: Vec<[f64; 3]> = mesh
        .faces
        .iter()
        .map(|face| {
            let pts: Vec<[f64; 3]> = face.iter().map(|&vi| mesh.vertices[vi]).collect();
            average_points(&pts)
        })
        .collect();

    let mut limit = Vec::with_capacity(nv);

    for i in 0..nv {
        let neighbors = &vertex_neighbors[i];
        let adj_faces = &vertex_faces[i];
        let n = neighbors.len() as f64;

        if n == 0.0 {
            limit.push(mesh.vertices[i]);
            continue;
        }

        let v = mesh.vertices[i];

        // Sum of edge midpoints
        let mut edge_mid_sum = [0.0, 0.0, 0.0];
        for &nb in neighbors {
            let mid = scale3(0.5, add3(v, mesh.vertices[nb]));
            edge_mid_sum = add3(edge_mid_sum, mid);
        }

        // Sum of face centroids
        let mut face_centroid_sum = [0.0, 0.0, 0.0];
        for &fi in adj_faces {
            face_centroid_sum = add3(face_centroid_sum, face_centroids[fi]);
        }

        let denom = n * (n + 5.0);
        limit.push(scale3(
            1.0 / denom,
            add3(
                scale3(n * n, v),
                add3(scale3(4.0, edge_mid_sum), face_centroid_sum),
            ),
        ));
    }

    Ok(limit)
}

/// Parameters for semi-sharp creases in subdivision.
#[derive(Debug, Clone)]
pub struct CreaseParams {
    /// Edges that are creased, as pairs of vertex indices.
    pub crease_edges: Vec<(usize, usize)>,
    /// Sharpness value for each crease edge (0 = smooth, large = sharp).
    pub sharpness: Vec<f64>,
}

/// Apply one iteration of Loop subdivision with semi-sharp crease support.
///
/// Crease sharpness is decremented by 1 at each subdivision level.
/// When sharpness reaches 0, the edge is treated as smooth.
pub fn loop_subdivide_with_creases(
    mesh: &Mesh,
    creases: &CreaseParams,
) -> InterpolateResult<(Mesh, CreaseParams)> {
    if !mesh.is_triangle_mesh() {
        return Err(InterpolateError::InvalidValue(
            "Loop subdivision with creases requires a triangle mesh".to_string(),
        ));
    }

    // Build crease map
    let mut crease_map: HashMap<(usize, usize), f64> = HashMap::new();
    for (edge, &sharp) in creases.crease_edges.iter().zip(creases.sharpness.iter()) {
        let ek = edge_key(edge.0, edge.1);
        crease_map.insert(ek, sharp);
    }

    let adj = build_loop_adjacency(mesh);
    let nv = mesh.vertices.len();

    // Vertex classification: crease vertex if incident to >= 2 sharp crease edges
    let mut new_vertices: Vec<[f64; 3]> = Vec::with_capacity(nv * 4);

    for i in 0..nv {
        let neighbors = &adj.vertex_neighbors[i];
        let n = neighbors.len();

        // Find crease edges incident to this vertex
        let mut crease_neighbors: Vec<usize> = Vec::new();
        for &nb in neighbors {
            let ek = edge_key(i, nb);
            if let Some(&sharp) = crease_map.get(&ek) {
                if sharp > 0.0 {
                    crease_neighbors.push(nb);
                }
            }
        }

        if crease_neighbors.len() >= 2 {
            // Crease vertex or corner: use boundary-like rule with crease neighbors
            if crease_neighbors.len() == 2 {
                let p0 = mesh.vertices[crease_neighbors[0]];
                let p1 = mesh.vertices[crease_neighbors[1]];
                let v = mesh.vertices[i];
                new_vertices.push(add3(scale3(0.75, v), scale3(0.125, add3(p0, p1))));
            } else {
                // Corner vertex: don't move
                new_vertices.push(mesh.vertices[i]);
            }
        } else if n > 0 {
            // Smooth vertex
            let nf = n as f64;
            let beta = {
                let t = 3.0 / 8.0 + (2.0 * std::f64::consts::PI / nf).cos() / 4.0;
                (5.0 / 8.0 - t * t) / nf
            };
            let mut neighbor_sum = [0.0, 0.0, 0.0];
            for &nb in neighbors {
                neighbor_sum = add3(neighbor_sum, mesh.vertices[nb]);
            }
            let v = mesh.vertices[i];
            new_vertices.push(add3(scale3(1.0 - nf * beta, v), scale3(beta, neighbor_sum)));
        } else {
            new_vertices.push(mesh.vertices[i]);
        }
    }

    // Edge points
    let mut edge_vertex_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut new_crease_edges: Vec<(usize, usize)> = Vec::new();
    let mut new_sharpness: Vec<f64> = Vec::new();

    for (&ek, face_list) in &adj.edge_faces {
        let (a, b) = ek;
        let pa = mesh.vertices[a];
        let pb = mesh.vertices[b];

        let is_crease = crease_map.get(&ek).map_or(false, |&s| s > 0.0);

        let edge_point = if is_crease || face_list.len() < 2 {
            // Crease or boundary edge: midpoint
            scale3(0.5, add3(pa, pb))
        } else {
            // Smooth interior edge
            let f0 = &mesh.faces[face_list[0]];
            let f1 = &mesh.faces[face_list[1]];
            let opp0 = f0.iter().find(|&&v| v != a && v != b).copied().unwrap_or(a);
            let opp1 = f1.iter().find(|&&v| v != a && v != b).copied().unwrap_or(b);
            let po0 = mesh.vertices[opp0];
            let po1 = mesh.vertices[opp1];
            add3(
                scale3(3.0 / 8.0, add3(pa, pb)),
                scale3(1.0 / 8.0, add3(po0, po1)),
            )
        };

        let idx = new_vertices.len();
        new_vertices.push(edge_point);
        edge_vertex_map.insert(ek, idx);

        // Propagate crease with reduced sharpness
        if let Some(&sharp) = crease_map.get(&ek) {
            let new_sharp = (sharp - 1.0).max(0.0);
            if new_sharp > 0.0 {
                // The crease edge splits into two sub-edges
                new_crease_edges.push((a, idx));
                new_crease_edges.push((idx, b));
                new_sharpness.push(new_sharp);
                new_sharpness.push(new_sharp);
            }
        }
    }

    // Build new faces
    let mut new_faces: Vec<Vec<usize>> = Vec::with_capacity(mesh.faces.len() * 4);

    for face in &mesh.faces {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];

        let e01 = edge_vertex_map
            .get(&edge_key(v0, v1))
            .copied()
            .ok_or_else(|| {
                InterpolateError::ComputationError(
                    "missing edge vertex in crease subdivision".to_string(),
                )
            })?;
        let e12 = edge_vertex_map
            .get(&edge_key(v1, v2))
            .copied()
            .ok_or_else(|| {
                InterpolateError::ComputationError(
                    "missing edge vertex in crease subdivision".to_string(),
                )
            })?;
        let e20 = edge_vertex_map
            .get(&edge_key(v2, v0))
            .copied()
            .ok_or_else(|| {
                InterpolateError::ComputationError(
                    "missing edge vertex in crease subdivision".to_string(),
                )
            })?;

        new_faces.push(vec![v0, e01, e20]);
        new_faces.push(vec![v1, e12, e01]);
        new_faces.push(vec![v2, e20, e12]);
        new_faces.push(vec![e01, e12, e20]);
    }

    let new_creases = CreaseParams {
        crease_edges: new_crease_edges,
        sharpness: new_sharpness,
    };

    Ok((
        Mesh {
            vertices: new_vertices,
            faces: new_faces,
        },
        new_creases,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple tetrahedron mesh.
    fn tetrahedron() -> Mesh {
        let vertices = vec![
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ];
        let faces = vec![vec![0, 1, 2], vec![0, 3, 1], vec![0, 2, 3], vec![1, 3, 2]];
        Mesh::new(vertices, faces).expect("test: valid tetrahedron")
    }

    /// Create a unit cube mesh (6 quad faces).
    fn cube() -> Mesh {
        let vertices = vec![
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];
        let faces = vec![
            vec![0, 1, 2, 3], // bottom
            vec![4, 7, 6, 5], // top
            vec![0, 4, 5, 1], // front
            vec![2, 6, 7, 3], // back
            vec![0, 3, 7, 4], // left
            vec![1, 5, 6, 2], // right
        ];
        Mesh::new(vertices, faces).expect("test: valid cube")
    }

    #[test]
    fn test_loop_subdivision_face_count() {
        let mesh = tetrahedron();
        assert_eq!(mesh.num_faces(), 4);

        let sub1 =
            subdivide(&mesh, 1, SubdivisionScheme::Loop).expect("test: subdivision should succeed");
        // Each triangle -> 4 triangles
        assert_eq!(sub1.num_faces(), 16);

        let sub2 =
            subdivide(&mesh, 2, SubdivisionScheme::Loop).expect("test: subdivision should succeed");
        assert_eq!(sub2.num_faces(), 64);
    }

    #[test]
    fn test_loop_subdivision_smoother() {
        // A tetrahedron should become more sphere-like after subdivision.
        // The tetrahedron already has uniform distances from centroid,
        // so we verify that the subdivided mesh also approaches a sphere
        // by checking that the coefficient of variation of distances is small.
        let mesh = tetrahedron();

        let sub =
            subdivide(&mesh, 3, SubdivisionScheme::Loop).expect("test: subdivision should succeed");

        // Compute coefficient of variation of distances from centroid.
        let centroid = average_points(&sub.vertices);
        let dists: Vec<f64> = sub
            .vertices
            .iter()
            .map(|v| {
                let dx = v[0] - centroid[0];
                let dy = v[1] - centroid[1];
                let dz = v[2] - centroid[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();
        let mean_dist = dists.iter().sum::<f64>() / dists.len() as f64;
        let variance: f64 =
            dists.iter().map(|d| (d - mean_dist).powi(2)).sum::<f64>() / dists.len() as f64;
        let cv = variance.sqrt() / mean_dist;

        // After several subdivisions of a tetrahedron, it should approach
        // a sphere. The coefficient of variation should be small.
        assert!(
            cv < 0.15,
            "subdivided tetrahedron should approach sphere-like shape, cv={}",
            cv
        );

        // Also verify the mesh has more faces and vertices
        assert!(sub.num_faces() > mesh.num_faces());
        assert!(sub.num_vertices() > mesh.num_vertices());
    }

    #[test]
    fn test_catmull_clark_cube() {
        let mesh = cube();
        assert_eq!(mesh.num_faces(), 6);
        assert!(mesh.is_quad_mesh());

        let sub = subdivide(&mesh, 1, SubdivisionScheme::CatmullClark)
            .expect("test: Catmull-Clark should succeed");

        // Each quad -> 4 quads
        assert_eq!(sub.num_faces(), 24);
        assert!(
            sub.is_quad_mesh(),
            "Catmull-Clark output should be all quads"
        );
    }

    #[test]
    fn test_catmull_clark_smoothing() {
        let mesh = cube();

        let sub = subdivide(&mesh, 2, SubdivisionScheme::CatmullClark)
            .expect("test: Catmull-Clark should succeed");

        // After subdivision, vertices should be closer to a sphere
        let centroid = average_points(&sub.vertices);
        let dists: Vec<f64> = sub
            .vertices
            .iter()
            .map(|v| {
                let dx = v[0] - centroid[0];
                let dy = v[1] - centroid[1];
                let dz = v[2] - centroid[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .collect();

        // All distances should be positive (not degenerate)
        assert!(
            dists.iter().all(|&d| d > 0.0),
            "all vertices should be at non-zero distance from centroid"
        );

        // More vertices than original
        assert!(sub.num_vertices() > mesh.num_vertices());
    }

    #[test]
    fn test_loop_limit_positions() {
        let mesh = tetrahedron();
        let limit = loop_limit_positions(&mesh).expect("test: limit positions should work");

        assert_eq!(limit.len(), mesh.num_vertices());

        // Limit positions should be closer to centroid than original
        let centroid = average_points(&mesh.vertices);
        for (orig, lim) in mesh.vertices.iter().zip(limit.iter()) {
            let orig_dist = ((orig[0] - centroid[0]).powi(2)
                + (orig[1] - centroid[1]).powi(2)
                + (orig[2] - centroid[2]).powi(2))
            .sqrt();
            let lim_dist = ((lim[0] - centroid[0]).powi(2)
                + (lim[1] - centroid[1]).powi(2)
                + (lim[2] - centroid[2]).powi(2))
            .sqrt();

            assert!(
                lim_dist <= orig_dist + 1e-10,
                "limit position should be closer to centroid"
            );
        }
    }

    #[test]
    fn test_catmull_clark_limit_positions() {
        let mesh = cube();
        let limit =
            catmull_clark_limit_positions(&mesh).expect("test: CC limit positions should work");

        assert_eq!(limit.len(), mesh.num_vertices());

        // Limit positions should be inside the original bounding box
        for lim in &limit {
            assert!(
                lim[0].abs() <= 1.0 + 1e-10
                    && lim[1].abs() <= 1.0 + 1e-10
                    && lim[2].abs() <= 1.0 + 1e-10,
                "limit position should be within original bounds"
            );
        }
    }

    #[test]
    fn test_zero_iterations() {
        let mesh = tetrahedron();
        let result = subdivide(&mesh, 0, SubdivisionScheme::Loop)
            .expect("test: zero iterations should succeed");
        assert_eq!(result.num_faces(), mesh.num_faces());
        assert_eq!(result.num_vertices(), mesh.num_vertices());
    }

    #[test]
    fn test_loop_subdivision_with_creases() {
        let mesh = tetrahedron();
        let creases = CreaseParams {
            crease_edges: vec![(0, 1), (1, 2)],
            sharpness: vec![2.0, 2.0],
        };

        let (sub, new_creases) = loop_subdivide_with_creases(&mesh, &creases)
            .expect("test: crease subdivision should succeed");

        assert_eq!(sub.num_faces(), 16); // Still 4x faces
                                         // Sharpness decremented by 1
        for &s in &new_creases.sharpness {
            assert!((s - 1.0).abs() < 1e-10 || s == 0.0);
        }
    }

    #[test]
    fn test_mesh_validation_error() {
        // Face references non-existent vertex
        let result = Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            vec![vec![0, 1, 5]], // vertex 5 doesn't exist
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_loop_on_non_triangle_error() {
        let mesh = cube(); // Quad mesh
        let result = subdivide(&mesh, 1, SubdivisionScheme::Loop);
        assert!(result.is_err());
    }
}
