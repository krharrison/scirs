//! 3D Surface Extraction
//!
//! Provides algorithms for extracting and analyzing surfaces from volumetric
//! data:
//!
//! * [`marching_cubes`]          – Lorensen–Cline marching cubes surface mesh
//! * [`estimate_surface_normals`]– Area-weighted normal estimation from a triangle mesh
//! * [`isosurface_extraction`]   – High-level wrapper: volume → mesh with normals
//!
//! # Mesh Representation
//!
//! Surfaces are represented using [`SurfaceMesh`] which stores:
//!
//! - `vertices`: list of `[x, y, z]` floating-point positions
//! - `triangles`: list of `[i0, i1, i2]` index triples into `vertices`
//! - `normals`: per-vertex normals (populated by `estimate_surface_normals`)

use crate::error::{NdimageError, NdimageResult};
use scirs2_core::ndarray::Array3;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A 3-D surface mesh produced by marching cubes.
#[derive(Debug, Clone)]
pub struct SurfaceMesh {
    /// Vertex positions as `[x, y, z]` triples (in voxel coordinates).
    pub vertices: Vec<[f64; 3]>,
    /// Triangle faces as triples of vertex indices into `vertices`.
    pub triangles: Vec<[usize; 3]>,
    /// Per-vertex normals (unit vectors); populated by
    /// [`estimate_surface_normals`].  Empty until that function is called.
    pub normals: Vec<[f64; 3]>,
}

impl SurfaceMesh {
    /// Create an empty mesh.
    pub fn new() -> Self {
        SurfaceMesh {
            vertices: Vec::new(),
            triangles: Vec::new(),
            normals: Vec::new(),
        }
    }

    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of triangle faces.
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Approximate surface area in squared voxel units.
    pub fn surface_area(&self) -> f64 {
        self.triangles.iter().map(|&[i0, i1, i2]| {
            let v0 = self.vertices[i0];
            let v1 = self.vertices[i1];
            let v2 = self.vertices[i2];
            triangle_area(v0, v1, v2)
        }).sum()
    }
}

// ---------------------------------------------------------------------------
// Marching cubes
// ---------------------------------------------------------------------------

/// Extract an isosurface from a scalar volume at the given `level` using the
/// Lorensen–Cline marching cubes algorithm.
///
/// The function processes each `2×2×2` cube of voxels, classifies it via the
/// 256-case lookup table, and emits triangles whose vertices are linearly
/// interpolated along cube edges.
///
/// # Arguments
///
/// * `volume`  – Scalar 3-D array with shape `(depth, height, width)`.
/// * `level`   – Isovalue defining the surface.
///
/// # Returns
///
/// A [`SurfaceMesh`] without normals (call [`estimate_surface_normals`] to
/// populate them).
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when the volume is smaller than
/// `2×2×2`.
pub fn marching_cubes(volume: &Array3<f64>, level: f64) -> NdimageResult<SurfaceMesh> {
    let sh = volume.shape();
    let (nz, ny, nx) = (sh[0], sh[1], sh[2]);
    if nz < 2 || ny < 2 || nx < 2 {
        return Err(NdimageError::InvalidInput(
            "Volume must be at least 2×2×2 for marching cubes".to_string(),
        ));
    }

    let mut mesh = SurfaceMesh::new();
    // Cache of edge-midpoint vertex indices to avoid duplicate vertices.
    // Key: (cube_flat_index, edge_index) → vertex index
    let mut edge_cache: HashMap<(usize, usize), usize> = HashMap::new();

    for z in 0..(nz - 1) {
        for y in 0..(ny - 1) {
            for x in 0..(nx - 1) {
                // Cube corner values (v0..v7 in Lorensen's ordering)
                let corners = cube_corners(volume, z, y, x);
                let cube_index = classify_cube(&corners, level);

                if cube_index == 0 || cube_index == 255 {
                    continue; // all inside or all outside
                }

                let flat = z * (ny - 1) * (nx - 1) + y * (nx - 1) + x;

                // Compute edge vertex positions (up to 12 edges)
                let edge_verts = compute_edge_verts(&corners, level, z, y, x);

                // Emit triangles from the lookup table
                let tris = &TRIANGLE_TABLE[cube_index];
                let mut i = 0;
                while i < tris.len() && tris[i] != -1 {
                    let e0 = tris[i] as usize;
                    let e1 = tris[i + 1] as usize;
                    let e2 = tris[i + 2] as usize;

                    let v0 = get_or_insert_vertex(&mut mesh, &mut edge_cache, flat, e0, edge_verts[e0]);
                    let v1 = get_or_insert_vertex(&mut mesh, &mut edge_cache, flat, e1, edge_verts[e1]);
                    let v2 = get_or_insert_vertex(&mut mesh, &mut edge_cache, flat, e2, edge_verts[e2]);

                    // Degenerate triangle guard
                    if v0 != v1 && v1 != v2 && v0 != v2 {
                        mesh.triangles.push([v0, v1, v2]);
                    }
                    i += 3;
                }
            }
        }
    }

    Ok(mesh)
}

fn get_or_insert_vertex(
    mesh: &mut SurfaceMesh,
    cache: &mut HashMap<(usize, usize), usize>,
    flat: usize,
    edge: usize,
    pos: [f64; 3],
) -> usize {
    *cache.entry((flat, edge)).or_insert_with(|| {
        let idx = mesh.vertices.len();
        mesh.vertices.push(pos);
        idx
    })
}

// ---------------------------------------------------------------------------
// Surface normal estimation
// ---------------------------------------------------------------------------

/// Estimate per-vertex normals for a surface mesh using area-weighted averaging
/// of adjacent triangle normals.
///
/// The normals are stored in `mesh.normals` (one per vertex).  Existing normals
/// are overwritten.
///
/// # Errors
///
/// Returns [`NdimageError::InvalidInput`] when the mesh has no vertices.
pub fn estimate_surface_normals(mesh: &mut SurfaceMesh) -> NdimageResult<()> {
    if mesh.vertices.is_empty() {
        return Err(NdimageError::InvalidInput(
            "Mesh has no vertices".to_string(),
        ));
    }

    let nv = mesh.vertices.len();
    let mut normals = vec![[0.0_f64; 3]; nv];

    for &[i0, i1, i2] in &mesh.triangles {
        let v0 = mesh.vertices[i0];
        let v1 = mesh.vertices[i1];
        let v2 = mesh.vertices[i2];

        let e1 = sub3(v1, v0);
        let e2 = sub3(v2, v0);
        let cross = cross3(e1, e2);
        let area = mag3(cross);

        // Weight the face normal by the triangle's area
        for k in 0..3 {
            normals[i0][k] += cross[k] * area;
            normals[i1][k] += cross[k] * area;
            normals[i2][k] += cross[k] * area;
        }
    }

    // Normalize per-vertex normals
    for n in normals.iter_mut() {
        let m = mag3(*n);
        if m > 1e-14 {
            n[0] /= m;
            n[1] /= m;
            n[2] /= m;
        }
    }

    mesh.normals = normals;
    Ok(())
}

// ---------------------------------------------------------------------------
// High-level isosurface extraction
// ---------------------------------------------------------------------------

/// Extract an isosurface from a scalar volume and compute vertex normals.
///
/// This is a convenience wrapper around [`marching_cubes`] followed by
/// [`estimate_surface_normals`].
///
/// # Arguments
///
/// * `volume` – Scalar 3-D volume.
/// * `level`  – Isovalue.
///
/// # Returns
///
/// A [`SurfaceMesh`] with vertices, triangles, and per-vertex normals.
///
/// # Errors
///
/// Forwards any errors from the underlying functions.
pub fn isosurface_extraction(volume: &Array3<f64>, level: f64) -> NdimageResult<SurfaceMesh> {
    let mut mesh = marching_cubes(volume, level)?;
    if mesh.vertex_count() > 0 {
        estimate_surface_normals(&mut mesh)?;
    }
    Ok(mesh)
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

#[inline]
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn mag3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn triangle_area(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> f64 {
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    mag3(cross3(e1, e2)) * 0.5
}

// ---------------------------------------------------------------------------
// Marching cubes internals
// ---------------------------------------------------------------------------

/// Read the 8 corner values of the cube at `(z, y, x)` in Lorensen ordering.
///
/// Corner numbering (Lorensen 1987):
/// ```text
///   4----5
///   |  7-|--6
///   | /  | /
///   0----1
///     3----2  (bottom face: y=0; top face: y=1)
/// ```
/// Mapping in `(dz, dy, dx)`:
///   0: (0,0,0), 1: (0,0,1), 2: (0,1,1), 3: (0,1,0),
///   4: (1,0,0), 5: (1,0,1), 6: (1,1,1), 7: (1,1,0)
fn cube_corners(volume: &Array3<f64>, z: usize, y: usize, x: usize) -> [f64; 8] {
    [
        volume[[z, y, x]],
        volume[[z, y, x + 1]],
        volume[[z, y + 1, x + 1]],
        volume[[z, y + 1, x]],
        volume[[z + 1, y, x]],
        volume[[z + 1, y, x + 1]],
        volume[[z + 1, y + 1, x + 1]],
        volume[[z + 1, y + 1, x]],
    ]
}

/// Classify a cube: build an 8-bit index where bit `i` is set iff corner `i`
/// is at or above `level`.
fn classify_cube(corners: &[f64; 8], level: f64) -> usize {
    let mut idx = 0_usize;
    for (i, &v) in corners.iter().enumerate() {
        if v >= level {
            idx |= 1 << i;
        }
    }
    idx
}

/// Compute the positions of edge-intersection vertices for a cube at `(z, y, x)`.
///
/// Returns an array of 12 positions (one per edge).  Edges where neither
/// endpoint crosses the isosurface get a zeroed-out position (they will never
/// be referenced by the triangle table).
fn compute_edge_verts(
    corners: &[f64; 8],
    level: f64,
    z: usize,
    y: usize,
    x: usize,
) -> [[f64; 3]; 12] {
    // Corner positions in (z, y, x) space
    let pos: [[f64; 3]; 8] = [
        [z as f64, y as f64, x as f64],
        [z as f64, y as f64, x as f64 + 1.0],
        [z as f64, y as f64 + 1.0, x as f64 + 1.0],
        [z as f64, y as f64 + 1.0, x as f64],
        [z as f64 + 1.0, y as f64, x as f64],
        [z as f64 + 1.0, y as f64, x as f64 + 1.0],
        [z as f64 + 1.0, y as f64 + 1.0, x as f64 + 1.0],
        [z as f64 + 1.0, y as f64 + 1.0, x as f64],
    ];

    // Edge list: pairs of corner indices (Lorensen standard)
    const EDGES: [(usize, usize); 12] = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];

    let mut result = [[0.0_f64; 3]; 12];
    for (ei, &(a, b)) in EDGES.iter().enumerate() {
        let va = corners[a];
        let vb = corners[b];
        // Interpolate only when the edge actually crosses the isosurface
        let t = if (vb - va).abs() > 1e-15 {
            (level - va) / (vb - va)
        } else {
            0.5
        };
        let t = t.clamp(0.0, 1.0);
        result[ei] = [
            pos[a][0] + t * (pos[b][0] - pos[a][0]),
            pos[a][1] + t * (pos[b][1] - pos[a][1]),
            pos[a][2] + t * (pos[b][2] - pos[a][2]),
        ];
    }
    result
}

// ---------------------------------------------------------------------------
// Triangle table (256 entries, up to 16 i8 values per entry; -1 = sentinel)
// ---------------------------------------------------------------------------
//
// This is the standard Lorensen–Cline lookup table.  Only a subset of the 256
// cases contains triangles; the rest are all-(-1).

const TRIANGLE_TABLE: [&[i8]; 256] = [
    &[],                                           // 0
    &[0, 8, 3, -1],
    &[0, 1, 9, -1],
    &[1, 8, 3, 9, 8, 1, -1],
    &[1, 2, 10, -1],
    &[0, 8, 3, 1, 2, 10, -1],
    &[9, 2, 10, 0, 2, 9, -1],
    &[2, 8, 3, 2, 10, 8, 10, 9, 8, -1],
    &[3, 11, 2, -1],
    &[0, 11, 2, 8, 11, 0, -1],
    &[1, 9, 0, 2, 3, 11, -1],                     // 10
    &[1, 11, 2, 1, 9, 11, 9, 8, 11, -1],
    &[3, 10, 1, 11, 10, 3, -1],
    &[0, 10, 1, 0, 8, 10, 8, 11, 10, -1],
    &[3, 9, 0, 3, 11, 9, 11, 10, 9, -1],
    &[9, 8, 10, 10, 8, 11, -1],
    &[4, 7, 8, -1],
    &[4, 3, 0, 7, 3, 4, -1],
    &[0, 1, 9, 8, 4, 7, -1],
    &[4, 1, 9, 4, 7, 1, 7, 3, 1, -1],
    &[1, 2, 10, 8, 4, 7, -1],                     // 20
    &[3, 4, 7, 3, 0, 4, 1, 2, 10, -1],
    &[9, 2, 10, 9, 0, 2, 8, 4, 7, -1],
    &[2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1],
    &[8, 4, 7, 3, 11, 2, -1],
    &[11, 4, 7, 11, 2, 4, 2, 0, 4, -1],
    &[9, 0, 1, 8, 4, 7, 2, 3, 11, -1],
    &[4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1],
    &[3, 10, 1, 3, 11, 10, 7, 8, 4, -1],
    &[1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1],
    &[4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1], // 30
    &[4, 7, 11, 4, 11, 9, 9, 11, 10, -1],
    &[9, 5, 4, -1],
    &[9, 5, 4, 0, 8, 3, -1],
    &[0, 5, 4, 1, 5, 0, -1],
    &[8, 5, 4, 8, 3, 5, 3, 1, 5, -1],
    &[1, 2, 10, 9, 5, 4, -1],
    &[3, 0, 8, 1, 2, 10, 4, 9, 5, -1],
    &[5, 2, 10, 5, 4, 2, 4, 0, 2, -1],
    &[2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1],
    &[9, 5, 4, 2, 3, 11, -1],                     // 40
    &[0, 11, 2, 0, 8, 11, 4, 9, 5, -1],
    &[0, 5, 4, 0, 1, 5, 2, 3, 11, -1],
    &[2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1],
    &[10, 3, 11, 10, 1, 3, 9, 5, 4, -1],
    &[4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1],
    &[5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1],
    &[5, 4, 8, 5, 8, 10, 10, 8, 11, -1],
    &[9, 7, 8, 5, 7, 9, -1],
    &[9, 3, 0, 9, 5, 3, 5, 7, 3, -1],
    &[0, 7, 8, 0, 1, 7, 1, 5, 7, -1],             // 50
    &[1, 5, 3, 3, 5, 7, -1],
    &[9, 7, 8, 9, 5, 7, 10, 1, 2, -1],
    &[10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1],
    &[8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1],
    &[2, 10, 5, 2, 5, 3, 3, 5, 7, -1],
    &[7, 9, 5, 7, 8, 9, 3, 11, 2, -1],
    &[9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1],
    &[2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1],
    &[11, 2, 1, 11, 1, 7, 7, 1, 5, -1],
    &[9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1], // 60
    &[5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    &[11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    &[11, 10, 5, 7, 11, 5, -1],
    &[10, 6, 5, -1],
    &[0, 8, 3, 5, 10, 6, -1],
    &[9, 0, 1, 5, 10, 6, -1],
    &[1, 8, 3, 1, 9, 8, 5, 10, 6, -1],
    &[1, 6, 5, 2, 6, 1, -1],
    &[1, 6, 5, 1, 2, 6, 3, 0, 8, -1],
    &[9, 6, 5, 9, 0, 6, 0, 2, 6, -1],             // 70
    &[5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1],
    &[2, 3, 11, 10, 6, 5, -1],
    &[11, 0, 8, 11, 2, 0, 10, 6, 5, -1],
    &[0, 1, 9, 2, 3, 11, 5, 10, 6, -1],
    &[5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1],
    &[6, 3, 11, 6, 5, 3, 5, 1, 3, -1],
    &[0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1],
    &[3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1],
    &[6, 5, 9, 6, 9, 11, 11, 9, 8, -1],
    &[5, 10, 6, 4, 7, 8, -1],                     // 80
    &[4, 3, 0, 4, 7, 3, 6, 5, 10, -1],
    &[1, 9, 0, 5, 10, 6, 8, 4, 7, -1],
    &[10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1],
    &[6, 1, 2, 6, 5, 1, 4, 7, 8, -1],
    &[1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1],
    &[8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1],
    &[7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    &[3, 11, 2, 7, 8, 4, 10, 6, 5, -1],
    &[5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1],
    &[0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1],  // 90
    &[9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    &[8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1],
    &[5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    &[0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    &[6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1],
    &[10, 4, 9, 6, 4, 10, -1],
    &[4, 10, 6, 4, 9, 10, 0, 8, 3, -1],
    &[10, 0, 1, 10, 6, 0, 6, 4, 0, -1],
    &[8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1],
    &[1, 4, 9, 1, 2, 4, 2, 6, 4, -1],             // 100
    &[3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1],
    &[0, 2, 4, 4, 2, 6, -1],
    &[8, 3, 2, 8, 2, 4, 4, 2, 6, -1],
    &[10, 4, 9, 10, 6, 4, 11, 2, 3, -1],
    &[0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1],
    &[3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1],
    &[6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    &[9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1],
    &[8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    &[3, 11, 6, 3, 6, 0, 0, 6, 4, -1],             // 110
    &[6, 4, 8, 11, 6, 8, -1],
    &[7, 10, 6, 7, 8, 10, 8, 9, 10, -1],
    &[0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1],
    &[10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1],
    &[10, 6, 7, 10, 7, 1, 1, 7, 3, -1],
    &[1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1],
    &[2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    &[7, 8, 0, 7, 0, 6, 6, 0, 2, -1],
    &[7, 3, 2, 6, 7, 2, -1],
    &[2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1],  // 120
    &[2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    &[1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    &[11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1],
    &[8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    &[0, 9, 1, 11, 6, 7, -1],
    &[7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1],
    &[7, 11, 6, -1],
    &[7, 6, 11, -1],
    &[3, 0, 8, 11, 7, 6, -1],
    &[0, 1, 9, 11, 7, 6, -1],                     // 130
    &[8, 1, 9, 8, 3, 1, 11, 7, 6, -1],
    &[10, 1, 2, 6, 11, 7, -1],
    &[1, 2, 10, 3, 0, 8, 6, 11, 7, -1],
    &[2, 9, 0, 2, 10, 9, 6, 11, 7, -1],
    &[6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1],
    &[7, 2, 3, 6, 2, 7, -1],
    &[7, 0, 8, 7, 6, 0, 6, 2, 0, -1],
    &[2, 7, 6, 2, 3, 7, 0, 1, 9, -1],
    &[1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1],
    &[10, 7, 6, 10, 1, 7, 1, 3, 7, -1],           // 140
    &[10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1],
    &[0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1],
    &[7, 6, 10, 7, 10, 8, 8, 10, 9, -1],
    &[6, 8, 4, 11, 8, 6, -1],
    &[3, 6, 11, 3, 0, 6, 0, 4, 6, -1],
    &[8, 6, 11, 8, 4, 6, 9, 0, 1, -1],
    &[9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1],
    &[6, 8, 4, 6, 11, 8, 2, 10, 1, -1],
    &[1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1],
    &[4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1], // 150
    &[10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    &[8, 2, 3, 8, 4, 2, 4, 6, 2, -1],
    &[0, 4, 2, 4, 6, 2, -1],
    &[1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1],
    &[1, 9, 4, 1, 4, 2, 2, 4, 6, -1],
    &[8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1],
    &[10, 1, 0, 10, 0, 6, 6, 0, 4, -1],
    &[4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    &[10, 9, 4, 6, 10, 4, -1],
    &[4, 9, 5, 7, 6, 11, -1],                     // 160
    &[0, 8, 3, 4, 9, 5, 11, 7, 6, -1],
    &[5, 0, 1, 5, 4, 0, 7, 6, 11, -1],
    &[11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1],
    &[9, 5, 4, 10, 1, 2, 7, 6, 11, -1],
    &[6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1],
    &[7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1],
    &[3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    &[7, 2, 3, 7, 6, 2, 5, 4, 9, -1],
    &[9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1],
    &[3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1],   // 170
    &[6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    &[9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1],
    &[1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    &[4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    &[7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1],
    &[6, 9, 5, 6, 11, 9, 11, 8, 9, -1],
    &[3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1],
    &[0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1],
    &[6, 11, 3, 6, 3, 5, 5, 3, 1, -1],
    &[1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1], // 180
    &[0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    &[11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    &[6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1],
    &[5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1],
    &[9, 5, 6, 9, 6, 0, 0, 6, 2, -1],
    &[1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    &[1, 5, 6, 2, 1, 6, -1],
    &[1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    &[10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1],
    &[0, 3, 8, 5, 6, 10, -1],                     // 190
    &[10, 5, 6, -1],
    &[11, 5, 10, 7, 5, 11, -1],
    &[11, 5, 10, 11, 7, 5, 8, 3, 0, -1],
    &[5, 11, 7, 5, 10, 11, 1, 9, 0, -1],
    &[10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1],
    &[11, 1, 2, 11, 7, 1, 7, 5, 1, -1],
    &[0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1],
    &[9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1],
    &[7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    &[2, 5, 10, 2, 3, 5, 3, 7, 5, -1],            // 200
    &[8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1],
    &[9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1],
    &[9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    &[1, 3, 5, 3, 7, 5, -1],
    &[0, 8, 7, 0, 7, 1, 1, 7, 5, -1],
    &[9, 0, 3, 9, 3, 5, 5, 3, 7, -1],
    &[9, 8, 7, 5, 9, 7, -1],
    &[5, 8, 4, 5, 10, 8, 10, 11, 8, -1],
    &[5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1],
    &[0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1], // 210
    &[10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    &[2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1],
    &[0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    &[0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    &[9, 4, 5, 2, 11, 3, -1],
    &[2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1],
    &[5, 10, 2, 5, 2, 4, 4, 2, 0, -1],
    &[3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    &[5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1],
    &[8, 4, 5, 8, 5, 3, 3, 5, 1, -1],             // 220
    &[0, 4, 5, 1, 0, 5, -1],
    &[8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1],
    &[9, 4, 5, -1],
    &[4, 11, 7, 4, 9, 11, 9, 10, 11, -1],
    &[0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1],
    &[1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1],
    &[3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    &[4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1],
    &[9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    &[11, 7, 4, 11, 4, 2, 2, 4, 0, -1],           // 230
    &[11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1],
    &[2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1],
    &[9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    &[3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    &[1, 10, 2, 8, 7, 4, -1],
    &[4, 9, 1, 4, 1, 7, 7, 1, 3, -1],
    &[4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1],
    &[4, 0, 3, 7, 4, 3, -1],
    &[4, 8, 7, -1],
    &[9, 10, 8, 10, 11, 8, -1],                   // 240
    &[3, 0, 9, 3, 9, 11, 11, 9, 10, -1],
    &[0, 1, 10, 0, 10, 8, 8, 10, 11, -1],
    &[3, 1, 10, 11, 3, 10, -1],
    &[1, 2, 11, 1, 11, 9, 9, 11, 8, -1],
    &[3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1],
    &[0, 2, 11, 8, 0, 11, -1],
    &[3, 2, 11, -1],
    &[2, 3, 8, 2, 8, 10, 10, 8, 9, -1],
    &[9, 10, 2, 0, 9, 2, -1],
    &[2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1],  // 250
    &[1, 10, 2, -1],
    &[1, 3, 8, 9, 1, 8, -1],
    &[0, 9, 1, -1],
    &[0, 3, 8, -1],
    &[],                                            // 255
];

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    fn sphere_volume(sz: usize, radius: f64) -> Array3<f64> {
        let c = sz as f64 / 2.0;
        Array3::from_shape_fn((sz, sz, sz), |(z, y, x)| {
            let dz = z as f64 - c;
            let dy = y as f64 - c;
            let dx = x as f64 - c;
            // SDF: positive inside the sphere
            radius - (dz * dz + dy * dy + dx * dx).sqrt()
        })
    }

    #[test]
    fn marching_cubes_sphere_produces_mesh() {
        let vol = sphere_volume(20, 7.0);
        let mesh = marching_cubes(&vol, 0.0).expect("marching cubes failed");
        assert!(
            mesh.vertex_count() > 0,
            "Expected non-empty mesh for sphere SDF"
        );
        assert!(
            mesh.triangle_count() > 0,
            "Expected non-zero triangles for sphere SDF"
        );
    }

    #[test]
    fn marching_cubes_empty_volume_rejected() {
        let vol = Array3::<f64>::zeros((1, 1, 1));
        assert!(marching_cubes(&vol, 0.5).is_err());
    }

    #[test]
    fn estimate_normals_populated() {
        let vol = sphere_volume(16, 5.0);
        let mut mesh = marching_cubes(&vol, 0.0).expect("marching cubes failed");
        assert!(mesh.normals.is_empty());
        estimate_surface_normals(&mut mesh).expect("normals failed");
        assert_eq!(mesh.normals.len(), mesh.vertex_count());
    }

    #[test]
    fn normals_are_unit_vectors() {
        let vol = sphere_volume(16, 5.0);
        let mut mesh = marching_cubes(&vol, 0.0).expect("marching cubes failed");
        estimate_surface_normals(&mut mesh).expect("normals failed");
        for n in &mesh.normals {
            let mag = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (mag - 1.0).abs() < 1e-9 || mag < 1e-14,
                "Normal magnitude should be ≈1, got {mag}"
            );
        }
    }

    #[test]
    fn estimate_normals_rejects_empty_mesh() {
        let mut mesh = SurfaceMesh::new();
        assert!(estimate_surface_normals(&mut mesh).is_err());
    }

    #[test]
    fn isosurface_extraction_full_pipeline() {
        let vol = sphere_volume(18, 6.0);
        let mesh = isosurface_extraction(&vol, 0.0).expect("isosurface extraction failed");
        assert!(mesh.vertex_count() > 0);
        assert_eq!(mesh.normals.len(), mesh.vertex_count());
    }

    #[test]
    fn surface_area_positive() {
        let vol = sphere_volume(18, 6.0);
        let mesh = isosurface_extraction(&vol, 0.0).expect("isosurface failed");
        let area = mesh.surface_area();
        assert!(area > 0.0, "Surface area must be positive, got {area}");
    }

    #[test]
    fn all_background_produces_no_triangles() {
        // A volume entirely below level → no surface
        let vol = Array3::<f64>::from_elem((8, 8, 8), -1.0);
        let mesh = marching_cubes(&vol, 0.0).expect("marching cubes failed");
        assert_eq!(mesh.triangle_count(), 0, "All-background volume has no triangles");
    }
}
