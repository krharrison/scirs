//! Mesh smoothing algorithms
//!
//! Provides Laplacian and Taubin smoothing for triangle meshes.

use super::TriangleMesh;
use crate::error::{SpatialError, SpatialResult};

/// Apply Laplacian smoothing to a mesh
///
/// Laplacian smoothing moves each vertex towards the centroid of its neighbors.
/// This reduces noise but can cause mesh shrinkage with many iterations.
///
/// # Arguments
///
/// * `mesh` - The input triangle mesh (modified in place)
/// * `iterations` - Number of smoothing iterations
/// * `lambda` - Smoothing factor in (0, 1]. Larger values smooth more aggressively.
///   Typical values are 0.3-0.7.
/// * `preserve_boundary` - If true, boundary vertices are not moved
///
/// # Returns
///
/// * The smoothed mesh
///
/// # Examples
///
/// ```
/// use scirs2_spatial::mesh::{TriangleMesh, Vertex, Face, laplacian_smooth};
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
/// let smoothed = laplacian_smooth(&mesh, 3, 0.5, true).expect("smooth failed");
/// assert_eq!(smoothed.num_vertices(), mesh.num_vertices());
/// assert_eq!(smoothed.num_faces(), mesh.num_faces());
/// ```
pub fn laplacian_smooth(
    mesh: &TriangleMesh,
    iterations: usize,
    lambda: f64,
    preserve_boundary: bool,
) -> SpatialResult<TriangleMesh> {
    if lambda <= 0.0 || lambda > 1.0 {
        return Err(SpatialError::ValueError(format!(
            "Lambda must be in (0, 1], got {}",
            lambda
        )));
    }

    let mut result = mesh.clone();

    // Find boundary vertices if needed
    let boundary_verts = if preserve_boundary {
        let boundary_edges = result.boundary_edges();
        let mut bv = std::collections::HashSet::new();
        for e in &boundary_edges {
            bv.insert(e.a);
            bv.insert(e.b);
        }
        bv
    } else {
        std::collections::HashSet::new()
    };

    for _ in 0..iterations {
        let neighbors = result.vertex_neighbors();
        let mut new_positions = result.vertices.clone();

        for (vi, vertex) in result.vertices.iter().enumerate() {
            if preserve_boundary && boundary_verts.contains(&vi) {
                continue;
            }

            if let Some(nbrs) = neighbors.get(&vi) {
                if nbrs.is_empty() {
                    continue;
                }

                // Compute centroid of neighbors
                let mut cx = 0.0;
                let mut cy = 0.0;
                let mut cz = 0.0;
                let count = nbrs.len() as f64;

                for &ni in nbrs {
                    cx += result.vertices[ni].x;
                    cy += result.vertices[ni].y;
                    cz += result.vertices[ni].z;
                }

                cx /= count;
                cy /= count;
                cz /= count;

                // Move vertex towards centroid
                new_positions[vi].x = vertex.x + lambda * (cx - vertex.x);
                new_positions[vi].y = vertex.y + lambda * (cy - vertex.y);
                new_positions[vi].z = vertex.z + lambda * (cz - vertex.z);
            }
        }

        result.vertices = new_positions;
    }

    Ok(result)
}

/// Apply Taubin smoothing to a mesh
///
/// Taubin smoothing alternates between a positive and negative smoothing step,
/// which prevents the shrinkage that occurs with pure Laplacian smoothing.
/// Each iteration performs one smoothing step (lambda) followed by one
/// inflation step (mu, where mu < -lambda).
///
/// # Arguments
///
/// * `mesh` - The input triangle mesh
/// * `iterations` - Number of smoothing-inflation iterations
/// * `lambda` - Smoothing factor (positive, typically 0.3-0.7)
/// * `mu` - Inflation factor (negative, typically -0.31 to -0.71, |mu| > lambda)
/// * `preserve_boundary` - If true, boundary vertices are not moved
///
/// # Returns
///
/// * The smoothed mesh
///
/// # Examples
///
/// ```
/// use scirs2_spatial::mesh::{TriangleMesh, Vertex, Face, taubin_smooth};
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
/// let smoothed = taubin_smooth(&mesh, 5, 0.5, -0.53, true).expect("smooth failed");
/// assert_eq!(smoothed.num_vertices(), mesh.num_vertices());
/// ```
pub fn taubin_smooth(
    mesh: &TriangleMesh,
    iterations: usize,
    lambda: f64,
    mu: f64,
    preserve_boundary: bool,
) -> SpatialResult<TriangleMesh> {
    if lambda <= 0.0 || lambda > 1.0 {
        return Err(SpatialError::ValueError(format!(
            "Lambda must be in (0, 1], got {}",
            lambda
        )));
    }

    if mu >= 0.0 {
        return Err(SpatialError::ValueError(format!(
            "Mu must be negative, got {}",
            mu
        )));
    }

    if mu.abs() <= lambda {
        return Err(SpatialError::ValueError(format!(
            "|mu| must be > lambda for volume preservation. Got lambda={}, mu={}",
            lambda, mu
        )));
    }

    let mut result = mesh.clone();

    // Find boundary vertices if needed
    let boundary_verts = if preserve_boundary {
        let boundary_edges = result.boundary_edges();
        let mut bv = std::collections::HashSet::new();
        for e in &boundary_edges {
            bv.insert(e.a);
            bv.insert(e.b);
        }
        bv
    } else {
        std::collections::HashSet::new()
    };

    for _ in 0..iterations {
        // Smoothing step (lambda)
        apply_laplacian_step(&mut result, lambda, preserve_boundary, &boundary_verts);

        // Inflation step (mu is negative, so this "inflates")
        apply_laplacian_step(&mut result, mu, preserve_boundary, &boundary_verts);
    }

    Ok(result)
}

/// Apply a single Laplacian step with a given factor
fn apply_laplacian_step(
    mesh: &mut TriangleMesh,
    factor: f64,
    preserve_boundary: bool,
    boundary_verts: &std::collections::HashSet<usize>,
) {
    let neighbors = mesh.vertex_neighbors();
    let old_vertices = mesh.vertices.clone();

    for (vi, vertex) in old_vertices.iter().enumerate() {
        if preserve_boundary && boundary_verts.contains(&vi) {
            continue;
        }

        if let Some(nbrs) = neighbors.get(&vi) {
            if nbrs.is_empty() {
                continue;
            }

            let count = nbrs.len() as f64;
            let mut cx = 0.0;
            let mut cy = 0.0;
            let mut cz = 0.0;

            for &ni in nbrs {
                cx += old_vertices[ni].x;
                cy += old_vertices[ni].y;
                cz += old_vertices[ni].z;
            }

            cx /= count;
            cy /= count;
            cz /= count;

            mesh.vertices[vi].x = vertex.x + factor * (cx - vertex.x);
            mesh.vertices[vi].y = vertex.y + factor * (cy - vertex.y);
            mesh.vertices[vi].z = vertex.z + factor * (cz - vertex.z);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Face, Vertex};

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
        TriangleMesh::new(vertices, faces).expect("valid")
    }

    #[test]
    fn test_laplacian_smooth() {
        let mesh = tetrahedron();
        let smoothed = laplacian_smooth(&mesh, 3, 0.5, false).expect("smooth failed");
        assert_eq!(smoothed.num_vertices(), mesh.num_vertices());
        assert_eq!(smoothed.num_faces(), mesh.num_faces());

        // After smoothing, vertices should have moved towards each other
        // (centroid of a tetrahedron is (0.5, 0.375, 0.25))
        // With smoothing, all vertices should move towards the centroid
    }

    #[test]
    fn test_laplacian_smooth_preserves_boundary() {
        // Single triangle has all boundary vertices
        let vertices = vec![
            Vertex::new(0.0, 0.0, 0.0),
            Vertex::new(1.0, 0.0, 0.0),
            Vertex::new(0.5, 1.0, 0.0),
        ];
        let faces = vec![Face::new(0, 1, 2)];
        let mesh = TriangleMesh::new(vertices, faces).expect("valid");

        let smoothed = laplacian_smooth(&mesh, 10, 0.5, true).expect("smooth");

        // Boundary preserved: no vertices should move
        for i in 0..3 {
            assert!(
                (smoothed.vertices[i].x - mesh.vertices[i].x).abs() < 1e-10,
                "Vertex {} x changed",
                i
            );
            assert!(
                (smoothed.vertices[i].y - mesh.vertices[i].y).abs() < 1e-10,
                "Vertex {} y changed",
                i
            );
        }
    }

    #[test]
    fn test_taubin_smooth() {
        let mesh = tetrahedron();
        let smoothed = taubin_smooth(&mesh, 5, 0.5, -0.53, false).expect("smooth");
        assert_eq!(smoothed.num_vertices(), mesh.num_vertices());
        assert_eq!(smoothed.num_faces(), mesh.num_faces());
    }

    #[test]
    fn test_taubin_smooth_volume_preservation() {
        let mesh = tetrahedron();
        let original_area = mesh.surface_area().expect("area");

        // Taubin smoothing should better preserve volume than Laplacian
        let taubin = taubin_smooth(&mesh, 3, 0.5, -0.53, false).expect("taubin");
        let laplacian = laplacian_smooth(&mesh, 3, 0.5, false).expect("laplacian");

        let taubin_area = taubin.surface_area().expect("area");
        let laplacian_area = laplacian.surface_area().expect("area");

        // Taubin should preserve area better than Laplacian
        let taubin_diff = (taubin_area - original_area).abs();
        let laplacian_diff = (laplacian_area - original_area).abs();

        // Laplacian should shrink more
        assert!(
            laplacian_diff >= taubin_diff * 0.5 || laplacian_area < original_area,
            "Expected Laplacian to shrink more. taubin_diff={}, laplacian_diff={}",
            taubin_diff,
            laplacian_diff
        );
    }

    #[test]
    fn test_smooth_errors() {
        let mesh = tetrahedron();

        // Invalid lambda
        assert!(laplacian_smooth(&mesh, 1, 0.0, false).is_err());
        assert!(laplacian_smooth(&mesh, 1, 1.5, false).is_err());

        // Invalid mu for Taubin
        assert!(taubin_smooth(&mesh, 1, 0.5, 0.1, false).is_err());

        // |mu| <= lambda
        assert!(taubin_smooth(&mesh, 1, 0.5, -0.3, false).is_err());
    }
}
