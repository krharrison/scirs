//! Incremental 3D convex hull.
//!
//! Computes the convex hull of a set of 3D points using an incremental
//! (gift-wrapping style) algorithm:
//! 1. Start with a valid tetrahedron from 4 non-coplanar points.
//! 2. For each remaining point, find all faces visible from that point.
//! 3. Remove visible faces and replace them with a cone of new faces.
//!
//! Returns a [`ConvexHull3D`] containing vertex positions and triangle faces.

use crate::error::{SpatialError, SpatialResult};
use std::collections::{HashMap, HashSet};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// A 3D convex hull.
#[derive(Debug, Clone)]
pub struct ConvexHull3D {
    /// Indices into the original point array of hull vertices.
    pub vertices: Vec<usize>,
    /// Triangular faces: each face is `[i, j, k]` where i, j, k are indices
    /// into the **original** point array, and the face normal points outward.
    pub faces: Vec<[usize; 3]>,
}

impl ConvexHull3D {
    /// Compute the signed volume of the hull using the divergence theorem.
    ///
    /// `points` must be the same slice passed to `convex_hull_3d`.
    pub fn volume(&self, points: &[[f64; 3]]) -> f64 {
        let mut vol = 0.0_f64;
        for &[i, j, k] in &self.faces {
            if i >= points.len() || j >= points.len() || k >= points.len() {
                continue;
            }
            let a = points[i];
            let b = points[j];
            let c = points[k];
            // Signed volume contribution = dot(a, cross(b, c)) / 6
            vol += signed_tet_volume([0.0; 3], a, b, c);
        }
        vol.abs()
    }

    /// Compute the surface area of the hull.
    ///
    /// `points` must be the same slice passed to `convex_hull_3d`.
    pub fn surface_area(&self, points: &[[f64; 3]]) -> f64 {
        self.faces
            .iter()
            .map(|&[i, j, k]| {
                if i >= points.len() || j >= points.len() || k >= points.len() {
                    return 0.0;
                }
                triangle_area(points[i], points[j], points[k])
            })
            .sum()
    }

    /// Number of hull faces.
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Number of hull vertices.
    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Main entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the 3D convex hull of `points`.
///
/// # Errors
///
/// Returns an error if fewer than 4 non-coplanar points are provided.
pub fn convex_hull_3d(points: &[[f64; 3]]) -> SpatialResult<ConvexHull3D> {
    if points.len() < 4 {
        return Err(SpatialError::InvalidInput(
            "At least 4 points required for 3D convex hull".into(),
        ));
    }

    // Find 4 non-coplanar seed points.
    let (i0, i1, i2, i3) = find_seed_tetrahedron(points)?;

    // Build the initial tetrahedron with correct outward-facing normals.
    let mut faces: Vec<[usize; 3]> = build_tetrahedron(points, i0, i1, i2, i3);

    // Use the centroid of the seed tetrahedron as a reliable interior point.
    let hull_interior = centroid4(points[i0], points[i1], points[i2], points[i3]);

    // Add remaining points incrementally.
    for (idx, pt) in points.iter().enumerate() {
        if idx == i0 || idx == i1 || idx == i2 || idx == i3 {
            continue;
        }

        // Find all faces visible from `pt`.
        let visible: Vec<usize> = (0..faces.len())
            .filter(|&fi| face_visible(points, &faces[fi], *pt))
            .collect();

        if visible.is_empty() {
            // Point is inside or on the hull – skip.
            continue;
        }

        // Collect the horizon edges (edges shared by exactly one visible face).
        let horizon = horizon_edges(&faces, &visible);

        // Remove visible faces (in reverse order to preserve indices).
        let mut vis_set: Vec<usize> = visible;
        vis_set.sort_unstable();
        for &fi in vis_set.iter().rev() {
            faces.swap_remove(fi);
        }

        // Create new cone faces from horizon edges to `idx`.
        for (a, b) in horizon {
            // Orient so that normal points away from the hull interior.
            let face = orient_face_with_interior(points, [a, b, idx], hull_interior);
            faces.push(face);
        }
    }

    // Collect unique vertex indices.
    let mut vert_set: HashSet<usize> = HashSet::new();
    for &[a, b, c] in &faces {
        vert_set.insert(a);
        vert_set.insert(b);
        vert_set.insert(c);
    }
    let mut vertices: Vec<usize> = vert_set.into_iter().collect();
    vertices.sort_unstable();

    Ok(ConvexHull3D { vertices, faces })
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Find 4 non-coplanar points to seed the tetrahedron.
fn find_seed_tetrahedron(
    points: &[[f64; 3]],
) -> SpatialResult<(usize, usize, usize, usize)> {
    let n = points.len();

    // i0: pick point with smallest x (or 0 by default).
    let i0 = (0..n)
        .min_by(|&a, &b| {
            points[a][0]
                .partial_cmp(&points[b][0])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    // i1: furthest from i0.
    let i1 = (0..n)
        .filter(|&j| j != i0)
        .max_by(|&a, &b| {
            sq_dist3(points[i0], points[a])
                .partial_cmp(&sq_dist3(points[i0], points[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| SpatialError::InvalidInput("Need more points".into()))?;

    // i2: maximise distance to line (i0, i1).
    let i2 = (0..n)
        .filter(|&j| j != i0 && j != i1)
        .max_by(|&a, &b| {
            dist_to_line(points[a], points[i0], points[i1])
                .partial_cmp(&dist_to_line(points[b], points[i0], points[i1]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| SpatialError::InvalidInput("Collinear points".into()))?;

    if dist_to_line(points[i2], points[i0], points[i1]) < 1e-10 {
        return Err(SpatialError::InvalidInput("All points are collinear".into()));
    }

    // i3: maximise distance to plane (i0, i1, i2).
    let i3 = (0..n)
        .filter(|&j| j != i0 && j != i1 && j != i2)
        .max_by(|&a, &b| {
            dist_to_plane(points[a], points[i0], points[i1], points[i2])
                .abs()
                .partial_cmp(
                    &dist_to_plane(points[b], points[i0], points[i1], points[i2]).abs(),
                )
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| SpatialError::InvalidInput("Coplanar points".into()))?;

    if dist_to_plane(points[i3], points[i0], points[i1], points[i2]).abs() < 1e-10 {
        return Err(SpatialError::InvalidInput("All points are coplanar".into()));
    }

    Ok((i0, i1, i2, i3))
}

/// Build the 4 outward-facing triangles of the initial tetrahedron.
fn build_tetrahedron(
    points: &[[f64; 3]],
    i0: usize,
    i1: usize,
    i2: usize,
    i3: usize,
) -> Vec<[usize; 3]> {
    let candidate_faces = [
        [i0, i1, i2],
        [i0, i1, i3],
        [i0, i2, i3],
        [i1, i2, i3],
    ];
    // Find centroid.
    let centroid = centroid4(
        points[i0],
        points[i1],
        points[i2],
        points[i3],
    );
    candidate_faces
        .iter()
        .map(|&f| {
            // Determine if we need to flip the face so normal points away from centroid.
            let n = face_normal(points[f[0]], points[f[1]], points[f[2]]);
            let to_centroid = sub3(centroid, points[f[0]]);
            if dot3(n, to_centroid) > 0.0 {
                // Normal points toward centroid → flip.
                [f[0], f[2], f[1]]
            } else {
                f
            }
        })
        .collect()
}

/// Collect boundary (horizon) edges of the set of visible faces.
///
/// A horizon edge appears in exactly one visible face.
fn horizon_edges(
    faces: &[[usize; 3]],
    visible: &[usize],
) -> Vec<(usize, usize)> {
    let vis_set: HashSet<usize> = visible.iter().copied().collect();
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

    for &fi in &vis_set {
        let f = faces[fi];
        let edges = [
            (f[0].min(f[1]), f[0].max(f[1])),
            (f[1].min(f[2]), f[1].max(f[2])),
            (f[0].min(f[2]), f[0].max(f[2])),
        ];
        for e in edges {
            *edge_count.entry(e).or_insert(0) += 1;
        }
    }

    edge_count
        .into_iter()
        .filter(|(_, cnt)| *cnt == 1)
        .map(|(e, _)| e)
        .collect()
}

/// Orient `face` so that its normal points away from the hull interior (away from
/// the centroid of the existing hull).
fn orient_face(points: &[[f64; 3]], face: [usize; 3]) -> [usize; 3] {
    orient_face_with_interior(points, face, interior_point(points))
}

/// Orient a face so its normal points away from the given interior point.
fn orient_face_with_interior(
    points: &[[f64; 3]],
    face: [usize; 3],
    interior: [f64; 3],
) -> [usize; 3] {
    let [a, b, c] = face;
    let n = face_normal(points[a], points[b], points[c]);
    // The normal should point away from the interior:
    // dot(normal, face_centroid - interior) should be > 0
    let face_cent = centroid3(points[a], points[b], points[c]);
    let out_dir = sub3(face_cent, interior);
    if dot3(n, out_dir) < 0.0 {
        [a, c, b]
    } else {
        face
    }
}

/// Compute a point guaranteed to be in the interior of the point cloud.
fn interior_point(points: &[[f64; 3]]) -> [f64; 3] {
    if points.is_empty() {
        return [0.0; 3];
    }
    let n = points.len() as f64;
    let mut c = [0.0f64; 3];
    for p in points {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    [c[0] / n, c[1] / n, c[2] / n]
}

/// Returns `true` if `pt` is strictly above the face (face normal side).
fn face_visible(points: &[[f64; 3]], face: &[usize; 3], pt: [f64; 3]) -> bool {
    let [a, b, c] = [face[0], face[1], face[2]];
    if a >= points.len() || b >= points.len() || c >= points.len() {
        return false;
    }
    let n = face_normal(points[a], points[b], points[c]);
    let v = sub3(pt, points[a]);
    dot3(n, v) > 1e-10
}

// ──────────────────────────────────────────────────────────────────────────────
// 3D vector math
// ──────────────────────────────────────────────────────────────────────────────

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn face_normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    cross3(sub3(b, a), sub3(c, a))
}

fn sq_dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub3(a, b);
    dot3(d, d)
}

fn dist_to_line(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let ab = sub3(b, a);
    let ap = sub3(p, a);
    let cross = cross3(ab, ap);
    norm3(cross) / norm3(ab).max(1e-14)
}

fn dist_to_plane(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let n = face_normal(a, b, c);
    let len = norm3(n);
    if len < 1e-14 {
        return 0.0;
    }
    dot3(n, sub3(p, a)) / len
}

fn centroid4(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0] + c[0] + d[0]) / 4.0,
        (a[1] + b[1] + c[1] + d[1]) / 4.0,
        (a[2] + b[2] + c[2] + d[2]) / 4.0,
    ]
}

fn centroid3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ]
}

fn triangle_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    norm3(face_normal(a, b, c)) / 2.0
}

fn signed_tet_volume(o: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    dot3(sub3(a, o), cross3(sub3(b, o), sub3(c, o))) / 6.0
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_tetrahedron() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn test_tetrahedron_faces() {
        let pts = unit_tetrahedron();
        let hull = convex_hull_3d(&pts).expect("ok");
        assert_eq!(hull.num_faces(), 4);
        assert_eq!(hull.num_vertices(), 4);
    }

    #[test]
    fn test_tetrahedron_volume() {
        let pts = unit_tetrahedron();
        let hull = convex_hull_3d(&pts).expect("ok");
        let vol = hull.volume(&pts);
        // Volume of unit tetrahedron = 1/6
        assert!((vol - 1.0 / 6.0).abs() < 1e-9, "vol={}", vol);
    }

    #[test]
    fn test_tetrahedron_surface_area() {
        let pts = unit_tetrahedron();
        let hull = convex_hull_3d(&pts).expect("ok");
        let sa = hull.surface_area(&pts);
        // Three right-triangle faces + one equilateral face.
        // Right triangles: area = 0.5 each → 3 × 0.5 = 1.5
        // Equilateral face: sqrt(3)/2 ≈ 0.866
        // Total ≈ 2.366
        assert!(sa > 2.0 && sa < 3.0, "sa={}", sa);
    }

    #[test]
    fn test_cube_hull() {
        let pts: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        ];
        let hull = convex_hull_3d(&pts).expect("ok");
        // Cube hull has 8 vertices and 12 triangle faces
        assert_eq!(hull.num_vertices(), 8);
        assert_eq!(hull.num_faces(), 12);
    }

    #[test]
    fn test_cube_volume() {
        let pts: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],
        ];
        let hull = convex_hull_3d(&pts).expect("ok");
        let vol = hull.volume(&pts);
        assert!((vol - 1.0).abs() < 0.1, "vol={}", vol);
    }

    #[test]
    fn test_too_few_points() {
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert!(convex_hull_3d(&pts).is_err());
    }

    #[test]
    fn test_interior_points_ignored() {
        let mut pts = unit_tetrahedron();
        pts.push([0.1, 0.1, 0.1]); // interior point
        let hull = convex_hull_3d(&pts).expect("ok");
        assert_eq!(hull.num_faces(), 4);
    }
}
