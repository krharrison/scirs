//! Mesh quality metrics
//!
//! Provides triangle quality assessment including aspect ratio and minimum angle.

use super::TriangleMesh;
use crate::error::SpatialResult;

/// Compute the aspect ratio of a triangle face
///
/// The aspect ratio is the ratio of the circumscribed circle radius to the
/// inscribed circle radius, normalized so that an equilateral triangle has
/// aspect ratio 1.0. Higher values indicate poorer quality.
///
/// # Arguments
///
/// * `mesh` - The triangle mesh
/// * `face_idx` - Index of the face
///
/// # Returns
///
/// * The aspect ratio (>= 1.0, where 1.0 is equilateral)
pub fn face_aspect_ratio(mesh: &TriangleMesh, face_idx: usize) -> SpatialResult<f64> {
    if face_idx >= mesh.faces.len() {
        return Err(crate::error::SpatialError::ValueError(format!(
            "Face index {} out of range (num faces = {})",
            face_idx,
            mesh.faces.len()
        )));
    }

    let face = &mesh.faces[face_idx];
    let v0 = &mesh.vertices[face.v0];
    let v1 = &mesh.vertices[face.v1];
    let v2 = &mesh.vertices[face.v2];

    // Edge lengths
    let a = v0.distance(v1);
    let b = v1.distance(v2);
    let c = v0.distance(v2);

    // Semi-perimeter
    let s = (a + b + c) * 0.5;

    // Area via Heron's formula
    let area_sq = s * (s - a) * (s - b) * (s - c);
    if area_sq <= 0.0 {
        // Degenerate triangle
        return Ok(f64::INFINITY);
    }
    let area = area_sq.sqrt();

    // Circumradius R = abc / (4 * area)
    let circum_r = (a * b * c) / (4.0 * area);

    // Inradius r = area / s
    let in_r = area / s;

    if in_r < 1e-15 {
        return Ok(f64::INFINITY);
    }

    // Aspect ratio normalized so equilateral = 1.0
    // For equilateral: R/r = 2, so we divide by 2
    Ok(circum_r / (2.0 * in_r))
}

/// Compute the minimum angle (in radians) of a triangle face
///
/// # Arguments
///
/// * `mesh` - The triangle mesh
/// * `face_idx` - Index of the face
///
/// # Returns
///
/// * The minimum angle in radians
pub fn face_min_angle(mesh: &TriangleMesh, face_idx: usize) -> SpatialResult<f64> {
    if face_idx >= mesh.faces.len() {
        return Err(crate::error::SpatialError::ValueError(format!(
            "Face index {} out of range (num faces = {})",
            face_idx,
            mesh.faces.len()
        )));
    }

    let face = &mesh.faces[face_idx];
    let v0 = &mesh.vertices[face.v0];
    let v1 = &mesh.vertices[face.v1];
    let v2 = &mesh.vertices[face.v2];

    // Edge lengths squared
    let a_sq = v0.distance_sq(v1);
    let b_sq = v1.distance_sq(v2);
    let c_sq = v0.distance_sq(v2);

    let a = a_sq.sqrt();
    let b = b_sq.sqrt();
    let c = c_sq.sqrt();

    if a < 1e-15 || b < 1e-15 || c < 1e-15 {
        return Ok(0.0);
    }

    // Angles via law of cosines
    let angle_at_0 = ((a_sq + c_sq - b_sq) / (2.0 * a * c))
        .clamp(-1.0, 1.0)
        .acos();
    let angle_at_1 = ((a_sq + b_sq - c_sq) / (2.0 * a * b))
        .clamp(-1.0, 1.0)
        .acos();
    let angle_at_2 = ((b_sq + c_sq - a_sq) / (2.0 * b * c))
        .clamp(-1.0, 1.0)
        .acos();

    Ok(angle_at_0.min(angle_at_1).min(angle_at_2))
}

/// Summary statistics for mesh quality
#[derive(Debug, Clone)]
pub struct QualityStats {
    /// Minimum aspect ratio across all faces
    pub min_aspect_ratio: f64,
    /// Maximum aspect ratio across all faces
    pub max_aspect_ratio: f64,
    /// Mean aspect ratio
    pub mean_aspect_ratio: f64,
    /// Minimum angle (radians) across all faces
    pub min_angle: f64,
    /// Maximum minimum-angle across all faces
    pub max_min_angle: f64,
    /// Mean minimum-angle
    pub mean_min_angle: f64,
    /// Number of faces analyzed
    pub num_faces: usize,
}

/// Compute quality statistics over the entire mesh
///
/// # Arguments
///
/// * `mesh` - The triangle mesh
///
/// # Returns
///
/// * Quality statistics summary
pub fn mesh_quality_stats(mesh: &TriangleMesh) -> SpatialResult<QualityStats> {
    let nf = mesh.num_faces();
    if nf == 0 {
        return Ok(QualityStats {
            min_aspect_ratio: 0.0,
            max_aspect_ratio: 0.0,
            mean_aspect_ratio: 0.0,
            min_angle: 0.0,
            max_min_angle: 0.0,
            mean_min_angle: 0.0,
            num_faces: 0,
        });
    }

    let mut min_ar = f64::INFINITY;
    let mut max_ar = f64::NEG_INFINITY;
    let mut sum_ar = 0.0;
    let mut min_ang = f64::INFINITY;
    let mut max_min_ang = f64::NEG_INFINITY;
    let mut sum_ang = 0.0;

    for i in 0..nf {
        let ar = face_aspect_ratio(mesh, i)?;
        let ang = face_min_angle(mesh, i)?;

        if ar.is_finite() {
            min_ar = min_ar.min(ar);
            max_ar = max_ar.max(ar);
            sum_ar += ar;
        }

        min_ang = min_ang.min(ang);
        max_min_ang = max_min_ang.max(ang);
        sum_ang += ang;
    }

    Ok(QualityStats {
        min_aspect_ratio: if min_ar.is_finite() { min_ar } else { 0.0 },
        max_aspect_ratio: if max_ar.is_finite() { max_ar } else { 0.0 },
        mean_aspect_ratio: sum_ar / nf as f64,
        min_angle: min_ang,
        max_min_angle: max_min_ang,
        mean_min_angle: sum_ang / nf as f64,
        num_faces: nf,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Face, Vertex};

    fn equilateral_mesh() -> TriangleMesh {
        let h = (3.0_f64).sqrt() / 2.0;
        let vertices = vec![
            Vertex::new(0.0, 0.0, 0.0),
            Vertex::new(1.0, 0.0, 0.0),
            Vertex::new(0.5, h, 0.0),
        ];
        let faces = vec![Face::new(0, 1, 2)];
        TriangleMesh::new(vertices, faces).expect("valid")
    }

    #[test]
    fn test_equilateral_aspect_ratio() {
        let mesh = equilateral_mesh();
        let ar = face_aspect_ratio(&mesh, 0).expect("aspect ratio failed");
        // Equilateral triangle should have aspect ratio 1.0
        assert!(
            (ar - 1.0).abs() < 1e-10,
            "Expected aspect ratio ~1.0, got {}",
            ar
        );
    }

    #[test]
    fn test_equilateral_min_angle() {
        let mesh = equilateral_mesh();
        let ang = face_min_angle(&mesh, 0).expect("min angle failed");
        // Equilateral triangle: all angles are pi/3
        let expected = std::f64::consts::PI / 3.0;
        assert!(
            (ang - expected).abs() < 1e-10,
            "Expected angle ~{}, got {}",
            expected,
            ang
        );
    }

    #[test]
    fn test_skinny_triangle() {
        let vertices = vec![
            Vertex::new(0.0, 0.0, 0.0),
            Vertex::new(10.0, 0.0, 0.0),
            Vertex::new(5.0, 0.01, 0.0),
        ];
        let faces = vec![Face::new(0, 1, 2)];
        let mesh = TriangleMesh::new(vertices, faces).expect("valid");

        let ar = face_aspect_ratio(&mesh, 0).expect("aspect ratio");
        // Skinny triangle has high aspect ratio
        assert!(ar > 10.0, "Expected high aspect ratio, got {}", ar);

        let ang = face_min_angle(&mesh, 0).expect("min angle");
        // Skinny triangle has very small minimum angle
        assert!(ang < 0.01, "Expected very small angle, got {}", ang);
    }

    #[test]
    fn test_quality_stats() {
        let h = (3.0_f64).sqrt() / 2.0;
        let vertices = vec![
            Vertex::new(0.0, 0.0, 0.0),
            Vertex::new(1.0, 0.0, 0.0),
            Vertex::new(0.5, h, 0.0),
            Vertex::new(0.5, 0.5, 1.0),
        ];
        let faces = vec![
            Face::new(0, 1, 2),
            Face::new(0, 1, 3),
            Face::new(1, 2, 3),
            Face::new(0, 2, 3),
        ];
        let mesh = TriangleMesh::new(vertices, faces).expect("valid");

        let stats = mesh_quality_stats(&mesh).expect("stats failed");
        assert_eq!(stats.num_faces, 4);
        assert!(stats.min_aspect_ratio >= 1.0 - 1e-10);
        assert!(stats.max_aspect_ratio >= stats.min_aspect_ratio);
        assert!(stats.min_angle > 0.0);
        assert!(stats.min_angle <= stats.max_min_angle);
    }
}
