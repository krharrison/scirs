//! 3D reconstruction utility functions.
//!
//! Provides:
//! - Direct Linear Transform (DLT) triangulation.
//! - Midpoint (ray intersection) triangulation.
//! - Disparity-to-depth conversion.
//! - Point cloud back-projection from depth maps.

use crate::error::VisionError;

// ─── Linear algebra helpers ───────────────────────────────────────────────────

type Mat34 = [[f64; 4]; 3];
type Mat3 = [[f64; 3]; 3];

/// Solve a 4×4 homogeneous system A·x = 0 via the Jacobi SVD of A^T·A.
/// Returns the last right singular vector (homogeneous solution).
fn null_space_4(a: &[[f64; 4]; 4]) -> [f64; 4] {
    // Build A^T A (4×4 symmetric).
    let mut ata = [[0.0_f64; 4]; 4];
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }
    jacobi_min_eigenvec_4(&ata)
}

/// Jacobi eigendecomposition of a 4×4 symmetric matrix.
/// Returns the eigenvector corresponding to the smallest eigenvalue.
fn jacobi_min_eigenvec_4(s: &[[f64; 4]; 4]) -> [f64; 4] {
    let n = 4_usize;
    let mut a = *s;
    let mut v = [[0.0_f64; 4]; 4];
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _ in 0..200 {
        let (mut p, mut q, mut mx) = (0, 1, 0.0_f64);
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            for j in i + 1..n {
                let av = a[i][j].abs();
                if av > mx {
                    mx = av;
                    p = i;
                    q = j;
                }
            }
        }
        if mx < 1e-14 {
            break;
        }
        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            1.0 / (tau - (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let sn = t * c;
        let app = c * c * a[p][p] - 2.0 * sn * c * a[p][q] + sn * sn * a[q][q];
        let aqq = sn * sn * a[p][p] + 2.0 * sn * c * a[p][q] + c * c * a[q][q];
        a[p][p] = app;
        a[q][q] = aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        #[allow(clippy::needless_range_loop)]
        for r in 0..n {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - sn * arq;
                a[p][r] = a[r][p];
                a[r][q] = sn * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }
        #[allow(clippy::needless_range_loop)]
        for r in 0..n {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = c * vrp - sn * vrq;
            v[r][q] = sn * vrp + c * vrq;
        }
    }

    let min_idx = (0..n)
        .min_by(|&i, &j| {
            a[i][i]
                .partial_cmp(&a[j][j])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    let mut ev = [0.0_f64; 4];
    for r in 0..n {
        ev[r] = v[r][min_idx];
    }
    let norm: f64 = ev.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for x in ev.iter_mut() {
            *x /= norm;
        }
    }
    ev
}

// ─── DLT triangulation ───────────────────────────────────────────────────────

/// Triangulate a 3D point from two views using the Direct Linear Transform.
///
/// `P1`, `P2` are 3×4 camera projection matrices.
/// `pt1`, `pt2` are observed image coordinates `(x, y)` (column, row).
///
/// Returns the inhomogeneous 3D point `[X, Y, Z]`.
///
/// # Errors
/// Returns [`VisionError`] if the system is degenerate (all-zero solution).
pub fn triangulate_dlt(
    p1: &Mat34,
    p2: &Mat34,
    pt1: (f64, f64),
    pt2: (f64, f64),
) -> Result<[f64; 3], VisionError> {
    let (x1, y1) = pt1;
    let (x2, y2) = pt2;

    // Each point gives 2 equations: cross product x × (P·X) = 0.
    // A·X_h = 0 where A is 4×4.
    let mut a = [[0.0_f64; 4]; 4];
    for j in 0..4 {
        a[0][j] = x1 * p1[2][j] - p1[0][j];
        a[1][j] = y1 * p1[2][j] - p1[1][j];
        a[2][j] = x2 * p2[2][j] - p2[0][j];
        a[3][j] = y2 * p2[2][j] - p2[1][j];
    }

    let x_h = null_space_4(&a);
    let w = x_h[3];
    if w.abs() < 1e-15 {
        return Err(VisionError::OperationError(
            "Degenerate DLT triangulation: point at infinity".into(),
        ));
    }
    Ok([x_h[0] / w, x_h[1] / w, x_h[2] / w])
}

// ─── Midpoint triangulation ───────────────────────────────────────────────────

/// Triangulate a 3D point by finding the midpoint of the closest approach of
/// two camera rays.
///
/// `R`, `t` describe the pose of camera 2 relative to camera 1
/// (camera 1 is at the origin looking along +Z).
/// `pt1`, `pt2` are normalised image coordinates (i.e., already divided by
/// focal length and centred at principal point).
pub fn triangulate_midpoint(r: &Mat3, t: &[f64; 3], pt1: (f64, f64), pt2: (f64, f64)) -> [f64; 3] {
    // Ray 1: origin = [0,0,0], direction d1 = [x1, y1, 1] (normalised).
    let d1 = normalise_vec3([pt1.0, pt1.1, 1.0]);
    // Ray 2: origin = camera2 position (=-R^T t) relative to cam1 frame,
    //        direction d2 = R^T * [x2, y2, 1].
    let d2_cam2 = normalise_vec3([pt2.0, pt2.1, 1.0]);
    let d2 = mat3_vec(r, &d2_cam2); // rotate into cam1 frame (R maps cam2→cam1)
                                    // Camera 2 centre in cam1 frame is simply t (the translation vector
                                    // gives the position of cam2 relative to cam1).
    let o2 = *t;

    // Solve: o1 + s*d1 = o2 + u*d2 (least squares).
    // |d1 -d2| * [s; u] = o2 - o1
    // Use the midpoint formula:
    let cross_d1_d2 = cross3(&d1, &d2);
    let denom = dot3(&cross_d1_d2, &cross_d1_d2);
    if denom < 1e-15 {
        // Rays are parallel: return midpoint of origins.
        return [o2[0] / 2.0, o2[1] / 2.0, o2[2] / 2.0];
    }

    let diff = [o2[0], o2[1], o2[2]];
    let s = dot3(&cross3(&diff, &d2), &cross_d1_d2) / denom;
    let u = dot3(&cross3(&diff, &d1), &cross_d1_d2) / denom;

    let p1 = [s * d1[0], s * d1[1], s * d1[2]];
    let p2 = [o2[0] + u * d2[0], o2[1] + u * d2[1], o2[2] + u * d2[2]];
    [
        (p1[0] + p2[0]) / 2.0,
        (p1[1] + p2[1]) / 2.0,
        (p1[2] + p2[2]) / 2.0,
    ]
}

// ─── Depth from disparity ─────────────────────────────────────────────────────

/// Compute depth from stereo disparity.
///
/// Z = f * B / d
///
/// # Arguments
/// - `disparity`: pixel disparity (must be > 0).
/// - `baseline`: stereo baseline in metres.
/// - `focal_length`: focal length in pixels.
///
/// # Errors
/// Returns [`VisionError`] if disparity is zero or negative.
pub fn depth_from_disparity(
    disparity: f64,
    baseline: f64,
    focal_length: f64,
) -> Result<f64, VisionError> {
    if disparity <= 0.0 {
        return Err(VisionError::InvalidInput(
            "Disparity must be positive for depth computation".into(),
        ));
    }
    Ok(focal_length * baseline / disparity)
}

// ─── Point cloud from depth map ───────────────────────────────────────────────

/// Back-project a depth map to a 3D point cloud.
///
/// `depth[r][c]` is the metric depth at pixel `(c, r)`.
/// `K` is the 3×3 camera intrinsic matrix:
/// ```text
/// K = [[fx, 0,  cx],
///      [0,  fy, cy],
///      [0,  0,  1 ]]
/// ```
/// Returns `[X, Y, Z]` for each pixel where `depth > 0`.
pub fn point_cloud_from_depth(depth: &[Vec<f32>], k: &Mat3) -> Vec<[f64; 3]> {
    let fx = k[0][0];
    let fy = k[1][1];
    let cx = k[0][2];
    let cy = k[1][2];

    let mut pts = Vec::new();
    for (r, row) in depth.iter().enumerate() {
        for (c, &d) in row.iter().enumerate() {
            if d <= 0.0 {
                continue;
            }
            let z = d as f64;
            let x = (c as f64 - cx) / fx * z;
            let y = (r as f64 - cy) / fy * z;
            pts.push([x, y, z]);
        }
    }
    pts
}

// ─── Small vector helpers ────────────────────────────────────────────────────

#[inline]
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn normalise_vec3(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 1e-15 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        v
    }
}

#[inline]
fn mat3_vec(m: &Mat3, v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
fn transpose3(m: &Mat3) -> Mat3 {
    let mut t = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[j][i] = m[i][j];
        }
    }
    t
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Make a canonical projection matrix for a camera at origin facing +Z.
    fn identity_proj() -> Mat34 {
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    }

    /// Translation-only camera (baseline b along X).
    fn translated_proj(b: f64) -> Mat34 {
        [
            [1.0, 0.0, 0.0, -b],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn test_triangulate_dlt_known_point() {
        // Point at (1, 2, 5).  Camera1 = identity proj, camera2 = translate -3 on X.
        let p1 = identity_proj();
        let p2 = translated_proj(3.0);
        let pt3d = [1.0_f64, 2.0, 5.0];
        // Project.
        let proj = |p: &Mat34, pt: &[f64; 3]| -> (f64, f64) {
            let w = p[2][0] * pt[0] + p[2][1] * pt[1] + p[2][2] * pt[2] + p[2][3];
            (
                (p[0][0] * pt[0] + p[0][1] * pt[1] + p[0][2] * pt[2] + p[0][3]) / w,
                (p[1][0] * pt[0] + p[1][1] * pt[1] + p[1][2] * pt[2] + p[1][3]) / w,
            )
        };
        let obs1 = proj(&p1, &pt3d);
        let obs2 = proj(&p2, &pt3d);
        let recon = triangulate_dlt(&p1, &p2, obs1, obs2)
            .expect("triangulate_dlt should succeed with valid projection matrices");
        for (r, &g) in recon.iter().zip(pt3d.iter()) {
            assert!((r - g).abs() < 1e-6, "Component mismatch: {} vs {}", r, g);
        }
    }

    #[test]
    fn test_depth_from_disparity_basic() {
        let d = depth_from_disparity(10.0, 0.1, 500.0)
            .expect("depth_from_disparity should succeed with valid inputs");
        assert!((d - 5.0).abs() < 1e-10, "Expected 5.0, got {}", d);
    }

    #[test]
    fn test_depth_from_disparity_zero_error() {
        assert!(depth_from_disparity(0.0, 0.1, 500.0).is_err());
    }

    #[test]
    fn test_point_cloud_from_depth() {
        let depth: Vec<Vec<f32>> = vec![vec![0.0, 1.0], vec![2.0, 0.0]];
        let k: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let pts = point_cloud_from_depth(&depth, &k);
        // Only pixels with depth > 0 are included.
        assert_eq!(pts.len(), 2);
        // Pixel (r=0, c=1, d=1) → X=(1-0)/1*1=1, Y=(0-0)/1*1=0, Z=1
        assert!((pts[0][2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangulate_midpoint_forward() {
        // Camera 1 at origin, camera 2 translated 1 unit along X.
        let r: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let t = [1.0, 0.0, 0.0];
        // Both cameras see point (0, 0, 5) in their respective frames.
        let pt1 = (0.0, 0.0); // cam1: x=0/5=0, y=0/5=0
        let pt2 = (-1.0 / 5.0, 0.0); // cam2: x=(0-1)/5=-0.2, y=0/5=0
        let result = triangulate_midpoint(&r, &t, pt1, pt2);
        assert!(
            (result[2] - 5.0).abs() < 0.1,
            "Z expected ~5, got {}",
            result[2]
        );
    }
}
