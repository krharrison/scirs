//! Camera intrinsics, extrinsics and stereo camera system.
//!
//! This module provides the canonical `CameraIntrinsics` / `CameraExtrinsics` /
//! `StereoCameraSystem` trio used across the 3-D vision pipeline.  They share
//! the same Brown-Conrady distortion model as [`super::PinholeCamera`] but are
//! expressed with plain Rust arrays (`[f64; 3]`, `[[f64; 3]; 3]`) so that no
//! ndarray dependency is needed at the call site.

use crate::error::{Result, VisionError};

// ─────────────────────────────────────────────────────────────────────────────
// CameraIntrinsics
// ─────────────────────────────────────────────────────────────────────────────

/// Pinhole camera intrinsic parameters with Brown-Conrady distortion.
///
/// Projection chain (ideal then distorted):
/// ```text
/// xn = X / Z,  yn = Y / Z            (normalised coords)
/// r² = xn² + yn²
/// xd = xn*(1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*xn*yn + p2*(r²+2*xn²)
/// yd = yn*(1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r²+2*yn²) + 2*p2*xn*yn
/// u  = fx*xd + cx,  v = fy*yd + cy
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in pixels along the X axis.
    pub fx: f64,
    /// Focal length in pixels along the Y axis.
    pub fy: f64,
    /// Principal-point X coordinate (pixels).
    pub cx: f64,
    /// Principal-point Y coordinate (pixels).
    pub cy: f64,
    /// Radial distortion coefficient k1.
    pub k1: f64,
    /// Radial distortion coefficient k2.
    pub k2: f64,
    /// Radial distortion coefficient k3.
    pub k3: f64,
    /// Tangential distortion coefficient p1.
    pub p1: f64,
    /// Tangential distortion coefficient p2.
    pub p2: f64,
}

impl CameraIntrinsics {
    /// Create new intrinsics with all parameters.
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        k1: f64,
        k2: f64,
        k3: f64,
        p1: f64,
        p2: f64,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            k3,
            p1,
            p2,
        }
    }

    /// Create distortion-free intrinsics.
    pub fn ideal(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self::new(fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Return the 3×3 calibration matrix K.
    ///
    /// ```
    /// # use scirs2_vision::camera::CameraIntrinsics;
    /// let k = CameraIntrinsics::ideal(800.0, 600.0, 320.0, 240.0);
    /// let mat = k.calibration_matrix();
    /// assert!((mat[0][0] - 800.0).abs() < 1e-12);
    /// assert!((mat[1][1] - 600.0).abs() < 1e-12);
    /// assert!((mat[2][2] - 1.0).abs() < 1e-12);
    /// ```
    pub fn calibration_matrix(&self) -> [[f64; 3]; 3] {
        [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Apply radial + tangential distortion to normalised image coordinates.
    ///
    /// ```
    /// # use scirs2_vision::camera::CameraIntrinsics;
    /// let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
    /// let d = cam.distort([0.0, 0.0]);
    /// assert!((d[0]).abs() < 1e-12);
    /// assert!((d[1]).abs() < 1e-12);
    /// ```
    pub fn distort(&self, normalized: [f64; 2]) -> [f64; 2] {
        let xn = normalized[0];
        let yn = normalized[1];
        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let xd = xn * radial + 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn);
        let yd = yn * radial + self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn;
        [xd, yd]
    }

    /// Project a 3-D point `[X, Y, Z]` (in camera frame) to a pixel `[u, v]`
    /// using the full distortion model.
    ///
    /// Returns `Err` when `Z ≤ 0`.
    ///
    /// ```
    /// # use scirs2_vision::camera::CameraIntrinsics;
    /// let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
    /// let px = cam.project([0.0, 0.0, 1.0]).unwrap();
    /// assert!((px[0] - 320.0).abs() < 1e-9);
    /// assert!((px[1] - 240.0).abs() < 1e-9);
    /// ```
    pub fn project(&self, point3d: [f64; 3]) -> Result<[f64; 2]> {
        let z = point3d[2];
        if z <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "Z must be positive for projection".to_string(),
            ));
        }
        let xn = point3d[0] / z;
        let yn = point3d[1] / z;
        let [xd, yd] = self.distort([xn, yn]);
        Ok([self.fx * xd + self.cx, self.fy * yd + self.cy])
    }

    /// Back-project pixel `[u, v]` to a unit ray in the camera frame.
    ///
    /// The returned vector has unit L2 norm.  Distortion is NOT undone here;
    /// use [`Self::undistort`] first for accurate results.
    ///
    /// ```
    /// # use scirs2_vision::camera::CameraIntrinsics;
    /// let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
    /// let ray = cam.unproject([320.0, 240.0]);
    /// assert!((ray[0]).abs() < 1e-12);
    /// assert!((ray[1]).abs() < 1e-12);
    /// assert!((ray[2] - 1.0).abs() < 1e-12);
    /// ```
    pub fn unproject(&self, pixel: [f64; 2]) -> [f64; 3] {
        let xn = (pixel[0] - self.cx) / self.fx;
        let yn = (pixel[1] - self.cy) / self.fy;
        let len = (xn * xn + yn * yn + 1.0).sqrt();
        [xn / len, yn / len, 1.0 / len]
    }

    /// Undistort a distorted pixel using Newton iterations.
    ///
    /// Inverts the Brown-Conrady model (max 20 iterations, stops when the
    /// residual falls below 1e-10 pixels).
    ///
    /// ```
    /// # use scirs2_vision::camera::CameraIntrinsics;
    /// let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
    /// let u = cam.undistort([320.0, 240.0]);
    /// assert!((u[0] - 320.0).abs() < 1e-9);
    /// assert!((u[1] - 240.0).abs() < 1e-9);
    /// ```
    pub fn undistort(&self, pixel: [f64; 2]) -> [f64; 2] {
        // Initial guess: normalised undistorted = normalised distorted
        let mut xn = (pixel[0] - self.cx) / self.fx;
        let mut yn = (pixel[1] - self.cy) / self.fy;

        for _ in 0..20 {
            let [xd, yd] = self.distort([xn, yn]);
            // Residual in pixel space
            let ex = pixel[0] - (self.fx * xd + self.cx);
            let ey = pixel[1] - (self.fy * yd + self.cy);
            if ex * ex + ey * ey < 1e-20 {
                break;
            }
            // Newton step (Jacobian ≈ identity for small distortions)
            xn += ex / self.fx;
            yn += ey / self.fy;
        }

        [self.fx * xn + self.cx, self.fy * yn + self.cy]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CameraExtrinsics
// ─────────────────────────────────────────────────────────────────────────────

/// Camera extrinsic parameters: a 3×3 rotation and a 3-element translation.
///
/// Together with [`CameraIntrinsics`] they define the complete camera model:
/// `p_cam = R * p_world + t`.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraExtrinsics {
    /// 3×3 rotation matrix (world → camera).
    pub rotation: [[f64; 3]; 3],
    /// 3-element translation vector (world → camera).
    pub translation: [f64; 3],
}

impl CameraExtrinsics {
    /// Create new extrinsics.
    pub fn new(rotation: [[f64; 3]; 3], translation: [f64; 3]) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Identity extrinsics (R = I₃, t = 0).
    pub fn identity() -> Self {
        Self {
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
        }
    }

    /// Transform a world-frame point into the camera frame.
    pub fn transform(&self, world_point: [f64; 3]) -> [f64; 3] {
        let r = &self.rotation;
        let t = &self.translation;
        let x =
            r[0][0] * world_point[0] + r[0][1] * world_point[1] + r[0][2] * world_point[2] + t[0];
        let y =
            r[1][0] * world_point[0] + r[1][1] * world_point[1] + r[1][2] * world_point[2] + t[1];
        let z =
            r[2][0] * world_point[0] + r[2][1] * world_point[1] + r[2][2] * world_point[2] + t[2];
        [x, y, z]
    }

    /// 4×4 homogeneous transformation matrix `[R | t; 0 0 0 1]`.
    pub fn as_matrix4(&self) -> [[f64; 4]; 4] {
        let r = &self.rotation;
        let t = &self.translation;
        [
            [r[0][0], r[0][1], r[0][2], t[0]],
            [r[1][0], r[1][1], r[1][2], t[1]],
            [r[2][0], r[2][1], r[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StereoCameraSystem
// ─────────────────────────────────────────────────────────────────────────────

/// A stereo camera rig consisting of left and right pinhole cameras.
///
/// The stereo geometry is described by a relative rotation `R` and translation
/// `T` such that a point `P_l` in the left-camera frame maps to the right
/// frame as `P_r = R * P_l + T`.
///
/// For a typical horizontal stereo setup `T ≈ [-baseline, 0, 0]`.
#[derive(Debug, Clone)]
pub struct StereoCameraSystem {
    /// Left camera intrinsics.
    pub left: CameraIntrinsics,
    /// Right camera intrinsics.
    pub right: CameraIntrinsics,
    /// Baseline distance between optical centres (metres).
    pub baseline: f64,
    /// Rotation from left to right camera frame.
    pub r: [[f64; 3]; 3],
    /// Translation from left to right camera frame (metres).
    pub t: [f64; 3],
}

impl StereoCameraSystem {
    /// Create a new stereo system.  The baseline is derived from `‖T‖`.
    pub fn new(
        left: CameraIntrinsics,
        right: CameraIntrinsics,
        r: [[f64; 3]; 3],
        t: [f64; 3],
    ) -> Self {
        let baseline = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
        Self {
            left,
            right,
            baseline,
            r,
            t,
        }
    }

    /// Convert disparity to metric depth using `depth = baseline * fx / disparity`.
    ///
    /// Returns `None` when `disparity ≤ 0`.
    ///
    /// ```
    /// # use scirs2_vision::camera::{CameraIntrinsics, StereoCameraSystem};
    /// let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
    /// let stereo = StereoCameraSystem::new(
    ///     cam.clone(), cam,
    ///     [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
    ///     [-0.1, 0.0, 0.0],
    /// );
    /// let depth = stereo.disparity_to_depth(80.0).unwrap();
    /// assert!((depth - 1.0).abs() < 1e-9);
    /// ```
    pub fn disparity_to_depth(&self, disparity: f64) -> Option<f64> {
        if disparity <= 0.0 {
            return None;
        }
        Some(self.baseline * self.left.fx / disparity)
    }

    /// Triangulate a 3-D point from stereo pixel correspondences using the
    /// **Direct Linear Transform** (mid-point method on the two rays).
    ///
    /// Both pixels are assumed to be in a **rectified** coordinate frame so
    /// that epipolar lines are horizontal.
    ///
    /// Returns `None` when the rays are nearly parallel.
    ///
    /// ```
    /// # use scirs2_vision::camera::{CameraIntrinsics, StereoCameraSystem};
    /// let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
    /// let stereo = StereoCameraSystem::new(
    ///     cam.clone(), cam,
    ///     [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
    ///     [-0.1, 0.0, 0.0],
    /// );
    /// // A point at (0, 0, 1 m): left pixel = (320, 240), right pixel = (240, 240)
    /// let pt = stereo.triangulate([320.0, 240.0], [240.0, 240.0]).unwrap();
    /// assert!((pt[2] - 1.0).abs() < 0.05);
    /// ```
    pub fn triangulate(&self, left_px: [f64; 2], right_px: [f64; 2]) -> Option<[f64; 3]> {
        // Left ray direction in left-camera frame
        let d1 = self.left.unproject(left_px);
        // Right ray in left-camera frame: d2 = R^T * right_unproject
        let d_r = self.right.unproject(right_px);
        let rt = mat3_transpose(&self.r);
        let d2 = mat3_vec3_mul(&rt, d_r);

        // Origin of right camera in left-camera frame: O2 = -R^T * t
        let neg_t = [-self.t[0], -self.t[1], -self.t[2]];
        let o2 = mat3_vec3_mul(&rt, neg_t);

        // Solve: o2 = s1*d1 - s2*d2  (mid-point method via least squares)
        // s1*(d1·d1) - s2*(d1·d2) = o2·d1
        // s1*(d1·d2) - s2*(d2·d2) = o2·d2
        let a = dot3(d1, d1);
        let b = dot3(d1, d2);
        let c = dot3(d2, d2);
        let det = a * c - b * b;
        if det.abs() < 1e-12 {
            return None;
        }
        let e = dot3(o2, d1);
        let f = dot3(o2, d2);
        let s1 = (e * c - f * b) / det;
        let s2 = (e * b - f * a) / det;

        // Mid-point between the two closest points on the rays
        let p1 = [d1[0] * s1, d1[1] * s1, d1[2] * s1];
        let p2_world = [o2[0] + d2[0] * s2, o2[1] + d2[1] * s2, o2[2] + d2[2] * s2];

        Some([
            (p1[0] + p2_world[0]) * 0.5,
            (p1[1] + p2_world[1]) * 0.5,
            (p1[2] + p2_world[2]) * 0.5,
        ])
    }

    /// Compute a disparity map from a rectified stereo pair using block matching.
    ///
    /// Both images must have the same dimensions.  Returns a disparity map of
    /// identical dimensions; pixels where no match was found carry value `0.0`.
    ///
    /// # Arguments
    /// * `left_img`  – Row-major grayscale image `[row][col]`.
    /// * `right_img` – Row-major grayscale image `[row][col]`.
    /// * `max_disparity` – Maximum disparity to search (pixels).
    /// * `block_size`    – Half-window radius (full window = 2*r+1 × 2*r+1).
    pub fn compute_disparity_map(
        &self,
        left_img: &[Vec<f64>],
        right_img: &[Vec<f64>],
        max_disparity: usize,
        block_size: usize,
    ) -> Vec<Vec<f64>> {
        let rows = left_img.len();
        if rows == 0 {
            return Vec::new();
        }
        let cols = left_img[0].len();
        let r = block_size;
        let mut disp = vec![vec![0.0f64; cols]; rows];

        #[allow(clippy::needless_range_loop)]
        for row in r..rows.saturating_sub(r) {
            for col in r..cols.saturating_sub(r) {
                let mut best_disp = 0usize;
                let mut best_sad = f64::INFINITY;

                let max_d = max_disparity.min(col.saturating_sub(r) + 1);
                for d in 0..max_d {
                    let right_col = col - d;
                    if right_col < r {
                        break;
                    }
                    let mut sad = 0.0f64;
                    'win: for dr in 0..=(2 * r) {
                        let lr = row + dr - r;
                        if lr >= rows {
                            break 'win;
                        }
                        for dc in 0..=(2 * r) {
                            let lc = col + dc - r;
                            let rc = right_col + dc - r;
                            if lc >= cols || rc >= cols {
                                continue;
                            }
                            sad += (left_img[lr][lc] - right_img[lr][rc]).abs();
                        }
                    }
                    if sad < best_sad {
                        best_sad = sad;
                        best_disp = d;
                    }
                }
                disp[row][col] = best_disp as f64;
            }
        }
        disp
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Small matrix helpers (private)
// ─────────────────────────────────────────────────────────────────────────────

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn mat3_transpose(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[inline]
fn mat3_vec3_mul(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_matrix() {
        let cam = CameraIntrinsics::ideal(500.0, 600.0, 320.0, 240.0);
        let k = cam.calibration_matrix();
        assert!((k[0][0] - 500.0).abs() < 1e-12);
        assert!((k[1][1] - 600.0).abs() < 1e-12);
        assert!((k[0][2] - 320.0).abs() < 1e-12);
        assert!((k[1][2] - 240.0).abs() < 1e-12);
        assert!((k[2][2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_project_undistorted() {
        let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
        let px = cam
            .project([1.0, 0.0, 2.0])
            .expect("project should succeed for point in front of camera");
        assert!((px[0] - 720.0).abs() < 1e-9, "u={}", px[0]);
        assert!((px[1] - 240.0).abs() < 1e-9, "v={}", px[1]);
    }

    #[test]
    fn test_project_negative_z_err() {
        let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
        assert!(cam.project([0.0, 0.0, -1.0]).is_err());
        assert!(cam.project([0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_unproject_principal_ray() {
        let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
        let ray = cam.unproject([320.0, 240.0]);
        assert!((ray[0]).abs() < 1e-12);
        assert!((ray[1]).abs() < 1e-12);
        // ray[2] should be 1/sqrt(1) = 1
        assert!((ray[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_undistort_principal_point() {
        let cam = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0, 0.1, 0.05, 0.0, 0.001, 0.001);
        let u = cam.undistort([320.0, 240.0]);
        assert!((u[0] - 320.0).abs() < 1e-9, "u={}", u[0]);
        assert!((u[1] - 240.0).abs() < 1e-9, "v={}", u[1]);
    }

    #[test]
    fn test_distort_zero() {
        let cam = CameraIntrinsics::new(800.0, 800.0, 320.0, 240.0, 0.2, 0.05, 0.01, 0.001, 0.001);
        let d = cam.distort([0.0, 0.0]);
        assert!((d[0]).abs() < 1e-12);
        assert!((d[1]).abs() < 1e-12);
    }

    #[test]
    fn test_extrinsics_identity() {
        let ext = CameraExtrinsics::identity();
        let pt = [1.0, 2.0, 3.0];
        let out = ext.transform(pt);
        assert!((out[0] - pt[0]).abs() < 1e-12);
        assert!((out[1] - pt[1]).abs() < 1e-12);
        assert!((out[2] - pt[2]).abs() < 1e-12);
    }

    #[test]
    fn test_disparity_to_depth() {
        let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
        let stereo = StereoCameraSystem::new(
            cam.clone(),
            cam,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [-0.1, 0.0, 0.0],
        );
        // depth = baseline * fx / disparity = 0.1 * 800 / 80 = 1.0
        let depth = stereo
            .disparity_to_depth(80.0)
            .expect("disparity_to_depth should return Some for positive disparity");
        assert!((depth - 1.0).abs() < 1e-9, "depth={}", depth);
    }

    #[test]
    fn test_disparity_zero_returns_none() {
        let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
        let stereo = StereoCameraSystem::new(
            cam.clone(),
            cam,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [-0.1, 0.0, 0.0],
        );
        assert!(stereo.disparity_to_depth(0.0).is_none());
        assert!(stereo.disparity_to_depth(-5.0).is_none());
    }

    #[test]
    fn test_triangulate_known_point() {
        // Horizontal stereo: left at origin, right shifted -0.1 m on X.
        // A point at (0, 0, 1 m):
        //   left  pixel = (320, 240)   (principal point, f=800)
        //   right pixel = (320 - 80, 240) = (240, 240)  (disparity = 80 px)
        let cam = CameraIntrinsics::ideal(800.0, 800.0, 320.0, 240.0);
        let stereo = StereoCameraSystem::new(
            cam.clone(),
            cam,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [-0.1, 0.0, 0.0],
        );
        let pt = stereo
            .triangulate([320.0, 240.0], [240.0, 240.0])
            .expect("triangulate should succeed for valid stereo observations");
        assert!((pt[2] - 1.0).abs() < 0.01, "Z={}", pt[2]);
    }
}
