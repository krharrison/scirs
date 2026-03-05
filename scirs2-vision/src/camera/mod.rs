//! Camera models for 3D vision
//!
//! Provides a [`PinholeCamera`] with radial/tangential distortion (Brown–Conrady model),
//! 3-D ↔ 2-D projection, iterative undistortion, and a [`StereoPair`] that bundles two
//! cameras with their relative pose.

pub mod intrinsics;
pub use intrinsics::{CameraExtrinsics, CameraIntrinsics, StereoCameraSystem};

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array1, Array2};

// ─────────────────────────────────────────────────────────────────────────────
// PinholeCamera
// ─────────────────────────────────────────────────────────────────────────────

/// Pinhole camera with radial and tangential (Brown–Conrady) distortion.
///
/// Coordinate convention: X right, Y down, Z forward (optical axis).
/// Projection (ideal):  `u = fx * X/Z + cx`,  `v = fy * Y/Z + cy`.
///
/// Distortion model (normalised coordinates `xn = X/Z`, `yn = Y/Z`):
///
/// ```text
/// r² = xn² + yn²
/// xd = xn*(1 + k1*r² + k2*r⁴) + 2*p1*xn*yn + p2*(r² + 2*xn²)
/// yd = yn*(1 + k1*r² + k2*r⁴) + p1*(r² + 2*yn²) + 2*p2*xn*yn
/// u = fx*xd + cx,  v = fy*yd + cy
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PinholeCamera {
    /// Focal length in pixels along X.
    pub fx: f64,
    /// Focal length in pixels along Y.
    pub fy: f64,
    /// Principal point X (pixels).
    pub cx: f64,
    /// Principal point Y (pixels).
    pub cy: f64,
    /// Radial distortion coefficient k1.
    pub k1: f64,
    /// Radial distortion coefficient k2.
    pub k2: f64,
    /// Tangential distortion coefficient p1.
    pub p1: f64,
    /// Tangential distortion coefficient p2.
    pub p2: f64,
}

impl PinholeCamera {
    /// Create a new camera with all parameters explicit.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, k1: f64, k2: f64, p1: f64, p2: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            k1,
            k2,
            p1,
            p2,
        }
    }

    /// Create a camera with no distortion (k1=k2=p1=p2=0).
    pub fn ideal(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self::new(fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0)
    }

    /// Project a 3-D point to a 2-D pixel using the **ideal** pinhole model
    /// (distortion coefficients are ignored).
    ///
    /// Returns `Err` if `Z ≤ 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_vision::camera::PinholeCamera;
    ///
    /// let cam = PinholeCamera::ideal(800.0, 800.0, 320.0, 240.0);
    /// let px = cam.project(&[0.0, 0.0, 1.0]).unwrap();
    /// assert!((px[0] - 320.0).abs() < 1e-9);
    /// assert!((px[1] - 240.0).abs() < 1e-9);
    /// ```
    pub fn project(&self, point_3d: &[f64; 3]) -> Result<[f64; 2]> {
        let z = point_3d[2];
        if z <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "point_3d[2] (Z) must be positive".to_string(),
            ));
        }
        let xn = point_3d[0] / z;
        let yn = point_3d[1] / z;
        Ok([self.fx * xn + self.cx, self.fy * yn + self.cy])
    }

    /// Project a 3-D point to a 2-D pixel applying radial and tangential distortion.
    ///
    /// Returns `Err` if `Z ≤ 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_vision::camera::PinholeCamera;
    ///
    /// let cam = PinholeCamera::new(800.0, 800.0, 320.0, 240.0, 0.1, 0.0, 0.0, 0.0);
    /// let px = cam.project_distorted(&[0.0, 0.0, 1.0]).unwrap();
    /// // On the principal axis distortion has no effect.
    /// assert!((px[0] - 320.0).abs() < 1e-9);
    /// assert!((px[1] - 240.0).abs() < 1e-9);
    /// ```
    pub fn project_distorted(&self, point_3d: &[f64; 3]) -> Result<[f64; 2]> {
        let z = point_3d[2];
        if z <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "point_3d[2] (Z) must be positive".to_string(),
            ));
        }
        let xn = point_3d[0] / z;
        let yn = point_3d[1] / z;
        let (xd, yd) = apply_distortion(self.k1, self.k2, self.p1, self.p2, xn, yn);
        Ok([self.fx * xd + self.cx, self.fy * yd + self.cy])
    }

    /// Back-project a 2-D pixel plus a known depth into a 3-D point (inverse of
    /// the ideal projection, distortion not reversed here).
    ///
    /// # Arguments
    ///
    /// * `pixel` – `[u, v]` pixel coordinates.
    /// * `depth` – depth along the optical axis (Z), must be positive.
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_vision::camera::PinholeCamera;
    ///
    /// let cam = PinholeCamera::ideal(800.0, 800.0, 320.0, 240.0);
    /// let pt = cam.backproject(&[320.0, 240.0], 5.0);
    /// assert!((pt[2] - 5.0).abs() < 1e-9);
    /// ```
    pub fn backproject(&self, pixel: &[f64; 2], depth: f64) -> [f64; 3] {
        let xn = (pixel[0] - self.cx) / self.fx;
        let yn = (pixel[1] - self.cy) / self.fy;
        [xn * depth, yn * depth, depth]
    }

    /// Iteratively undistort a distorted pixel coordinate.
    ///
    /// Uses Newton-style fixed-point iteration (max 20 steps) to invert the
    /// Brown–Conrady distortion model.
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_vision::camera::PinholeCamera;
    ///
    /// let cam = PinholeCamera::new(800.0, 800.0, 320.0, 240.0, 0.1, 0.0, 0.0, 0.0);
    /// // A point on the principal axis is unaffected by distortion.
    /// let undist = cam.undistort_point(&[320.0, 240.0]);
    /// assert!((undist[0] - 320.0).abs() < 1e-9);
    /// assert!((undist[1] - 240.0).abs() < 1e-9);
    /// ```
    pub fn undistort_point(&self, pixel: &[f64; 2]) -> [f64; 2] {
        // Normalised distorted coordinates.
        let mut xn = (pixel[0] - self.cx) / self.fx;
        let mut yn = (pixel[1] - self.cy) / self.fy;

        // Fixed-point iteration: find (xn0, yn0) such that distort(xn0, yn0) = (xn, yn).
        for _ in 0..20 {
            let r2 = xn * xn + yn * yn;
            let rad = 1.0 + self.k1 * r2 + self.k2 * r2 * r2;
            let dx = 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn);
            let dy = self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn;

            // Distorted value at current estimate.
            let xd_curr = xn * rad + dx;
            let yd_curr = yn * rad + dy;

            // Observed normalised distorted coords.
            let xd_obs = (pixel[0] - self.cx) / self.fx;
            let yd_obs = (pixel[1] - self.cy) / self.fy;

            // Residual-based correction.
            xn += (xd_obs - xd_curr) / rad.max(1e-8);
            yn += (yd_obs - yd_curr) / rad.max(1e-8);
        }

        [self.fx * xn + self.cx, self.fy * yn + self.cy]
    }

    /// Return the 3×3 intrinsic (calibration) matrix K as an `Array2<f64>`.
    ///
    /// ```text
    /// K = [[fx,  0, cx],
    ///      [ 0, fy, cy],
    ///      [ 0,  0,  1]]
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_vision::camera::PinholeCamera;
    ///
    /// let cam = PinholeCamera::ideal(400.0, 400.0, 200.0, 150.0);
    /// let k = cam.intrinsic_matrix();
    /// assert_eq!(k.shape(), &[3, 3]);
    /// assert!((k[[0, 0]] - 400.0).abs() < 1e-9);
    /// ```
    pub fn intrinsic_matrix(&self) -> Array2<f64> {
        let mut k = Array2::<f64>::zeros((3, 3));
        k[[0, 0]] = self.fx;
        k[[0, 2]] = self.cx;
        k[[1, 1]] = self.fy;
        k[[1, 2]] = self.cy;
        k[[2, 2]] = 1.0;
        k
    }

    /// Distortion coefficient vector `[k1, k2, p1, p2]`.
    pub fn distortion_coeffs(&self) -> [f64; 4] {
        [self.k1, self.k2, self.p1, self.p2]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Distortion helper
// ─────────────────────────────────────────────────────────────────────────────

/// Apply Brown–Conrady radial + tangential distortion to normalised image coords.
#[inline]
fn apply_distortion(k1: f64, k2: f64, p1: f64, p2: f64, xn: f64, yn: f64) -> (f64, f64) {
    let r2 = xn * xn + yn * yn;
    let rad = 1.0 + k1 * r2 + k2 * r2 * r2;
    let xd = xn * rad + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
    let yd = yn * rad + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
    (xd, yd)
}

// ─────────────────────────────────────────────────────────────────────────────
// StereoPair
// ─────────────────────────────────────────────────────────────────────────────

/// A rectified stereo camera pair with a known relative pose.
///
/// The relative pose satisfies: `P_right = R * P_left + t`, where `P` denotes
/// a 3-D point expressed in each camera's coordinate frame.
#[derive(Debug, Clone)]
pub struct StereoPair {
    /// Left camera intrinsics and distortion.
    pub left: PinholeCamera,
    /// Right camera intrinsics and distortion.
    pub right: PinholeCamera,
    /// 3×3 rotation matrix from left to right camera frame.
    pub rotation: Array2<f64>,
    /// 3-vector translation from left to right camera frame (metres).
    pub translation: Array1<f64>,
}

impl StereoPair {
    /// Create a new `StereoPair`.
    ///
    /// # Arguments
    ///
    /// * `left`        – Left camera intrinsics/distortion.
    /// * `right`       – Right camera intrinsics/distortion.
    /// * `rotation`    – 3×3 rotation matrix `R` (right = R * left + t).
    /// * `translation` – 3-vector translation `t`.
    ///
    /// # Errors
    ///
    /// Returns [`VisionError::InvalidParameter`] when `rotation` is not 3×3 or
    /// `translation` does not have length 3.
    pub fn new(
        left: PinholeCamera,
        right: PinholeCamera,
        rotation: Array2<f64>,
        translation: Array1<f64>,
    ) -> Result<Self> {
        if rotation.shape() != [3, 3] {
            return Err(VisionError::InvalidParameter(
                "rotation must be a 3×3 matrix".to_string(),
            ));
        }
        if translation.len() != 3 {
            return Err(VisionError::InvalidParameter(
                "translation must have length 3".to_string(),
            ));
        }
        Ok(Self {
            left,
            right,
            rotation,
            translation,
        })
    }

    /// Approximate stereo baseline (‖t‖).
    pub fn baseline(&self) -> f64 {
        self.translation.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    /// Project a 3-D point (in the **left** camera frame) to both image planes.
    ///
    /// Returns `Ok(([ul, vl], [ur, vr]))` on success.
    pub fn project_both(&self, point_left: &[f64; 3]) -> Result<([f64; 2], [f64; 2])> {
        // Left projection.
        let px_l = self.left.project(point_left)?;

        // Transform to right frame: P_r = R*P_l + t.
        let r = &self.rotation;
        let t = &self.translation;
        let x = r[[0, 0]] * point_left[0]
            + r[[0, 1]] * point_left[1]
            + r[[0, 2]] * point_left[2]
            + t[0];
        let y = r[[1, 0]] * point_left[0]
            + r[[1, 1]] * point_left[1]
            + r[[1, 2]] * point_left[2]
            + t[1];
        let z = r[[2, 0]] * point_left[0]
            + r[[2, 1]] * point_left[1]
            + r[[2, 2]] * point_left[2]
            + t[2];

        let px_r = self.right.project(&[x, y, z])?;
        Ok((px_l, px_r))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_principal_axis() {
        let cam = PinholeCamera::ideal(800.0, 800.0, 320.0, 240.0);
        let px = cam
            .project(&[0.0, 0.0, 1.0])
            .expect("project should succeed for point in front of camera");
        assert!((px[0] - 320.0).abs() < 1e-9, "u = {}", px[0]);
        assert!((px[1] - 240.0).abs() < 1e-9, "v = {}", px[1]);
    }

    #[test]
    fn test_project_behind_camera() {
        let cam = PinholeCamera::ideal(800.0, 800.0, 320.0, 240.0);
        assert!(cam.project(&[0.0, 0.0, 0.0]).is_err());
        assert!(cam.project(&[0.0, 0.0, -1.0]).is_err());
    }

    #[test]
    fn test_backproject_roundtrip() {
        let cam = PinholeCamera::ideal(800.0, 800.0, 320.0, 240.0);
        let pt3d = [1.0, -0.5, 3.0];
        let px = cam
            .project(&pt3d)
            .expect("project should succeed for valid 3D point");
        let back = cam.backproject(&px, pt3d[2]);
        assert!((back[0] - pt3d[0]).abs() < 1e-9, "X = {}", back[0]);
        assert!((back[1] - pt3d[1]).abs() < 1e-9, "Y = {}", back[1]);
        assert!((back[2] - pt3d[2]).abs() < 1e-9, "Z = {}", back[2]);
    }

    #[test]
    fn test_distorted_on_axis() {
        // Points on the principal axis should be unaffected by distortion.
        let cam = PinholeCamera::new(800.0, 800.0, 320.0, 240.0, 0.2, 0.05, 0.001, 0.001);
        let ideal = cam
            .project(&[0.0, 0.0, 1.0])
            .expect("project should succeed for principal axis");
        let dist = cam
            .project_distorted(&[0.0, 0.0, 1.0])
            .expect("project_distorted should succeed for principal axis");
        assert!((ideal[0] - dist[0]).abs() < 1e-12);
        assert!((ideal[1] - dist[1]).abs() < 1e-12);
    }

    #[test]
    fn test_distorted_off_axis() {
        // With radial distortion the distorted pixel should differ from ideal.
        let cam = PinholeCamera::new(800.0, 800.0, 320.0, 240.0, 0.2, 0.0, 0.0, 0.0);
        let ideal = cam
            .project(&[1.0, 1.0, 2.0])
            .expect("project should succeed for off-axis point");
        let dist = cam
            .project_distorted(&[1.0, 1.0, 2.0])
            .expect("project_distorted should succeed for off-axis point");
        // k1 > 0 → barrel distortion → pixel moves outward from principal point.
        assert!(
            (dist[0] - 320.0).abs() > (ideal[0] - 320.0).abs(),
            "dist.u={}, ideal.u={}",
            dist[0],
            ideal[0]
        );
    }

    #[test]
    fn test_undistort_point_principal_axis() {
        let cam = PinholeCamera::new(800.0, 800.0, 320.0, 240.0, 0.1, 0.02, 0.0, 0.0);
        let undist = cam.undistort_point(&[320.0, 240.0]);
        assert!((undist[0] - 320.0).abs() < 1e-9);
        assert!((undist[1] - 240.0).abs() < 1e-9);
    }

    #[test]
    fn test_intrinsic_matrix() {
        let cam = PinholeCamera::ideal(400.0, 500.0, 200.0, 150.0);
        let k = cam.intrinsic_matrix();
        assert_eq!(k.shape(), &[3, 3]);
        assert!((k[[0, 0]] - 400.0).abs() < 1e-12);
        assert!((k[[1, 1]] - 500.0).abs() < 1e-12);
        assert!((k[[0, 2]] - 200.0).abs() < 1e-12);
        assert!((k[[1, 2]] - 150.0).abs() < 1e-12);
        assert!((k[[2, 2]] - 1.0).abs() < 1e-12);
        assert!((k[[0, 1]]).abs() < 1e-12); // skew = 0
    }

    #[test]
    fn test_stereo_pair_baseline() {
        use scirs2_core::ndarray::{Array1, Array2};
        let cam = PinholeCamera::ideal(800.0, 800.0, 320.0, 240.0);
        let r = Array2::<f64>::eye(3);
        let t = Array1::from_vec(vec![-0.1, 0.0, 0.0]);
        let stereo = StereoPair::new(cam.clone(), cam, r, t)
            .expect("StereoPair::new should succeed with valid inputs");
        assert!((stereo.baseline() - 0.1).abs() < 1e-9);
    }
}
