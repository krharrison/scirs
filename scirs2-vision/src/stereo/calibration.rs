//! Stereo camera calibration data structures and reprojection utilities.
//!
//! Provides [`CameraIntrinsics`], [`StereoCalibration`], and the Q-matrix
//! (`reprojectImageTo3D`) for converting disparity maps to 3-D point clouds.

use crate::stereo::disparity::DisparityMap;
use crate::stereo::reconstruction::PointCloud;

// ─────────────────────────────────────────────────────────────────────────────
// CameraIntrinsics
// ─────────────────────────────────────────────────────────────────────────────

/// Pinhole camera intrinsic parameters with Brown–Conrady distortion.
#[derive(Debug, Clone)]
pub struct CameraIntrinsics {
    /// Horizontal focal length in pixels.
    pub fx: f64,
    /// Vertical focal length in pixels.
    pub fy: f64,
    /// Principal point x.
    pub cx: f64,
    /// Principal point y.
    pub cy: f64,
    /// Radial (k1, k2, k3) and tangential (p1, p2) distortion coefficients
    /// stored as `[k1, k2, p1, p2, k3]`.
    pub distortion: [f64; 5],
}

impl CameraIntrinsics {
    /// Create an ideal (zero-distortion) pinhole camera.
    pub fn ideal(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            distortion: [0.0; 5],
        }
    }

    /// Undistort a single normalised image point using the Brown–Conrady model.
    ///
    /// `(u, v)` are pixel coordinates; the returned pair is the corrected pixel.
    pub fn undistort_point(&self, u: f64, v: f64) -> (f64, f64) {
        let xn = (u - self.cx) / self.fx;
        let yn = (v - self.cy) / self.fy;
        let r2 = xn * xn + yn * yn;
        let [k1, k2, p1, p2, k3] = self.distortion;

        let radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
        let xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
        let yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;

        (xd * self.fx + self.cx, yd * self.fy + self.cy)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StereoCalibration
// ─────────────────────────────────────────────────────────────────────────────

/// Stereo camera calibration: intrinsics for each camera, extrinsic rotation R
/// and translation T from right to left camera frame, and derived baseline.
#[derive(Debug, Clone)]
pub struct StereoCalibration {
    /// Left camera intrinsics.
    pub left: CameraIntrinsics,
    /// Right camera intrinsics.
    pub right: CameraIntrinsics,
    /// 3×3 rotation matrix — right camera frame expressed in left frame.
    pub rotation: [[f64; 3]; 3],
    /// 3-vector translation from left camera origin to right camera origin,
    /// in left camera coordinates.  `translation[0]` is the baseline along X.
    pub translation: [f64; 3],
    /// Stereo baseline magnitude in the same unit as `translation`.
    pub baseline: f64,
}

impl StereoCalibration {
    /// Construct from explicit intrinsics and extrinsics.
    pub fn new(
        left: CameraIntrinsics,
        right: CameraIntrinsics,
        rotation: [[f64; 3]; 3],
        translation: [f64; 3],
    ) -> Self {
        let baseline = translation[0].abs();
        Self {
            left,
            right,
            rotation,
            translation,
            baseline,
        }
    }

    /// Convenience constructor: identical cameras separated by `baseline` along X.
    ///
    /// Suitable for an ideal rectified stereo rig.
    pub fn from_baseline(fx: f64, fy: f64, cx: f64, cy: f64, baseline: f64) -> Self {
        let cam = CameraIntrinsics::ideal(fx, fy, cx, cy);
        Self {
            left: cam.clone(),
            right: cam,
            rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            translation: [-baseline, 0.0, 0.0],
            baseline,
        }
    }

    /// Compute the 4×4 **Q** matrix (Hartley & Zisserman §11.12) used by
    /// `reprojectImageTo3D`.
    ///
    /// For a rectified parallel stereo pair the Q matrix is:
    ///
    /// ```text
    /// Q = | 1   0   0   -cx        |
    ///     | 0   1   0   -cy        |
    ///     | 0   0   0    f         |
    ///     | 0   0  -1/b  (cx-cx')/b|
    /// ```
    ///
    /// where `cx' = cx` for identical cameras (the last entry is 0).
    pub fn q_matrix(&self) -> [[f64; 4]; 4] {
        let f = self.left.fx;
        let cx = self.left.cx;
        let cy = self.left.cy;
        let cx_prime = self.right.cx;
        let b = self.baseline;
        [
            [1.0, 0.0, 0.0, -cx],
            [0.0, 1.0, 0.0, -cy],
            [0.0, 0.0, 0.0, f],
            [0.0, 0.0, -1.0 / b, (cx - cx_prime) / b],
        ]
    }

    /// Reproject a disparity map to a 3-D point cloud using the Q matrix.
    ///
    /// Pixels with disparity ≤ 0 are skipped.
    pub fn reproject_to_3d(&self, disparity: &DisparityMap) -> PointCloud {
        let q = self.q_matrix();
        let mut points = Vec::new();

        for row in 0..disparity.height {
            for col in 0..disparity.width {
                let d = disparity.get(row, col) as f64;
                if d <= 0.0 {
                    continue;
                }

                let xu = col as f64;
                let yu = row as f64;

                // Homogeneous 3-D point:  P_h = Q * [x, y, d, 1]^T
                let wx = q[0][0] * xu + q[0][3];
                let wy = q[1][1] * yu + q[1][3];
                let wz = q[2][3]; // q[2][3]*1.0 (last column element)
                let ww = q[3][2] * d + q[3][3];

                if ww.abs() > 1e-10 {
                    // Ensure positive depth: the Q matrix convention may
                    // produce negative ww. We take |ww| and keep the sign
                    // consistent so that Z = f·B/d > 0 (in front of the
                    // camera).
                    let x3d = wx / ww;
                    let y3d = wy / ww;
                    let z3d = (wz / ww).abs(); // depth is always positive
                    points.push([x3d as f32, y3d as f32, z3d as f32]);
                }
            }
        }

        PointCloud {
            points,
            colors: None,
            normals: None,
        }
    }

    /// Compute the expected disparity for a point at known depth `z`.
    ///
    /// Useful for evaluating calibration accuracy.
    pub fn depth_to_disparity(&self, z: f64) -> f64 {
        if z.abs() < 1e-10 {
            return 0.0;
        }
        self.baseline * self.left.fx / z
    }

    /// Compute depth from a disparity value.
    pub fn disparity_to_depth(&self, disparity: f64) -> f64 {
        if disparity.abs() < 1e-10 {
            return 0.0;
        }
        self.baseline * self.left.fx / disparity
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_baseline_baseline_field() {
        let cal = StereoCalibration::from_baseline(500.0, 500.0, 320.0, 240.0, 0.12);
        assert!((cal.baseline - 0.12).abs() < 1e-9);
    }

    #[test]
    fn test_q_matrix_dimensions() {
        let cal = StereoCalibration::from_baseline(500.0, 500.0, 320.0, 240.0, 0.1);
        let q = cal.q_matrix();
        assert_eq!(q.len(), 4);
        assert_eq!(q[0].len(), 4);
        // Q[2][3] should be focal length.
        assert!((q[2][3] - 500.0).abs() < 1e-9, "Q[2][3]={}", q[2][3]);
        // Q[3][2] should be -1/baseline.
        assert!((q[3][2] - (-1.0 / 0.1)).abs() < 1e-6, "Q[3][2]={}", q[3][2]);
    }

    #[test]
    fn test_reproject_to_3d_basic() {
        let cal = StereoCalibration::from_baseline(500.0, 500.0, 32.0, 24.0, 0.1);
        let mut disp = DisparityMap::new(64, 48, 0, 32);
        // Set a disparity at the principal point.
        disp.set(24, 32, 10.0); // depth = 0.1 * 500 / 10 = 5 m

        let pc = cal.reproject_to_3d(&disp);
        assert_eq!(pc.len(), 1);
        let p = pc.points[0];
        assert!(p[2] > 0.0, "z should be positive (in front of camera)");
    }

    #[test]
    fn test_depth_disparity_roundtrip() {
        let cal = StereoCalibration::from_baseline(600.0, 600.0, 320.0, 240.0, 0.15);
        let z = 3.0;
        let d = cal.depth_to_disparity(z);
        let z2 = cal.disparity_to_depth(d);
        assert!((z - z2).abs() < 1e-6, "roundtrip z={z} → d={d} → z={z2}");
    }

    #[test]
    fn test_undistort_point_zero_distortion() {
        let cam = CameraIntrinsics::ideal(500.0, 500.0, 320.0, 240.0);
        let (u2, v2) = cam.undistort_point(350.0, 270.0);
        // Zero distortion → point is unchanged.
        assert!((u2 - 350.0).abs() < 1e-4, "u={u2}");
        assert!((v2 - 270.0).abs() < 1e-4, "v={v2}");
    }
}
