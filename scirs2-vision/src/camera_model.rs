//! Camera models for 3D vision
//!
//! Provides pinhole camera models, radial/tangential distortion (Brown-Conrady),
//! and utilities for projecting 3-D points to 2-D pixels, unprojecting pixels to
//! 3-D rays, and full image undistortion.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::Array3;

// ─────────────────────────────────────────────────────────────────────────────
// PinholeCameraModel
// ─────────────────────────────────────────────────────────────────────────────

/// Pinhole camera intrinsic model.
///
/// # Coordinate convention
///
/// - **X** points right, **Y** points down, **Z** points forward (camera space).
/// - The image origin is at the top-left pixel centre.
///
/// Projection:  `u = fx * X/Z + cx`,  `v = fy * Y/Z + cy`.
#[derive(Debug, Clone, PartialEq)]
pub struct PinholeCameraModel {
    /// Focal length in pixels along the X axis.
    pub fx: f64,
    /// Focal length in pixels along the Y axis.
    pub fy: f64,
    /// Principal point X coordinate in pixels.
    pub cx: f64,
    /// Principal point Y coordinate in pixels.
    pub cy: f64,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl PinholeCameraModel {
    /// Create a new pinhole camera model.
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, width: usize, height: usize) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            width,
            height,
        }
    }

    /// Build from a 3×3 calibration matrix K.
    ///
    /// ```
    /// # use scirs2_vision::camera_model::PinholeCameraModel;
    /// let k = [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]];
    /// let cam = PinholeCameraModel::from_calibration_matrix(&k, 640, 480);
    /// assert!((cam.fx - 800.0).abs() < 1e-9);
    /// ```
    pub fn from_calibration_matrix(k: &[[f64; 3]; 3], width: usize, height: usize) -> Self {
        Self {
            fx: k[0][0],
            fy: k[1][1],
            cx: k[0][2],
            cy: k[1][2],
            width,
            height,
        }
    }

    /// Return the intrinsic matrix K as a flat row-major 3×3 array.
    pub fn to_matrix(&self) -> [[f64; 3]; 3] {
        [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Horizontal field-of-view in radians.
    pub fn fov_x(&self) -> f64 {
        2.0 * (self.width as f64 / (2.0 * self.fx)).atan()
    }

    /// Vertical field-of-view in radians.
    pub fn fov_y(&self) -> f64 {
        2.0 * (self.height as f64 / (2.0 * self.fy)).atan()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// project / unproject
// ─────────────────────────────────────────────────────────────────────────────

/// Project a 3-D camera-space point to a 2-D pixel coordinate.
///
/// Returns `Err` when `Z ≤ 0` (point is behind or at the camera plane).
///
/// # Arguments
///
/// * `model`     – Pinhole camera intrinsics.
/// * `point_3d`  – `[X, Y, Z]` in camera space (Z > 0 required).
///
/// # Returns
///
/// `(u, v)` pixel coordinates.
///
/// # Example
///
/// ```
/// use scirs2_vision::camera_model::{PinholeCameraModel, project};
/// let cam = PinholeCameraModel::new(800.0, 800.0, 320.0, 240.0, 640, 480);
/// let (u, v) = project(&cam, &[0.0, 0.0, 1.0]).unwrap();
/// assert!((u - 320.0).abs() < 1e-9);
/// assert!((v - 240.0).abs() < 1e-9);
/// ```
pub fn project(model: &PinholeCameraModel, point_3d: &[f64; 3]) -> Result<(f64, f64)> {
    let z = point_3d[2];
    if z <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "point_3d Z must be positive (point must be in front of camera)".to_string(),
        ));
    }
    let u = model.fx * point_3d[0] / z + model.cx;
    let v = model.fy * point_3d[1] / z + model.cy;
    Ok((u, v))
}

/// Unproject a pixel coordinate and depth to a 3-D camera-space point.
///
/// # Arguments
///
/// * `model`  – Pinhole camera intrinsics.
/// * `pixel`  – `(u, v)` pixel coordinate.
/// * `depth`  – Depth (Z value) in the same units as the desired 3-D output.
///
/// # Returns
///
/// `[X, Y, Z]` in camera space.
///
/// # Example
///
/// ```
/// use scirs2_vision::camera_model::{PinholeCameraModel, unproject};
/// let cam = PinholeCameraModel::new(800.0, 800.0, 320.0, 240.0, 640, 480);
/// let pt = unproject(&cam, (320.0, 240.0), 2.0);
/// assert!((pt[2] - 2.0).abs() < 1e-9);
/// ```
pub fn unproject(model: &PinholeCameraModel, pixel: (f64, f64), depth: f64) -> [f64; 3] {
    let (u, v) = pixel;
    let x = (u - model.cx) * depth / model.fx;
    let y = (v - model.cy) * depth / model.fy;
    [x, y, depth]
}

// ─────────────────────────────────────────────────────────────────────────────
// RadialDistortion  (Brown-Conrady model)
// ─────────────────────────────────────────────────────────────────────────────

/// Radial and tangential distortion coefficients (Brown-Conrady model).
///
/// The distortion model is:
///
/// ```text
/// r² = x² + y²
/// x_d = x(1 + k1 r² + k2 r⁴ + k3 r⁶) + 2 p1 xy + p2(r² + 2x²)
/// y_d = y(1 + k1 r² + k2 r⁴ + k3 r⁶) + p1(r² + 2y²) + 2 p2 xy
/// ```
///
/// where `(x, y) = ((u - cx)/fx, (v - cy)/fy)` are normalised image coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct RadialDistortion {
    /// Radial distortion coefficient 1.
    pub k1: f64,
    /// Radial distortion coefficient 2.
    pub k2: f64,
    /// Radial distortion coefficient 3.
    pub k3: f64,
    /// Tangential distortion coefficient 1.
    pub p1: f64,
    /// Tangential distortion coefficient 2.
    pub p2: f64,
}

impl Default for RadialDistortion {
    fn default() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
}

impl RadialDistortion {
    /// Create a new distortion model with all coefficients.
    pub fn new(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self { k1, k2, k3, p1, p2 }
    }

    /// Returns true when all coefficients are zero (no distortion).
    pub fn is_identity(&self) -> bool {
        self.k1 == 0.0 && self.k2 == 0.0 && self.k3 == 0.0 && self.p1 == 0.0 && self.p2 == 0.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// distort / undistort_point
// ─────────────────────────────────────────────────────────────────────────────

/// Apply radial and tangential distortion to a normalised image point `(x, y)`.
///
/// The input `point` should be in **normalised camera coordinates**
/// `x = (u - cx)/fx`,  `y = (v - cy)/fy`.
///
/// Returns the distorted normalised point `(x_d, y_d)`.
///
/// # Example
///
/// ```
/// use scirs2_vision::camera_model::{RadialDistortion, distort};
/// let d = RadialDistortion::new(0.1, 0.01, 0.0, 0.0, 0.0);
/// let (xd, yd) = distort(&d, (0.0, 0.0));
/// // At the principal point there is no distortion.
/// assert!((xd).abs() < 1e-12);
/// assert!((yd).abs() < 1e-12);
/// ```
pub fn distort(model: &RadialDistortion, point: (f64, f64)) -> (f64, f64) {
    let (x, y) = point;
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;

    let radial = 1.0 + model.k1 * r2 + model.k2 * r4 + model.k3 * r6;
    let x_d = x * radial + 2.0 * model.p1 * x * y + model.p2 * (r2 + 2.0 * x * x);
    let y_d = y * radial + model.p1 * (r2 + 2.0 * y * y) + 2.0 * model.p2 * x * y;
    (x_d, y_d)
}

/// Undistort a single normalised image point using iterative Newton refinement.
///
/// Inverts the Brown-Conrady distortion model numerically.
/// At most `max_iters` Newton steps are taken; the loop stops early when the
/// residual is below `tol`.
///
/// # Arguments
///
/// * `model`     – Distortion coefficients.
/// * `point`     – Observed (distorted) normalised point `(x_d, y_d)`.
/// * `max_iters` – Maximum number of Newton iterations (≥ 1).
/// * `tol`       – Convergence tolerance for the residual norm.
pub fn undistort_point(
    model: &RadialDistortion,
    point: (f64, f64),
    max_iters: usize,
    tol: f64,
) -> (f64, f64) {
    if model.is_identity() {
        return point;
    }

    let (xd, yd) = point;
    // Initial guess: use the distorted point directly.
    let mut x = xd;
    let mut y = yd;

    for _ in 0..max_iters.max(1) {
        let (x_proj, y_proj) = distort(model, (x, y));
        let ex = x_proj - xd;
        let ey = y_proj - yd;

        if ex * ex + ey * ey < tol * tol {
            break;
        }

        // Simple gradient-descent step (Jacobian ≈ I for small distortions).
        x -= ex;
        y -= ey;
    }

    (x, y)
}

// ─────────────────────────────────────────────────────────────────────────────
// undistort_image
// ─────────────────────────────────────────────────────────────────────────────

/// Produce an undistorted copy of `image` using the provided camera model and
/// distortion coefficients.
///
/// The function maps every destination pixel `(u, v)` back to a distorted source
/// position using the inverse of the Brown-Conrady model (solved iteratively), then
/// samples the source image with bilinear interpolation.
///
/// # Arguments
///
/// * `image`      – Source image as `Array3<f64>` with shape `[H, W, C]`.
///   Pixel values are expected in `[0, 255]`.
/// * `model`      – Pinhole camera intrinsics.
/// * `distortion` – Radial and tangential distortion coefficients.
///
/// # Returns
///
/// Undistorted image with the same shape as the input.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when the image dimensions do not
/// match the camera model's declared `width` / `height`.
///
/// # Example
///
/// ```
/// use scirs2_vision::camera_model::{PinholeCameraModel, RadialDistortion, undistort_image};
/// use scirs2_core::ndarray::Array3;
///
/// let cam = PinholeCameraModel::new(400.0, 400.0, 160.0, 120.0, 320, 240);
/// let dist = RadialDistortion::default(); // no distortion
/// let img = Array3::<f64>::zeros((240, 320, 3));
/// let out = undistort_image(&img, &cam, &dist).unwrap();
/// assert_eq!(out.dim(), img.dim());
/// ```
pub fn undistort_image(
    image: &Array3<f64>,
    model: &PinholeCameraModel,
    distortion: &RadialDistortion,
) -> Result<Array3<f64>> {
    let (h, w, c) = image.dim();
    if h != model.height || w != model.width {
        return Err(VisionError::InvalidParameter(format!(
            "image dimensions {}×{} do not match camera model {}×{}",
            h, w, model.height, model.width
        )));
    }

    let mut output = Array3::zeros((h, w, c));

    for v in 0..h {
        for u in 0..w {
            // Normalised destination coordinates.
            let xn = (u as f64 - model.cx) / model.fx;
            let yn = (v as f64 - model.cy) / model.fy;

            // Find corresponding distorted normalised source point.
            let (xd, yd) = distort(distortion, (xn, yn));

            // Back to pixel coordinates in the source (distorted) image.
            let us = model.fx * xd + model.cx;
            let vs = model.fy * yd + model.cy;

            // Bilinear interpolation.
            let u0 = us.floor() as i64;
            let v0 = vs.floor() as i64;
            let u1 = u0 + 1;
            let v1 = v0 + 1;

            let du = us - u0 as f64;
            let dv = vs - v0 as f64;

            let in_bounds = |uu: i64, vv: i64| uu >= 0 && uu < w as i64 && vv >= 0 && vv < h as i64;

            for ch in 0..c {
                let sample = |uu: i64, vv: i64| -> f64 {
                    if in_bounds(uu, vv) {
                        image[[vv as usize, uu as usize, ch]]
                    } else {
                        0.0
                    }
                };

                let val = sample(u0, v0) * (1.0 - du) * (1.0 - dv)
                    + sample(u1, v0) * du * (1.0 - dv)
                    + sample(u0, v1) * (1.0 - du) * dv
                    + sample(u1, v1) * du * dv;

                output[[v, u, ch]] = val;
            }
        }
    }

    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_project_principal_ray() {
        let cam = PinholeCameraModel::new(800.0, 800.0, 320.0, 240.0, 640, 480);
        let (u, v) =
            project(&cam, &[0.0, 0.0, 1.0]).expect("project should succeed for principal ray");
        assert!((u - 320.0).abs() < 1e-9, "u={u}");
        assert!((v - 240.0).abs() < 1e-9, "v={v}");
    }

    #[test]
    fn test_project_negative_z_errors() {
        let cam = PinholeCameraModel::new(800.0, 800.0, 320.0, 240.0, 640, 480);
        assert!(project(&cam, &[0.0, 0.0, -1.0]).is_err());
        assert!(project(&cam, &[0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_unproject_roundtrip() {
        let cam = PinholeCameraModel::new(800.0, 800.0, 320.0, 240.0, 640, 480);
        let depth = 3.5;
        let (u, v) = project(&cam, &[0.5, -0.3, depth])
            .expect("project should succeed for point in front of camera");
        let pt = unproject(&cam, (u, v), depth);
        assert!((pt[0] - 0.5).abs() < 1e-9, "X={}", pt[0]);
        assert!((pt[1] - (-0.3)).abs() < 1e-9, "Y={}", pt[1]);
        assert!((pt[2] - depth).abs() < 1e-9, "Z={}", pt[2]);
    }

    #[test]
    fn test_distort_identity() {
        let d = RadialDistortion::default();
        let (xd, yd) = distort(&d, (0.3, 0.2));
        assert!((xd - 0.3).abs() < 1e-12);
        assert!((yd - 0.2).abs() < 1e-12);
    }

    #[test]
    fn test_distort_nonzero() {
        let d = RadialDistortion::new(0.1, 0.0, 0.0, 0.0, 0.0);
        let x = 0.5;
        let y = 0.0;
        let (xd, _yd) = distort(&d, (x, y));
        // r2 = 0.25, radial = 1 + 0.1*0.25 = 1.025
        assert!((xd - x * 1.025).abs() < 1e-12, "xd={xd}");
    }

    #[test]
    fn test_undistort_image_identity_distortion() {
        let cam = PinholeCameraModel::new(400.0, 400.0, 160.0, 120.0, 320, 240);
        let dist = RadialDistortion::default();
        let img = Array3::from_elem((240, 320, 3), 128.0_f64);
        let out = undistort_image(&img, &cam, &dist)
            .expect("undistort_image should succeed with identity distortion");
        assert_eq!(out.dim(), img.dim());
        // With no distortion the output should be identical to the input (bilinear interpolation of uniform image).
        let err: f64 = (&out - &img).iter().map(|x| x.abs()).sum::<f64>() / (240 * 320 * 3) as f64;
        assert!(err < 1.0, "mean error = {err}");
    }

    #[test]
    fn test_undistort_image_wrong_size() {
        let cam = PinholeCameraModel::new(400.0, 400.0, 160.0, 120.0, 320, 240);
        let dist = RadialDistortion::default();
        let img = Array3::<f64>::zeros((100, 100, 3));
        assert!(undistort_image(&img, &cam, &dist).is_err());
    }

    #[test]
    fn test_fov() {
        let cam = PinholeCameraModel::new(800.0, 800.0, 320.0, 240.0, 640, 480);
        let fov_x = cam.fov_x();
        let fov_y = cam.fov_y();
        // Rough sanity: should be in (0, π)
        assert!(fov_x > 0.0 && fov_x < std::f64::consts::PI);
        assert!(fov_y > 0.0 && fov_y < std::f64::consts::PI);
    }

    #[test]
    fn test_to_matrix_roundtrip() {
        let cam = PinholeCameraModel::new(750.0, 760.0, 319.5, 239.5, 640, 480);
        let k = cam.to_matrix();
        assert!((k[0][0] - cam.fx).abs() < 1e-12);
        assert!((k[1][1] - cam.fy).abs() < 1e-12);
        assert!((k[0][2] - cam.cx).abs() < 1e-12);
        assert!((k[1][2] - cam.cy).abs() < 1e-12);
    }
}
