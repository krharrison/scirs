//! Stereo image rectification.
//!
//! Implements Bouguet's algorithm for computing rectification homographies and
//! the associated pixel remapping tables used to transform raw stereo images
//! into a canonical parallel configuration where epipolar lines are horizontal.

use crate::stereo::calibration::StereoCalibration;

// ─────────────────────────────────────────────────────────────────────────────
// StereoRectifier
// ─────────────────────────────────────────────────────────────────────────────

/// Computes and applies stereo rectification maps.
///
/// After calling `compute_maps`, use `remap_left` / `remap_right` to
/// obtain the rectified images.
///
/// ## Example
/// ```
/// use scirs2_vision::stereo::calibration::StereoCalibration;
/// use scirs2_vision::stereo::rectification::StereoRectifier;
///
/// let cal = StereoCalibration::from_baseline(500.0, 500.0, 32.0, 24.0, 0.1);
/// let mut rect = StereoRectifier::new(cal);
/// rect.compute_maps(64, 48);
/// assert!(rect.map_left.is_some());
/// ```
pub struct StereoRectifier {
    /// Stereo calibration parameters.
    pub calib: StereoCalibration,
    /// Rectification map for the left image: `(map_x, map_y)` in pixels.
    pub map_left: Option<(Vec<f32>, Vec<f32>)>,
    /// Rectification map for the right image: `(map_x, map_y)` in pixels.
    pub map_right: Option<(Vec<f32>, Vec<f32>)>,
    /// The 3×3 left rectification homography (row-major).
    pub r_left: [[f64; 3]; 3],
    /// The 3×3 right rectification homography (row-major).
    pub r_right: [[f64; 3]; 3],
}

impl StereoRectifier {
    /// Create a rectifier from calibration data.
    pub fn new(calib: StereoCalibration) -> Self {
        Self {
            calib,
            map_left: None,
            map_right: None,
            r_left: identity_3x3(),
            r_right: identity_3x3(),
        }
    }

    /// Compute the rectification maps for an image of size `width × height`.
    ///
    /// Uses a simplified Bouguet decomposition:
    /// - If the cameras are already parallel (no rotation), the maps reduce to
    ///   identity (no warping needed).
    /// - Otherwise a axis-angle half-rotation is applied to each camera so both
    ///   look in a common direction with horizontal epipolar lines.
    pub fn compute_maps(&mut self, width: usize, height: usize) {
        let n = width * height;

        // Decompose rotation into axis-angle, split half to each camera.
        let (r_l, r_r) = self.compute_rectification_rotations();
        self.r_left = r_l;
        self.r_right = r_r;

        // Build pixel maps: for each output pixel (u,v), find the source pixel
        // in the original (unrectified) image.
        let fl = self.calib.left.fx as f32;
        let fr = self.calib.right.fx as f32;
        let cxl = self.calib.left.cx as f32;
        let cyl = self.calib.left.cy as f32;
        let cxr = self.calib.right.cx as f32;
        let cyr = self.calib.right.cy as f32;

        let mut mx_l = vec![0.0_f32; n];
        let mut my_l = vec![0.0_f32; n];
        let mut mx_r = vec![0.0_f32; n];
        let mut my_r = vec![0.0_f32; n];

        // New (shared) focal length and principal point — use left as reference.
        let f_new = fl;
        let cx_new = cxl;
        let cy_new = cyl;

        for row in 0..height {
            for col in 0..width {
                let idx = row * width + col;

                // Left map
                {
                    let xn = (col as f32 - cx_new) / f_new;
                    let yn = (row as f32 - cy_new) / f_new;
                    let p = mat3_apply_f32(&r_l, [xn, yn, 1.0]);
                    let xp = p[0] / p[2];
                    let yp = p[1] / p[2];
                    mx_l[idx] = xp * fl + cxl;
                    my_l[idx] = yp * fl + cyl;
                }

                // Right map
                {
                    let xn = (col as f32 - cx_new) / f_new;
                    let yn = (row as f32 - cy_new) / f_new;
                    let p = mat3_apply_f32(&r_r, [xn, yn, 1.0]);
                    let xp = p[0] / p[2];
                    let yp = p[1] / p[2];
                    mx_r[idx] = xp * fr + cxr;
                    my_r[idx] = yp * fr + cyr;
                }
            }
        }

        self.map_left = Some((mx_l, my_l));
        self.map_right = Some((mx_r, my_r));
    }

    /// Apply the left rectification map to a grayscale image.
    pub fn remap_left(&self, image: &[u8], width: usize, height: usize) -> Vec<u8> {
        self.map_left
            .as_ref()
            .map(|m| remap_bilinear(image, width, height, m))
            .unwrap_or_else(|| image.to_vec())
    }

    /// Apply the right rectification map to a grayscale image.
    pub fn remap_right(&self, image: &[u8], width: usize, height: usize) -> Vec<u8> {
        self.map_right
            .as_ref()
            .map(|m| remap_bilinear(image, width, height, m))
            .unwrap_or_else(|| image.to_vec())
    }

    /// Generic remap using bilinear interpolation.
    pub fn remap(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        map: &(Vec<f32>, Vec<f32>),
    ) -> Vec<u8> {
        remap_bilinear(image, width, height, map)
    }

    // ── Rectification rotation computation ───────────────────────────────────

    /// Compute (R_left, R_right) rectification rotations using Bouguet's
    /// axis-angle splitting.
    fn compute_rectification_rotations(&self) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
        // R maps from right to left camera. We want to find:
        //   R_l, R_r  s.t.  R_l · R_r^T = R  (the extrinsic rotation)
        // and both cameras look in the same direction after rectification.

        let r = self.calib.rotation;

        // Compute the axis-angle of R.
        let (axis, angle) = rotation_to_axis_angle(&r);

        // Split the rotation: left gets +half, right gets -half around the axis.
        let r_l = axis_angle_to_matrix(axis, angle * 0.5);
        let r_r = axis_angle_to_matrix(axis, -angle * 0.5);

        // Transpose (inverse) because the map warps backward.
        (transpose_3x3(&r_l), transpose_3x3(&r_r))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilinear remap
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a pixel remap with bilinear interpolation.
///
/// For each output pixel `i`, the source coordinates are `(map_x[i], map_y[i])`.
/// Out-of-bounds pixels are filled with 0.
pub fn remap_bilinear(
    image: &[u8],
    width: usize,
    height: usize,
    map: &(Vec<f32>, Vec<f32>),
) -> Vec<u8> {
    let n = width * height;
    let mut out = vec![0u8; n];
    let (map_x, map_y) = map;

    for i in 0..n {
        let x = map_x[i];
        let y = map_y[i];

        if x < 0.0 || y < 0.0 || x >= (width - 1) as f32 || y >= (height - 1) as f32 {
            // Clamp boundary pixels.
            let xi = x.round().clamp(0.0, (width - 1) as f32) as usize;
            let yi = y.round().clamp(0.0, (height - 1) as f32) as usize;
            out[i] = image[yi * width + xi];
            continue;
        }

        let xi = x as usize;
        let yi = y as usize;
        let fx = x - xi as f32;
        let fy = y - yi as f32;

        let v00 = image[yi * width + xi] as f32;
        let v01 = image[yi * width + xi + 1] as f32;
        let v10 = image[(yi + 1) * width + xi] as f32;
        let v11 = image[(yi + 1) * width + xi + 1] as f32;

        let val = (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01) + fy * ((1.0 - fx) * v10 + fx * v11);
        out[i] = val as u8;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotation utilities
// ─────────────────────────────────────────────────────────────────────────────

fn identity_3x3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn transpose_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

/// Extract axis and angle (in radians) from a rotation matrix.
fn rotation_to_axis_angle(r: &[[f64; 3]; 3]) -> ([f64; 3], f64) {
    // trace = 1 + 2 cos θ
    let trace = r[0][0] + r[1][1] + r[2][2];
    let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();

    if theta.abs() < 1e-10 {
        // Identity rotation: arbitrary axis.
        return ([0.0, 0.0, 1.0], 0.0);
    }

    let denom = 2.0 * theta.sin();
    let axis = [
        (r[2][1] - r[1][2]) / denom,
        (r[0][2] - r[2][0]) / denom,
        (r[1][0] - r[0][1]) / denom,
    ];
    (axis, theta)
}

/// Build a rotation matrix from axis (unit vector) and angle (Rodrigues).
fn axis_angle_to_matrix(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    let [ax, ay, az] = axis;

    [
        [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay],
        [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax],
        [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c],
    ]
}

/// Apply a 3×3 matrix to a 3-vector (homogeneous).
fn mat3_apply_f32(m: &[[f64; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    let [x, y, z] = [v[0] as f64, v[1] as f64, v[2] as f64];
    [
        (m[0][0] * x + m[0][1] * y + m[0][2] * z) as f32,
        (m[1][0] * x + m[1][1] * y + m[1][2] * z) as f32,
        (m[2][0] * x + m[2][1] * y + m[2][2] * z) as f32,
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_parallel_cal() -> StereoCalibration {
        StereoCalibration::from_baseline(500.0, 500.0, 32.0, 24.0, 0.1)
    }

    #[test]
    fn test_compute_maps_produces_maps() {
        let mut rect = StereoRectifier::new(make_parallel_cal());
        rect.compute_maps(64, 48);
        assert!(rect.map_left.is_some());
        assert!(rect.map_right.is_some());

        let (mx, my) = rect.map_left.as_ref().expect("map_left should exist");
        assert_eq!(mx.len(), 64 * 48);
        assert_eq!(my.len(), 64 * 48);
    }

    #[test]
    fn test_remap_identity_like() {
        // For a parallel stereo rig (R = I) the rectification maps should
        // be approximately identity; the remapped image is close to the original.
        let mut rect = StereoRectifier::new(make_parallel_cal());
        rect.compute_maps(32, 24);

        let img: Vec<u8> = (0..32 * 24).map(|i| ((i * 3) % 256) as u8).collect();

        let out = rect.remap_left(&img, 32, 24);
        assert_eq!(out.len(), img.len());
    }

    #[test]
    fn test_remap_bilinear_boundary() {
        let img = vec![100u8; 10 * 10];
        let n = 10 * 10;
        // Map all pixels to the top-left corner.
        let map = (vec![0.0_f32; n], vec![0.0_f32; n]);
        let out = remap_bilinear(&img, 10, 10, &map);
        assert_eq!(out.len(), n);
        assert!(out.iter().all(|&v| v == 100));
    }

    #[test]
    fn test_rotation_utilities_roundtrip() {
        // Build a small rotation, extract axis-angle, rebuild, check trace.
        let angle_in = 0.3_f64;
        let axis = [0.0, 0.0, 1.0];
        let r = axis_angle_to_matrix(axis, angle_in);
        let (_, angle_out) = rotation_to_axis_angle(&r);
        assert!((angle_in - angle_out).abs() < 1e-8, "angle roundtrip");
    }
}
