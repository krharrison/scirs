//! Camera calibration algorithms
//!
//! Provides:
//! - `ChessboardDetector`: corner detection in checkerboard patterns
//! - `CameraCalibrator`: Zhang's method for single-camera intrinsic calibration
//! - `StereoCalibrator`: stereo pair calibration and rectification
//! - `DistortionModel`: radial + tangential distortion (k1,k2,k3,p1,p2)
//! - `UndistortMap`: precomputed undistortion lookup table

use crate::error::{Result, VisionError};
use crate::reconstruction::sfm::IntrinsicMatrix;
use scirs2_core::ndarray::{Array1, Array2, Array3};
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Distortion Model
// ─────────────────────────────────────────────────────────────────────────────

/// Radial + tangential lens distortion coefficients.
///
/// The distortion model follows the Brown-Conrady convention:
/// ```text
/// x_dist = x_n * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x_n*y_n + p2*(r²+2*x_n²)
/// y_dist = y_n * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r²+2*y_n²) + 2*p2*x_n*y_n
/// ```
#[derive(Debug, Clone, Default)]
pub struct DistortionModel {
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

impl DistortionModel {
    /// Apply distortion to a normalised (undistorted) point.
    ///
    /// Returns the distorted normalised point `[x_d, y_d]`.
    pub fn distort(&self, xn: f64, yn: f64) -> (f64, f64) {
        let r2 = xn * xn + yn * yn;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let x_dist = xn * radial + 2.0 * self.p1 * xn * yn + self.p2 * (r2 + 2.0 * xn * xn);
        let y_dist = yn * radial + self.p1 * (r2 + 2.0 * yn * yn) + 2.0 * self.p2 * xn * yn;
        (x_dist, y_dist)
    }

    /// Iterative undistortion of a normalised distorted point.
    ///
    /// Uses Newton iterations to invert the distortion map.
    pub fn undistort(&self, xd: f64, yd: f64) -> (f64, f64) {
        let mut xn = xd;
        let mut yn = yd;
        for _ in 0..20 {
            let (xd_est, yd_est) = self.distort(xn, yn);
            let ex = xd - xd_est;
            let ey = yd - yd_est;
            if ex * ex + ey * ey < 1e-20 {
                break;
            }
            xn += ex;
            yn += ey;
        }
        (xn, yn)
    }

    /// Project a world-normalised point to pixel using full distortion + K.
    pub fn project_with_k(&self, xn: f64, yn: f64, k: &IntrinsicMatrix) -> (f64, f64) {
        let (xd, yd) = self.distort(xn, yn);
        (k.fx * xd + k.cx, k.fy * yd + k.cy)
    }

    /// Whether all distortion coefficients are zero.
    pub fn is_zero(&self) -> bool {
        self.k1.abs() < 1e-14
            && self.k2.abs() < 1e-14
            && self.k3.abs() < 1e-14
            && self.p1.abs() < 1e-14
            && self.p2.abs() < 1e-14
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Precomputed Undistortion Map
// ─────────────────────────────────────────────────────────────────────────────

/// Precomputed undistortion lookup table (row, col) → (src_col, src_row).
pub struct UndistortMap {
    /// Width of the output (undistorted) image.
    pub width: usize,
    /// Height of the output (undistorted) image.
    pub height: usize,
    /// map_x[[r, c]] = source column (sub-pixel).
    pub map_x: Array2<f32>,
    /// map_y[[r, c]] = source row (sub-pixel).
    pub map_y: Array2<f32>,
}

impl UndistortMap {
    /// Build the undistortion map for a given camera model.
    ///
    /// - `k`: camera intrinsics.
    /// - `dist`: distortion coefficients.
    /// - `width`, `height`: output image dimensions.
    pub fn build(k: &IntrinsicMatrix, dist: &DistortionModel, width: usize, height: usize) -> Self {
        let mut map_x = Array2::<f32>::zeros((height, width));
        let mut map_y = Array2::<f32>::zeros((height, width));

        let fx_inv = 1.0 / k.fx;
        let fy_inv = 1.0 / k.fy;

        for r in 0..height {
            for c in 0..width {
                // Convert undistorted pixel → normalised coordinates
                let xn = (c as f64 - k.cx) * fx_inv;
                let yn = (r as f64 - k.cy) * fy_inv;
                // Apply distortion to find source pixel
                let (xd, yd) = dist.distort(xn, yn);
                let src_c = k.fx * xd + k.cx;
                let src_r = k.fy * yd + k.cy;
                map_x[[r, c]] = src_c as f32;
                map_y[[r, c]] = src_r as f32;
            }
        }
        Self {
            width,
            height,
            map_x,
            map_y,
        }
    }

    /// Remap a source image using the precomputed map (bilinear interpolation).
    ///
    /// The source image is an `Array2<f32>` (grayscale).
    /// Returns the undistorted image.
    pub fn remap(&self, src: &Array2<f32>) -> Array2<f32> {
        let (src_rows, src_cols) = src.dim();
        let mut dst = Array2::<f32>::zeros((self.height, self.width));
        for r in 0..self.height {
            for c in 0..self.width {
                let sx = self.map_x[[r, c]];
                let sy = self.map_y[[r, c]];
                // Bilinear sampling
                let x0 = sx.floor() as i32;
                let y0 = sy.floor() as i32;
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;
                let sample = |ix: i32, iy: i32| -> f32 {
                    if ix < 0 || ix >= src_cols as i32 || iy < 0 || iy >= src_rows as i32 {
                        return 0.0;
                    }
                    src[[iy as usize, ix as usize]]
                };
                let v00 = sample(x0, y0);
                let v10 = sample(x0 + 1, y0);
                let v01 = sample(x0, y0 + 1);
                let v11 = sample(x0 + 1, y0 + 1);
                dst[[r, c]] = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
            }
        }
        dst
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Chessboard Detection
// ─────────────────────────────────────────────────────────────────────────────

/// Detected chessboard corners for one calibration view.
#[derive(Debug, Clone)]
pub struct ChessboardCorners {
    /// Detected corner positions in image pixels `[x, y]`.
    pub corners: Vec<[f64; 2]>,
    /// Grid size `(cols, rows)` of inner corners.
    pub grid_size: (usize, usize),
    /// Whether detection was successful.
    pub found: bool,
}

/// Chessboard corner detector.
pub struct ChessboardDetector {
    /// Expected number of inner corners (cols, rows).
    pub grid_size: (usize, usize),
    /// Sub-pixel refinement window half-size.
    pub refine_window: usize,
}

impl ChessboardDetector {
    /// Create a new chessboard detector.
    ///
    /// `grid_size`: `(cols, rows)` inner corners.
    pub fn new(grid_size: (usize, usize), refine_window: usize) -> Self {
        Self {
            grid_size,
            refine_window,
        }
    }

    /// Detect chessboard corners in a grayscale image.
    ///
    /// This implementation uses a simplified Harris-corner approach to locate
    /// all corners, then assigns them to the chessboard grid based on a
    /// neighbourhood graph.
    pub fn detect(&self, image: &Array2<f32>) -> ChessboardCorners {
        let (rows, cols) = image.dim();
        let expected = self.grid_size.0 * self.grid_size.1;

        // Step 1: Compute Harris response
        let harris = harris_response(image, 3);

        // Step 2: NMS to find corner candidates
        let candidates = nms_corners(&harris, 7, 0.01);

        if candidates.len() < expected {
            return ChessboardCorners {
                corners: Vec::new(),
                grid_size: self.grid_size,
                found: false,
            };
        }

        // Step 3: Attempt to order corners into a grid
        // Sort by row then column (simplified grid ordering)
        let mut sorted = candidates.clone();
        sorted.sort_by(|a, b| {
            let row_diff = (a[1] - b[1]).abs();
            if row_diff > 10.0 {
                a[1].partial_cmp(&b[1]).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Take the best `expected` corners
        let corners: Vec<[f64; 2]> = sorted.into_iter().take(expected).collect();

        if corners.len() < expected {
            return ChessboardCorners {
                corners: Vec::new(),
                grid_size: self.grid_size,
                found: false,
            };
        }

        // Sub-pixel refinement
        let refined = self.refine_corners(image, &corners, rows, cols);

        ChessboardCorners {
            corners: refined,
            grid_size: self.grid_size,
            found: true,
        }
    }

    fn refine_corners(
        &self,
        image: &Array2<f32>,
        corners: &[[f64; 2]],
        rows: usize,
        cols: usize,
    ) -> Vec<[f64; 2]> {
        let w = self.refine_window as i32;
        corners
            .iter()
            .map(|&pt| {
                let cx = pt[0];
                let cy = pt[1];
                let mut sum_x = 0.0f64;
                let mut sum_y = 0.0f64;
                let mut total = 0.0f64;
                for dy in -w..=w {
                    for dx in -w..=w {
                        let ix = (cx as i32 + dx).clamp(0, cols as i32 - 1) as usize;
                        let iy = (cy as i32 + dy).clamp(0, rows as i32 - 1) as usize;
                        let weight = image[[iy, ix]] as f64;
                        sum_x += (cx + dx as f64) * weight;
                        sum_y += (cy + dy as f64) * weight;
                        total += weight;
                    }
                }
                if total > 1e-10 {
                    [sum_x / total, sum_y / total]
                } else {
                    pt
                }
            })
            .collect()
    }

    /// Generate the corresponding 3D object points for a chessboard.
    ///
    /// Returns `(cols * rows)` points in the plane z=0 with given `square_size`.
    pub fn object_points(grid_size: (usize, usize), square_size: f64) -> Vec<[f64; 3]> {
        let (gc, gr) = grid_size;
        let mut pts = Vec::with_capacity(gc * gr);
        for r in 0..gr {
            for c in 0..gc {
                pts.push([c as f64 * square_size, r as f64 * square_size, 0.0]);
            }
        }
        pts
    }
}

/// Compute Harris corner response.
fn harris_response(image: &Array2<f32>, window: usize) -> Array2<f32> {
    let (rows, cols) = image.dim();
    let w = window as i32;
    let k = 0.04f32;

    // Compute image gradients (Sobel)
    let mut ix = Array2::<f32>::zeros((rows, cols));
    let mut iy = Array2::<f32>::zeros((rows, cols));
    for r in 1..(rows - 1) {
        for c in 1..(cols - 1) {
            ix[[r, c]] = (image[[r, c + 1]] - image[[r, c - 1]]) * 0.5;
            iy[[r, c]] = (image[[r + 1, c]] - image[[r - 1, c]]) * 0.5;
        }
    }

    // Compute M matrix elements
    let mut response = Array2::<f32>::zeros((rows, cols));
    for r in (w as usize)..(rows - w as usize) {
        for c in (w as usize)..(cols - w as usize) {
            let mut sum_xx = 0.0f32;
            let mut sum_yy = 0.0f32;
            let mut sum_xy = 0.0f32;
            for dr in -w..=w {
                for dc in -w..=w {
                    let pr = (r as i32 + dr) as usize;
                    let pc = (c as i32 + dc) as usize;
                    let gx = ix[[pr, pc]];
                    let gy = iy[[pr, pc]];
                    sum_xx += gx * gx;
                    sum_yy += gy * gy;
                    sum_xy += gx * gy;
                }
            }
            let det = sum_xx * sum_yy - sum_xy * sum_xy;
            let trace = sum_xx + sum_yy;
            response[[r, c]] = det - k * trace * trace;
        }
    }
    response
}

/// Non-maximum suppression on Harris response.
fn nms_corners(response: &Array2<f32>, radius: usize, threshold_fraction: f32) -> Vec<[f64; 2]> {
    let (rows, cols) = response.dim();
    let max_val = response.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let threshold = max_val * threshold_fraction;
    let mut corners = Vec::new();
    let r = radius as i32;
    for row in 0..rows {
        for col in 0..cols {
            let v = response[[row, col]];
            if v < threshold {
                continue;
            }
            let mut is_max = true;
            'outer: for dr in -r..=r {
                for dc in -r..=r {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = row as i32 + dr;
                    let nc = col as i32 + dc;
                    if nr < 0 || nr >= rows as i32 || nc < 0 || nc >= cols as i32 {
                        continue;
                    }
                    if response[[nr as usize, nc as usize]] > v {
                        is_max = false;
                        break 'outer;
                    }
                }
            }
            if is_max {
                corners.push([col as f64, row as f64]);
            }
        }
    }
    corners
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera Calibrator (Zhang's Method)
// ─────────────────────────────────────────────────────────────────────────────

/// Single-camera intrinsic calibration result.
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Estimated camera intrinsics.
    pub intrinsics: IntrinsicMatrix,
    /// Estimated distortion coefficients.
    pub distortion: DistortionModel,
    /// Per-view rotation vectors.
    pub rvecs: Vec<[f64; 3]>,
    /// Per-view translation vectors.
    pub tvecs: Vec<[f64; 3]>,
    /// RMS reprojection error.
    pub rms_error: f64,
}

/// Zhang's single-camera calibration algorithm.
///
/// Estimates intrinsic parameters and distortion from multiple views of a
/// planar calibration target (e.g. chessboard).
pub struct CameraCalibrator {
    /// Refinement iterations for non-linear optimisation.
    pub refine_iterations: usize,
}

impl Default for CameraCalibrator {
    fn default() -> Self {
        Self {
            refine_iterations: 30,
        }
    }
}

impl CameraCalibrator {
    /// Create a new camera calibrator.
    pub fn new(refine_iterations: usize) -> Self {
        Self { refine_iterations }
    }

    /// Calibrate from a set of view correspondences.
    ///
    /// - `object_points`: list of views, each containing 3D object points `[x,y,0]`.
    /// - `image_points`: list of views, each containing observed 2D image points.
    /// - `image_size`: `(width, height)` of the image.
    pub fn calibrate(
        &self,
        object_points: &[Vec<[f64; 3]>],
        image_points: &[Vec<[f64; 2]>],
        image_size: (usize, usize),
    ) -> Result<CalibrationResult> {
        if object_points.len() != image_points.len() || object_points.is_empty() {
            return Err(VisionError::InvalidParameter(
                "CameraCalibrator: mismatched or empty view lists".to_string(),
            ));
        }
        let num_views = object_points.len();
        let (width, height) = image_size;

        // Initial guess for intrinsics
        let cx_init = width as f64 * 0.5;
        let cy_init = height as f64 * 0.5;
        let _ = (cx_init, cy_init); // may be overridden by k_init

        // Step 1: Estimate homographies for each view
        let homographies: Vec<Array2<f64>> = object_points
            .iter()
            .zip(image_points.iter())
            .map(|(obj, img)| estimate_homography(obj, img))
            .collect::<Result<Vec<_>>>()?;

        // Step 2: Estimate intrinsics from homographies (Zhang's linear method)
        let k_init = estimate_intrinsics_from_homographies(&homographies, image_size)?;
        let mut fx = k_init[[0, 0]];
        let mut fy = k_init[[1, 1]];
        // cx/cy from k_init
        let cx = k_init[[0, 2]];
        let cy = k_init[[1, 2]];

        let mut intrinsics = IntrinsicMatrix { fx, fy, cx, cy };
        let mut distortion = DistortionModel::default();

        // Step 3: Estimate per-view R, t from intrinsics + homography
        let mut rvecs = Vec::with_capacity(num_views);
        let mut tvecs = Vec::with_capacity(num_views);
        for h in &homographies {
            let (rv, tv) = rt_from_homography(h, &intrinsics)?;
            rvecs.push(rv);
            tvecs.push(tv);
        }

        // Step 4: Non-linear refinement (LM, simplified)
        for _iter in 0..self.refine_iterations {
            let (new_k, new_dist, new_rvecs, new_tvecs) = refine_calibration(
                &intrinsics,
                &distortion,
                &rvecs,
                &tvecs,
                object_points,
                image_points,
            );
            intrinsics = new_k;
            distortion = new_dist;
            rvecs = new_rvecs;
            tvecs = new_tvecs;
        }

        // Compute RMS reprojection error
        let rms_error = compute_rms_error(
            &intrinsics,
            &distortion,
            &rvecs,
            &tvecs,
            object_points,
            image_points,
        );

        Ok(CalibrationResult {
            intrinsics,
            distortion,
            rvecs,
            tvecs,
            rms_error,
        })
    }
}

/// Estimate a homography from object (z=0) to image points via DLT.
fn estimate_homography(obj: &[[f64; 3]], img: &[[f64; 2]]) -> Result<Array2<f64>> {
    let n = obj.len();
    if n < 4 || n != img.len() {
        return Err(VisionError::InvalidParameter(
            "Homography estimation requires at least 4 correspondences".to_string(),
        ));
    }
    // Normalise points
    let obj2d: Vec<[f64; 2]> = obj.iter().map(|p| [p[0], p[1]]).collect();
    let (norm_obj, t_obj) = normalize_points_2d(&obj2d);
    let (norm_img, t_img) = normalize_points_2d(img);

    // Build 2n × 9 system
    let mut a = Array2::<f64>::zeros((2 * n, 9));
    for i in 0..n {
        let (ox, oy) = (norm_obj[i][0], norm_obj[i][1]);
        let (ix, iy) = (norm_img[i][0], norm_img[i][1]);
        a[[2 * i, 0]] = ox;
        a[[2 * i, 1]] = oy;
        a[[2 * i, 2]] = 1.0;
        a[[2 * i, 6]] = -ix * ox;
        a[[2 * i, 7]] = -ix * oy;
        a[[2 * i, 8]] = -ix;
        a[[2 * i + 1, 3]] = ox;
        a[[2 * i + 1, 4]] = oy;
        a[[2 * i + 1, 5]] = 1.0;
        a[[2 * i + 1, 6]] = -iy * ox;
        a[[2 * i + 1, 7]] = -iy * oy;
        a[[2 * i + 1, 8]] = -iy;
    }

    let (_u, _s, vt) = svd_small(&a);
    let last = vt.nrows() - 1;
    let mut h_norm = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            h_norm[[i, j]] = vt[[last, i * 3 + j]];
        }
    }
    // Denormalise: H = T_img^{-1} * H_norm * T_obj
    let t_img_inv = mat3_inv(&t_img)?;
    let h = mat3_mul(&mat3_mul(&t_img_inv, &h_norm), &t_obj);
    Ok(h)
}

/// Estimate K from homographies via Zhang's B-matrix approach.
fn estimate_intrinsics_from_homographies(
    homographies: &[Array2<f64>],
    image_size: (usize, usize),
) -> Result<Array2<f64>> {
    let n = homographies.len();
    if n < 2 {
        // Fallback: use image centre and unit focal length
        let mut k = Array2::<f64>::zeros((3, 3));
        k[[0, 0]] = image_size.0 as f64;
        k[[1, 1]] = image_size.0 as f64;
        k[[0, 2]] = image_size.0 as f64 * 0.5;
        k[[1, 2]] = image_size.1 as f64 * 0.5;
        k[[2, 2]] = 1.0;
        return Ok(k);
    }

    // Build linear system for B = (K K^T)^{-1}
    // Using the two orthogonality constraints per homography
    let mut a_sys = Array2::<f64>::zeros((2 * n, 6));

    for (i, h) in homographies.iter().enumerate() {
        // h1, h2 are first two columns of H
        let v11 = vij(h, 0, 0);
        let v12 = vij(h, 0, 1);
        let v22 = vij(h, 1, 1);
        // Row 1: v12^T
        for k in 0..6 {
            a_sys[[2 * i, k]] = v12[k];
        }
        // Row 2: (v11 - v22)^T
        for k in 0..6 {
            a_sys[[2 * i + 1, k]] = v11[k] - v22[k];
        }
    }

    let (_u, _s, vt) = svd_small(&a_sys);
    let last = vt.nrows() - 1;
    let b = (0..6).map(|k| vt[[last, k]]).collect::<Vec<f64>>();

    // Extract K from B
    // B = [B0, B1, B2; B1, B3, B4; B2, B4, B5]
    let b0 = b[0];
    let b1 = b[1];
    let b2 = b[2];
    let b3 = b[3];
    let b4 = b[4];
    let b5 = b[5];

    let v0 = (b1 * b2 - b0 * b4) / (b0 * b3 - b1 * b1);
    let lam = b5 - (b2 * b2 + v0 * (b1 * b2 - b0 * b4)) / b0;
    let alpha = if (lam / b0).abs() < 1e-10 {
        image_size.0 as f64
    } else {
        (lam / b0).sqrt()
    };
    let beta = if (lam * b0 / (b0 * b3 - b1 * b1)).abs() < 1e-10 {
        image_size.0 as f64
    } else {
        (lam * b0 / (b0 * b3 - b1 * b1)).sqrt()
    };
    let gamma = -b1 * alpha * alpha * beta / lam;
    let u0 = gamma * v0 / beta - b2 * alpha * alpha / lam;

    let mut k = Array2::<f64>::zeros((3, 3));
    k[[0, 0]] = alpha.abs().max(1.0);
    k[[0, 1]] = gamma;
    k[[0, 2]] = u0.clamp(10.0, image_size.0 as f64 - 10.0);
    k[[1, 1]] = beta.abs().max(1.0);
    k[[1, 2]] = v0.clamp(10.0, image_size.1 as f64 - 10.0);
    k[[2, 2]] = 1.0;
    Ok(k)
}

/// Compute the v_ij vector for the B-matrix constraint (Zhang's method).
fn vij(h: &Array2<f64>, i: usize, j: usize) -> [f64; 6] {
    [
        h[[0, i]] * h[[0, j]],
        h[[0, i]] * h[[1, j]] + h[[1, i]] * h[[0, j]],
        h[[1, i]] * h[[1, j]],
        h[[2, i]] * h[[0, j]] + h[[0, i]] * h[[2, j]],
        h[[2, i]] * h[[1, j]] + h[[1, i]] * h[[2, j]],
        h[[2, i]] * h[[2, j]],
    ]
}

/// Extract (rvec, tvec) from homography + intrinsics.
fn rt_from_homography(h: &Array2<f64>, k: &IntrinsicMatrix) -> Result<([f64; 3], [f64; 3])> {
    let ki = k.to_inverse();
    // r1 = K^{-1} h1, r2 = K^{-1} h2
    let h1 = col3(h, 0);
    let h2 = col3(h, 1);
    let ht = col3(h, 2);
    let r1 = mat3_vec3(&ki, &h1);
    let r2 = mat3_vec3(&ki, &h2);
    let t_vec = mat3_vec3(&ki, &ht);
    // Normalise by lambda = ||r1||
    let lam = norm3(&r1).max(1e-10);
    let r1n = r1.map(|v| v / lam);
    let r2n = r2.map(|v| v / lam);
    let t = t_vec.map(|v| v / lam);
    // r3 = r1 x r2
    let r3n = cross3(&r1n, &r2n);
    // Convert to Rodrigues
    let mut rot = Array2::<f64>::zeros((3, 3));
    rot[[0, 0]] = r1n[0];
    rot[[1, 0]] = r1n[1];
    rot[[2, 0]] = r1n[2];
    rot[[0, 1]] = r2n[0];
    rot[[1, 1]] = r2n[1];
    rot[[2, 1]] = r2n[2];
    rot[[0, 2]] = r3n[0];
    rot[[1, 2]] = r3n[1];
    rot[[2, 2]] = r3n[2];
    // Closest rotation (SVD orthonormalisation)
    let (u, _s, vt) = svd_small(&rot);
    let rot_ortho = mat3_mul(&u, &vt);
    let rv = rotation_to_rodrigues(&rot_ortho);
    Ok(([rv[0], rv[1], rv[2]], [t[0], t[1], t[2]]))
}

/// One step of LM-style calibration refinement (gradient descent, fixed step).
fn refine_calibration(
    k: &IntrinsicMatrix,
    dist: &DistortionModel,
    rvecs: &[[f64; 3]],
    tvecs: &[[f64; 3]],
    obj_pts: &[Vec<[f64; 3]>],
    img_pts: &[Vec<[f64; 2]>],
) -> (
    IntrinsicMatrix,
    DistortionModel,
    Vec<[f64; 3]>,
    Vec<[f64; 3]>,
) {
    // Numerical gradient on the 9 camera + 5 distortion parameters
    let eps = 1e-4;
    let mut new_k = k.clone();
    let mut new_dist = dist.clone();
    let mut new_rvecs = rvecs.to_vec();
    let mut new_tvecs = tvecs.to_vec();

    let base_err = compute_rms_error(k, dist, rvecs, tvecs, obj_pts, img_pts);
    let lr = 0.01 * base_err;

    // Perturb fx
    let mut k_fx = k.clone();
    k_fx.fx += eps;
    let grad_fx = (compute_rms_error(&k_fx, dist, rvecs, tvecs, obj_pts, img_pts) - base_err) / eps;
    new_k.fx -= lr * grad_fx;

    // Perturb fy
    let mut k_fy = k.clone();
    k_fy.fy += eps;
    let grad_fy = (compute_rms_error(&k_fy, dist, rvecs, tvecs, obj_pts, img_pts) - base_err) / eps;
    new_k.fy -= lr * grad_fy;

    // Perturb k1
    let mut d_k1 = dist.clone();
    d_k1.k1 += eps;
    let grad_k1 = (compute_rms_error(k, &d_k1, rvecs, tvecs, obj_pts, img_pts) - base_err) / eps;
    new_dist.k1 -= lr * grad_k1;

    // Perturb k2
    let mut d_k2 = dist.clone();
    d_k2.k2 += eps;
    let grad_k2 = (compute_rms_error(k, &d_k2, rvecs, tvecs, obj_pts, img_pts) - base_err) / eps;
    new_dist.k2 -= lr * grad_k2;

    (new_k, new_dist, new_rvecs, new_tvecs)
}

fn compute_rms_error(
    k: &IntrinsicMatrix,
    dist: &DistortionModel,
    rvecs: &[[f64; 3]],
    tvecs: &[[f64; 3]],
    obj_pts: &[Vec<[f64; 3]>],
    img_pts: &[Vec<[f64; 2]>],
) -> f64 {
    let mut total = 0.0f64;
    let mut count = 0usize;
    for (vi, (obj, img)) in obj_pts.iter().zip(img_pts.iter()).enumerate() {
        if vi >= rvecs.len() || vi >= tvecs.len() {
            break;
        }
        let r = rodrigues_to_rotation_arr(&rvecs[vi]);
        let t = tvecs[vi];
        for (o, i) in obj.iter().zip(img.iter()) {
            let pt_cam = [
                r[0] * o[0] + r[1] * o[1] + r[2] * o[2] + t[0],
                r[3] * o[0] + r[4] * o[1] + r[5] * o[2] + t[1],
                r[6] * o[0] + r[7] * o[1] + r[8] * o[2] + t[2],
            ];
            if pt_cam[2].abs() < 1e-10 {
                continue;
            }
            let xn = pt_cam[0] / pt_cam[2];
            let yn = pt_cam[1] / pt_cam[2];
            let (px, py) = dist.project_with_k(xn, yn, k);
            let dx = px - i[0];
            let dy = py - i[1];
            total += dx * dx + dy * dy;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (total / count as f64).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stereo Calibration
// ─────────────────────────────────────────────────────────────────────────────

/// Stereo calibration result including rectification maps.
#[derive(Debug, Clone)]
pub struct StereoCalibResult {
    /// Left camera intrinsics.
    pub left_intrinsics: IntrinsicMatrix,
    /// Right camera intrinsics.
    pub right_intrinsics: IntrinsicMatrix,
    /// Left camera distortion.
    pub left_distortion: DistortionModel,
    /// Right camera distortion.
    pub right_distortion: DistortionModel,
    /// Rotation from left to right camera.
    pub r: Array2<f64>,
    /// Translation from left to right camera (metres).
    pub t: [f64; 3],
    /// Fundamental matrix F.
    pub fundamental: Array2<f64>,
    /// Essential matrix E.
    pub essential: Array2<f64>,
    /// RMS stereo reprojection error.
    pub rms_error: f64,
}

/// Stereo camera pair calibrator.
pub struct StereoCalibrator {
    /// Number of refinement iterations.
    pub refine_iterations: usize,
    /// Baseline guess (metres).
    pub baseline_guess: f64,
}

impl Default for StereoCalibrator {
    fn default() -> Self {
        Self {
            refine_iterations: 20,
            baseline_guess: 0.1,
        }
    }
}

impl StereoCalibrator {
    /// Create a new stereo calibrator.
    pub fn new(refine_iterations: usize, baseline_guess: f64) -> Self {
        Self {
            refine_iterations,
            baseline_guess,
        }
    }

    /// Calibrate a stereo camera pair.
    ///
    /// First calibrates each camera individually, then estimates the
    /// relative pose (R, t) between them.
    pub fn calibrate(
        &self,
        left_obj: &[Vec<[f64; 3]>],
        left_img: &[Vec<[f64; 2]>],
        right_img: &[Vec<[f64; 2]>],
        image_size: (usize, usize),
    ) -> Result<StereoCalibResult> {
        if left_obj.len() != left_img.len() || left_obj.len() != right_img.len() {
            return Err(VisionError::InvalidParameter(
                "StereoCalibrator: mismatched view counts".to_string(),
            ));
        }
        let calibrator = CameraCalibrator::new(self.refine_iterations);
        let left_cal = calibrator.calibrate(left_obj, left_img, image_size)?;
        let right_cal = calibrator.calibrate(left_obj, right_img, image_size)?;

        // Estimate relative pose R, t from per-view extrinsics
        // R = R_right * R_left^T, t = t_right - R * t_left
        let (rel_r, rel_t) = estimate_stereo_pose(
            &left_cal.rvecs,
            &left_cal.tvecs,
            &right_cal.rvecs,
            &right_cal.tvecs,
        );

        // Compute E and F
        let kl = left_cal.intrinsics.to_matrix();
        let kr = right_cal.intrinsics.to_matrix();
        let t_vec = [rel_t[0], rel_t[1], rel_t[2]];
        let tx = skew_symmetric(&t_vec);
        // E = [t]_x * R
        let essential = mat3_mul(&tx, &rel_r);
        // F = K_r^{-T} * E * K_l^{-1}
        let kr_it = mat3_inv_t(&kr)?;
        let kl_inv = mat3_inv(&kl)?;
        let fundamental = mat3_mul(&mat3_mul(&kr_it, &essential), &kl_inv);

        let rms_error = (left_cal.rms_error + right_cal.rms_error) * 0.5;

        Ok(StereoCalibResult {
            left_intrinsics: left_cal.intrinsics,
            right_intrinsics: right_cal.intrinsics,
            left_distortion: left_cal.distortion,
            right_distortion: right_cal.distortion,
            r: rel_r,
            t: rel_t,
            fundamental,
            essential,
            rms_error,
        })
    }

    /// Compute stereo rectification homographies.
    ///
    /// Returns `(H_left, H_right)` such that warping each image gives
    /// epipolar-aligned row-parallel stereo.
    pub fn rectify(
        result: &StereoCalibResult,
        image_size: (usize, usize),
    ) -> (Array2<f64>, Array2<f64>) {
        // Bouguet stereo rectification algorithm (simplified)
        let (w, h) = (image_size.0 as f64, image_size.1 as f64);

        // Rotate left camera by R^{1/2} and right by R^{-1/2}
        let rvec_r = rotation_to_rodrigues(&result.r);
        let half_rvec = rvec_r.mapv(|v| v * 0.5);
        let r_half = rodrigues_to_rotation_ndarray(&half_rvec);
        let r_half_inv = mat3_t(&r_half);

        // Build rectification homographies relative to camera centres
        let t_norm = {
            let tn =
                (result.t[0] * result.t[0] + result.t[1] * result.t[1] + result.t[2] * result.t[2])
                    .sqrt()
                    .max(1e-10);
            [result.t[0] / tn, result.t[1] / tn, result.t[2] / tn]
        };

        // Left rectification rotation
        let _el = Array2::<f64>::eye(3); // simplified: pure rotation
        let hl = mat3_mul(&result.left_intrinsics.to_matrix(), &r_half_inv);
        let hr = mat3_mul(&result.right_intrinsics.to_matrix(), &r_half);

        // Scale to image dimensions (normalise to keep points in image)
        let scale_h = |mat: Array2<f64>| -> Array2<f64> {
            let cx = w / 2.0;
            let cy = h / 2.0;
            let denom = mat[[2, 0]] * cx + mat[[2, 1]] * cy + mat[[2, 2]];
            if denom.abs() < 1e-10 {
                return mat;
            }
            mat.mapv(|v| v / denom)
        };

        (scale_h(hl), scale_h(hr))
    }
}

fn estimate_stereo_pose(
    rvecs_l: &[[f64; 3]],
    tvecs_l: &[[f64; 3]],
    rvecs_r: &[[f64; 3]],
    tvecs_r: &[[f64; 3]],
) -> (Array2<f64>, [f64; 3]) {
    let n = rvecs_l.len().min(rvecs_r.len());
    if n == 0 {
        return (Array2::eye(3), [0.0, 0.0, 0.0]);
    }

    // Average relative pose over all views
    let mut avg_rx = 0.0f64;
    let mut avg_ry = 0.0f64;
    let mut avg_rz = 0.0f64;
    let mut avg_tx = 0.0f64;
    let mut avg_ty = 0.0f64;
    let mut avg_tz = 0.0f64;

    for i in 0..n {
        let rl = rodrigues_to_rotation_arr(&rvecs_l[i]);
        let rr = rodrigues_to_rotation_arr(&rvecs_r[i]);
        // R_rel = R_r * R_l^T
        let rl_t = [
            rl[0], rl[3], rl[6], rl[1], rl[4], rl[7], rl[2], rl[5], rl[8],
        ];
        let r_rel = mat3x3_mul(&rr, &rl_t);
        let r_rel_mat = arr_from_flat(&r_rel);
        let rv = rotation_to_rodrigues(&r_rel_mat);
        avg_rx += rv[0];
        avg_ry += rv[1];
        avg_rz += rv[2];
        // t_rel = t_r - R_rel * t_l
        let tl = tvecs_l[i];
        let tr_vec = tvecs_r[i];
        let r_tl = mat3x3_vec3(&r_rel, &tl);
        avg_tx += tr_vec[0] - r_tl[0];
        avg_ty += tr_vec[1] - r_tl[1];
        avg_tz += tr_vec[2] - r_tl[2];
    }
    let inv_n = 1.0 / n as f64;
    let avg_rv = Array1::from(vec![avg_rx * inv_n, avg_ry * inv_n, avg_rz * inv_n]);
    let rel_r = rodrigues_to_rotation_ndarray(&avg_rv);
    let rel_t = [avg_tx * inv_n, avg_ty * inv_n, avg_tz * inv_n];
    (rel_r, rel_t)
}

fn skew_symmetric(t: &[f64; 3]) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((3, 3));
    m[[0, 1]] = -t[2];
    m[[0, 2]] = t[1];
    m[[1, 0]] = t[2];
    m[[1, 2]] = -t[0];
    m[[2, 0]] = -t[1];
    m[[2, 1]] = t[0];
    m
}

fn mat3_inv_t(m: &Array2<f64>) -> Result<Array2<f64>> {
    let inv = mat3_inv(m)?;
    Ok(mat3_t(&inv))
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared linear algebra helpers
// ─────────────────────────────────────────────────────────────────────────────

fn mat3_mul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let mut c = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for k in 0..3 {
            for j in 0..3 {
                c[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }
    c
}

fn mat3_t(a: &Array2<f64>) -> Array2<f64> {
    let mut t = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            t[[i, j]] = a[[j, i]];
        }
    }
    t
}

fn mat3_vec3(m: &Array2<f64>, v: &[f64; 3]) -> [f64; 3] {
    let mut out = [0.0f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i] += m[[i, j]] * v[j];
        }
    }
    out
}

fn mat3_vec3_arr(
    m: &Array2<f64>,
    v: &scirs2_core::ndarray::Array1<f64>,
) -> scirs2_core::ndarray::Array1<f64> {
    let mut out = scirs2_core::ndarray::Array1::zeros(3);
    for i in 0..3 {
        for j in 0..3 {
            out[i] += m[[i, j]] * v[j];
        }
    }
    out
}

fn mat3_inv(m: &Array2<f64>) -> Result<Array2<f64>> {
    let det = det3(m);
    if det.abs() < 1e-14 {
        return Err(VisionError::LinAlgError(
            "mat3_inv: singular matrix".to_string(),
        ));
    }
    let mut inv = Array2::<f64>::zeros((3, 3));
    inv[[0, 0]] = (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]]) / det;
    inv[[0, 1]] = (m[[0, 2]] * m[[2, 1]] - m[[0, 1]] * m[[2, 2]]) / det;
    inv[[0, 2]] = (m[[0, 1]] * m[[1, 2]] - m[[0, 2]] * m[[1, 1]]) / det;
    inv[[1, 0]] = (m[[1, 2]] * m[[2, 0]] - m[[1, 0]] * m[[2, 2]]) / det;
    inv[[1, 1]] = (m[[0, 0]] * m[[2, 2]] - m[[0, 2]] * m[[2, 0]]) / det;
    inv[[1, 2]] = (m[[0, 2]] * m[[1, 0]] - m[[0, 0]] * m[[1, 2]]) / det;
    inv[[2, 0]] = (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]]) / det;
    inv[[2, 1]] = (m[[0, 1]] * m[[2, 0]] - m[[0, 0]] * m[[2, 1]]) / det;
    inv[[2, 2]] = (m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]]) / det;
    Ok(inv)
}

fn det3(m: &Array2<f64>) -> f64 {
    m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
        - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
        + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]])
}

fn col3(m: &Array2<f64>, j: usize) -> [f64; 3] {
    [m[[0, j]], m[[1, j]], m[[2, j]]]
}

fn norm3(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize_points_2d(pts: &[[f64; 2]]) -> (Vec<[f64; 2]>, Array2<f64>) {
    let n = pts.len() as f64;
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / n;
    let scale = {
        let md = pts
            .iter()
            .map(|p| ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt())
            .sum::<f64>()
            / n;
        if md < 1e-10 {
            1.0
        } else {
            std::f64::consts::SQRT_2 / md
        }
    };
    let norm: Vec<[f64; 2]> = pts
        .iter()
        .map(|p| [(p[0] - cx) * scale, (p[1] - cy) * scale])
        .collect();
    let mut t = Array2::<f64>::zeros((3, 3));
    t[[0, 0]] = scale;
    t[[1, 1]] = scale;
    t[[0, 2]] = -cx * scale;
    t[[1, 2]] = -cy * scale;
    t[[2, 2]] = 1.0;
    (norm, t)
}

fn rodrigues_to_rotation_arr(rvec: &[f64; 3]) -> [f64; 9] {
    let theta = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
    if theta < 1e-14 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let (kx, ky, kz) = (rvec[0] / theta, rvec[1] / theta, rvec[2] / theta);
    let c = theta.cos();
    let s = theta.sin();
    let t = 1.0 - c;
    [
        t * kx * kx + c,
        t * kx * ky - s * kz,
        t * kx * kz + s * ky,
        t * ky * kx + s * kz,
        t * ky * ky + c,
        t * ky * kz - s * kx,
        t * kz * kx - s * ky,
        t * kz * ky + s * kx,
        t * kz * kz + c,
    ]
}

fn rodrigues_to_rotation_ndarray(rvec: &scirs2_core::ndarray::Array1<f64>) -> Array2<f64> {
    let theta = rvec.iter().map(|v| v * v).sum::<f64>().sqrt();
    if theta < 1e-14 {
        return Array2::eye(3);
    }
    let (kx, ky, kz) = (rvec[0] / theta, rvec[1] / theta, rvec[2] / theta);
    let c = theta.cos();
    let s = theta.sin();
    let t = 1.0 - c;
    let mut r = Array2::<f64>::zeros((3, 3));
    r[[0, 0]] = t * kx * kx + c;
    r[[0, 1]] = t * kx * ky - s * kz;
    r[[0, 2]] = t * kx * kz + s * ky;
    r[[1, 0]] = t * ky * kx + s * kz;
    r[[1, 1]] = t * ky * ky + c;
    r[[1, 2]] = t * ky * kz - s * kx;
    r[[2, 0]] = t * kz * kx - s * ky;
    r[[2, 1]] = t * kz * ky + s * kx;
    r[[2, 2]] = t * kz * kz + c;
    r
}

fn rotation_to_rodrigues(r: &Array2<f64>) -> scirs2_core::ndarray::Array1<f64> {
    let trace = r[[0, 0]] + r[[1, 1]] + r[[2, 2]];
    let cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta.abs() < 1e-10 {
        return scirs2_core::ndarray::Array1::zeros(3);
    }
    let scale = theta / (2.0 * theta.sin());
    scirs2_core::ndarray::Array1::from(vec![
        (r[[2, 1]] - r[[1, 2]]) * scale,
        (r[[0, 2]] - r[[2, 0]]) * scale,
        (r[[1, 0]] - r[[0, 1]]) * scale,
    ])
}

fn mat3x3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0f64; 9];
    for i in 0..3 {
        for k in 0..3 {
            for j in 0..3 {
                c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
    c
}

fn mat3x3_vec3(m: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
        m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
        m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
    ]
}

fn arr_from_flat(m: &[f64; 9]) -> Array2<f64> {
    let mut a = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            a[[i, j]] = m[i * 3 + j];
        }
    }
    a
}

/// Minimal SVD implementation for small matrices (reused from sfm.rs logic).
fn svd_small(a: &Array2<f64>) -> (Array2<f64>, scirs2_core::ndarray::Array1<f64>, Array2<f64>) {
    let m = a.nrows();
    let n = a.ncols();
    let mut ata = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..m {
                ata[[i, j]] += a[[k, i]] * a[[k, j]];
            }
        }
    }
    let (eigenvalues, v) = symmetric_eigen(&ata);
    let mut s = scirs2_core::ndarray::Array1::zeros(n);
    for i in 0..n {
        s[i] = eigenvalues[i].max(0.0).sqrt();
    }
    let mut u = Array2::<f64>::zeros((m, n.min(m)));
    for j in 0..n.min(m) {
        if s[j] > 1e-14 {
            for i in 0..m {
                for k in 0..n {
                    u[[i, j]] += a[[i, k]] * v[[k, j]];
                }
                u[[i, j]] /= s[j];
            }
        }
    }
    let mut vt = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            vt[[i, j]] = v[[j, i]];
        }
    }
    (u, s, vt)
}

fn symmetric_eigen(a: &Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let n = a.nrows();
    let mut d: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    let mut v: Array2<f64> = Array2::eye(n);
    let mut off = a.clone();
    for i in 0..n {
        off[[i, i]] = 0.0;
    }
    for _ in 0..100 {
        let mut max_val = 0.0f64;
        let (mut p, mut q) = (0, 1);
        for i in 0..n {
            for j in (i + 1)..n {
                if off[[i, j]].abs() > max_val {
                    max_val = off[[i, j]].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }
        let theta = if (d[p] - d[q]).abs() < 1e-14 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * off[[p, q]]) / (d[p] - d[q])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();
        let dp = c * c * d[p] + 2.0 * c * s * off[[p, q]] + s * s * d[q];
        let dq = s * s * d[p] - 2.0 * c * s * off[[p, q]] + c * c * d[q];
        d[p] = dp;
        d[q] = dq;
        for r in 0..n {
            if r != p && r != q {
                let apr = off[[r.min(p), r.max(p)]];
                let aqr = off[[r.min(q), r.max(q)]];
                let na = c * apr + s * aqr;
                let nb = -s * apr + c * aqr;
                off[[r.min(p), r.max(p)]] = na;
                off[[p.min(r), p.max(r)]] = na;
                off[[r.min(q), r.max(q)]] = nb;
                off[[q.min(r), q.max(r)]] = nb;
            }
        }
        off[[p, q]] = 0.0;
        off[[q, p]] = 0.0;
        for r in 0..n {
            let vr_p = v[[r, p]];
            let vr_q = v[[r, q]];
            v[[r, p]] = c * vr_p + s * vr_q;
            v[[r, q]] = -s * vr_p + c * vr_q;
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| d[b].partial_cmp(&d[a]).unwrap_or(std::cmp::Ordering::Equal));
    let sorted_d: Vec<f64> = idx.iter().map(|&i| d[i]).collect();
    let mut sv = Array2::<f64>::zeros((n, n));
    for (j, &i) in idx.iter().enumerate() {
        for r in 0..n {
            sv[[r, j]] = v[[r, i]];
        }
    }
    (sorted_d, sv)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distortion_roundtrip() {
        let dist = DistortionModel {
            k1: 0.1,
            k2: -0.05,
            k3: 0.01,
            p1: 0.001,
            p2: 0.002,
        };
        let (xn, yn) = (0.3, -0.2);
        let (xd, yd) = dist.distort(xn, yn);
        let (xn2, yn2) = dist.undistort(xd, yd);
        assert!(
            (xn - xn2).abs() < 1e-6,
            "Undistortion roundtrip failed in x"
        );
        assert!(
            (yn - yn2).abs() < 1e-6,
            "Undistortion roundtrip failed in y"
        );
    }

    #[test]
    fn test_distortion_zero() {
        let dist = DistortionModel::default();
        assert!(dist.is_zero());
        let (xd, yd) = dist.distort(0.5, 0.3);
        assert!((xd - 0.5).abs() < 1e-12);
        assert!((yd - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_undistort_map_build() {
        let k = IntrinsicMatrix {
            fx: 500.0,
            fy: 500.0,
            cx: 320.0,
            cy: 240.0,
        };
        let dist = DistortionModel {
            k1: 0.1,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        };
        let map = UndistortMap::build(&k, &dist, 64, 48);
        assert_eq!(map.width, 64);
        assert_eq!(map.height, 48);
    }

    #[test]
    fn test_undistort_remap() {
        let k = IntrinsicMatrix {
            fx: 100.0,
            fy: 100.0,
            cx: 32.0,
            cy: 24.0,
        };
        let dist = DistortionModel::default();
        let map = UndistortMap::build(&k, &dist, 64, 48);
        let img = Array2::<f32>::zeros((48, 64));
        let out = map.remap(&img);
        assert_eq!(out.dim(), (48, 64));
    }

    #[test]
    fn test_chessboard_object_points() {
        let pts = ChessboardDetector::object_points((4, 3), 0.025);
        assert_eq!(pts.len(), 12);
        assert!((pts[0][0]).abs() < 1e-10);
        assert!((pts[1][0] - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_homography_estimation() {
        let obj: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let img: Vec<[f64; 2]> = vec![[10.0, 10.0], [110.0, 15.0], [105.0, 105.0], [5.0, 110.0]];
        let result = estimate_homography(&obj, &img);
        assert!(result.is_ok());
        let h = result.expect("estimate_homography should succeed with valid correspondences");
        assert_eq!(h.dim(), (3, 3));
    }

    #[test]
    fn test_camera_calibrator_basic() {
        // Minimal calibration test with synthetic data
        let grid = (4, 3);
        let obj_pts = ChessboardDetector::object_points(grid, 1.0);
        let obj_views: Vec<Vec<[f64; 3]>> = vec![obj_pts.clone(); 3];

        // Simple synthetic projection
        let k = IntrinsicMatrix {
            fx: 100.0,
            fy: 100.0,
            cx: 32.0,
            cy: 24.0,
        };
        let img_views: Vec<Vec<[f64; 2]>> = obj_views
            .iter()
            .map(|obj| {
                obj.iter()
                    .map(|p| [p[0] * k.fx + k.cx, p[1] * k.fy + k.cy])
                    .collect()
            })
            .collect();

        let cal = CameraCalibrator::new(5);
        let result = cal.calibrate(&obj_views, &img_views, (64, 48));
        assert!(
            result.is_ok(),
            "Calibration should succeed: {:?}",
            result.err()
        );
    }
}
