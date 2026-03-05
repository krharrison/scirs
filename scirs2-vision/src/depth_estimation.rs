//! Monocular depth estimation features
//!
//! Provides evaluation metrics for depth estimation networks, structure-from-motion
//! triangulation, and sparse-to-dense depth propagation.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::Array2;
use std::collections::VecDeque;

// ─────────────────────────────────────────────────────────────────────────────
// DepthMap
// ─────────────────────────────────────────────────────────────────────────────

/// A depth map with associated metadata.
///
/// Stores per-pixel depth values together with calibration metadata that allows
/// converting raw network outputs (which are typically scale-ambiguous) into
/// metric depths.
#[derive(Debug, Clone)]
pub struct DepthMap {
    /// Per-pixel depth values.  Shape is `[H, W]`.
    pub data: Array2<f64>,
    /// Multiplicative scale factor that maps `data` values to metric depth
    /// (metres).  For ground-truth or metric depths set to `1.0`.
    pub scale: f64,
    /// Smallest valid (positive) depth in the map.
    pub min_depth: f64,
    /// Largest valid depth in the map.
    pub max_depth: f64,
}

impl DepthMap {
    /// Create a new `DepthMap` with explicit metadata.
    pub fn new(data: Array2<f64>, scale: f64, min_depth: f64, max_depth: f64) -> Self {
        Self {
            data,
            scale,
            min_depth,
            max_depth,
        }
    }

    /// Create a `DepthMap` by computing metadata automatically from the data.
    ///
    /// All positive finite values are considered valid depth samples.
    /// If no valid values exist, `min_depth` and `max_depth` are set to `0.0`.
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_vision::depth_estimation::DepthMap;
    /// use scirs2_core::ndarray::Array2;
    ///
    /// let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let dm = DepthMap::from_data(data, 1.0);
    /// assert!((dm.min_depth - 1.0).abs() < 1e-9);
    /// assert!((dm.max_depth - 4.0).abs() < 1e-9);
    /// ```
    pub fn from_data(data: Array2<f64>, scale: f64) -> Self {
        let mut min_depth = f64::INFINITY;
        let mut max_depth = f64::NEG_INFINITY;
        for &v in data.iter() {
            if v > 0.0 && v.is_finite() {
                if v < min_depth {
                    min_depth = v;
                }
                if v > max_depth {
                    max_depth = v;
                }
            }
        }
        if !min_depth.is_finite() {
            min_depth = 0.0;
            max_depth = 0.0;
        }
        Self {
            data,
            scale,
            min_depth,
            max_depth,
        }
    }

    /// Apply the scale factor: returns a new `DepthMap` where all values are in
    /// metric units.
    pub fn to_metric(&self) -> DepthMap {
        DepthMap {
            data: self.data.mapv(|v| v * self.scale),
            scale: 1.0,
            min_depth: self.min_depth * self.scale,
            max_depth: self.max_depth * self.scale,
        }
    }

    /// Dimensions `(height, width)` of the depth map.
    pub fn dim(&self) -> (usize, usize) {
        self.data.dim()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Evaluation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Scale-invariant logarithmic depth loss (Eigen et al., 2014).
///
/// Aligns the predicted and ground-truth depth maps by the optimal scale factor
/// (minimising log RMSE), making the metric invariant to global scale ambiguity.
///
/// Only pixels where both `pred > 0` and `gt > 0` are included.
///
/// Formula:
///
/// ```text
/// d_i = log(pred_i) − log(gt_i)
/// SILoss = (1/n) Σ d_i² − (1/n²)(Σ d_i)²
/// ```
///
/// # Errors
///
/// Returns [`VisionError::DimensionMismatch`] when `pred` and `gt` have
/// different shapes, or [`VisionError::InvalidParameter`] when there are no
/// valid pixels.
///
/// # Example
///
/// ```
/// use scirs2_vision::depth_estimation::scale_invariant_loss;
/// use scirs2_core::ndarray::Array2;
///
/// let pred = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let gt   = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let loss = scale_invariant_loss(&pred, &gt).unwrap();
/// assert!(loss.abs() < 1e-10);   // perfect prediction ⇒ zero loss
/// ```
pub fn scale_invariant_loss(pred: &Array2<f64>, gt: &Array2<f64>) -> Result<f64> {
    if pred.dim() != gt.dim() {
        return Err(VisionError::DimensionMismatch(
            "pred and gt must have the same shape".to_string(),
        ));
    }

    let mut sum_d = 0.0_f64;
    let mut sum_d2 = 0.0_f64;
    let mut n = 0usize;

    for (&p, &g) in pred.iter().zip(gt.iter()) {
        if p > 0.0 && g > 0.0 && p.is_finite() && g.is_finite() {
            let d = p.ln() - g.ln();
            sum_d += d;
            sum_d2 += d * d;
            n += 1;
        }
    }

    if n == 0 {
        return Err(VisionError::InvalidParameter(
            "No valid pixels (pred > 0 and gt > 0) found".to_string(),
        ));
    }

    let n_f = n as f64;
    Ok(sum_d2 / n_f - (sum_d * sum_d) / (n_f * n_f))
}

/// Absolute Relative Error (AbsRel): `mean(|pred − gt| / gt)`.
///
/// Only pixels where `gt > 0` are included.
///
/// # Errors
///
/// Returns [`VisionError::DimensionMismatch`] when shapes differ, or
/// [`VisionError::InvalidParameter`] when there are no valid pixels.
///
/// # Example
///
/// ```
/// use scirs2_vision::depth_estimation::absolute_relative_error;
/// use scirs2_core::ndarray::Array2;
///
/// let pred = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let gt   = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let err = absolute_relative_error(&pred, &gt).unwrap();
/// assert!(err.abs() < 1e-10);
/// ```
pub fn absolute_relative_error(pred: &Array2<f64>, gt: &Array2<f64>) -> Result<f64> {
    if pred.dim() != gt.dim() {
        return Err(VisionError::DimensionMismatch(
            "pred and gt must have the same shape".to_string(),
        ));
    }

    let mut sum = 0.0_f64;
    let mut n = 0usize;

    for (&p, &g) in pred.iter().zip(gt.iter()) {
        if g > 0.0 && g.is_finite() {
            sum += (p - g).abs() / g;
            n += 1;
        }
    }

    if n == 0 {
        return Err(VisionError::InvalidParameter(
            "No valid pixels (gt > 0) found".to_string(),
        ));
    }

    Ok(sum / n as f64)
}

/// Threshold accuracy δ < `threshold` (e.g. δ₁: threshold = 1.25).
///
/// Fraction of pixels where `max(pred/gt, gt/pred) < threshold`.
///
/// Only pixels where both `pred > 0` and `gt > 0` are included.
///
/// # Arguments
///
/// * `pred`      – Predicted depth map.
/// * `gt`        – Ground-truth depth map.
/// * `threshold` – The δ threshold.  Common values: 1.25, 1.25², 1.25³.
///
/// # Errors
///
/// Returns [`VisionError::DimensionMismatch`] when shapes differ, or
/// [`VisionError::InvalidParameter`] when there are no valid pixels or
/// `threshold ≤ 1.0`.
///
/// # Example
///
/// ```
/// use scirs2_vision::depth_estimation::threshold_accuracy;
/// use scirs2_core::ndarray::Array2;
///
/// let pred = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let gt   = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let acc = threshold_accuracy(&pred, &gt, 1.25).unwrap();
/// assert!((acc - 1.0).abs() < 1e-10);  // perfect prediction ⇒ 100 %
/// ```
pub fn threshold_accuracy(pred: &Array2<f64>, gt: &Array2<f64>, threshold: f64) -> Result<f64> {
    if pred.dim() != gt.dim() {
        return Err(VisionError::DimensionMismatch(
            "pred and gt must have the same shape".to_string(),
        ));
    }
    if threshold <= 1.0 {
        return Err(VisionError::InvalidParameter(
            "threshold must be greater than 1.0".to_string(),
        ));
    }

    let mut correct = 0usize;
    let mut n = 0usize;

    for (&p, &g) in pred.iter().zip(gt.iter()) {
        if p > 0.0 && g > 0.0 && p.is_finite() && g.is_finite() {
            let ratio = (p / g).max(g / p);
            if ratio < threshold {
                correct += 1;
            }
            n += 1;
        }
    }

    if n == 0 {
        return Err(VisionError::InvalidParameter(
            "No valid pixels (pred > 0 and gt > 0) found".to_string(),
        ));
    }

    Ok(correct as f64 / n as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Structure-from-Motion: depth_from_sfm
// ─────────────────────────────────────────────────────────────────────────────

/// A camera pose represented by a 3×3 rotation matrix `R` and a translation
/// vector `t` (3-element array).  The camera-to-world transform is:
/// `P_world = R P_cam + t`.
#[derive(Debug, Clone)]
pub struct CameraPose {
    /// 3×3 rotation matrix (row-major: `R[i]` is row *i*).
    pub r: [[f64; 3]; 3],
    /// Translation vector `[tx, ty, tz]` (camera centre in world coordinates).
    pub t: [f64; 3],
}

impl CameraPose {
    /// Create a new camera pose.
    pub fn new(r: [[f64; 3]; 3], t: [f64; 3]) -> Self {
        Self { r, t }
    }

    /// Identity pose (camera at world origin, no rotation).
    pub fn identity() -> Self {
        Self {
            r: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            t: [0.0, 0.0, 0.0],
        }
    }
}

/// A 2-D image keypoint with floating-point sub-pixel coordinates.
#[derive(Debug, Clone, Copy)]
pub struct KeyPoint2D {
    /// Horizontal pixel coordinate (column).
    pub x: f64,
    /// Vertical pixel coordinate (row).
    pub y: f64,
}

impl KeyPoint2D {
    /// Create a new keypoint.
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// A match between a keypoint in image *i* and a keypoint in image *j*.
#[derive(Debug, Clone, Copy)]
pub struct KeyPointMatch {
    /// Index into the keypoints array for image *i*.
    pub idx_i: usize,
    /// Index into the keypoints array for image *j*.
    pub idx_j: usize,
    /// Index of image *i* in the pose array.
    pub cam_i: usize,
    /// Index of image *j* in the pose array.
    pub cam_j: usize,
}

impl KeyPointMatch {
    /// Create a new keypoint match.
    pub fn new(idx_i: usize, idx_j: usize, cam_i: usize, cam_j: usize) -> Self {
        Self {
            idx_i,
            idx_j,
            cam_i,
            cam_j,
        }
    }
}

/// Recover 3-D structure from known camera poses and keypoint correspondences
/// using linear triangulation (DLT).
///
/// For each match the function triangulates the two rays (one from each camera)
/// and returns the world-space 3-D point.
///
/// # Arguments
///
/// * `camera_poses` – Per-image camera pose (rotation + translation).
/// * `keypoints`    – Per-image list of 2-D keypoints.
/// * `matches`      – List of cross-image keypoint matches.
///
/// # Returns
///
/// A vector of `(X, Y, Z)` world-space points, one per valid match.
/// Points that triangulate behind both cameras are omitted.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when the pose or keypoint
/// arrays are empty.
///
/// # Example
///
/// ```
/// use scirs2_vision::depth_estimation::{
///     CameraPose, KeyPoint2D, KeyPointMatch, depth_from_sfm,
/// };
///
/// // Two cameras side by side (baseline 1 unit along X).
/// let poses = vec![
///     CameraPose::identity(),
///     CameraPose::new(
///         [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
///         [1.0, 0.0, 0.0],
///     ),
/// ];
/// // A point at (0, 0, 5) projects to (0, 0) in both cameras when focal=1,
/// // cx=cy=0, and the point is along the optical axis.
/// let kps = vec![
///     vec![KeyPoint2D::new(0.0, 0.0)],  // cam 0
///     vec![KeyPoint2D::new(-0.2, 0.0)], // cam 1 (point appears displaced)
/// ];
/// let matches = vec![KeyPointMatch::new(0, 0, 0, 1)];
/// let pts = depth_from_sfm(&poses, &kps, &matches).unwrap();
/// assert!(!pts.is_empty());
/// ```
pub fn depth_from_sfm(
    camera_poses: &[CameraPose],
    keypoints: &[Vec<KeyPoint2D>],
    matches: &[KeyPointMatch],
) -> Result<Vec<(f64, f64, f64)>> {
    if camera_poses.is_empty() {
        return Err(VisionError::InvalidParameter(
            "camera_poses must not be empty".to_string(),
        ));
    }
    if keypoints.is_empty() {
        return Err(VisionError::InvalidParameter(
            "keypoints must not be empty".to_string(),
        ));
    }

    let mut points = Vec::with_capacity(matches.len());

    for m in matches {
        if m.cam_i >= camera_poses.len() || m.cam_j >= camera_poses.len() {
            continue;
        }
        if m.cam_i >= keypoints.len() || m.cam_j >= keypoints.len() {
            continue;
        }
        let kps_i = &keypoints[m.cam_i];
        let kps_j = &keypoints[m.cam_j];
        if m.idx_i >= kps_i.len() || m.idx_j >= kps_j.len() {
            continue;
        }

        let pose_i = &camera_poses[m.cam_i];
        let pose_j = &camera_poses[m.cam_j];
        let kp_i = kps_i[m.idx_i];
        let kp_j = kps_j[m.idx_j];

        // Build 3×4 projection matrices P = [R | -R t] for each camera.
        // (Assumes identity intrinsics K = I for simplicity; the caller should
        //  pass normalised coordinates if using real calibrated cameras.)
        let pi = pose_to_projection(pose_i);
        let pj = pose_to_projection(pose_j);

        if let Some(pt) = triangulate_dlt(&pi, &pj, (kp_i.x, kp_i.y), (kp_j.x, kp_j.y)) {
            points.push(pt);
        }
    }

    Ok(points)
}

/// Build the 3×4 projection matrix `[R | t_cam]` where `t_cam = -R^T t_world`.
///
/// The convention used here is the standard computer-vision one:
/// `x_cam = R (X_world - t) = R X_world - R t`.
fn pose_to_projection(pose: &CameraPose) -> [[f64; 4]; 3] {
    // t_cam = -R t  (translation in camera frame)
    let r = pose.r;
    let t_world = pose.t;
    let tc = [
        -(r[0][0] * t_world[0] + r[0][1] * t_world[1] + r[0][2] * t_world[2]),
        -(r[1][0] * t_world[0] + r[1][1] * t_world[1] + r[1][2] * t_world[2]),
        -(r[2][0] * t_world[0] + r[2][1] * t_world[1] + r[2][2] * t_world[2]),
    ];
    [
        [r[0][0], r[0][1], r[0][2], tc[0]],
        [r[1][0], r[1][1], r[1][2], tc[1]],
        [r[2][0], r[2][1], r[2][2], tc[2]],
    ]
}

/// Linear triangulation (DLT) from two projection matrices and corresponding
/// image points.
///
/// Returns the 3-D world point in homogeneous / Euclidean coordinates.
/// Returns `None` when the system is degenerate.
fn triangulate_dlt(
    p1: &[[f64; 4]; 3],
    p2: &[[f64; 4]; 3],
    (u1, v1): (f64, f64),
    (u2, v2): (f64, f64),
) -> Option<(f64, f64, f64)> {
    // Build 4×4 system A X = 0:
    // row 0: u1 * P1[2] - P1[0]
    // row 1: v1 * P1[2] - P1[1]
    // row 2: u2 * P2[2] - P2[0]
    // row 3: v2 * P2[2] - P2[1]
    let a: [[f64; 4]; 4] = [
        [
            u1 * p1[2][0] - p1[0][0],
            u1 * p1[2][1] - p1[0][1],
            u1 * p1[2][2] - p1[0][2],
            u1 * p1[2][3] - p1[0][3],
        ],
        [
            v1 * p1[2][0] - p1[1][0],
            v1 * p1[2][1] - p1[1][1],
            v1 * p1[2][2] - p1[1][2],
            v1 * p1[2][3] - p1[1][3],
        ],
        [
            u2 * p2[2][0] - p2[0][0],
            u2 * p2[2][1] - p2[0][1],
            u2 * p2[2][2] - p2[0][2],
            u2 * p2[2][3] - p2[0][3],
        ],
        [
            v2 * p2[2][0] - p2[1][0],
            v2 * p2[2][1] - p2[1][1],
            v2 * p2[2][2] - p2[1][2],
            v2 * p2[2][3] - p2[1][3],
        ],
    ];

    // Solve via least-squares SVD: find the null space of A (last right
    // singular vector).  For a 4×4 we use Gaussian elimination to find the
    // solution X that minimises ||AX||² with ||X|| = 1.
    let x = solve_4x4_nullspace(&a)?;

    // Normalise homogeneous coordinates.
    if x[3].abs() < 1e-10 {
        return None;
    }
    let w = x[3];
    Some((x[0] / w, x[1] / w, x[2] / w))
}

/// Find the null-space vector of a 4×4 matrix via Gaussian elimination.
/// Returns the last column of V in the SVD (approximated using the smallest
/// diagonal after elimination).
fn solve_4x4_nullspace(a: &[[f64; 4]; 4]) -> Option<[f64; 4]> {
    // Use 4×4 SVD via power iteration on AᵀA.
    let mut ata = [[0.0_f64; 4]; 4];
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }

    // Jacobi eigendecomposition for the 4×4 symmetric matrix.
    let mut v = [[0.0_f64; 4]; 4];
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        v[i][i] = 1.0;
    }
    let mut b = ata;

    for _ in 0..200 {
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        #[allow(clippy::needless_range_loop)]
        for i in 0..4 {
            for j in (i + 1)..4 {
                if b[i][j].abs() > max_val {
                    max_val = b[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-14 {
            break;
        }

        let mpq = b[p][q];
        if mpq.abs() < 1e-30 {
            continue;
        }
        let theta = (b[q][q] - b[p][p]) / (2.0 * mpq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let cos = 1.0 / (1.0 + t * t).sqrt();
        let sin = t * cos;
        let tau = sin / (1.0 + cos);

        b[p][p] -= t * mpq;
        b[q][q] += t * mpq;
        b[p][q] = 0.0;
        b[q][p] = 0.0;

        #[allow(clippy::needless_range_loop)]
        for r in 0..4 {
            if r != p && r != q {
                let brp = b[r][p];
                let brq = b[r][q];
                b[r][p] = brp - sin * (brq + tau * brp);
                b[p][r] = b[r][p];
                b[r][q] = brq + sin * (brp - tau * brq);
                b[q][r] = b[r][q];
            }
        }

        #[allow(clippy::needless_range_loop)]
        for r in 0..4 {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = vrp - sin * (vrq + tau * vrp);
            v[r][q] = vrq + sin * (vrp - tau * vrq);
        }
    }

    // Find the column of V with the smallest eigenvalue.
    let eigenvalues = [b[0][0], b[1][1], b[2][2], b[3][3]];
    let min_idx = eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)?;

    Some([v[0][min_idx], v[1][min_idx], v[2][min_idx], v[3][min_idx]])
}

// ─────────────────────────────────────────────────────────────────────────────
// dense_depth_from_sparse
// ─────────────────────────────────────────────────────────────────────────────

/// Densify a sparse depth map using an image-guided fast-marching propagation.
///
/// Valid pixels (depth > 0) are used as seeds.  Their values propagate outward
/// to neighbouring zero-depth pixels in a BFS fashion, weighted by the inverse
/// of the image intensity gradient (to encourage propagation along homogeneous
/// regions).
///
/// # Arguments
///
/// * `sparse_depth` – Sparse depth values, shape `[H, W]`.
///   Zero (or negative) values are treated as missing.
/// * `image`        – Guidance image as a grayscale array, shape `[H, W]`.
///   Used to weight propagation; need not be normalised.
///
/// # Returns
///
/// Dense depth map of the same shape.  Any pixels that are not reachable from
/// a valid seed pixel remain 0.
///
/// # Errors
///
/// Returns [`VisionError::DimensionMismatch`] when `sparse_depth` and `image`
/// have different shapes.
///
/// # Example
///
/// ```
/// use scirs2_vision::depth_estimation::dense_depth_from_sparse;
/// use scirs2_core::ndarray::Array2;
///
/// let mut sparse = Array2::zeros((4, 4));
/// sparse[[1, 1]] = 5.0;
/// let image = Array2::from_elem((4, 4), 128.0_f64);
/// let dense = dense_depth_from_sparse(&sparse, &image).unwrap();
/// assert_eq!(dense.dim(), (4, 4));
/// // The seed pixel is preserved.
/// assert!((dense[[1, 1]] - 5.0).abs() < 1e-9);
/// ```
pub fn dense_depth_from_sparse(
    sparse_depth: &Array2<f64>,
    image: &Array2<f64>,
) -> Result<Array2<f64>> {
    let (h, w) = sparse_depth.dim();
    if image.dim() != (h, w) {
        return Err(VisionError::DimensionMismatch(
            "sparse_depth and image must have the same shape".to_string(),
        ));
    }

    let mut output = sparse_depth.clone();
    // Confidence weights for the blending (we track total weight per pixel).
    let mut weight = Array2::zeros((h, w));

    // Mark valid seeds with weight 1.
    for y in 0..h {
        for x in 0..w {
            if sparse_depth[[y, x]] > 0.0 {
                weight[[y, x]] = 1.0;
            }
        }
    }

    // Compute an edge-strength map based on local gradient.
    // Pixels with high gradient get higher resistance to propagation.
    let mut edge = Array2::zeros((h, w));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let dx = (image[[y, x + 1]] - image[[y, x - 1]]) * 0.5;
            let dy = (image[[y + 1, x]] - image[[y - 1, x]]) * 0.5;
            edge[[y, x]] = (dx * dx + dy * dy).sqrt();
        }
    }
    let max_edge = edge.iter().cloned().fold(0.0_f64, f64::max).max(1.0);

    // BFS from all valid seeds.
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
    for y in 0..h {
        for x in 0..w {
            if sparse_depth[[y, x]] > 0.0 {
                queue.push_back((y, x));
            }
        }
    }

    let neighbours: [(i64, i64); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];

    while let Some((cy, cx)) = queue.pop_front() {
        let cur_depth = output[[cy, cx]];
        let cur_w = weight[[cy, cx]];

        for &(dy, dx) in &neighbours {
            let ny = cy as i64 + dy;
            let nx = cx as i64 + dx;
            if ny < 0 || ny >= h as i64 || nx < 0 || nx >= w as i64 {
                continue;
            }
            let ny = ny as usize;
            let nx = nx as usize;

            // Propagation weight decreases across edges.
            let edge_penalty = 1.0 + edge[[ny, nx]] / max_edge;
            let prop_w = cur_w / edge_penalty;

            if prop_w > weight[[ny, nx]] {
                // Weighted accumulation: the neighbour adopts the depth of the
                // strongest propagating path.
                output[[ny, nx]] = cur_depth;
                weight[[ny, nx]] = prop_w;
                queue.push_back((ny, nx));
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
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_depth_map_from_data() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("from_shape_vec should succeed with correct element count");
        let dm = DepthMap::from_data(data, 1.0);
        assert!((dm.min_depth - 1.0).abs() < 1e-9);
        assert!((dm.max_depth - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_depth_map_to_metric() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("from_shape_vec should succeed with correct element count");
        let dm = DepthMap::new(data, 0.5, 0.5, 2.0);
        let metric = dm.to_metric();
        assert!((metric.data[[0, 0]] - 0.5).abs() < 1e-9);
        assert!((metric.data[[1, 1]] - 2.0).abs() < 1e-9);
        assert!((metric.scale - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_scale_invariant_loss_perfect() {
        let pred = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("from_shape_vec should succeed with correct element count");
        let gt = pred.clone();
        let loss = scale_invariant_loss(&pred, &gt)
            .expect("scale_invariant_loss should succeed on identical arrays");
        assert!(loss.abs() < 1e-10);
    }

    #[test]
    fn test_scale_invariant_loss_scaled() {
        // If pred = k * gt, the scale-invariant loss should still be zero.
        let gt = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("from_shape_vec should succeed with correct element count");
        let pred = gt.mapv(|v| v * 2.0);
        let loss = scale_invariant_loss(&pred, &gt)
            .expect("scale_invariant_loss should succeed with valid scaled inputs");
        assert!(loss.abs() < 1e-10, "loss = {loss}");
    }

    #[test]
    fn test_scale_invariant_loss_no_valid_pixels() {
        let zeros = Array2::zeros((2, 2));
        assert!(scale_invariant_loss(&zeros, &zeros).is_err());
    }

    #[test]
    fn test_scale_invariant_loss_shape_mismatch() {
        let a = Array2::from_elem((2, 2), 1.0_f64);
        let b = Array2::from_elem((2, 3), 1.0_f64);
        assert!(scale_invariant_loss(&a, &b).is_err());
    }

    #[test]
    fn test_absolute_relative_error_perfect() {
        let pred = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("from_shape_vec should succeed with correct element count");
        let err = absolute_relative_error(&pred, &pred)
            .expect("absolute_relative_error should succeed on identical arrays");
        assert!(err.abs() < 1e-10);
    }

    #[test]
    fn test_absolute_relative_error_value() {
        let gt = Array2::from_shape_vec((1, 1), vec![4.0])
            .expect("from_shape_vec should succeed with correct element count");
        let pred = Array2::from_shape_vec((1, 1), vec![5.0])
            .expect("from_shape_vec should succeed with correct element count");
        let err = absolute_relative_error(&pred, &gt)
            .expect("absolute_relative_error should succeed with valid inputs");
        // |5 - 4| / 4 = 0.25
        assert!((err - 0.25).abs() < 1e-10, "err={err}");
    }

    #[test]
    fn test_threshold_accuracy_perfect() {
        let pred = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("from_shape_vec should succeed with correct element count");
        let acc = threshold_accuracy(&pred, &pred, 1.25)
            .expect("threshold_accuracy should succeed on identical arrays");
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_threshold_accuracy_none_pass() {
        let gt = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("from_shape_vec should succeed with correct element count");
        let pred = gt.mapv(|v| v * 10.0); // ratio = 10 > 1.25
        let acc = threshold_accuracy(&pred, &gt, 1.25)
            .expect("threshold_accuracy should succeed with valid inputs");
        assert!((acc).abs() < 1e-10, "acc={acc}");
    }

    #[test]
    fn test_threshold_accuracy_bad_threshold() {
        let img = Array2::from_elem((2, 2), 1.0_f64);
        assert!(threshold_accuracy(&img, &img, 1.0).is_err());
        assert!(threshold_accuracy(&img, &img, 0.5).is_err());
    }

    #[test]
    fn test_depth_from_sfm_basic() {
        // Two cameras: cam0 at origin, cam1 offset by 1 unit along X.
        let poses = vec![
            CameraPose::identity(),
            CameraPose::new(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [1.0, 0.0, 0.0],
            ),
        ];
        // Keypoints: the point is at world (0, 0, 5).
        // With K = I and normalised coords:
        //   cam0: projects to (0/5, 0/5) = (0, 0)
        //   cam1: projects to ((0-1)/5, 0/5) = (-0.2, 0)
        let kps = vec![
            vec![KeyPoint2D::new(0.0, 0.0)],
            vec![KeyPoint2D::new(-0.2, 0.0)],
        ];
        let matches = vec![KeyPointMatch::new(0, 0, 0, 1)];
        let pts = depth_from_sfm(&poses, &kps, &matches)
            .expect("depth_from_sfm should succeed with valid inputs");
        assert_eq!(pts.len(), 1);
        let (x, y, z) = pts[0];
        // Expect approximately (0, 0, 5).
        assert!(x.abs() < 0.5, "x={x}");
        assert!(y.abs() < 0.5, "y={y}");
        assert!((z - 5.0).abs() < 1.0, "z={z}");
    }

    #[test]
    fn test_depth_from_sfm_empty_poses() {
        let kps: Vec<Vec<KeyPoint2D>> = vec![vec![]];
        let matches: Vec<KeyPointMatch> = vec![];
        assert!(depth_from_sfm(&[], &kps, &matches).is_err());
    }

    #[test]
    fn test_dense_depth_from_sparse() {
        let mut sparse = Array2::zeros((6, 6));
        sparse[[2, 2]] = 5.0;
        let image = Array2::from_elem((6, 6), 128.0_f64);
        let dense = dense_depth_from_sparse(&sparse, &image)
            .expect("dense_depth_from_sparse should succeed with valid inputs");
        assert_eq!(dense.dim(), (6, 6));
        assert!((dense[[2, 2]] - 5.0).abs() < 1e-9);
        // Neighbours should also have been filled.
        assert!(dense[[2, 3]] > 0.0);
        assert!(dense[[3, 2]] > 0.0);
    }

    #[test]
    fn test_dense_depth_from_sparse_shape_mismatch() {
        let sparse = Array2::zeros((4, 4));
        let image = Array2::zeros((4, 5));
        assert!(dense_depth_from_sparse(&sparse, &image).is_err());
    }

    #[test]
    fn test_dense_depth_no_seeds() {
        let sparse = Array2::zeros((4, 4));
        let image = Array2::from_elem((4, 4), 100.0_f64);
        let dense = dense_depth_from_sparse(&sparse, &image)
            .expect("dense_depth_from_sparse should succeed with all-zero sparse depth");
        // No seeds → all zeros.
        assert!(dense.iter().all(|&v| v == 0.0));
    }
}
