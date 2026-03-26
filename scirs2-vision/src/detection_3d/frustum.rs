//! Frustum-based 3D object detection.
//!
//! Given a 2D bounding box and a set of LiDAR points in camera coordinates,
//! this module extracts the 3D "frustum" of points that project inside the box
//! and estimates a 3D bounding box via PCA.

use crate::error::{Result, VisionError};

// ---------------------------------------------------------------------------
// CameraIntrinsics
// ---------------------------------------------------------------------------

/// Pinhole camera intrinsic parameters.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CameraIntrinsics {
    /// Horizontal focal length in pixels.
    pub fx: f64,
    /// Vertical focal length in pixels.
    pub fy: f64,
    /// Principal point x-coordinate.
    pub cx: f64,
    /// Principal point y-coordinate.
    pub cy: f64,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl Default for CameraIntrinsics {
    /// KITTI left-camera defaults.
    fn default() -> Self {
        Self {
            fx: 718.856,
            fy: 718.856,
            cx: 607.1928,
            cy: 185.2157,
            width: 1242,
            height: 375,
        }
    }
}

// ---------------------------------------------------------------------------
// Bbox2D
// ---------------------------------------------------------------------------

/// Axis-aligned 2D bounding box in image space (pixel coordinates).
#[derive(Debug, Clone)]
pub struct Bbox2D {
    /// Left edge.
    pub x_min: f64,
    /// Top edge.
    pub y_min: f64,
    /// Right edge (exclusive).
    pub x_max: f64,
    /// Bottom edge (exclusive).
    pub y_max: f64,
}

impl Bbox2D {
    /// Create a new 2D bounding box. Returns an error if the box is degenerate.
    pub fn new(x_min: f64, y_min: f64, x_max: f64, y_max: f64) -> Result<Self> {
        if x_max <= x_min || y_max <= y_min {
            return Err(VisionError::InvalidParameter(
                "Bbox2D: x_max must exceed x_min and y_max must exceed y_min".into(),
            ));
        }
        Ok(Self {
            x_min,
            y_min,
            x_max,
            y_max,
        })
    }
}

// ---------------------------------------------------------------------------
// extract_frustum_points
// ---------------------------------------------------------------------------

/// Extract the subset of `points_3d` (in **camera** coordinates, i.e. Z is
/// depth) that project inside `bbox_2d` using the pinhole camera model.
///
/// Points with Z ≤ 0 are automatically discarded (behind / at the camera).
///
/// # Arguments
///
/// * `points_3d` — LiDAR/depth points expressed in camera frame `[X, Y, Z]`.
/// * `bbox_2d` — 2D detection box in pixel coordinates.
/// * `intrinsics` — pinhole camera parameters.
pub fn extract_frustum_points(
    points_3d: &[[f64; 3]],
    bbox_2d: &Bbox2D,
    intrinsics: &CameraIntrinsics,
) -> Result<Vec<[f64; 3]>> {
    let mut frustum = Vec::new();

    for &[x, y, z] in points_3d {
        // Discard points behind the camera.
        if z <= 0.0 {
            continue;
        }
        // Pinhole projection.
        let u = intrinsics.fx * x / z + intrinsics.cx;
        let v = intrinsics.fy * y / z + intrinsics.cy;

        if u >= bbox_2d.x_min && u <= bbox_2d.x_max && v >= bbox_2d.y_min && v <= bbox_2d.y_max {
            frustum.push([x, y, z]);
        }
    }

    Ok(frustum)
}

// ---------------------------------------------------------------------------
// BboxEstimationConfig / BBox3DEstimate
// ---------------------------------------------------------------------------

/// Configuration for the frustum-based 3D bounding-box estimation.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BboxEstimationConfig {
    /// Minimum number of points required to attempt estimation.
    pub min_points: usize,
    /// Number of angular bins for heading discretisation (unused in simplified
    /// PCA variant, but retained for API completeness).
    pub heading_bins: usize,
}

impl Default for BboxEstimationConfig {
    fn default() -> Self {
        Self {
            min_points: 5,
            heading_bins: 12,
        }
    }
}

/// Axis-aligned 3D bounding-box estimate in the PCA-aligned frame, reported
/// back in the original coordinate system.
#[derive(Debug, Clone)]
pub struct BBox3DEstimate {
    /// Centre of the estimated box.
    pub center: [f64; 3],
    /// Box dimensions `[length, width, height]`.
    pub dimensions: [f64; 3],
    /// Heading angle in radians (atan2 of first principal component, XY plane).
    pub heading_angle: f64,
    /// Number of points used to produce this estimate.
    pub n_points: usize,
}

// ---------------------------------------------------------------------------
// estimate_3d_bbox_from_frustum
// ---------------------------------------------------------------------------

/// Estimate a 3D bounding box from frustum points using PCA.
///
/// # Algorithm (simplified T-Net-free approach)
///
/// 1. Compute the centroid of the frustum points.
/// 2. Build the 3×3 scatter (covariance) matrix of centred points.
/// 3. Find principal components via Jacobi iteration.
/// 4. Rotate points into the PCA frame and compute the axis-aligned AABB.
/// 5. Report heading angle = `atan2(e0.y, e0.x)` where `e0` is the first PC.
pub fn estimate_3d_bbox_from_frustum(
    frustum_points: &[[f64; 3]],
    config: &BboxEstimationConfig,
) -> Result<BBox3DEstimate> {
    let n = frustum_points.len();
    if n < config.min_points {
        return Err(VisionError::InvalidParameter(format!(
            "Not enough frustum points ({n}) to estimate bounding box \
             (min_points={})",
            config.min_points
        )));
    }

    // 1. Centroid.
    let (sx, sy, sz) = frustum_points.iter().fold((0.0, 0.0, 0.0), |acc, &p| {
        (acc.0 + p[0], acc.1 + p[1], acc.2 + p[2])
    });
    let nf = n as f64;
    let mean = [sx / nf, sy / nf, sz / nf];

    // 2. Covariance matrix.
    let mut cov = [[0.0_f64; 3]; 3];
    for &p in frustum_points {
        let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        for r in 0..3 {
            for c in 0..3 {
                cov[r][c] += d[r] * d[c];
            }
        }
    }
    for row in cov.iter_mut() {
        for v in row.iter_mut() {
            *v /= nf;
        }
    }

    // 3. Eigendecomposition (Jacobi).
    let (_eigenvalues, eigenvectors) = jacobi_eigen_3x3(cov)?;

    // eigenvectors[i] = i-th eigenvector (row).  Sort by descending eigenvalue.
    let mut ev_pairs: Vec<(f64, [f64; 3])> = _eigenvalues
        .iter()
        .zip(eigenvectors.iter())
        .map(|(&val, &vec)| (val, vec))
        .collect();
    ev_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let e0 = ev_pairs[0].1; // largest variance direction
    let e1 = ev_pairs[1].1;
    let e2 = ev_pairs[2].1; // smallest variance direction

    // Heading angle from dominant XY direction.
    let heading_angle = e0[1].atan2(e0[0]);

    // 4. Project points onto PCA axes.
    let proj: Vec<[f64; 3]> = frustum_points
        .iter()
        .map(|&p| {
            let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
            [dot3(d, e0), dot3(d, e1), dot3(d, e2)]
        })
        .collect();

    // AABB in PCA frame.
    let (mut min0, mut max0) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min1, mut max1) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min2, mut max2) = (f64::INFINITY, f64::NEG_INFINITY);
    for &[a, b, c] in &proj {
        if a < min0 {
            min0 = a;
        }
        if a > max0 {
            max0 = a;
        }
        if b < min1 {
            min1 = b;
        }
        if b > max1 {
            max1 = b;
        }
        if c < min2 {
            min2 = c;
        }
        if c > max2 {
            max2 = c;
        }
    }

    let length = (max0 - min0).max(1e-6);
    let width = (max1 - min1).max(1e-6);
    let height = (max2 - min2).max(1e-6);

    // Centre in world coordinates.
    let pca_ctr = [
        (min0 + max0) * 0.5,
        (min1 + max1) * 0.5,
        (min2 + max2) * 0.5,
    ];
    let cx = mean[0] + pca_ctr[0] * e0[0] + pca_ctr[1] * e1[0] + pca_ctr[2] * e2[0];
    let cy = mean[1] + pca_ctr[0] * e0[1] + pca_ctr[1] * e1[1] + pca_ctr[2] * e2[1];
    let cz = mean[2] + pca_ctr[0] * e0[2] + pca_ctr[1] * e1[2] + pca_ctr[2] * e2[2];

    Ok(BBox3DEstimate {
        center: [cx, cy, cz],
        dimensions: [length, width, height],
        heading_angle,
        n_points: n,
    })
}

// ---------------------------------------------------------------------------
// nms_3d_oriented
// ---------------------------------------------------------------------------

/// Non-maximum suppression for oriented 3D bounding boxes.
///
/// Boxes are sorted by their associated `scores` in descending order. A box is
/// suppressed when its BEV IoU with any previously kept box exceeds
/// `iou_threshold`.
///
/// Returns the indices of **kept** boxes in the order they were selected.
pub fn nms_3d_oriented(
    boxes: &[BBox3DEstimate],
    scores: &[f64],
    iou_threshold: f64,
) -> Result<Vec<usize>> {
    if boxes.len() != scores.len() {
        return Err(VisionError::DimensionMismatch(format!(
            "nms_3d_oriented: boxes.len()={} != scores.len()={}",
            boxes.len(),
            scores.len()
        )));
    }

    // Sort indices by descending score.
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<usize> = Vec::new();

    'outer: for &idx in &order {
        for &kept_idx in &kept {
            if bev_iou_estimate(&boxes[idx], &boxes[kept_idx]) > iou_threshold {
                continue 'outer;
            }
        }
        kept.push(idx);
    }

    Ok(kept)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#[inline]
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Axis-aligned BEV IoU approximation for `BBox3DEstimate`.
///
/// Each box is treated as having its `heading_angle` yaw; its BEV footprint is
/// approximated by a rectangle with half-extents `(length/2, width/2)` rotated
/// by `heading_angle` and then further approximated by its AABB for the overlap
/// computation.  This matches the approach used in `detection_head::iou_3d_bev`.
fn bev_iou_estimate(a: &BBox3DEstimate, b: &BBox3DEstimate) -> f64 {
    let a_aabb = bev_aabb_estimate(a);
    let b_aabb = bev_aabb_estimate(b);

    let x_overlap = (a_aabb.1[0].min(b_aabb.1[0]) - a_aabb.0[0].max(b_aabb.0[0])).max(0.0);
    let y_overlap = (a_aabb.1[1].min(b_aabb.1[1]) - a_aabb.0[1].max(b_aabb.0[1])).max(0.0);

    let inter = x_overlap * y_overlap;
    let area_a = (a_aabb.1[0] - a_aabb.0[0]) * (a_aabb.1[1] - a_aabb.0[1]);
    let area_b = (b_aabb.1[0] - b_aabb.0[0]) * (b_aabb.1[1] - b_aabb.0[1]);
    let union = area_a + area_b - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Compute the AABB of a rotated BEV rectangle for a `BBox3DEstimate`.
///
/// Returns `(min_xy, max_xy)`.
fn bev_aabb_estimate(b: &BBox3DEstimate) -> ([f64; 2], [f64; 2]) {
    let (cos_h, sin_h) = b.heading_angle.cos_sin();
    let hl = b.dimensions[0] / 2.0;
    let hw = b.dimensions[1] / 2.0;

    // Four corners of the rotated rectangle.
    let corners = [
        (hl * cos_h - hw * sin_h, hl * sin_h + hw * cos_h),
        (-hl * cos_h - hw * sin_h, -hl * sin_h + hw * cos_h),
        (-hl * cos_h + hw * sin_h, -hl * sin_h - hw * cos_h),
        (hl * cos_h + hw * sin_h, hl * sin_h - hw * cos_h),
    ];

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for (cx, cy) in corners {
        let wx = b.center[0] + cx;
        let wy = b.center[1] + cy;
        if wx < min_x {
            min_x = wx;
        }
        if wx > max_x {
            max_x = wx;
        }
        if wy < min_y {
            min_y = wy;
        }
        if wy > max_y {
            max_y = wy;
        }
    }

    ([min_x, min_y], [max_x, max_y])
}

// Convenience trait for `(cos, sin)` tuple.
trait CosSin {
    fn cos_sin(self) -> (f64, f64);
}
impl CosSin for f64 {
    #[inline]
    fn cos_sin(self) -> (f64, f64) {
        (self.cos(), self.sin())
    }
}

/// Jacobi eigenvalue decomposition for a 3×3 symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where row `i` of `eigenvectors` is
/// the eigenvector for `eigenvalues[i]`.
fn jacobi_eigen_3x3(a_in: [[f64; 3]; 3]) -> Result<([f64; 3], [[f64; 3]; 3])> {
    let mut a = a_in;
    let mut v = [[0.0_f64; 3]; 3];
    for (i, v_row) in v.iter_mut().enumerate() {
        v_row[i] = 1.0;
    }

    const MAX_ITER: usize = 100;
    const EPS: f64 = 1e-12;

    for _ in 0..MAX_ITER {
        // Find largest off-diagonal element.
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for (r, a_row) in a.iter().enumerate() {
            for (c, &a_val) in a_row.iter().enumerate().skip(r + 1) {
                if a_val.abs() > max_val {
                    max_val = a_val.abs();
                    p = r;
                    q = c;
                }
            }
        }
        if max_val < EPS {
            break;
        }

        let theta = if (a[q][q] - a[p][p]).abs() < EPS {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * a[p][q]) / (a[q][q] - a[p][p])).atan()
        };

        let (s, cs) = theta.sin_cos();

        let mut a2 = a;
        a2[p][p] = cs * cs * a[p][p] - 2.0 * s * cs * a[p][q] + s * s * a[q][q];
        a2[q][q] = s * s * a[p][p] + 2.0 * s * cs * a[p][q] + cs * cs * a[q][q];
        a2[p][q] = 0.0;
        a2[q][p] = 0.0;

        let r = 3 - p - q;
        a2[p][r] = cs * a[p][r] - s * a[q][r];
        a2[r][p] = a2[p][r];
        a2[q][r] = s * a[p][r] + cs * a[q][r];
        a2[r][q] = a2[q][r];

        a = a2;

        for v_row in &mut v {
            let vi_p = v_row[p];
            let vi_q = v_row[q];
            v_row[p] = cs * vi_p - s * vi_q;
            v_row[q] = s * vi_p + cs * vi_q;
        }
    }

    Ok(([a[0][0], a[1][1], a[2][2]], v))
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_intrinsics() -> CameraIntrinsics {
        CameraIntrinsics::default()
    }

    /// Create a simple grid of 3D points that project inside a 2D box.
    fn make_frustum_cloud() -> (Vec<[f64; 3]>, Bbox2D) {
        let intrinsics = make_intrinsics();
        // All points at depth Z=10, centred around the optical axis.
        // u = fx * x/z + cx ≈ cx for x=0
        let z = 10.0;
        let pts: Vec<[f64; 3]> = (-2i32..=2)
            .flat_map(|ix| {
                (-2i32..=2).map(move |iy| {
                    let x = ix as f64 * 0.5; // small lateral offset
                    let y = iy as f64 * 0.3;
                    [x, y, z]
                })
            })
            .collect();

        // 2D bbox centred on the projection of (0,0,10) with generous margin.
        let u0 = intrinsics.cx;
        let v0 = intrinsics.cy;
        let bbox = Bbox2D::new(u0 - 100.0, v0 - 100.0, u0 + 100.0, v0 + 100.0).expect("valid bbox");
        (pts, bbox)
    }

    #[test]
    fn test_camera_intrinsics_defaults() {
        let intr = CameraIntrinsics::default();
        assert!((intr.fx - 718.856).abs() < 1e-6);
        assert_eq!(intr.width, 1242);
        assert_eq!(intr.height, 375);
    }

    #[test]
    fn test_bbox_estimation_config_defaults() {
        let cfg = BboxEstimationConfig::default();
        assert_eq!(cfg.min_points, 5);
        assert_eq!(cfg.heading_bins, 12);
    }

    #[test]
    fn test_frustum_extract_all_inside() {
        let (pts, bbox) = make_frustum_cloud();
        let intr = make_intrinsics();
        let result =
            extract_frustum_points(&pts, &bbox, &intr).expect("frustum extraction should succeed");
        // All 25 points should project inside the generous bbox.
        assert_eq!(
            result.len(),
            pts.len(),
            "all 25 points should be inside frustum"
        );
    }

    #[test]
    fn test_frustum_extract_filters_behind_camera() {
        let intr = make_intrinsics();
        let pts = vec![
            [0.0, 0.0, 5.0],  // valid
            [0.0, 0.0, -1.0], // behind camera
            [0.0, 0.0, 0.0],  // at camera
        ];
        let bbox = Bbox2D::new(0.0, 0.0, 1280.0, 720.0).expect("valid bbox");
        let result =
            extract_frustum_points(&pts, &bbox, &intr).expect("frustum extraction should succeed");
        assert_eq!(result.len(), 1); // only the Z=5 point survives
    }

    #[test]
    fn test_frustum_extract_filters_outside_bbox() {
        let intr = make_intrinsics();
        // Point projects far outside the box.
        let pts = vec![[1000.0, 0.0, 10.0]]; // u ≫ box
        let bbox = Bbox2D::new(100.0, 100.0, 200.0, 200.0).expect("valid bbox");
        let result =
            extract_frustum_points(&pts, &bbox, &intr).expect("frustum extraction should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_bbox_estimate_non_empty() {
        let (pts, bbox) = make_frustum_cloud();
        let intr = make_intrinsics();
        let frustum =
            extract_frustum_points(&pts, &bbox, &intr).expect("frustum extraction should succeed");
        let cfg = BboxEstimationConfig::default();
        let est =
            estimate_3d_bbox_from_frustum(&frustum, &cfg).expect("bbox estimation should succeed");
        assert_eq!(est.n_points, frustum.len());
        // All dimensions should be positive.
        assert!(est.dimensions[0] > 0.0);
        assert!(est.dimensions[1] > 0.0);
        assert!(est.dimensions[2] > 0.0);
        // heading angle in [-pi, pi].
        assert!(est.heading_angle >= -std::f64::consts::PI);
        assert!(est.heading_angle <= std::f64::consts::PI);
    }

    #[test]
    fn test_bbox_estimate_min_points_error() {
        let pts: Vec<[f64; 3]> = vec![[0.0, 0.0, 1.0]; 3]; // only 3 points
        let cfg = BboxEstimationConfig::default(); // min_points = 5
        let result = estimate_3d_bbox_from_frustum(&pts, &cfg);
        assert!(
            result.is_err(),
            "should fail with fewer points than min_points"
        );
    }

    #[test]
    fn test_nms_reduces_overlapping_boxes() {
        // Two heavily overlapping boxes and one far away.
        let box_a = BBox3DEstimate {
            center: [0.0, 0.0, 0.0],
            dimensions: [2.0, 1.0, 1.0],
            heading_angle: 0.0,
            n_points: 10,
        };
        let box_b = BBox3DEstimate {
            center: [0.05, 0.0, 0.0],
            dimensions: [2.0, 1.0, 1.0],
            heading_angle: 0.0,
            n_points: 8,
        };
        let box_c = BBox3DEstimate {
            center: [100.0, 100.0, 0.0],
            dimensions: [2.0, 1.0, 1.0],
            heading_angle: 0.0,
            n_points: 6,
        };
        let boxes = vec![box_a, box_b, box_c];
        let scores = vec![0.9, 0.7, 0.5];
        let kept = nms_3d_oriented(&boxes, &scores, 0.5).expect("NMS should succeed");
        // box_a and box_b overlap → only box_a kept; box_c is separate.
        assert_eq!(kept.len(), 2, "NMS should keep 2 boxes, got {kept:?}");
        assert!(kept.contains(&0)); // highest-score box_a
        assert!(kept.contains(&2)); // non-overlapping box_c
    }

    #[test]
    fn test_nms_mismatched_lengths_error() {
        let boxes = vec![BBox3DEstimate {
            center: [0.0, 0.0, 0.0],
            dimensions: [1.0, 1.0, 1.0],
            heading_angle: 0.0,
            n_points: 1,
        }];
        let scores = vec![0.9, 0.8]; // length mismatch
        let result = nms_3d_oriented(&boxes, &scores, 0.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_bbox2d_degenerate_error() {
        assert!(Bbox2D::new(10.0, 10.0, 5.0, 20.0).is_err());
        assert!(Bbox2D::new(10.0, 10.0, 20.0, 5.0).is_err());
    }
}
