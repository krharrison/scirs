//! 3D bounding box IoU computations.
//!
//! Supports:
//! - Axis-aligned IoU (fast)
//! - Bird's-Eye View (BEV) IoU with rotated rectangles (Sutherland-Hodgman)
//! - Full 3D IoU (BEV intersection * z-overlap)
//! - AP computation at configurable IoU threshold

/// A 3D bounding box with yaw rotation around the Z-axis.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct BBox3D {
    /// Center X
    pub cx: f64,
    /// Center Y
    pub cy: f64,
    /// Center Z
    pub cz: f64,
    /// Extent along X axis (length)
    pub lx: f64,
    /// Extent along Y axis (width)
    pub ly: f64,
    /// Extent along Z axis (height)
    pub lz: f64,
    /// Yaw rotation around Z-axis in radians
    pub yaw: f64,
}

impl BBox3D {
    /// Construct a new 3D bounding box.
    pub fn new(cx: f64, cy: f64, cz: f64, lx: f64, ly: f64, lz: f64, yaw: f64) -> Self {
        Self {
            cx,
            cy,
            cz,
            lx,
            ly,
            lz,
            yaw,
        }
    }

    /// Volume of this box.
    pub fn volume(&self) -> f64 {
        self.lx * self.ly * self.lz
    }

    /// Axis-aligned IoU (ignores yaw). Fast conservative bound.
    pub fn iou_axis_aligned(&self, other: &BBox3D) -> f64 {
        // Extents from centers
        let ax1 = self.cx - self.lx * 0.5;
        let ax2 = self.cx + self.lx * 0.5;
        let ay1 = self.cy - self.ly * 0.5;
        let ay2 = self.cy + self.ly * 0.5;
        let az1 = self.cz - self.lz * 0.5;
        let az2 = self.cz + self.lz * 0.5;

        let bx1 = other.cx - other.lx * 0.5;
        let bx2 = other.cx + other.lx * 0.5;
        let by1 = other.cy - other.ly * 0.5;
        let by2 = other.cy + other.ly * 0.5;
        let bz1 = other.cz - other.lz * 0.5;
        let bz2 = other.cz + other.lz * 0.5;

        let ix = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
        let iy = (ay2.min(by2) - ay1.max(by1)).max(0.0);
        let iz = (az2.min(bz2) - az1.max(bz1)).max(0.0);

        let intersection = ix * iy * iz;
        if intersection <= 0.0 {
            return 0.0;
        }
        let union = self.volume() + other.volume() - intersection;
        if union <= 0.0 {
            0.0
        } else {
            intersection / union
        }
    }

    /// Bird's-Eye View (BEV) IoU using Sutherland-Hodgman polygon clipping for
    /// rotated rectangles, combined with exact Z-axis overlap.
    pub fn iou_bev(&self, other: &BBox3D) -> f64 {
        // Z-axis overlap
        let az1 = self.cz - self.lz * 0.5;
        let az2 = self.cz + self.lz * 0.5;
        let bz1 = other.cz - other.lz * 0.5;
        let bz2 = other.cz + other.lz * 0.5;
        let z_overlap = (az2.min(bz2) - az1.max(bz1)).max(0.0);

        if z_overlap <= 0.0 {
            return 0.0;
        }

        // BEV polygon intersection
        let corners_a = get_corners_2d(self);
        let corners_b = get_corners_2d(other);
        let inter_area = polygon_intersection_area(&corners_a, &corners_b);

        if inter_area <= 0.0 {
            return 0.0;
        }

        let bev_area_a = self.lx * self.ly;
        let bev_area_b = other.lx * other.ly;
        let bev_union = bev_area_a + bev_area_b - inter_area;

        let inter_vol = inter_area * z_overlap;
        let union_vol = bev_union * z_overlap
            + (bev_area_a * self.lz + bev_area_b * other.lz - inter_area * z_overlap);

        // Properly: union_vol = vol_a + vol_b - intersection_vol
        let intersection_vol = inter_area * z_overlap;
        let true_union = self.volume() + other.volume() - intersection_vol;

        if true_union <= 0.0 {
            0.0
        } else {
            inter_vol / true_union
        }
    }

    /// Full 3D IoU (same as iou_bev for boxes with z-axis yaw only).
    /// This correctly handles rotation in the horizontal plane.
    pub fn iou_3d(&self, other: &BBox3D) -> f64 {
        self.iou_bev(other)
    }
}

/// Get the 4 corners of a rotated rectangle in BEV (XY plane).
pub(crate) fn get_corners_2d(bbox: &BBox3D) -> [[f64; 2]; 4] {
    let cos_yaw = bbox.yaw.cos();
    let sin_yaw = bbox.yaw.sin();
    let hx = bbox.lx * 0.5;
    let hy = bbox.ly * 0.5;

    // Local corners before rotation: (±hx, ±hy)
    let local: [[f64; 2]; 4] = [[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]];

    let mut corners = [[0.0f64; 2]; 4];
    for (i, lc) in local.iter().enumerate() {
        corners[i][0] = bbox.cx + cos_yaw * lc[0] - sin_yaw * lc[1];
        corners[i][1] = bbox.cy + sin_yaw * lc[0] + cos_yaw * lc[1];
    }
    corners
}

/// Compute area of polygon intersection using Sutherland-Hodgman clipping.
pub(crate) fn polygon_intersection_area(corners_a: &[[f64; 2]], corners_b: &[[f64; 2]]) -> f64 {
    if corners_a.is_empty() || corners_b.is_empty() {
        return 0.0;
    }

    // Start with corners_a as subject polygon, clip by each edge of corners_b
    let mut clipped: Vec<[f64; 2]> = corners_a.to_vec();

    let n_b = corners_b.len();
    for i in 0..n_b {
        if clipped.is_empty() {
            break;
        }
        let p1 = corners_b[i];
        let p2 = corners_b[(i + 1) % n_b];
        clipped = clip_polygon_by_halfplane(&clipped, p1, p2);
    }

    polygon_area(&clipped)
}

/// Sutherland-Hodgman polygon clipping against one half-plane defined by edge p1->p2.
/// Points on the left side (interior) are kept.
pub(crate) fn clip_polygon_by_halfplane(
    poly: &[[f64; 2]],
    p1: [f64; 2],
    p2: [f64; 2],
) -> Vec<[f64; 2]> {
    if poly.is_empty() {
        return vec![];
    }

    let mut output = Vec::with_capacity(poly.len() + 1);
    let n = poly.len();

    for i in 0..n {
        let current = poly[i];
        let prev = poly[(i + n - 1) % n];

        let current_inside = is_inside(current, p1, p2);
        let prev_inside = is_inside(prev, p1, p2);

        if current_inside {
            if !prev_inside {
                // prev outside -> current inside: add intersection then current
                if let Some(pt) = line_intersection(prev, current, p1, p2) {
                    output.push(pt);
                }
            }
            output.push(current);
        } else if prev_inside {
            // prev inside -> current outside: add intersection only
            if let Some(pt) = line_intersection(prev, current, p1, p2) {
                output.push(pt);
            }
        }
    }

    output
}

/// Test whether point p is on the left (interior) side of directed edge p1->p2.
#[inline]
fn is_inside(p: [f64; 2], p1: [f64; 2], p2: [f64; 2]) -> bool {
    // Cross product of (p2-p1) x (p-p1) >= 0 means left or on the edge
    let cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0]);
    cross >= 0.0
}

/// Compute intersection of segment (a->b) with the infinite line through (p1, p2).
fn line_intersection(a: [f64; 2], b: [f64; 2], p1: [f64; 2], p2: [f64; 2]) -> Option<[f64; 2]> {
    let d_ab = [b[0] - a[0], b[1] - a[1]];
    let d_p = [p2[0] - p1[0], p2[1] - p1[1]];

    let denom = d_ab[0] * d_p[1] - d_ab[1] * d_p[0];
    if denom.abs() < 1e-12 {
        return None; // Parallel
    }

    let t = ((p1[0] - a[0]) * d_p[1] - (p1[1] - a[1]) * d_p[0]) / denom;
    Some([a[0] + t * d_ab[0], a[1] + t * d_ab[1]])
}

/// Shoelace formula for polygon area (absolute value).
pub(crate) fn polygon_area(corners: &[[f64; 2]]) -> f64 {
    let n = corners.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0f64;
    for i in 0..n {
        let j = (i + 1) % n;
        area += corners[i][0] * corners[j][1];
        area -= corners[j][0] * corners[i][1];
    }
    area.abs() * 0.5
}

/// Metrics for 3D object detection evaluation.
#[derive(Debug, Clone)]
pub struct Detection3dMetrics {
    /// Average precision (mean over IoU thresholds 0.25, 0.5, 0.75 if multi-threshold)
    pub ap: f64,
    /// AP at IoU threshold 0.5
    pub ap_iou50: f64,
    /// AP at IoU threshold 0.25
    pub ap_iou25: f64,
    /// True positives at default threshold
    pub tp: usize,
    /// False positives at default threshold
    pub fp: usize,
    /// False negatives at default threshold
    pub fn_: usize,
}

/// Compute average precision for 3D detection at the given IoU threshold.
///
/// Predictions should be sorted by score (descending) if possible; otherwise
/// the function sorts internally by `pred_scores`.
pub fn compute_ap_3d(
    gt: &[BBox3D],
    pred: &[BBox3D],
    pred_scores: &[f64],
    iou_threshold: f64,
) -> Detection3dMetrics {
    if gt.is_empty() || pred.is_empty() {
        return Detection3dMetrics {
            ap: 0.0,
            ap_iou50: 0.0,
            ap_iou25: 0.0,
            tp: 0,
            fp: pred.len(),
            fn_: gt.len(),
        };
    }

    let ap_iou50 = compute_ap_at_threshold(gt, pred, pred_scores, 0.5);
    let ap_iou25 = compute_ap_at_threshold(gt, pred, pred_scores, 0.25);
    let ap_main = compute_ap_at_threshold(gt, pred, pred_scores, iou_threshold);

    // Compute TP/FP/FN at specified threshold for reporting
    let (tp, fp, fn_) = compute_tp_fp_fn(gt, pred, pred_scores, iou_threshold);

    Detection3dMetrics {
        ap: ap_main,
        ap_iou50,
        ap_iou25,
        tp,
        fp,
        fn_,
    }
}

/// Compute average precision (11-point interpolation) at a single IoU threshold.
fn compute_ap_at_threshold(
    gt: &[BBox3D],
    pred: &[BBox3D],
    pred_scores: &[f64],
    iou_threshold: f64,
) -> f64 {
    if pred.is_empty() || gt.is_empty() {
        return 0.0;
    }

    // Sort predictions by descending score
    let mut indices: Vec<usize> = (0..pred.len()).collect();
    indices.sort_by(|&a, &b| {
        pred_scores[b]
            .partial_cmp(&pred_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_gt = gt.len();
    let mut gt_matched = vec![false; n_gt];

    let mut tp_cum = Vec::with_capacity(pred.len());
    let mut fp_cum = Vec::with_capacity(pred.len());
    let mut tp_count = 0usize;
    let mut fp_count = 0usize;

    for &pi in &indices {
        // Find best-matching GT
        let mut best_iou = iou_threshold - 1e-9;
        let mut best_gt = None;

        for (gi, g) in gt.iter().enumerate() {
            if gt_matched[gi] {
                continue;
            }
            let iou = pred[pi].iou_3d(g);
            if iou > best_iou {
                best_iou = iou;
                best_gt = Some(gi);
            }
        }

        if let Some(gi) = best_gt {
            gt_matched[gi] = true;
            tp_count += 1;
        } else {
            fp_count += 1;
        }

        tp_cum.push(tp_count);
        fp_cum.push(fp_count);
    }

    // Compute precision/recall curve
    let n_pred = pred.len();
    let mut precisions = Vec::with_capacity(n_pred);
    let mut recalls = Vec::with_capacity(n_pred);

    for i in 0..n_pred {
        let rec = tp_cum[i] as f64 / n_gt as f64;
        let prec = tp_cum[i] as f64 / (tp_cum[i] + fp_cum[i]) as f64;
        recalls.push(rec);
        precisions.push(prec);
    }

    // 11-point interpolated AP
    let mut ap = 0.0f64;
    for t in 0..=10 {
        let recall_threshold = t as f64 / 10.0;
        let max_prec = precisions
            .iter()
            .zip(recalls.iter())
            .filter(|(_, &r)| r >= recall_threshold)
            .map(|(&p, _)| p)
            .fold(0.0f64, f64::max);
        ap += max_prec;
    }
    ap / 11.0
}

/// Compute TP/FP/FN at a given threshold (greedy matching by descending score).
fn compute_tp_fp_fn(
    gt: &[BBox3D],
    pred: &[BBox3D],
    pred_scores: &[f64],
    iou_threshold: f64,
) -> (usize, usize, usize) {
    let mut indices: Vec<usize> = (0..pred.len()).collect();
    indices.sort_by(|&a, &b| {
        pred_scores[b]
            .partial_cmp(&pred_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_gt = gt.len();
    let mut gt_matched = vec![false; n_gt];
    let mut tp = 0usize;
    let mut fp = 0usize;

    for &pi in &indices {
        let mut best_iou = iou_threshold - 1e-9;
        let mut best_gi = None;

        for (gi, g) in gt.iter().enumerate() {
            if gt_matched[gi] {
                continue;
            }
            let iou = pred[pi].iou_3d(g);
            if iou > best_iou {
                best_iou = iou;
                best_gi = Some(gi);
            }
        }

        if let Some(gi) = best_gi {
            gt_matched[gi] = true;
            tp += 1;
        } else {
            fp += 1;
        }
    }

    let fn_ = n_gt - gt_matched.iter().filter(|&&m| m).count();
    (tp, fp, fn_)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identical_box() -> (BBox3D, BBox3D) {
        let a = BBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0);
        let b = BBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0);
        (a, b)
    }

    #[test]
    fn test_iou_axis_aligned_identical() {
        let (a, b) = identical_box();
        let iou = a.iou_axis_aligned(&b);
        assert!(
            (iou - 1.0).abs() < 1e-9,
            "identical boxes should have IoU=1, got {iou}"
        );
    }

    #[test]
    fn test_iou_axis_aligned_nonoverlapping() {
        let a = BBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0);
        let b = BBox3D::new(10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 0.0);
        let iou = a.iou_axis_aligned(&b);
        assert_eq!(iou, 0.0);
    }

    #[test]
    fn test_iou_axis_aligned_contained() {
        // Small box entirely inside large box
        let large = BBox3D::new(0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 0.0);
        let small = BBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0);
        let iou = small.iou_axis_aligned(&large);
        // intersection = small.volume() = 8, union = 64 + 8 - 8 = 64
        let expected = 8.0 / 64.0;
        assert!(
            (iou - expected).abs() < 1e-9,
            "contained box IoU should be {expected}, got {iou}"
        );
    }

    #[test]
    fn test_volume_correct() {
        let b = BBox3D::new(0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0);
        assert!((b.volume() - 60.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_bev_zero_yaw_equals_axis_aligned() {
        let a = BBox3D::new(0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0);
        let b = BBox3D::new(0.5, 0.5, 0.0, 2.0, 2.0, 2.0, 0.0);
        let bev = a.iou_bev(&b);
        let aa = a.iou_axis_aligned(&b);
        assert!(
            (bev - aa).abs() < 1e-6,
            "BEV IoU ({bev}) should equal axis-aligned ({aa}) when yaw=0"
        );
    }

    #[test]
    fn test_iou_bev_nonoverlapping() {
        let a = BBox3D::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0);
        let b = BBox3D::new(100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 0.0);
        assert_eq!(a.iou_bev(&b), 0.0);
    }

    #[test]
    fn test_sutherland_hodgman_clip_full_polygon() {
        // Clip a unit square by the halfplane y >= 0 (line from (0,0) to (1,0), interior is above)
        let square = vec![[0.0f64, -1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 1.0]];
        let clipped = clip_polygon_by_halfplane(&square, [0.0, 0.0], [1.0, 0.0]);
        let area = polygon_area(&clipped);
        // Should be the upper half: area = 1.0
        assert!(
            (area - 1.0).abs() < 1e-9,
            "clipped area should be 1.0, got {area}"
        );
    }

    #[test]
    fn test_polygon_area_unit_square() {
        let sq = vec![[0.0f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let area = polygon_area(&sq);
        assert!((area - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_iou_3d_identical() {
        let (a, b) = identical_box();
        let iou = a.iou_3d(&b);
        assert!(
            (iou - 1.0).abs() < 1e-9,
            "iou_3d identical should be 1.0, got {iou}"
        );
    }
}
