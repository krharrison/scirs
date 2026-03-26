//! Detection head: anchor generation, prediction decoding, NMS, and losses.

use super::types::{BoundingBox3D, Detection3D};
use crate::error::{Result, VisionError};

// ---------------------------------------------------------------------------
// Anchor generation
// ---------------------------------------------------------------------------

/// Generates anchor boxes at every position of a BEV grid.
#[derive(Debug, Clone)]
pub struct AnchorGenerator {
    /// Anchor sizes: \[length, width, height\] per anchor type.
    pub anchor_sizes: Vec<[f64; 3]>,
    /// Yaw rotations to use for each anchor (radians).
    pub rotations: Vec<f64>,
    /// Z-centre of anchors.
    pub anchor_z: f64,
}

impl Default for AnchorGenerator {
    fn default() -> Self {
        Self {
            anchor_sizes: vec![
                [4.0, 1.8, 1.5], // car-like
                [0.8, 0.6, 1.7], // pedestrian-like
                [2.0, 0.6, 1.4], // cyclist-like
            ],
            rotations: vec![0.0, std::f64::consts::FRAC_PI_2],
            anchor_z: -1.0,
        }
    }
}

impl AnchorGenerator {
    /// Generate all anchors for a grid of size `(grid_h, grid_w)` with given
    /// spatial resolution `(res_x, res_y)` and origin `(x0, y0)`.
    ///
    /// Returns a flat vector of `BoundingBox3D`.
    pub fn generate(
        &self,
        grid_h: usize,
        grid_w: usize,
        x0: f64,
        y0: f64,
        res_x: f64,
        res_y: f64,
    ) -> Vec<BoundingBox3D> {
        let mut anchors =
            Vec::with_capacity(grid_h * grid_w * self.anchor_sizes.len() * self.rotations.len());
        for iy in 0..grid_h {
            let cy = y0 + (iy as f64 + 0.5) * res_y;
            for ix in 0..grid_w {
                let cx = x0 + (ix as f64 + 0.5) * res_x;
                for size in &self.anchor_sizes {
                    for &rot in &self.rotations {
                        anchors.push(BoundingBox3D {
                            center: [cx, cy, self.anchor_z],
                            size: *size,
                            yaw: rot,
                        });
                    }
                }
            }
        }
        anchors
    }
}

// ---------------------------------------------------------------------------
// Detection head
// ---------------------------------------------------------------------------

/// Anchor-based detection head.
///
/// For each anchor it predicts:
/// * classification logits (`num_classes` values)
/// * bounding-box regression (`7` values: dx, dy, dz, dl, dw, dh, d_yaw)
#[derive(Debug, Clone)]
pub struct DetectionHead {
    /// Number of object classes.
    pub num_classes: usize,
    /// Classification weights: `(feat_dim, num_classes)`.
    cls_weights: Vec<Vec<f64>>,
    /// Regression weights: `(feat_dim, 7)`.
    reg_weights: Vec<Vec<f64>>,
    /// Feature dimension.
    feat_dim: usize,
}

impl DetectionHead {
    /// Create a detection head for the given feature dimension and number of
    /// classes.
    pub fn new(feat_dim: usize, num_classes: usize) -> Self {
        let scale_cls = (2.0 / (feat_dim + num_classes) as f64).sqrt();
        let scale_reg = (2.0 / (feat_dim + 7) as f64).sqrt();
        let cls_weights = (0..feat_dim)
            .map(|i| {
                (0..num_classes)
                    .map(|j| ((i * 11 + j * 17 + 7) as f64).sin() * scale_cls)
                    .collect()
            })
            .collect();
        let reg_weights = (0..feat_dim)
            .map(|i| {
                (0..7)
                    .map(|j| ((i * 11 + j * 17 + 3) as f64).sin() * scale_reg)
                    .collect()
            })
            .collect();
        Self {
            num_classes,
            cls_weights,
            reg_weights,
            feat_dim,
        }
    }

    /// Run the detection head on a feature vector of length `feat_dim`.
    ///
    /// Returns `(cls_logits[num_classes], reg_offsets[7])`.
    pub fn predict(&self, features: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        if features.len() != self.feat_dim {
            return Err(VisionError::DimensionMismatch(format!(
                "DetectionHead: expected feat_dim={}, got {}",
                self.feat_dim,
                features.len()
            )));
        }
        let mut cls = vec![0.0f64; self.num_classes];
        let mut reg = vec![0.0f64; 7];

        for (i, &fv) in features.iter().enumerate() {
            for (j, cls_val) in cls.iter_mut().enumerate().take(self.num_classes) {
                *cls_val += fv * self.cls_weights[i][j];
            }
            for (j, reg_val) in reg.iter_mut().enumerate().take(7) {
                *reg_val += fv * self.reg_weights[i][j];
            }
        }
        Ok((cls, reg))
    }
}

// ---------------------------------------------------------------------------
// Decode predictions
// ---------------------------------------------------------------------------

/// Decode classification logits and regression offsets into `Detection3D`
/// instances.
///
/// * `anchors` – anchor boxes (one per prediction).
/// * `cls_logits` – `[n_anchors][num_classes]` raw logits.
/// * `bbox_reg` – `[n_anchors][7]` regression offsets `(dx,dy,dz,dl,dw,dh,dyaw)`.
/// * `score_threshold` – discard detections below this confidence.
pub fn decode_predictions(
    anchors: &[BoundingBox3D],
    cls_logits: &[Vec<f64>],
    bbox_reg: &[Vec<f64>],
    score_threshold: f64,
) -> Result<Vec<Detection3D>> {
    if anchors.len() != cls_logits.len() || anchors.len() != bbox_reg.len() {
        return Err(VisionError::DimensionMismatch(
            "decode_predictions: mismatched lengths".to_string(),
        ));
    }

    let mut detections = Vec::new();

    for (i, anchor) in anchors.iter().enumerate() {
        let logits = &cls_logits[i];
        let reg = &bbox_reg[i];
        if reg.len() < 7 {
            continue;
        }

        // Find best class via sigmoid of logits.
        let mut best_cls = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (c, &logit) in logits.iter().enumerate() {
            let score = sigmoid(logit);
            if score > best_score {
                best_score = score;
                best_cls = c;
            }
        }

        if best_score < score_threshold {
            continue;
        }

        // Decode box.
        let cx = anchor.center[0] + reg[0] * anchor.size[0];
        let cy = anchor.center[1] + reg[1] * anchor.size[1];
        let cz = anchor.center[2] + reg[2] * anchor.size[2];
        let l = anchor.size[0] * reg[3].exp();
        let w = anchor.size[1] * reg[4].exp();
        let h = anchor.size[2] * reg[5].exp();
        let yaw = anchor.yaw + reg[6];

        detections.push(Detection3D {
            bbox: BoundingBox3D::new([cx, cy, cz], [l, w, h], yaw),
            class_id: best_cls,
            confidence: best_score,
            class_name: None,
        });
    }

    Ok(detections)
}

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------

/// Non-maximum suppression using BEV IoU.
///
/// Detections are sorted by confidence and greedily kept when their BEV IoU
/// with all previously-kept detections is below `iou_threshold`.
pub fn nms_3d(detections: &[Detection3D], iou_threshold: f64) -> Vec<Detection3D> {
    let mut indices: Vec<usize> = (0..detections.len()).collect();
    indices.sort_by(|&a, &b| {
        detections[b]
            .confidence
            .partial_cmp(&detections[a].confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep: Vec<Detection3D> = Vec::new();

    for &idx in &indices {
        let det = &detections[idx];
        let dominated = keep
            .iter()
            .any(|kept| iou_3d_bev(&det.bbox, &kept.bbox) > iou_threshold);
        if !dominated {
            keep.push(det.clone());
        }
    }
    keep
}

// ---------------------------------------------------------------------------
// IoU
// ---------------------------------------------------------------------------

/// Compute intersection-over-union in the BEV plane (2D rotated rectangle
/// approximation).
///
/// For simplicity this uses an axis-aligned bounding-box approximation of the
/// rotated rectangles. A full Sutherland-Hodgman implementation would be more
/// precise but this is sufficient for a reference detector.
pub fn iou_3d_bev(a: &BoundingBox3D, b: &BoundingBox3D) -> f64 {
    // Compute AABB of each rotated rectangle in BEV.
    let a_corners = bev_aabb(a);
    let b_corners = bev_aabb(b);

    let x_overlap =
        (a_corners.1[0].min(b_corners.1[0]) - a_corners.0[0].max(b_corners.0[0])).max(0.0);
    let y_overlap =
        (a_corners.1[1].min(b_corners.1[1]) - a_corners.0[1].max(b_corners.0[1])).max(0.0);

    let inter = x_overlap * y_overlap;
    let area_a = (a_corners.1[0] - a_corners.0[0]) * (a_corners.1[1] - a_corners.0[1]);
    let area_b = (b_corners.1[0] - b_corners.0[0]) * (b_corners.1[1] - b_corners.0[1]);
    let union = area_a + area_b - inter;

    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Axis-aligned bounding box of a rotated 3D box projected onto BEV.
fn bev_aabb(b: &BoundingBox3D) -> ([f64; 2], [f64; 2]) {
    let corners = b.corners();
    let mut min_xy = [f64::INFINITY; 2];
    let mut max_xy = [f64::NEG_INFINITY; 2];
    // Only need the 4 bottom corners (top corners have same x,y).
    for corner in corners.iter().take(4) {
        min_xy[0] = min_xy[0].min(corner[0]);
        min_xy[1] = min_xy[1].min(corner[1]);
        max_xy[0] = max_xy[0].max(corner[0]);
        max_xy[1] = max_xy[1].max(corner[1]);
    }
    (min_xy, max_xy)
}

// ---------------------------------------------------------------------------
// Losses
// ---------------------------------------------------------------------------

/// Focal loss for classification.
///
/// `focal_loss = -alpha * (1 - p_t)^gamma * log(p_t)`
///
/// * `logits` and `targets` should have the same length.
/// * `targets` are 0 or 1 ground-truth labels.
pub fn focal_loss(logits: &[f64], targets: &[f64], alpha: f64, gamma: f64) -> f64 {
    let n = logits.len().min(targets.len());
    if n == 0 {
        return 0.0;
    }
    let mut loss = 0.0;
    for i in 0..n {
        let p = sigmoid(logits[i]);
        let t = targets[i];
        let p_t = t * p + (1.0 - t) * (1.0 - p);
        let p_t_clamped = p_t.max(1e-12); // avoid log(0)
        loss += -alpha * (1.0 - p_t).powf(gamma) * p_t_clamped.ln();
    }
    loss / n as f64
}

/// Smooth-L1 (Huber) loss.
///
/// `smooth_l1(x) = 0.5 * x^2 / beta` if `|x| < beta`, else `|x| - 0.5 * beta`.
pub fn smooth_l1_loss(pred: &[f64], target: &[f64], beta: f64) -> f64 {
    let n = pred.len().min(target.len());
    if n == 0 {
        return 0.0;
    }
    let beta = beta.max(1e-15);
    let mut loss = 0.0;
    for i in 0..n {
        let diff = (pred[i] - target[i]).abs();
        if diff < beta {
            loss += 0.5 * diff * diff / beta;
        } else {
            loss += diff - 0.5 * beta;
        }
    }
    loss / n as f64
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Numerically stable sigmoid.
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn anchor_generation_count() {
        let gen = AnchorGenerator::default();
        let anchors = gen.generate(4, 4, -10.0, -10.0, 5.0, 5.0);
        // 4*4 grid * 3 sizes * 2 rotations = 96
        assert_eq!(anchors.len(), 96);
    }

    #[test]
    fn detection_head_predict() {
        let head = DetectionHead::new(16, 3);
        let feat = vec![0.1; 16];
        let (cls, reg) = head.predict(&feat).expect("predict");
        assert_eq!(cls.len(), 3);
        assert_eq!(reg.len(), 7);
    }

    #[test]
    fn decode_empty() {
        let dets = decode_predictions(&[], &[], &[], 0.5).expect("decode");
        assert!(dets.is_empty());
    }

    #[test]
    fn decode_basic() {
        let anchor = BoundingBox3D::new([0.0, 0.0, 0.0], [4.0, 2.0, 1.5], 0.0);
        // High logit → high confidence.
        let cls = vec![vec![5.0, -5.0, -5.0]];
        let reg = vec![vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
        let dets = decode_predictions(&[anchor], &cls, &reg, 0.1).expect("decode");
        assert_eq!(dets.len(), 1);
        assert!(dets[0].confidence > 0.99);
        assert_eq!(dets[0].class_id, 0);
    }

    #[test]
    fn nms_removes_overlap() {
        let d1 = Detection3D {
            bbox: BoundingBox3D::new([0.0, 0.0, 0.0], [4.0, 2.0, 1.5], 0.0),
            class_id: 0,
            confidence: 0.9,
            class_name: None,
        };
        let d2 = Detection3D {
            bbox: BoundingBox3D::new([0.1, 0.0, 0.0], [4.0, 2.0, 1.5], 0.0),
            class_id: 0,
            confidence: 0.8,
            class_name: None,
        };
        let kept = nms_3d(&[d1, d2], 0.3);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].confidence - 0.9).abs() < 1e-12);
    }

    #[test]
    fn iou_identical_boxes() {
        let b = BoundingBox3D::new([0.0, 0.0, 0.0], [4.0, 2.0, 1.5], 0.0);
        let iou = iou_3d_bev(&b, &b);
        assert!((iou - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_no_overlap() {
        let a = BoundingBox3D::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0);
        let b = BoundingBox3D::new([100.0, 100.0, 0.0], [1.0, 1.0, 1.0], 0.0);
        assert!(iou_3d_bev(&a, &b) < 1e-12);
    }

    #[test]
    fn focal_loss_perfect() {
        // Perfect prediction: logits → 1.0 for positive targets.
        let loss = focal_loss(&[10.0], &[1.0], 0.25, 2.0);
        assert!(loss < 1e-3);
    }

    #[test]
    fn smooth_l1_zero() {
        let loss = smooth_l1_loss(&[1.0, 2.0], &[1.0, 2.0], 1.0);
        assert!(loss.abs() < 1e-12);
    }

    #[test]
    fn smooth_l1_large_diff() {
        let loss = smooth_l1_loss(&[0.0], &[10.0], 1.0);
        // |10| - 0.5*1 = 9.5
        assert!((loss - 9.5).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_range() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }
}
