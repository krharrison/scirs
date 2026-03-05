//! # Object Detection Utilities
//!
//! Comprehensive object detection infrastructure providing bounding box operations,
//! non-maximum suppression algorithms, anchor generation, detection metrics,
//! and image augmentation utilities for detection pipelines.
//!
//! ## Features
//!
//! - **DetectionBox**: Full bounding box representation with IoU, GIoU, DIoU, CIoU metrics
//! - **NMS**: Standard, soft, batched (per-class), and weighted NMS algorithms
//! - **Anchor Generation**: SSD, YOLO, and configurable anchor generators
//! - **Detection Metrics**: Average Precision (AP), mAP, precision-recall curves
//! - **Augmentation**: Bounding box-aware horizontal flip, crop, scale, translate, clip
//!
//! ## Example
//!
//! ```rust
//! use scirs2_vision::detection::{DetectionBox, nms, compute_ap};
//!
//! // Create bounding boxes
//! let b1 = DetectionBox::new(10.0, 10.0, 50.0, 50.0)
//!     .with_confidence(0.9)
//!     .with_class(1, Some("cat".to_string()));
//! let b2 = DetectionBox::new(12.0, 12.0, 52.0, 52.0)
//!     .with_confidence(0.7)
//!     .with_class(1, Some("cat".to_string()));
//! let b3 = DetectionBox::new(200.0, 200.0, 260.0, 260.0)
//!     .with_confidence(0.85)
//!     .with_class(2, Some("dog".to_string()));
//!
//! // Non-maximum suppression
//! let kept = nms(&[b1.clone(), b2, b3], 0.5);
//! assert_eq!(kept.len(), 2); // b1 and b3 survive
//!
//! // IoU computation
//! let iou = b1.iou(&DetectionBox::new(10.0, 10.0, 50.0, 50.0));
//! assert!((iou - 1.0).abs() < 1e-10);
//! ```

use crate::error::{Result, VisionError};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// DetectionBox
// ---------------------------------------------------------------------------

/// A bounding box for object detection, stored as (x1, y1, x2, y2) corner format.
///
/// All coordinates use `f64` precision. The box additionally carries optional
/// confidence, class id, and class name fields used throughout detection pipelines.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectionBox {
    /// Top-left x coordinate
    pub x1: f64,
    /// Top-left y coordinate
    pub y1: f64,
    /// Bottom-right x coordinate
    pub x2: f64,
    /// Bottom-right y coordinate
    pub y2: f64,
    /// Detection confidence score in [0, 1]
    pub confidence: f64,
    /// Class identifier (0-indexed)
    pub class_id: usize,
    /// Optional human-readable class name
    pub class_name: Option<String>,
}

impl DetectionBox {
    /// Create a new detection box from corner coordinates.
    ///
    /// Coordinates are normalised so that `x1 <= x2` and `y1 <= y2`.
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self {
            x1: x1.min(x2),
            y1: y1.min(y2),
            x2: x1.max(x2),
            y2: y1.max(y2),
            confidence: 0.0,
            class_id: 0,
            class_name: None,
        }
    }

    /// Create a detection box from centre coordinates and dimensions.
    ///
    /// # Arguments
    /// * `cx` - Centre x
    /// * `cy` - Centre y
    /// * `w`  - Width  (must be non-negative)
    /// * `h`  - Height (must be non-negative)
    pub fn from_center(cx: f64, cy: f64, w: f64, h: f64) -> Self {
        let half_w = w.abs() / 2.0;
        let half_h = h.abs() / 2.0;
        Self::new(cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    }

    /// Builder: set confidence score.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Builder: set class id and optional name.
    #[must_use]
    pub fn with_class(mut self, class_id: usize, class_name: Option<String>) -> Self {
        self.class_id = class_id;
        self.class_name = class_name;
        self
    }

    /// Box area.
    #[inline]
    pub fn area(&self) -> f64 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    /// Centre of the box as `(cx, cy)`.
    #[inline]
    pub fn center(&self) -> (f64, f64) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    /// Width of the box.
    #[inline]
    pub fn width(&self) -> f64 {
        (self.x2 - self.x1).max(0.0)
    }

    /// Height of the box.
    #[inline]
    pub fn height(&self) -> f64 {
        (self.y2 - self.y1).max(0.0)
    }

    /// Aspect ratio (width / height). Returns 0.0 if height is zero.
    #[inline]
    pub fn aspect_ratio(&self) -> f64 {
        let h = self.height();
        if h == 0.0 {
            0.0
        } else {
            self.width() / h
        }
    }

    // -- overlap metrics ----------------------------------------------------

    /// Intersection area with another box.
    pub fn intersection_area(&self, other: &DetectionBox) -> f64 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0)
    }

    /// Union area with another box.
    pub fn union_area(&self, other: &DetectionBox) -> f64 {
        self.area() + other.area() - self.intersection_area(other)
    }

    /// Intersection over Union (IoU).
    ///
    /// Returns 0.0 when the union area is zero.
    pub fn iou(&self, other: &DetectionBox) -> f64 {
        let union = self.union_area(other);
        if union <= 0.0 {
            return 0.0;
        }
        self.intersection_area(other) / union
    }

    /// Generalized Intersection over Union (GIoU).
    ///
    /// GIoU = IoU - |C \ (A union B)| / |C|
    /// where C is the smallest enclosing box.
    /// Range: [-1, 1]
    pub fn giou(&self, other: &DetectionBox) -> f64 {
        let inter = self.intersection_area(other);
        let union = self.union_area(other);
        if union <= 0.0 {
            return 0.0;
        }

        // Enclosing box area
        let enc_x1 = self.x1.min(other.x1);
        let enc_y1 = self.y1.min(other.y1);
        let enc_x2 = self.x2.max(other.x2);
        let enc_y2 = self.y2.max(other.y2);
        let enc_area = (enc_x2 - enc_x1).max(0.0) * (enc_y2 - enc_y1).max(0.0);

        let iou_val = inter / union;
        if enc_area <= 0.0 {
            return iou_val;
        }
        iou_val - (enc_area - union) / enc_area
    }

    /// Distance-IoU (DIoU).
    ///
    /// DIoU = IoU - d^2 / c^2
    /// where d is the Euclidean distance between centres and c is the diagonal
    /// length of the smallest enclosing box.
    pub fn diou(&self, other: &DetectionBox) -> f64 {
        let union = self.union_area(other);
        if union <= 0.0 {
            return 0.0;
        }
        let iou_val = self.intersection_area(other) / union;

        let (cx1, cy1) = self.center();
        let (cx2, cy2) = other.center();
        let d_sq = (cx1 - cx2).powi(2) + (cy1 - cy2).powi(2);

        let enc_x1 = self.x1.min(other.x1);
        let enc_y1 = self.y1.min(other.y1);
        let enc_x2 = self.x2.max(other.x2);
        let enc_y2 = self.y2.max(other.y2);
        let c_sq = (enc_x2 - enc_x1).powi(2) + (enc_y2 - enc_y1).powi(2);

        if c_sq <= 0.0 {
            return iou_val;
        }
        iou_val - d_sq / c_sq
    }

    /// Complete-IoU (CIoU).
    ///
    /// CIoU = IoU - d^2/c^2 - alpha * v
    /// where v measures aspect-ratio consistency and alpha is a trade-off parameter.
    pub fn ciou(&self, other: &DetectionBox) -> f64 {
        let union = self.union_area(other);
        if union <= 0.0 {
            return 0.0;
        }
        let iou_val = self.intersection_area(other) / union;

        let (cx1, cy1) = self.center();
        let (cx2, cy2) = other.center();
        let d_sq = (cx1 - cx2).powi(2) + (cy1 - cy2).powi(2);

        let enc_x1 = self.x1.min(other.x1);
        let enc_y1 = self.y1.min(other.y1);
        let enc_x2 = self.x2.max(other.x2);
        let enc_y2 = self.y2.max(other.y2);
        let c_sq = (enc_x2 - enc_x1).powi(2) + (enc_y2 - enc_y1).powi(2);

        // Aspect-ratio consistency term
        let pi = std::f64::consts::PI;
        let v = {
            let atan_self = (self.width() / self.height().max(1e-12)).atan();
            let atan_other = (other.width() / other.height().max(1e-12)).atan();
            (4.0 / (pi * pi)) * (atan_self - atan_other).powi(2)
        };

        let alpha = if (1.0 - iou_val + v).abs() < 1e-12 {
            0.0
        } else {
            v / (1.0 - iou_val + v)
        };

        let distance_term = if c_sq > 0.0 { d_sq / c_sq } else { 0.0 };
        iou_val - distance_term - alpha * v
    }

    /// Check whether a point `(px, py)` lies inside this box (inclusive).
    pub fn contains_point(&self, px: f64, py: f64) -> bool {
        px >= self.x1 && px <= self.x2 && py >= self.y1 && py <= self.y2
    }
}

// ---------------------------------------------------------------------------
// Non-Maximum Suppression
// ---------------------------------------------------------------------------

/// Standard greedy Non-Maximum Suppression (NMS).
///
/// Returns the indices (into the input slice) of the boxes that survive
/// suppression. Boxes are processed in descending order of confidence.
///
/// # Arguments
/// * `boxes`         - Detection boxes.
/// * `iou_threshold` - Boxes with IoU > threshold relative to a kept box are removed.
pub fn nms(boxes: &[DetectionBox], iou_threshold: f64) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // Sort indices by descending confidence
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| {
        boxes[b]
            .confidence
            .partial_cmp(&boxes[a].confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for &idx in &indices {
        if suppressed[idx] {
            continue;
        }
        keep.push(idx);
        for &other in &indices {
            if other != idx && !suppressed[other] && boxes[idx].iou(&boxes[other]) > iou_threshold {
                suppressed[other] = true;
            }
        }
    }
    keep
}

/// Soft-NMS with Gaussian score decay.
///
/// Instead of hard suppression, overlapping boxes have their confidence
/// reduced by `exp(-iou^2 / sigma)`. Boxes whose confidence drops below
/// `score_threshold` are discarded.
///
/// The function modifies the confidence values in place and returns the
/// indices of surviving boxes.
pub fn soft_nms(boxes: &mut [DetectionBox], sigma: f64, score_threshold: f64) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }
    let n = boxes.len();
    let mut active: Vec<bool> = vec![true; n];
    let mut keep = Vec::new();

    for _ in 0..n {
        // Find the active box with the highest confidence
        let mut best_idx: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        for (i, &is_active) in active.iter().enumerate() {
            if is_active && boxes[i].confidence > best_score {
                best_score = boxes[i].confidence;
                best_idx = Some(i);
            }
        }
        let best = match best_idx {
            Some(i) => i,
            None => break,
        };

        if boxes[best].confidence < score_threshold {
            break;
        }

        keep.push(best);
        active[best] = false;

        // Decay overlapping boxes
        for j in 0..n {
            if active[j] {
                let iou_val = boxes[best].iou(&boxes[j]);
                if sigma > 0.0 {
                    boxes[j].confidence *= (-iou_val * iou_val / sigma).exp();
                }
                if boxes[j].confidence < score_threshold {
                    active[j] = false;
                }
            }
        }
    }
    keep
}

/// Batched (per-class) NMS.
///
/// Applies standard NMS independently within each class and merges the results.
pub fn batched_nms(boxes: &[DetectionBox], iou_threshold: f64) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // Group indices by class_id
    let mut class_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, b) in boxes.iter().enumerate() {
        class_map.entry(b.class_id).or_default().push(i);
    }

    let mut keep = Vec::new();
    for group_indices in class_map.values() {
        let group_boxes: Vec<DetectionBox> =
            group_indices.iter().map(|&i| boxes[i].clone()).collect();
        let class_keep = nms(&group_boxes, iou_threshold);
        for local_idx in class_keep {
            keep.push(group_indices[local_idx]);
        }
    }
    // Sort by confidence descending for deterministic output
    keep.sort_by(|&a, &b| {
        boxes[b]
            .confidence
            .partial_cmp(&boxes[a].confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    keep
}

/// Weighted NMS: merges overlapping boxes by confidence-weighted averaging.
///
/// For each cluster of overlapping boxes (IoU > threshold), produces a single
/// box whose coordinates are the confidence-weighted mean of the cluster.
pub fn weighted_nms(boxes: &[DetectionBox], iou_threshold: f64) -> Vec<DetectionBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| {
        boxes[b]
            .confidence
            .partial_cmp(&boxes[a].confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut used = vec![false; boxes.len()];
    let mut result = Vec::new();

    for &idx in &indices {
        if used[idx] {
            continue;
        }
        // Collect cluster
        let mut cluster: Vec<usize> = vec![idx];
        for &other in &indices {
            if other != idx && !used[other] && boxes[idx].iou(&boxes[other]) > iou_threshold {
                cluster.push(other);
            }
        }
        // Mark all as used
        for &c in &cluster {
            used[c] = true;
        }

        // Weighted merge
        let total_conf: f64 = cluster.iter().map(|&c| boxes[c].confidence).sum();
        if total_conf <= 0.0 {
            result.push(boxes[idx].clone());
            continue;
        }
        let mut wx1 = 0.0;
        let mut wy1 = 0.0;
        let mut wx2 = 0.0;
        let mut wy2 = 0.0;
        for &c in &cluster {
            let w = boxes[c].confidence;
            wx1 += boxes[c].x1 * w;
            wy1 += boxes[c].y1 * w;
            wx2 += boxes[c].x2 * w;
            wy2 += boxes[c].y2 * w;
        }
        let merged = DetectionBox {
            x1: wx1 / total_conf,
            y1: wy1 / total_conf,
            x2: wx2 / total_conf,
            y2: wy2 / total_conf,
            confidence: boxes[idx].confidence, // keep max confidence
            class_id: boxes[idx].class_id,
            class_name: boxes[idx].class_name.clone(),
        };
        result.push(merged);
    }
    result
}

// ---------------------------------------------------------------------------
// Anchor Generation
// ---------------------------------------------------------------------------

/// Configuration for anchor generation across feature map levels.
#[derive(Clone, Debug)]
pub struct AnchorConfig {
    /// Feature map spatial sizes at each level, e.g. `[(38,38), (19,19), (10,10)]`.
    pub feature_map_sizes: Vec<(usize, usize)>,
    /// Aspect ratios to generate, e.g. `[0.5, 1.0, 2.0]`.
    pub aspect_ratios: Vec<f64>,
    /// Scale multipliers at each level (same length as `feature_map_sizes`).
    pub scales: Vec<f64>,
    /// Original image size `(width, height)`.
    pub image_size: (usize, usize),
}

/// Generate anchors for all feature map levels according to `config`.
///
/// For each feature map cell and each (scale, aspect_ratio) combination,
/// an anchor centred on that cell is produced with the given scale and aspect ratio.
pub fn generate_anchors(config: &AnchorConfig) -> Result<Vec<DetectionBox>> {
    if config.feature_map_sizes.is_empty() {
        return Err(VisionError::InvalidParameter(
            "feature_map_sizes must not be empty".into(),
        ));
    }
    if config.aspect_ratios.is_empty() {
        return Err(VisionError::InvalidParameter(
            "aspect_ratios must not be empty".into(),
        ));
    }
    if config.scales.len() != config.feature_map_sizes.len() {
        return Err(VisionError::InvalidParameter(
            "scales length must match feature_map_sizes length".into(),
        ));
    }

    let (img_w, img_h) = config.image_size;
    let img_w = img_w as f64;
    let img_h = img_h as f64;

    let mut anchors = Vec::new();

    for (level, &(fm_w, fm_h)) in config.feature_map_sizes.iter().enumerate() {
        if fm_w == 0 || fm_h == 0 {
            continue;
        }
        let step_x = img_w / fm_w as f64;
        let step_y = img_h / fm_h as f64;
        let scale = config.scales[level];

        for row in 0..fm_h {
            for col in 0..fm_w {
                let cx = (col as f64 + 0.5) * step_x;
                let cy = (row as f64 + 0.5) * step_y;

                for &ar in &config.aspect_ratios {
                    let w = scale * ar.sqrt();
                    let h = scale / ar.sqrt();
                    anchors.push(DetectionBox::from_center(cx, cy, w, h));
                }
            }
        }
    }
    Ok(anchors)
}

/// Generate SSD-style anchors.
///
/// Uses default aspect ratios `[1.0, 2.0, 0.5]` and derives scales from the
/// ratio between the image size and each feature map size.
pub fn generate_ssd_anchors(
    image_size: (usize, usize),
    feature_maps: &[(usize, usize)],
) -> Result<Vec<DetectionBox>> {
    if feature_maps.is_empty() {
        return Err(VisionError::InvalidParameter(
            "feature_maps must not be empty".into(),
        ));
    }
    let aspect_ratios = vec![1.0, 2.0, 0.5, 3.0, 1.0 / 3.0];
    let min_scale = 0.2;
    let max_scale = 0.9;
    let num_levels = feature_maps.len();
    let scales: Vec<f64> = (0..num_levels)
        .map(|k| {
            let s = if num_levels > 1 {
                min_scale + (max_scale - min_scale) * (k as f64) / ((num_levels - 1) as f64)
            } else {
                (min_scale + max_scale) / 2.0
            };
            s * image_size.0.min(image_size.1) as f64
        })
        .collect();

    generate_anchors(&AnchorConfig {
        feature_map_sizes: feature_maps.to_vec(),
        aspect_ratios,
        scales,
        image_size,
    })
}

/// Generate YOLO-style anchors from pre-defined anchor dimensions.
///
/// Each entry in `anchor_wh` is `(width, height)` in pixel space. The anchors
/// are placed at every grid cell of the specified feature map. Returns one
/// `DetectionBox` per (cell, anchor) pair.
pub fn generate_yolo_anchors(
    image_size: (usize, usize),
    feature_map: (usize, usize),
    anchor_wh: &[(f64, f64)],
) -> Result<Vec<DetectionBox>> {
    if anchor_wh.is_empty() {
        return Err(VisionError::InvalidParameter(
            "anchor_wh must not be empty".into(),
        ));
    }
    let (fm_w, fm_h) = feature_map;
    if fm_w == 0 || fm_h == 0 {
        return Err(VisionError::InvalidParameter(
            "feature map dimensions must be > 0".into(),
        ));
    }
    let step_x = image_size.0 as f64 / fm_w as f64;
    let step_y = image_size.1 as f64 / fm_h as f64;

    let mut anchors = Vec::new();
    for row in 0..fm_h {
        for col in 0..fm_w {
            let cx = (col as f64 + 0.5) * step_x;
            let cy = (row as f64 + 0.5) * step_y;
            for &(aw, ah) in anchor_wh {
                anchors.push(DetectionBox::from_center(cx, cy, aw, ah));
            }
        }
    }
    Ok(anchors)
}

// ---------------------------------------------------------------------------
// Detection Metrics
// ---------------------------------------------------------------------------

/// Compute Average Precision (AP) for a single class using the all-points interpolation method.
///
/// Both `predictions` and `ground_truth` should belong to the same class.
/// A prediction is considered a true positive if it has IoU > `iou_threshold`
/// with a ground truth box that has not already been matched.
pub fn compute_ap(
    predictions: &[DetectionBox],
    ground_truth: &[DetectionBox],
    iou_threshold: f64,
) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }
    if predictions.is_empty() {
        return 0.0;
    }

    let (precisions, recalls) = precision_recall_curve(predictions, ground_truth, iou_threshold);
    if precisions.is_empty() {
        return 0.0;
    }

    // All-points interpolation (PASCAL VOC 2010+ / COCO style)
    ap_from_pr(&precisions, &recalls)
}

/// Compute mean Average Precision (mAP) across multiple images/classes and IoU thresholds.
///
/// `predictions[i]` and `ground_truth[i]` correspond to the same image/class.
/// The function computes AP for each (image, threshold) pair and returns
/// the mean over all.
pub fn compute_map(
    predictions: &[Vec<DetectionBox>],
    ground_truth: &[Vec<DetectionBox>],
    iou_thresholds: &[f64],
) -> f64 {
    if predictions.is_empty() || ground_truth.is_empty() || iou_thresholds.is_empty() {
        return 0.0;
    }
    let n = predictions.len().min(ground_truth.len());
    let mut total_ap = 0.0;
    let mut count = 0usize;

    for threshold in iou_thresholds {
        for i in 0..n {
            total_ap += compute_ap(&predictions[i], &ground_truth[i], *threshold);
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        total_ap / count as f64
    }
}

/// Compute the precision-recall curve for a single class.
///
/// Returns `(precisions, recalls)` sorted by decreasing confidence.
pub fn precision_recall_curve(
    predictions: &[DetectionBox],
    ground_truth: &[DetectionBox],
    iou_threshold: f64,
) -> (Vec<f64>, Vec<f64>) {
    if ground_truth.is_empty() || predictions.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Sort predictions by descending confidence
    let mut sorted_preds: Vec<(usize, f64)> = predictions
        .iter()
        .enumerate()
        .map(|(i, p)| (i, p.confidence))
        .collect();
    sorted_preds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let total_gt = ground_truth.len() as f64;
    let mut matched_gt = vec![false; ground_truth.len()];
    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut precisions = Vec::with_capacity(sorted_preds.len());
    let mut recalls = Vec::with_capacity(sorted_preds.len());

    for &(pred_idx, _conf) in &sorted_preds {
        let pred = &predictions[pred_idx];

        // Find the best matching GT box
        let mut best_iou = 0.0;
        let mut best_gt: Option<usize> = None;
        for (gt_idx, gt) in ground_truth.iter().enumerate() {
            if matched_gt[gt_idx] {
                continue;
            }
            let iou_val = pred.iou(gt);
            if iou_val > best_iou {
                best_iou = iou_val;
                best_gt = Some(gt_idx);
            }
        }

        if best_iou >= iou_threshold {
            if let Some(gt_idx) = best_gt {
                tp += 1.0;
                matched_gt[gt_idx] = true;
            } else {
                fp += 1.0;
            }
        } else {
            fp += 1.0;
        }

        let precision = if (tp + fp) > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = tp / total_gt;
        precisions.push(precision);
        recalls.push(recall);
    }

    (precisions, recalls)
}

/// Compute AP from precision-recall arrays using the all-points interpolation method.
fn ap_from_pr(precisions: &[f64], recalls: &[f64]) -> f64 {
    if precisions.is_empty() || recalls.is_empty() {
        return 0.0;
    }

    // Prepend (recall=0, precision=1) and append (recall=1, precision=0) sentinel values
    let n = precisions.len();
    let mut mrec = Vec::with_capacity(n + 2);
    let mut mprec = Vec::with_capacity(n + 2);
    mrec.push(0.0);
    mprec.push(0.0);
    for i in 0..n {
        mrec.push(recalls[i]);
        mprec.push(precisions[i]);
    }
    mrec.push(1.0);
    mprec.push(0.0);

    // Make precision monotonically decreasing (right to left envelope)
    for i in (0..mprec.len() - 1).rev() {
        if mprec[i + 1] > mprec[i] {
            mprec[i] = mprec[i + 1];
        }
    }

    // Sum up the rectangular areas where recall changes
    let mut ap = 0.0;
    for i in 1..mrec.len() {
        if (mrec[i] - mrec[i - 1]).abs() > 1e-15 {
            ap += (mrec[i] - mrec[i - 1]) * mprec[i];
        }
    }
    ap
}

/// Compute a simple confusion matrix for detection results.
///
/// Returns a 2D map: `class_id -> class_id -> count`, where the first key
/// is the ground truth class and the second key is the predicted class.
/// Unmatched ground truths appear under predicted class `usize::MAX` (missed).
/// False positive predictions appear under ground truth class `usize::MAX`.
pub fn confusion_matrix(
    predictions: &[DetectionBox],
    ground_truth: &[DetectionBox],
    iou_threshold: f64,
) -> HashMap<usize, HashMap<usize, usize>> {
    let mut matrix: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    let mut matched_gt = vec![false; ground_truth.len()];

    // Sort predictions by descending confidence
    let mut sorted_indices: Vec<usize> = (0..predictions.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        predictions[b]
            .confidence
            .partial_cmp(&predictions[a].confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for &pred_idx in &sorted_indices {
        let pred = &predictions[pred_idx];
        let mut best_iou = 0.0_f64;
        let mut best_gt: Option<usize> = None;

        for (gt_idx, gt) in ground_truth.iter().enumerate() {
            if matched_gt[gt_idx] {
                continue;
            }
            let iou_val = pred.iou(gt);
            if iou_val > best_iou {
                best_iou = iou_val;
                best_gt = Some(gt_idx);
            }
        }

        if best_iou >= iou_threshold {
            if let Some(gt_idx) = best_gt {
                matched_gt[gt_idx] = true;
                let gt_class = ground_truth[gt_idx].class_id;
                let pred_class = pred.class_id;
                *matrix
                    .entry(gt_class)
                    .or_default()
                    .entry(pred_class)
                    .or_insert(0) += 1;
            }
        } else {
            // False positive
            *matrix
                .entry(usize::MAX)
                .or_default()
                .entry(pred.class_id)
                .or_insert(0) += 1;
        }
    }

    // Record unmatched ground truths (missed detections)
    for (gt_idx, gt) in ground_truth.iter().enumerate() {
        if !matched_gt[gt_idx] {
            *matrix
                .entry(gt.class_id)
                .or_default()
                .entry(usize::MAX)
                .or_insert(0) += 1;
        }
    }

    matrix
}

// ---------------------------------------------------------------------------
// Augmentation helpers
// ---------------------------------------------------------------------------

/// Flip bounding boxes horizontally within an image of width `image_width`.
///
/// Uses a simple xorshift PRNG seeded with `seed` and flips with `probability` in `[0,1]`.
/// Modified in place.
pub fn random_horizontal_flip(
    boxes: &mut [DetectionBox],
    image_width: f64,
    probability: f64,
    seed: u64,
) {
    // Simple xorshift64 PRNG
    let mut state = if seed == 0 { 0x5DEECE66D } else { seed };
    let should_flip = {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let rand_val = (state as f64) / (u64::MAX as f64);
        rand_val.abs() < probability
    };

    if should_flip {
        for b in boxes.iter_mut() {
            let new_x1 = image_width - b.x2;
            let new_x2 = image_width - b.x1;
            b.x1 = new_x1;
            b.x2 = new_x2;
        }
    }
}

/// Return boxes that are fully or partially inside `crop_box`, with coordinates
/// adjusted relative to the crop region.
///
/// Boxes that fall entirely outside the crop region are discarded.
/// Partially visible boxes are clipped to the crop boundary.
pub fn random_crop_with_boxes(
    boxes: &[DetectionBox],
    crop_box: &DetectionBox,
) -> Vec<DetectionBox> {
    let mut result = Vec::new();
    for b in boxes {
        // Clip to crop region
        let clipped_x1 = b.x1.max(crop_box.x1);
        let clipped_y1 = b.y1.max(crop_box.y1);
        let clipped_x2 = b.x2.min(crop_box.x2);
        let clipped_y2 = b.y2.min(crop_box.y2);

        // Discard if no intersection
        if clipped_x1 >= clipped_x2 || clipped_y1 >= clipped_y2 {
            continue;
        }

        // Translate to crop-relative coordinates
        result.push(DetectionBox {
            x1: clipped_x1 - crop_box.x1,
            y1: clipped_y1 - crop_box.y1,
            x2: clipped_x2 - crop_box.x1,
            y2: clipped_y2 - crop_box.y1,
            confidence: b.confidence,
            class_id: b.class_id,
            class_name: b.class_name.clone(),
        });
    }
    result
}

/// Scale bounding box coordinates by `(sx, sy)`.
pub fn scale_boxes(boxes: &mut [DetectionBox], sx: f64, sy: f64) {
    for b in boxes.iter_mut() {
        b.x1 *= sx;
        b.y1 *= sy;
        b.x2 *= sx;
        b.y2 *= sy;
    }
}

/// Translate bounding box coordinates by `(tx, ty)`.
pub fn translate_boxes(boxes: &mut [DetectionBox], tx: f64, ty: f64) {
    for b in boxes.iter_mut() {
        b.x1 += tx;
        b.y1 += ty;
        b.x2 += tx;
        b.y2 += ty;
    }
}

/// Clip bounding boxes to image boundaries `[0, image_width] x [0, image_height]`.
///
/// Boxes that end up with zero or negative area after clipping are removed.
pub fn clip_boxes(boxes: &mut Vec<DetectionBox>, image_width: f64, image_height: f64) {
    for b in boxes.iter_mut() {
        b.x1 = b.x1.max(0.0).min(image_width);
        b.y1 = b.y1.max(0.0).min(image_height);
        b.x2 = b.x2.max(0.0).min(image_width);
        b.y2 = b.y2.max(0.0).min(image_height);
    }
    boxes.retain(|b| b.width() > 0.0 && b.height() > 0.0);
}

/// Filter detection boxes by confidence threshold, returning only those above.
pub fn filter_by_confidence(boxes: &[DetectionBox], threshold: f64) -> Vec<DetectionBox> {
    boxes
        .iter()
        .filter(|b| b.confidence >= threshold)
        .cloned()
        .collect()
}

/// Convert a slice of `DetectionBox` to `(x1, y1, x2, y2, confidence, class_id)` tuples.
pub fn boxes_to_tuples(boxes: &[DetectionBox]) -> Vec<(f64, f64, f64, f64, f64, usize)> {
    boxes
        .iter()
        .map(|b| (b.x1, b.y1, b.x2, b.y2, b.confidence, b.class_id))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    // -----------------------------------------------------------------------
    // DetectionBox basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_normalises_coordinates() {
        let b = DetectionBox::new(50.0, 40.0, 10.0, 20.0);
        assert!(b.x1 <= b.x2);
        assert!(b.y1 <= b.y2);
        assert_eq!(b.x1, 10.0);
        assert_eq!(b.y1, 20.0);
        assert_eq!(b.x2, 50.0);
        assert_eq!(b.y2, 40.0);
    }

    #[test]
    fn test_from_center() {
        let b = DetectionBox::from_center(100.0, 100.0, 40.0, 20.0);
        assert!(approx_eq(b.x1, 80.0, 1e-10));
        assert!(approx_eq(b.y1, 90.0, 1e-10));
        assert!(approx_eq(b.x2, 120.0, 1e-10));
        assert!(approx_eq(b.y2, 110.0, 1e-10));
    }

    #[test]
    fn test_area() {
        let b = DetectionBox::new(0.0, 0.0, 10.0, 20.0);
        assert!(approx_eq(b.area(), 200.0, 1e-10));
    }

    #[test]
    fn test_center() {
        let b = DetectionBox::new(10.0, 20.0, 30.0, 40.0);
        let (cx, cy) = b.center();
        assert!(approx_eq(cx, 20.0, 1e-10));
        assert!(approx_eq(cy, 30.0, 1e-10));
    }

    #[test]
    fn test_width_height() {
        let b = DetectionBox::new(5.0, 10.0, 25.0, 40.0);
        assert!(approx_eq(b.width(), 20.0, 1e-10));
        assert!(approx_eq(b.height(), 30.0, 1e-10));
    }

    #[test]
    fn test_aspect_ratio() {
        let b = DetectionBox::new(0.0, 0.0, 20.0, 10.0);
        assert!(approx_eq(b.aspect_ratio(), 2.0, 1e-10));
    }

    #[test]
    fn test_contains_point() {
        let b = DetectionBox::new(10.0, 10.0, 50.0, 50.0);
        assert!(b.contains_point(30.0, 30.0));
        assert!(b.contains_point(10.0, 10.0));
        assert!(!b.contains_point(5.0, 5.0));
        assert!(!b.contains_point(55.0, 55.0));
    }

    #[test]
    fn test_builder_methods() {
        let b = DetectionBox::new(0.0, 0.0, 10.0, 10.0)
            .with_confidence(0.95)
            .with_class(3, Some("person".to_string()));
        assert!(approx_eq(b.confidence, 0.95, 1e-10));
        assert_eq!(b.class_id, 3);
        assert_eq!(b.class_name.as_deref(), Some("person"));
    }

    // -----------------------------------------------------------------------
    // IoU variants
    // -----------------------------------------------------------------------

    #[test]
    fn test_iou_identical() {
        let b = DetectionBox::new(10.0, 10.0, 50.0, 50.0);
        assert!(approx_eq(b.iou(&b), 1.0, 1e-10));
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = DetectionBox::new(0.0, 0.0, 10.0, 10.0);
        let b = DetectionBox::new(20.0, 20.0, 30.0, 30.0);
        assert!(approx_eq(a.iou(&b), 0.0, 1e-10));
    }

    #[test]
    fn test_iou_partial_overlap() {
        let a = DetectionBox::new(0.0, 0.0, 10.0, 10.0);
        let b = DetectionBox::new(5.0, 5.0, 15.0, 15.0);
        // Intersection: 5x5 = 25. Union: 100 + 100 - 25 = 175. IoU = 25/175
        let expected = 25.0 / 175.0;
        assert!(approx_eq(a.iou(&b), expected, 1e-10));
    }

    #[test]
    fn test_giou_identical() {
        let b = DetectionBox::new(10.0, 10.0, 50.0, 50.0);
        assert!(approx_eq(b.giou(&b), 1.0, 1e-10));
    }

    #[test]
    fn test_giou_no_overlap() {
        let a = DetectionBox::new(0.0, 0.0, 10.0, 10.0);
        let b = DetectionBox::new(20.0, 20.0, 30.0, 30.0);
        // GIoU should be negative when boxes are far apart
        let val = a.giou(&b);
        assert!(val < 0.0);
    }

    #[test]
    fn test_giou_range() {
        // GIoU should be in [-1, 1]
        let a = DetectionBox::new(0.0, 0.0, 1.0, 1.0);
        let b = DetectionBox::new(100.0, 100.0, 101.0, 101.0);
        let val = a.giou(&b);
        assert!(val >= -1.0 && val <= 1.0);
    }

    #[test]
    fn test_diou_identical() {
        let b = DetectionBox::new(10.0, 10.0, 50.0, 50.0);
        assert!(approx_eq(b.diou(&b), 1.0, 1e-10));
    }

    #[test]
    fn test_ciou_identical() {
        let b = DetectionBox::new(10.0, 10.0, 50.0, 50.0);
        assert!(approx_eq(b.ciou(&b), 1.0, 1e-10));
    }

    #[test]
    fn test_ciou_different_aspect_ratios() {
        let a = DetectionBox::new(0.0, 0.0, 100.0, 50.0); // 2:1
        let b = DetectionBox::new(0.0, 0.0, 50.0, 100.0); // 1:2
        let ciou_val = a.ciou(&b);
        // CIoU penalises aspect-ratio difference, so it should be lower than IoU
        let iou_val = a.iou(&b);
        assert!(ciou_val <= iou_val);
    }

    // -----------------------------------------------------------------------
    // NMS
    // -----------------------------------------------------------------------

    #[test]
    fn test_nms_basic() {
        let boxes = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.9),
            DetectionBox::new(1.0, 1.0, 11.0, 11.0).with_confidence(0.7),
            DetectionBox::new(200.0, 200.0, 210.0, 210.0).with_confidence(0.8),
        ];
        let kept = nms(&boxes, 0.5);
        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&0)); // highest confidence overlapping
        assert!(kept.contains(&2)); // non-overlapping
    }

    #[test]
    fn test_nms_empty() {
        let kept = nms(&[], 0.5);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_nms_no_suppression() {
        let boxes = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.9),
            DetectionBox::new(100.0, 100.0, 110.0, 110.0).with_confidence(0.8),
        ];
        let kept = nms(&boxes, 0.5);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_soft_nms_reduces_scores() {
        let mut boxes = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.9),
            DetectionBox::new(1.0, 1.0, 11.0, 11.0).with_confidence(0.8),
            DetectionBox::new(200.0, 200.0, 210.0, 210.0).with_confidence(0.85),
        ];
        let original_conf = boxes[1].confidence;
        let kept = soft_nms(&mut boxes, 0.5, 0.01);
        // Overlapping box should have reduced confidence
        assert!(boxes[1].confidence < original_conf);
        assert!(!kept.is_empty());
    }

    #[test]
    fn test_batched_nms_separates_classes() {
        let boxes = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0)
                .with_confidence(0.9)
                .with_class(0, None),
            DetectionBox::new(1.0, 1.0, 11.0, 11.0)
                .with_confidence(0.8)
                .with_class(1, None), // different class - should not suppress
        ];
        let kept = batched_nms(&boxes, 0.5);
        assert_eq!(kept.len(), 2); // both kept since different classes
    }

    #[test]
    fn test_batched_nms_suppresses_same_class() {
        let boxes = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0)
                .with_confidence(0.9)
                .with_class(0, None),
            DetectionBox::new(1.0, 1.0, 11.0, 11.0)
                .with_confidence(0.7)
                .with_class(0, None),
        ];
        let kept = batched_nms(&boxes, 0.5);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn test_weighted_nms_merges() {
        let boxes = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.9),
            DetectionBox::new(1.0, 1.0, 11.0, 11.0).with_confidence(0.8),
        ];
        let merged = weighted_nms(&boxes, 0.3);
        assert_eq!(merged.len(), 1);
        // Merged box should be a weighted average
        assert!(merged[0].x1 > 0.0 && merged[0].x1 < 1.0);
    }

    // -----------------------------------------------------------------------
    // Anchor Generation
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_anchors_basic() {
        let config = AnchorConfig {
            feature_map_sizes: vec![(4, 4)],
            aspect_ratios: vec![1.0, 2.0],
            scales: vec![32.0],
            image_size: (128, 128),
        };
        let anchors = generate_anchors(&config);
        assert!(anchors.is_ok());
        let anchors = anchors.expect("should succeed");
        // 4*4 cells * 2 aspect ratios = 32 anchors
        assert_eq!(anchors.len(), 32);
    }

    #[test]
    fn test_generate_anchors_empty_feature_maps() {
        let config = AnchorConfig {
            feature_map_sizes: vec![],
            aspect_ratios: vec![1.0],
            scales: vec![],
            image_size: (100, 100),
        };
        let result = generate_anchors(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_ssd_anchors() {
        let anchors = generate_ssd_anchors((300, 300), &[(38, 38), (19, 19), (10, 10)]);
        assert!(anchors.is_ok());
        let anchors = anchors.expect("should succeed");
        // 38*38*5 + 19*19*5 + 10*10*5 = 7220 + 1805 + 500 = 9525
        assert_eq!(anchors.len(), 9525);
    }

    #[test]
    fn test_generate_yolo_anchors() {
        let anchor_wh = vec![(10.0, 13.0), (16.0, 30.0), (33.0, 23.0)];
        let anchors = generate_yolo_anchors((416, 416), (13, 13), &anchor_wh);
        assert!(anchors.is_ok());
        let anchors = anchors.expect("should succeed");
        assert_eq!(anchors.len(), 13 * 13 * 3);
    }

    #[test]
    fn test_generate_yolo_anchors_empty() {
        let result = generate_yolo_anchors((416, 416), (13, 13), &[]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Detection Metrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_ap_perfect() {
        let preds = vec![DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.9)];
        let gt = vec![DetectionBox::new(0.0, 0.0, 10.0, 10.0)];
        let ap = compute_ap(&preds, &gt, 0.5);
        assert!(approx_eq(ap, 1.0, 1e-10));
    }

    #[test]
    fn test_compute_ap_no_predictions() {
        let gt = vec![DetectionBox::new(0.0, 0.0, 10.0, 10.0)];
        let ap = compute_ap(&[], &gt, 0.5);
        assert!(approx_eq(ap, 0.0, 1e-10));
    }

    #[test]
    fn test_compute_ap_no_ground_truth() {
        let preds = vec![DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.9)];
        let ap = compute_ap(&preds, &[], 0.5);
        assert!(approx_eq(ap, 0.0, 1e-10));
    }

    #[test]
    fn test_compute_map() {
        let preds = vec![vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.9)
        ]];
        let gt = vec![vec![DetectionBox::new(0.0, 0.0, 10.0, 10.0)]];
        let thresholds = vec![0.5, 0.75];
        let map_val = compute_map(&preds, &gt, &thresholds);
        assert!(approx_eq(map_val, 1.0, 1e-10));
    }

    #[test]
    fn test_precision_recall_curve_basic() {
        let preds = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.9),
            DetectionBox::new(100.0, 100.0, 110.0, 110.0).with_confidence(0.5),
        ];
        let gt = vec![DetectionBox::new(0.0, 0.0, 10.0, 10.0)];
        let (prec, rec) = precision_recall_curve(&preds, &gt, 0.5);
        assert_eq!(prec.len(), 2);
        assert_eq!(rec.len(), 2);
        // First prediction matches: precision=1.0, recall=1.0
        assert!(approx_eq(prec[0], 1.0, 1e-10));
        assert!(approx_eq(rec[0], 1.0, 1e-10));
    }

    #[test]
    fn test_confusion_matrix_basic() {
        let preds = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0)
                .with_confidence(0.9)
                .with_class(0, None),
            DetectionBox::new(200.0, 200.0, 210.0, 210.0)
                .with_confidence(0.8)
                .with_class(1, None),
        ];
        let gt = vec![DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_class(0, None)];
        let cm = confusion_matrix(&preds, &gt, 0.5);
        // GT class 0 matched by pred class 0
        assert_eq!(*cm.get(&0).and_then(|m| m.get(&0)).unwrap_or(&0), 1);
        // Pred class 1 is a false positive (gt=usize::MAX)
        assert_eq!(
            *cm.get(&usize::MAX).and_then(|m| m.get(&1)).unwrap_or(&0),
            1
        );
    }

    // -----------------------------------------------------------------------
    // Augmentation
    // -----------------------------------------------------------------------

    #[test]
    fn test_scale_boxes() {
        let mut boxes = vec![DetectionBox::new(10.0, 20.0, 30.0, 40.0)];
        scale_boxes(&mut boxes, 2.0, 0.5);
        assert!(approx_eq(boxes[0].x1, 20.0, 1e-10));
        assert!(approx_eq(boxes[0].y1, 10.0, 1e-10));
        assert!(approx_eq(boxes[0].x2, 60.0, 1e-10));
        assert!(approx_eq(boxes[0].y2, 20.0, 1e-10));
    }

    #[test]
    fn test_translate_boxes() {
        let mut boxes = vec![DetectionBox::new(10.0, 20.0, 30.0, 40.0)];
        translate_boxes(&mut boxes, 5.0, -5.0);
        assert!(approx_eq(boxes[0].x1, 15.0, 1e-10));
        assert!(approx_eq(boxes[0].y1, 15.0, 1e-10));
        assert!(approx_eq(boxes[0].x2, 35.0, 1e-10));
        assert!(approx_eq(boxes[0].y2, 35.0, 1e-10));
    }

    #[test]
    fn test_clip_boxes() {
        let mut boxes = vec![
            DetectionBox::new(-5.0, -5.0, 50.0, 50.0),
            DetectionBox::new(-10.0, -10.0, -1.0, -1.0), // fully outside
        ];
        clip_boxes(&mut boxes, 100.0, 100.0);
        assert_eq!(boxes.len(), 1); // second box removed
        assert!(approx_eq(boxes[0].x1, 0.0, 1e-10));
        assert!(approx_eq(boxes[0].y1, 0.0, 1e-10));
    }

    #[test]
    fn test_random_crop_with_boxes() {
        let boxes = vec![
            DetectionBox::new(10.0, 10.0, 50.0, 50.0).with_confidence(0.9),
            DetectionBox::new(200.0, 200.0, 250.0, 250.0).with_confidence(0.8), // outside crop
        ];
        let crop = DetectionBox::new(0.0, 0.0, 100.0, 100.0);
        let result = random_crop_with_boxes(&boxes, &crop);
        assert_eq!(result.len(), 1);
        assert!(approx_eq(result[0].x1, 10.0, 1e-10));
        assert!(approx_eq(result[0].confidence, 0.9, 1e-10));
    }

    #[test]
    fn test_random_horizontal_flip_deterministic() {
        // With probability 1.0, flip should always happen
        let mut boxes = vec![DetectionBox::new(10.0, 0.0, 30.0, 20.0)];
        random_horizontal_flip(&mut boxes, 100.0, 1.0, 42);
        // After flip: new_x1 = 100 - 30 = 70, new_x2 = 100 - 10 = 90
        assert!(approx_eq(boxes[0].x1, 70.0, 1e-10));
        assert!(approx_eq(boxes[0].x2, 90.0, 1e-10));
    }

    #[test]
    fn test_filter_by_confidence() {
        let boxes = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.9),
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.3),
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.6),
        ];
        let filtered = filter_by_confidence(&boxes, 0.5);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_boxes_to_tuples() {
        let boxes = vec![DetectionBox::new(1.0, 2.0, 3.0, 4.0)
            .with_confidence(0.5)
            .with_class(7, None)];
        let tuples = boxes_to_tuples(&boxes);
        assert_eq!(tuples.len(), 1);
        let (x1, y1, x2, y2, c, cls) = tuples[0];
        assert!(approx_eq(x1, 1.0, 1e-10));
        assert!(approx_eq(y1, 2.0, 1e-10));
        assert!(approx_eq(x2, 3.0, 1e-10));
        assert!(approx_eq(y2, 4.0, 1e-10));
        assert!(approx_eq(c, 0.5, 1e-10));
        assert_eq!(cls, 7);
    }

    #[test]
    fn test_intersection_area() {
        let a = DetectionBox::new(0.0, 0.0, 10.0, 10.0);
        let b = DetectionBox::new(5.0, 5.0, 15.0, 15.0);
        assert!(approx_eq(a.intersection_area(&b), 25.0, 1e-10));
    }

    #[test]
    fn test_union_area() {
        let a = DetectionBox::new(0.0, 0.0, 10.0, 10.0);
        let b = DetectionBox::new(5.0, 5.0, 15.0, 15.0);
        assert!(approx_eq(a.union_area(&b), 175.0, 1e-10));
    }

    #[test]
    fn test_zero_area_box() {
        let b = DetectionBox::new(5.0, 5.0, 5.0, 5.0);
        assert!(approx_eq(b.area(), 0.0, 1e-10));
        assert!(approx_eq(b.width(), 0.0, 1e-10));
        assert!(approx_eq(b.height(), 0.0, 1e-10));
    }

    #[test]
    fn test_iou_symmetry() {
        let a = DetectionBox::new(0.0, 0.0, 10.0, 10.0);
        let b = DetectionBox::new(5.0, 0.0, 15.0, 10.0);
        assert!(approx_eq(a.iou(&b), b.iou(&a), 1e-12));
    }

    #[test]
    fn test_nms_all_identical() {
        let boxes = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.9),
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.8),
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.7),
        ];
        let kept = nms(&boxes, 0.5);
        // Only the highest-confidence box survives
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0], 0);
    }

    #[test]
    fn test_compute_ap_mixed() {
        // Two GT boxes, three predictions: one TP, one FP, one TP
        let preds = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0).with_confidence(0.9), // TP
            DetectionBox::new(500.0, 500.0, 510.0, 510.0).with_confidence(0.8), // FP
            DetectionBox::new(100.0, 100.0, 110.0, 110.0).with_confidence(0.7), // TP
        ];
        let gt = vec![
            DetectionBox::new(0.0, 0.0, 10.0, 10.0),
            DetectionBox::new(100.0, 100.0, 110.0, 110.0),
        ];
        let ap = compute_ap(&preds, &gt, 0.5);
        // AP should be reasonably high (2 out of 3 correct, ordered well)
        assert!(ap > 0.5);
    }

    #[test]
    fn test_anchor_centres_are_within_image() {
        let config = AnchorConfig {
            feature_map_sizes: vec![(8, 8)],
            aspect_ratios: vec![1.0],
            scales: vec![16.0],
            image_size: (256, 256),
        };
        let anchors = generate_anchors(&config).expect("should succeed");
        for a in &anchors {
            let (cx, cy) = a.center();
            assert!(cx > 0.0 && cx < 256.0, "cx={cx} out of range");
            assert!(cy > 0.0 && cy < 256.0, "cy={cy} out of range");
        }
    }
}
