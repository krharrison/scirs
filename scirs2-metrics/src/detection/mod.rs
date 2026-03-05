//! Object detection and segmentation evaluation metrics.
//!
//! Provides standard metrics for evaluating object detection models
//! (IoU, GIoU, DIoU, mAP, COCO AP) and segmentation models
//! (pixel accuracy, mIoU, Dice coefficient).
//!
//! Bounding boxes are represented as `(x1, y1, x2, y2)` tuples where
//! `(x1, y1)` is the top-left corner and `(x2, y2)` is the bottom-right corner.

use crate::error::{MetricsError, Result};

/// Intersection over Union for two axis-aligned bounding boxes.
///
/// # Arguments
/// * `box1` — `(x1, y1, x2, y2)` bounding box
/// * `box2` — `(x1, y1, x2, y2)` bounding box
///
/// # Returns
/// IoU in [0, 1].
pub fn iou(
    box1: (f64, f64, f64, f64),
    box2: (f64, f64, f64, f64),
) -> Result<f64> {
    let (x1a, y1a, x2a, y2a) = box1;
    let (x1b, y1b, x2b, y2b) = box2;

    validate_box(box1)?;
    validate_box(box2)?;

    let inter_x1 = x1a.max(x1b);
    let inter_y1 = y1a.max(y1b);
    let inter_x2 = x2a.min(x2b);
    let inter_y2 = y2a.min(y2b);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (x2a - x1a) * (y2a - y1a);
    let area_b = (x2b - x1b) * (y2b - y1b);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        return Ok(0.0);
    }
    Ok(inter_area / union_area)
}

/// Generalized IoU.
///
/// GIoU = IoU - |C \ (A ∪ B)| / |C|
///
/// where C is the smallest enclosing box of A and B.
/// GIoU ∈ [-1, 1].
pub fn giou(
    box1: (f64, f64, f64, f64),
    box2: (f64, f64, f64, f64),
) -> Result<f64> {
    let (x1a, y1a, x2a, y2a) = box1;
    let (x1b, y1b, x2b, y2b) = box2;

    validate_box(box1)?;
    validate_box(box2)?;

    let inter_x1 = x1a.max(x1b);
    let inter_y1 = y1a.max(y1b);
    let inter_x2 = x2a.min(x2b);
    let inter_y2 = y2a.min(y2b);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (x2a - x1a) * (y2a - y1a);
    let area_b = (x2b - x1b) * (y2b - y1b);
    let union_area = area_a + area_b - inter_area;

    // Enclosing box
    let enc_x1 = x1a.min(x1b);
    let enc_y1 = y1a.min(y1b);
    let enc_x2 = x2a.max(x2b);
    let enc_y2 = y2a.max(y2b);
    let enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1);

    if union_area <= 0.0 || enc_area <= 0.0 {
        return Ok(-1.0);
    }

    let iou_val = inter_area / union_area;
    let penalty = (enc_area - union_area) / enc_area;
    Ok(iou_val - penalty)
}

/// Distance IoU.
///
/// DIoU = IoU - ρ²(b, b_gt) / c²
///
/// where ρ is the Euclidean distance between box centres and c is the diagonal
/// length of the enclosing box.
/// DIoU ∈ [-1, 1].
pub fn diou(
    box1: (f64, f64, f64, f64),
    box2: (f64, f64, f64, f64),
) -> Result<f64> {
    let (x1a, y1a, x2a, y2a) = box1;
    let (x1b, y1b, x2b, y2b) = box2;

    validate_box(box1)?;
    validate_box(box2)?;

    let inter_x1 = x1a.max(x1b);
    let inter_y1 = y1a.max(y1b);
    let inter_x2 = x2a.min(x2b);
    let inter_y2 = y2a.min(y2b);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let area_a = (x2a - x1a) * (y2a - y1a);
    let area_b = (x2b - x1b) * (y2b - y1b);
    let union_area = area_a + area_b - inter_area;

    // Centre points
    let cx_a = (x1a + x2a) / 2.0;
    let cy_a = (y1a + y2a) / 2.0;
    let cx_b = (x1b + x2b) / 2.0;
    let cy_b = (y1b + y2b) / 2.0;
    let rho2 = (cx_a - cx_b).powi(2) + (cy_a - cy_b).powi(2);

    // Diagonal of enclosing box
    let enc_x1 = x1a.min(x1b);
    let enc_y1 = y1a.min(y1b);
    let enc_x2 = x2a.max(x2b);
    let enc_y2 = y2a.max(y2b);
    let c2 = (enc_x2 - enc_x1).powi(2) + (enc_y2 - enc_y1).powi(2);

    if union_area <= 0.0 {
        return Ok(-1.0);
    }
    let iou_val = inter_area / union_area;
    if c2 == 0.0 {
        return Ok(iou_val);
    }
    Ok(iou_val - rho2 / c2)
}

/// Mean Average Precision for object detection.
///
/// Computes AP for each image using the provided `iou_threshold` to determine
/// true/false positives, then averages over images.
///
/// # Arguments
/// * `predictions` — per-image prediction lists: `(x1, y1, x2, y2, confidence)`
/// * `ground_truth` — per-image ground-truth lists: `(x1, y1, x2, y2)`
/// * `iou_threshold` — IoU threshold for a prediction to count as a true positive
///
/// # Returns
/// Mean AP across all images.
pub fn mean_average_precision_detection(
    predictions: &[Vec<(f64, f64, f64, f64, f64)>],
    ground_truth: &[Vec<(f64, f64, f64, f64)>],
    iou_threshold: f64,
) -> Result<f64> {
    if predictions.len() != ground_truth.len() {
        return Err(MetricsError::InvalidInput(format!(
            "predictions ({}) and ground_truth ({}) have different lengths",
            predictions.len(),
            ground_truth.len()
        )));
    }
    if predictions.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No images provided".to_string(),
        ));
    }

    let ap = compute_detection_ap(predictions, ground_truth, iou_threshold)?;
    Ok(ap)
}

/// COCO-style AP: average over IoU thresholds 0.50:0.05:0.95.
pub fn coco_ap(
    predictions: &[Vec<(f64, f64, f64, f64, f64)>],
    ground_truth: &[Vec<(f64, f64, f64, f64)>],
) -> Result<f64> {
    if predictions.len() != ground_truth.len() {
        return Err(MetricsError::InvalidInput(format!(
            "predictions ({}) and ground_truth ({}) have different lengths",
            predictions.len(),
            ground_truth.len()
        )));
    }
    if predictions.is_empty() {
        return Err(MetricsError::InvalidInput(
            "No images provided".to_string(),
        ));
    }

    let thresholds: Vec<f64> = (0..10).map(|i| 0.50 + i as f64 * 0.05).collect();
    let mut sum = 0.0_f64;
    for &thresh in &thresholds {
        sum += compute_detection_ap(predictions, ground_truth, thresh)?;
    }
    Ok(sum / thresholds.len() as f64)
}

/// Pixel accuracy for semantic segmentation.
///
/// # Arguments
/// * `pred_mask` — predicted class label per pixel (row-major)
/// * `true_mask` — ground-truth class label per pixel
///
/// # Returns
/// Fraction of correctly classified pixels.
pub fn pixel_accuracy(pred_mask: &[Vec<usize>], true_mask: &[Vec<usize>]) -> Result<f64> {
    if pred_mask.len() != true_mask.len() {
        return Err(MetricsError::InvalidInput(
            "pred_mask and true_mask have different numbers of rows".to_string(),
        ));
    }
    if pred_mask.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty mask provided".to_string(),
        ));
    }

    let mut correct = 0_u64;
    let mut total = 0_u64;

    for (pred_row, true_row) in pred_mask.iter().zip(true_mask.iter()) {
        if pred_row.len() != true_row.len() {
            return Err(MetricsError::InvalidInput(
                "pred_mask and true_mask rows have different widths".to_string(),
            ));
        }
        for (&p, &t) in pred_row.iter().zip(true_row.iter()) {
            if p == t {
                correct += 1;
            }
            total += 1;
        }
    }

    if total == 0 {
        return Err(MetricsError::InvalidInput(
            "Masks contain no pixels".to_string(),
        ));
    }
    Ok(correct as f64 / total as f64)
}

/// Mean IoU for semantic segmentation.
///
/// mIoU = mean_{k=0}^{n_classes-1} IoU(class k)
///
/// where IoU(k) = TP_k / (TP_k + FP_k + FN_k), with pixels ignored
/// that appear in neither pred nor true for class k.
pub fn mean_iou_segmentation(
    pred_mask: &[Vec<usize>],
    true_mask: &[Vec<usize>],
    n_classes: usize,
) -> Result<f64> {
    if pred_mask.len() != true_mask.len() {
        return Err(MetricsError::InvalidInput(
            "pred_mask and true_mask have different numbers of rows".to_string(),
        ));
    }
    if pred_mask.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty mask provided".to_string(),
        ));
    }
    if n_classes == 0 {
        return Err(MetricsError::InvalidInput(
            "n_classes must be > 0".to_string(),
        ));
    }

    let mut tp = vec![0_u64; n_classes];
    let mut fp = vec![0_u64; n_classes];
    let mut fn_ = vec![0_u64; n_classes];

    for (pred_row, true_row) in pred_mask.iter().zip(true_mask.iter()) {
        if pred_row.len() != true_row.len() {
            return Err(MetricsError::InvalidInput(
                "pred_mask and true_mask rows have different widths".to_string(),
            ));
        }
        for (&p, &t) in pred_row.iter().zip(true_row.iter()) {
            if p < n_classes && t < n_classes {
                if p == t {
                    tp[p] += 1;
                } else {
                    fp[p] += 1;
                    fn_[t] += 1;
                }
            }
        }
    }

    let mut iou_sum = 0.0_f64;
    let mut count = 0_usize;
    for k in 0..n_classes {
        let denom = tp[k] + fp[k] + fn_[k];
        if denom > 0 {
            iou_sum += tp[k] as f64 / denom as f64;
            count += 1;
        }
    }

    if count == 0 {
        return Ok(0.0);
    }
    Ok(iou_sum / count as f64)
}

/// Dice coefficient for binary segmentation masks.
///
/// Dice = 2 * |P ∩ T| / (|P| + |T|)
pub fn dice_coefficient(
    pred_mask: &[Vec<bool>],
    true_mask: &[Vec<bool>],
) -> Result<f64> {
    if pred_mask.len() != true_mask.len() {
        return Err(MetricsError::InvalidInput(
            "pred_mask and true_mask have different numbers of rows".to_string(),
        ));
    }
    if pred_mask.is_empty() {
        return Err(MetricsError::InvalidInput(
            "Empty mask provided".to_string(),
        ));
    }

    let mut intersection = 0_u64;
    let mut pred_total = 0_u64;
    let mut true_total = 0_u64;

    for (pred_row, true_row) in pred_mask.iter().zip(true_mask.iter()) {
        if pred_row.len() != true_row.len() {
            return Err(MetricsError::InvalidInput(
                "pred_mask and true_mask rows have different widths".to_string(),
            ));
        }
        for (&p, &t) in pred_row.iter().zip(true_row.iter()) {
            if p && t {
                intersection += 1;
            }
            if p {
                pred_total += 1;
            }
            if t {
                true_total += 1;
            }
        }
    }

    let denominator = pred_total + true_total;
    if denominator == 0 {
        // Both masks are empty → define Dice as 1.0 (perfect agreement on "no foreground")
        return Ok(1.0);
    }
    Ok(2.0 * intersection as f64 / denominator as f64)
}

// ─── internal helpers ────────────────────────────────────────────────────────

fn validate_box(b: (f64, f64, f64, f64)) -> Result<()> {
    let (x1, y1, x2, y2) = b;
    if x1 > x2 || y1 > y2 {
        return Err(MetricsError::InvalidInput(format!(
            "Invalid bounding box: ({x1}, {y1}, {x2}, {y2}) — x1 must be <= x2 and y1 <= y2"
        )));
    }
    Ok(())
}

/// Compute AP (single IoU threshold) over a set of images.
///
/// Strategy: collect all predictions globally, sort by confidence descending,
/// then compute precision-recall curve.
fn compute_detection_ap(
    predictions: &[Vec<(f64, f64, f64, f64, f64)>],
    ground_truth: &[Vec<(f64, f64, f64, f64)>],
    iou_threshold: f64,
) -> Result<f64> {
    // Collect (confidence, is_tp) pairs for all predictions
    let mut det_results: Vec<(f64, bool)> = Vec::new();
    let mut total_gt = 0_usize;

    for (img_preds, img_gts) in predictions.iter().zip(ground_truth.iter()) {
        total_gt += img_gts.len();
        let mut matched_gt = vec![false; img_gts.len()];

        // Sort image predictions by confidence descending
        let mut sorted_preds: Vec<(f64, f64, f64, f64, f64)> = img_preds.clone();
        sorted_preds.sort_by(|a, b| {
            b.4.partial_cmp(&a.4)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for pred in &sorted_preds {
            let pred_box = (pred.0, pred.1, pred.2, pred.3);
            let confidence = pred.4;

            // Find the best matching GT box
            let mut best_iou = 0.0_f64;
            let mut best_gt_idx = None;

            for (gt_idx, gt) in img_gts.iter().enumerate() {
                if matched_gt[gt_idx] {
                    continue;
                }
                let gt_box = (gt.0, gt.1, gt.2, gt.3);
                // Silently skip invalid boxes in GT
                if let Ok(iou_val) = iou(pred_box, gt_box) {
                    if iou_val > best_iou {
                        best_iou = iou_val;
                        best_gt_idx = Some(gt_idx);
                    }
                }
            }

            let is_tp = if best_iou >= iou_threshold {
                if let Some(idx) = best_gt_idx {
                    matched_gt[idx] = true;
                    true
                } else {
                    false
                }
            } else {
                false
            };

            det_results.push((confidence, is_tp));
        }
    }

    if total_gt == 0 {
        return Ok(0.0);
    }

    // Sort by confidence descending
    det_results.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute precision-recall curve
    let mut tp_cumulative = 0_usize;
    let mut fp_cumulative = 0_usize;
    let mut precisions = Vec::new();
    let mut recalls = Vec::new();

    for (_, is_tp) in &det_results {
        if *is_tp {
            tp_cumulative += 1;
        } else {
            fp_cumulative += 1;
        }
        let precision = tp_cumulative as f64 / (tp_cumulative + fp_cumulative) as f64;
        let recall = tp_cumulative as f64 / total_gt as f64;
        precisions.push(precision);
        recalls.push(recall);
    }

    // Compute area under PR curve using 11-point interpolation (PASCAL VOC style)
    let ap = interpolated_ap(&recalls, &precisions);
    Ok(ap)
}

/// 11-point interpolated AP (PASCAL VOC 2007 style).
fn interpolated_ap(recalls: &[f64], precisions: &[f64]) -> f64 {
    let mut ap = 0.0_f64;
    for t in 0..=10 {
        let recall_level = t as f64 / 10.0;
        let max_prec = recalls
            .iter()
            .zip(precisions.iter())
            .filter(|(&r, _)| r >= recall_level)
            .map(|(_, &p)| p)
            .fold(0.0_f64, f64::max);
        ap += max_prec;
    }
    ap / 11.0
}


// ---------------------------------------------------------------------------
// Array-based BBox API (requested interface)
// ---------------------------------------------------------------------------

/// Bounding box in `[x1, y1, x2, y2]` format.
pub type BBox = [f64; 4];

/// Validate a `BBox` array; returns an error if x1>x2 or y1>y2.
fn validate_bbox(b: &BBox) -> Result<()> {
    if b[0] > b[2] || b[1] > b[3] {
        return Err(MetricsError::InvalidInput(format!(
            "Invalid BBox: [{}, {}, {}, {}] — x1 must be ≤ x2 and y1 ≤ y2",
            b[0], b[1], b[2], b[3]
        )));
    }
    Ok(())
}

/// Convert a `BBox` array to the internal tuple representation.
fn bbox_to_tuple(b: &BBox) -> (f64, f64, f64, f64) {
    (b[0], b[1], b[2], b[3])
}

/// Intersection over Union for two `BBox` arrays.
///
/// This is the array-based entry point; it delegates to the tuple-based `iou`.
///
/// # Returns
/// IoU in [0, 1].
pub fn iou_bbox(a: &BBox, b: &BBox) -> Result<f64> {
    validate_bbox(a)?;
    validate_bbox(b)?;
    iou(bbox_to_tuple(a), bbox_to_tuple(b))
}

/// Generalized IoU for two `BBox` arrays.
pub fn giou_bbox(a: &BBox, b: &BBox) -> Result<f64> {
    validate_bbox(a)?;
    validate_bbox(b)?;
    giou(bbox_to_tuple(a), bbox_to_tuple(b))
}

/// Average Precision for a single class at a given IoU threshold.
///
/// # Arguments
/// * `detections` — `(confidence, bbox)` pairs, sorted by **descending** confidence.
/// * `ground_truths` — ground-truth bounding boxes for this class / image.
/// * `iou_threshold` — minimum IoU for a detection to count as a true positive.
///
/// # Returns
/// AP in [0, 1] using 11-point interpolation.
pub fn average_precision_bbox(
    detections: &[(f64, BBox)],
    ground_truths: &[BBox],
    iou_threshold: f64,
) -> Result<f64> {
    let n_gt = ground_truths.len();
    if n_gt == 0 {
        return Ok(0.0);
    }

    let mut matched_gt = vec![false; n_gt];
    let mut tp_cum = 0_usize;
    let mut fp_cum = 0_usize;
    let mut precisions = Vec::with_capacity(detections.len());
    let mut recalls = Vec::with_capacity(detections.len());

    for (_, det_box) in detections {
        let mut best_iou = 0.0_f64;
        let mut best_idx: Option<usize> = None;

        for (gi, gt_box) in ground_truths.iter().enumerate() {
            if matched_gt[gi] {
                continue;
            }
            if let Ok(iou_val) = iou(bbox_to_tuple(det_box), bbox_to_tuple(gt_box)) {
                if iou_val > best_iou {
                    best_iou = iou_val;
                    best_idx = Some(gi);
                }
            }
        }

        if best_iou >= iou_threshold {
            if let Some(idx) = best_idx {
                matched_gt[idx] = true;
                tp_cum += 1;
            } else {
                fp_cum += 1;
            }
        } else {
            fp_cum += 1;
        }

        let p = tp_cum as f64 / (tp_cum + fp_cum) as f64;
        let r = tp_cum as f64 / n_gt as f64;
        precisions.push(p);
        recalls.push(r);
    }

    Ok(interpolated_ap(&recalls, &precisions))
}

/// Mean Average Precision over multiple classes.
///
/// # Arguments
/// * `per_class` — map from `class_id` to `(detections, ground_truths)`.
///   Detections must be sorted by descending confidence.
/// * `iou_threshold` — IoU threshold for TP/FP assignment.
///
/// # Returns
/// mAP in [0, 1] averaged over all classes present in the map.
pub fn mean_average_precision_per_class(
    per_class: &std::collections::HashMap<usize, (Vec<(f64, BBox)>, Vec<BBox>)>,
    iou_threshold: f64,
) -> Result<f64> {
    if per_class.is_empty() {
        return Err(MetricsError::InvalidInput(
            "per_class map must not be empty".to_string(),
        ));
    }

    let mut ap_sum = 0.0_f64;
    let mut count = 0_usize;

    for (_, (detections, ground_truths)) in per_class {
        let ap = average_precision_bbox(detections, ground_truths, iou_threshold)?;
        ap_sum += ap;
        count += 1;
    }

    Ok(if count == 0 { 0.0 } else { ap_sum / count as f64 })
}

/// COCO-style mAP: mean over IoU thresholds [0.50, 0.55, …, 0.95].
///
/// # Arguments
/// * `per_class` — same format as [`mean_average_precision_per_class`].
pub fn coco_map_per_class(
    per_class: &std::collections::HashMap<usize, (Vec<(f64, BBox)>, Vec<BBox>)>,
) -> Result<f64> {
    let thresholds: Vec<f64> = (0..=9).map(|i| 0.50 + i as f64 * 0.05).collect();
    let mut sum = 0.0_f64;
    for &thresh in &thresholds {
        sum += mean_average_precision_per_class(per_class, thresh)?;
    }
    Ok(sum / thresholds.len() as f64)
}

/// Non-Maximum Suppression.
///
/// Removes detections that overlap with a higher-confidence detection by more
/// than `iou_threshold`.  The `detections` slice is sorted in-place by
/// descending confidence and filtered.
///
/// # Arguments
/// * `detections`    — mutable vector of `(confidence, bbox)` to filter in-place.
/// * `iou_threshold` — IoU threshold above which a lower-confidence detection
///                     is suppressed.
pub fn nms(detections: &mut Vec<(f64, BBox)>, iou_threshold: f64) {
    // Sort by descending confidence.
    detections.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = vec![true; detections.len()];

    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..detections.len() {
            if !keep[j] {
                continue;
            }
            let iou_val = iou(
                bbox_to_tuple(&detections[i].1),
                bbox_to_tuple(&detections[j].1),
            )
            .unwrap_or(0.0);
            if iou_val > iou_threshold {
                keep[j] = false;
            }
        }
    }

    let mut idx = 0;
    detections.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou_identical() {
        let b = (0.0, 0.0, 1.0, 1.0);
        let val = iou(b, b).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_iou_no_overlap() {
        let b1 = (0.0, 0.0, 1.0, 1.0);
        let b2 = (2.0, 2.0, 3.0, 3.0);
        let val = iou(b1, b2).expect("should succeed");
        assert!((val - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let b1 = (0.0, 0.0, 2.0, 2.0);
        let b2 = (1.0, 1.0, 3.0, 3.0);
        let val = iou(b1, b2).expect("should succeed");
        // Intersection: 1x1=1, Union: 4+4-1=7
        assert!((val - 1.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_giou_identical() {
        let b = (0.0, 0.0, 1.0, 1.0);
        let val = giou(b, b).expect("should succeed");
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_giou_no_overlap() {
        let b1 = (0.0, 0.0, 1.0, 1.0);
        let b2 = (2.0, 2.0, 3.0, 3.0);
        let val = giou(b1, b2).expect("should succeed");
        // IoU=0, penalty = (9-2)/9 = 7/9; GIoU = 0 - 7/9 = -7/9 ≈ -0.778
        assert!(val < 0.0, "GIoU should be negative for non-overlapping boxes, got {val}");
    }

    #[test]
    fn test_diou_identical() {
        let b = (0.0, 0.0, 1.0, 1.0);
        let val = diou(b, b).expect("should succeed");
        // IoU=1, rho2=0 → DIoU=1
        assert!((val - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pixel_accuracy_perfect() {
        let mask = vec![vec![0, 1, 2], vec![1, 2, 0]];
        let acc = pixel_accuracy(&mask, &mask).expect("should succeed");
        assert!((acc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pixel_accuracy_half() {
        let pred = vec![vec![0, 0]];
        let true_ = vec![vec![0, 1]];
        let acc = pixel_accuracy(&pred, &true_).expect("should succeed");
        assert!((acc - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mean_iou_perfect() {
        let mask = vec![vec![0, 0, 1, 1]];
        let miou = mean_iou_segmentation(&mask, &mask, 2).expect("should succeed");
        assert!((miou - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dice_coefficient_perfect() {
        let mask = vec![vec![true, false, true]];
        let dice = dice_coefficient(&mask, &mask).expect("should succeed");
        assert!((dice - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dice_coefficient_no_overlap() {
        let pred = vec![vec![true, false]];
        let true_ = vec![vec![false, true]];
        let dice = dice_coefficient(&pred, &true_).expect("should succeed");
        assert!((dice - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_mean_average_precision_detection_perfect() {
        // One image, one GT box, one perfectly-matching prediction
        let preds = vec![vec![(0.0, 0.0, 1.0, 1.0, 0.9)]];
        let gts = vec![vec![(0.0, 0.0, 1.0, 1.0)]];
        let map = mean_average_precision_detection(&preds, &gts, 0.5).expect("should succeed");
        assert!(map > 0.5, "mAP should be high for perfect detection, got {map}");
    }

    #[test]
    fn test_coco_ap_no_detection() {
        // One image, one GT, prediction doesn't overlap
        let preds = vec![vec![(10.0, 10.0, 11.0, 11.0, 0.9)]];
        let gts = vec![vec![(0.0, 0.0, 1.0, 1.0)]];
        let ap = coco_ap(&preds, &gts).expect("should succeed");
        assert!((ap - 0.0).abs() < 1e-10, "COCO AP should be 0 for wrong detection, got {ap}");
    }

    #[test]
    fn test_invalid_box() {
        // x1 > x2 should fail
        assert!(iou((1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)).is_err());
    }
}
