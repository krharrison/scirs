//! Segmentation evaluation metrics.
//!
//! Provides standard metrics for semantic segmentation and instance segmentation:
//! - Pixel-wise confusion matrix (mIoU, pixel accuracy, mean pixel accuracy,
//!   frequency-weighted IoU, Dice coefficient)
//! - Instance segmentation AP computed over multiple IoU thresholds.

use crate::error::{MetricsError, Result};

// ---------------------------------------------------------------------------
// Semantic segmentation
// ---------------------------------------------------------------------------

/// Pixel-wise confusion matrix for semantic segmentation.
///
/// Rows correspond to ground-truth classes, columns to predicted classes.
/// Accumulated with successive calls to [`SegmentationConfusionMatrix::update`].
pub struct SegmentationConfusionMatrix {
    /// Flat storage in row-major order: `matrix[gt * n + pred]`.
    matrix: Vec<u64>,
    n_classes: usize,
}

impl SegmentationConfusionMatrix {
    /// Create a new confusion matrix for `n_classes` classes.
    pub fn new(n_classes: usize) -> Self {
        Self {
            matrix: vec![0u64; n_classes * n_classes],
            n_classes,
        }
    }

    /// Accumulate predictions for one mini-batch.
    ///
    /// `pred` and `gt` must have the same length; each element is a class
    /// index in `[0, n_classes)`.
    ///
    /// # Errors
    /// Returns an error if the lengths differ or if any index is out of range.
    pub fn update(&mut self, pred: &[usize], gt: &[usize]) -> Result<()> {
        if pred.len() != gt.len() {
            return Err(MetricsError::InvalidInput(format!(
                "pred ({}) and gt ({}) have different lengths",
                pred.len(),
                gt.len()
            )));
        }
        for (&p, &g) in pred.iter().zip(gt.iter()) {
            if p >= self.n_classes {
                return Err(MetricsError::InvalidInput(format!(
                    "pred index {p} is out of range for n_classes={}",
                    self.n_classes
                )));
            }
            if g >= self.n_classes {
                return Err(MetricsError::InvalidInput(format!(
                    "gt index {g} is out of range for n_classes={}",
                    self.n_classes
                )));
            }
            self.matrix[g * self.n_classes + p] += 1;
        }
        Ok(())
    }

    /// Reset all counts to zero.
    pub fn reset(&mut self) {
        self.matrix.iter_mut().for_each(|v| *v = 0);
    }

    // -------------------------------------------------------------------
    // Per-class helpers
    // -------------------------------------------------------------------

    /// True positives for class `c`: diagonal element.
    fn tp(&self, c: usize) -> u64 {
        self.matrix[c * self.n_classes + c]
    }

    /// Number of ground-truth pixels for class `c`: row sum.
    fn gt_sum(&self, c: usize) -> u64 {
        (0..self.n_classes)
            .map(|p| self.matrix[c * self.n_classes + p])
            .sum()
    }

    /// Number of predicted pixels for class `c`: column sum.
    fn pred_sum(&self, c: usize) -> u64 {
        (0..self.n_classes)
            .map(|g| self.matrix[g * self.n_classes + c])
            .sum()
    }

    // -------------------------------------------------------------------
    // Aggregate metrics
    // -------------------------------------------------------------------

    /// Per-class Intersection over Union.
    ///
    /// IoU_c = TP_c / (GT_c + Pred_c - TP_c).
    /// Classes absent from both prediction and ground-truth are skipped
    /// (they would produce 0/0).
    pub fn per_class_iou(&self) -> Vec<f64> {
        (0..self.n_classes)
            .map(|c| {
                let tp = self.tp(c);
                let gt = self.gt_sum(c);
                let pred = self.pred_sum(c);
                let denom = gt + pred - tp;
                if denom == 0 {
                    // Class absent — return NaN so callers can decide to skip it.
                    f64::NAN
                } else {
                    tp as f64 / denom as f64
                }
            })
            .collect()
    }

    /// Mean Intersection over Union (average over present classes).
    pub fn mean_iou(&self) -> f64 {
        let iou = self.per_class_iou();
        let (sum, count) = iou.iter().fold((0.0_f64, 0_usize), |(s, n), &v| {
            if v.is_nan() {
                (s, n)
            } else {
                (s + v, n + 1)
            }
        });
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// Overall pixel accuracy: fraction of correctly classified pixels.
    pub fn pixel_accuracy(&self) -> f64 {
        let total_correct: u64 = (0..self.n_classes).map(|c| self.tp(c)).sum();
        let total: u64 = self.matrix.iter().sum();
        if total == 0 {
            0.0
        } else {
            total_correct as f64 / total as f64
        }
    }

    /// Mean pixel accuracy: per-class accuracy averaged over present classes.
    ///
    /// accuracy_c = TP_c / GT_c.
    pub fn mean_pixel_accuracy(&self) -> f64 {
        let (sum, count) = (0..self.n_classes).fold((0.0_f64, 0_usize), |(s, n), c| {
            let gt = self.gt_sum(c);
            if gt == 0 {
                (s, n)
            } else {
                (s + self.tp(c) as f64 / gt as f64, n + 1)
            }
        });
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// Frequency-weighted IoU.
    ///
    /// FWIoU = Σ_c (GT_c / Total) * IoU_c, summed over present classes.
    pub fn frequency_weighted_iou(&self) -> f64 {
        let total: u64 = self.matrix.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let iou = self.per_class_iou();
        iou.iter()
            .enumerate()
            .filter(|(_, v)| !v.is_nan())
            .map(|(c, &iou_c)| {
                let freq = self.gt_sum(c) as f64 / total as f64;
                freq * iou_c
            })
            .sum()
    }

    /// Dice coefficient (equal to F1 score), macro-averaged over present classes.
    ///
    /// Dice_c = 2*TP_c / (GT_c + Pred_c).
    pub fn dice(&self) -> f64 {
        let (sum, count) = (0..self.n_classes).fold((0.0_f64, 0_usize), |(s, n), c| {
            let gt = self.gt_sum(c);
            let pred = self.pred_sum(c);
            let denom = gt + pred;
            if denom == 0 {
                (s, n)
            } else {
                let dice_c = 2.0 * self.tp(c) as f64 / denom as f64;
                (s + dice_c, n + 1)
            }
        });
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Instance segmentation
// ---------------------------------------------------------------------------

/// A single instance prediction or ground-truth mask.
pub struct InstanceMask {
    /// Binary mask: `mask[row][col]` is `true` if the pixel belongs to this instance.
    pub mask: Vec<Vec<bool>>,
    /// Confidence score (for predictions; set to 1.0 for ground-truth).
    pub score: f64,
    /// Class identifier.
    pub class_id: usize,
}

/// Compute mask IoU between two instance masks.
///
/// Returns 0.0 when both masks are empty.
fn mask_iou(a: &Vec<Vec<bool>>, b: &Vec<Vec<bool>>) -> f64 {
    let rows = a.len().max(b.len());
    let cols = if rows == 0 {
        0
    } else {
        a.first().map(|r| r.len()).unwrap_or(0)
            .max(b.first().map(|r| r.len()).unwrap_or(0))
    };

    let mut intersection = 0u64;
    let mut union = 0u64;

    for r in 0..rows {
        let a_row: &[bool] = a.get(r).map(|v| v.as_slice()).unwrap_or(&[]);
        let b_row: &[bool] = b.get(r).map(|v| v.as_slice()).unwrap_or(&[]);
        for c in 0..cols {
            let av = a_row.get(c).copied().unwrap_or(false);
            let bv = b_row.get(c).copied().unwrap_or(false);
            if av || bv {
                union += 1;
                if av && bv {
                    intersection += 1;
                }
            }
        }
    }

    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

/// Average Precision for instance segmentation over multiple IoU thresholds.
///
/// For each threshold `t` the function uses a greedy matching strategy
/// (sort predictions by descending score, match each prediction to the
/// highest-IoU unmatched ground-truth whose IoU ≥ t).  The AP at each
/// threshold is computed as the area under the precision-recall curve
/// (101-point interpolation).  The final result is the mean over all
/// provided thresholds.
///
/// # Arguments
/// * `pred_masks`  — predicted instance masks (will be sorted by descending score).
/// * `gt_masks`    — ground-truth instance masks.
/// * `iou_thresholds` — list of IoU thresholds; must not be empty.
///
/// # Returns
/// Mean AP across all thresholds.
pub fn instance_ap(
    pred_masks: &[InstanceMask],
    gt_masks: &[InstanceMask],
    iou_thresholds: &[f64],
) -> Result<f64> {
    if iou_thresholds.is_empty() {
        return Err(MetricsError::InvalidInput(
            "iou_thresholds must not be empty".to_string(),
        ));
    }

    if gt_masks.is_empty() {
        // No ground truth → AP is 0 if there are predictions, undefined otherwise.
        return Ok(0.0);
    }

    // Sort predictions by descending score.
    let mut sorted_preds: Vec<&InstanceMask> = pred_masks.iter().collect();
    sorted_preds.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Pre-compute all pairwise IoU values: pred x gt.
    let n_pred = sorted_preds.len();
    let n_gt = gt_masks.len();
    let mut iou_matrix = vec![0.0_f64; n_pred * n_gt];
    for (pi, pred) in sorted_preds.iter().enumerate() {
        for (gi, gt) in gt_masks.iter().enumerate() {
            iou_matrix[pi * n_gt + gi] = mask_iou(&pred.mask, &gt.mask);
        }
    }

    let mut ap_sum = 0.0_f64;

    for &thresh in iou_thresholds {
        let mut matched_gt = vec![false; n_gt];
        // (confidence, is_tp)
        let mut det_results: Vec<(f64, bool)> = Vec::with_capacity(n_pred);

        for pi in 0..n_pred {
            let pred = sorted_preds[pi];
            let mut best_iou = 0.0_f64;
            let mut best_gi: Option<usize> = None;

            for gi in 0..n_gt {
                if matched_gt[gi] {
                    continue;
                }
                // Class filtering: only match same class.
                if gt_masks[gi].class_id != pred.class_id {
                    continue;
                }
                let iou_val = iou_matrix[pi * n_gt + gi];
                if iou_val > best_iou {
                    best_iou = iou_val;
                    best_gi = Some(gi);
                }
            }

            let is_tp = if best_iou >= thresh {
                if let Some(gi) = best_gi {
                    matched_gt[gi] = true;
                    true
                } else {
                    false
                }
            } else {
                false
            };
            det_results.push((pred.score, is_tp));
        }

        // Compute PR curve.
        let mut tp_cum = 0_usize;
        let mut fp_cum = 0_usize;
        let mut precisions = Vec::with_capacity(n_pred);
        let mut recalls = Vec::with_capacity(n_pred);

        for (_, is_tp) in &det_results {
            if *is_tp {
                tp_cum += 1;
            } else {
                fp_cum += 1;
            }
            let p = tp_cum as f64 / (tp_cum + fp_cum) as f64;
            let r = tp_cum as f64 / n_gt as f64;
            precisions.push(p);
            recalls.push(r);
        }

        ap_sum += interpolated_ap_101(&recalls, &precisions);
    }

    Ok(ap_sum / iou_thresholds.len() as f64)
}

/// 101-point interpolated AP (COCO style).
fn interpolated_ap_101(recalls: &[f64], precisions: &[f64]) -> f64 {
    let mut ap = 0.0_f64;
    for t in 0..=100 {
        let recall_level = t as f64 / 100.0;
        let max_prec = recalls
            .iter()
            .zip(precisions.iter())
            .filter(|(&r, _)| r >= recall_level)
            .map(|(_, &p)| p)
            .fold(0.0_f64, f64::max);
        ap += max_prec;
    }
    ap / 101.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_perfect() {
        let mut cm = SegmentationConfusionMatrix::new(3);
        let pred = vec![0, 1, 2, 0, 1, 2];
        let gt = vec![0, 1, 2, 0, 1, 2];
        cm.update(&pred, &gt).expect("update failed");

        assert!((cm.mean_iou() - 1.0).abs() < 1e-10, "mIoU should be 1.0 for perfect prediction");
        assert!((cm.pixel_accuracy() - 1.0).abs() < 1e-10);
        assert!((cm.mean_pixel_accuracy() - 1.0).abs() < 1e-10);
        assert!((cm.frequency_weighted_iou() - 1.0).abs() < 1e-10);
        assert!((cm.dice() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_all_wrong() {
        let mut cm = SegmentationConfusionMatrix::new(2);
        // Predict class 1 when GT is class 0 and vice versa.
        let pred = vec![1, 1, 0, 0];
        let gt = vec![0, 0, 1, 1];
        cm.update(&pred, &gt).expect("update failed");

        assert!((cm.mean_iou() - 0.0).abs() < 1e-10);
        assert!((cm.pixel_accuracy() - 0.0).abs() < 1e-10);
        assert!((cm.dice() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_partial() {
        let mut cm = SegmentationConfusionMatrix::new(2);
        // 3 correct out of 4.
        let pred = vec![0, 0, 1, 1];
        let gt = vec![0, 1, 1, 1];
        cm.update(&pred, &gt).expect("update failed");

        assert!((cm.pixel_accuracy() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_reset() {
        let mut cm = SegmentationConfusionMatrix::new(2);
        cm.update(&[0], &[1]).expect("update failed");
        cm.reset();
        assert!((cm.pixel_accuracy() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_length_mismatch() {
        let mut cm = SegmentationConfusionMatrix::new(2);
        assert!(cm.update(&[0, 1], &[0]).is_err());
    }

    #[test]
    fn test_per_class_iou() {
        let mut cm = SegmentationConfusionMatrix::new(2);
        // Class 0: 2 TP; Class 1: 1 TP, 1 FN, 1 FP.
        let pred = vec![0, 0, 1, 0];
        let gt = vec![0, 0, 1, 1];
        cm.update(&pred, &gt).expect("update failed");

        let iou = cm.per_class_iou();
        // Class 0: TP=2, GT=2, Pred=3 → IoU = 2/(2+3-2)=2/3
        assert!((iou[0] - 2.0 / 3.0).abs() < 1e-10, "class 0 IoU wrong: {}", iou[0]);
        // Class 1: TP=1, GT=2, Pred=1 → IoU = 1/(2+1-1)=1/2
        assert!((iou[1] - 0.5).abs() < 1e-10, "class 1 IoU wrong: {}", iou[1]);
    }

    fn make_mask(rows: usize, cols: usize, val: bool) -> Vec<Vec<bool>> {
        vec![vec![val; cols]; rows]
    }

    #[test]
    fn test_mask_iou_identical() {
        let m = make_mask(3, 3, true);
        assert!((mask_iou(&m, &m) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mask_iou_no_overlap() {
        let a = vec![vec![true, false], vec![true, false]];
        let b = vec![vec![false, true], vec![false, true]];
        assert!((mask_iou(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_instance_ap_perfect() {
        let mask = make_mask(4, 4, true);
        let pred = vec![InstanceMask {
            mask: mask.clone(),
            score: 0.99,
            class_id: 0,
        }];
        let gt = vec![InstanceMask {
            mask: mask.clone(),
            score: 1.0,
            class_id: 0,
        }];
        let ap = instance_ap(&pred, &gt, &[0.5]).expect("instance_ap failed");
        assert!(ap > 0.9, "AP should be near 1.0 for perfect prediction, got {ap}");
    }

    #[test]
    fn test_instance_ap_no_gt() {
        let ap = instance_ap(&[], &[], &[0.5]).expect("instance_ap failed");
        assert!((ap - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_instance_ap_empty_thresholds_error() {
        assert!(instance_ap(&[], &[], &[]).is_err());
    }

    #[test]
    fn test_frequency_weighted_iou_single_class() {
        let mut cm = SegmentationConfusionMatrix::new(1);
        cm.update(&[0, 0, 0], &[0, 0, 0]).expect("update failed");
        // Single class: FW-IoU should equal 1.0
        assert!((cm.frequency_weighted_iou() - 1.0).abs() < 1e-10);
    }
}
