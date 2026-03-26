//! Keypoint Detection Metrics
//!
//! This module provides evaluation metrics for human pose estimation and
//! keypoint detection tasks, following COCO benchmark conventions:
//!
//! - **OKS** (Object Keypoint Similarity): COCO-standard per-instance metric
//! - **PCK** (Percentage of Correct Keypoints): threshold-based accuracy
//! - **PCKh**: PCK normalised by head size
//! - **Mean OKS**: dataset-level OKS averaging
//! - **Mean Keypoint Error**: average Euclidean distance for visible keypoints

use crate::error::{MetricsError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Data Structures
// ─────────────────────────────────────────────────────────────────────────────

/// A single pose annotation with 2-D keypoint coordinates and visibility flags.
#[derive(Debug, Clone)]
pub struct KeypointAnnotation {
    /// `(x, y)` coordinates for each keypoint.
    pub keypoints: Vec<[f64; 2]>,
    /// Visibility flag per keypoint: `0` = absent, `1` = occluded, `2` = visible.
    pub visibility: Vec<u8>,
    /// Square root of the object area (used for OKS scale normalisation).
    pub scale: f64,
}

impl KeypointAnnotation {
    /// Validate that `keypoints` and `visibility` have the same length.
    pub fn validate(&self) -> Result<()> {
        if self.keypoints.len() != self.visibility.len() {
            return Err(MetricsError::DimensionMismatch(format!(
                "keypoints len {} != visibility len {}",
                self.keypoints.len(),
                self.visibility.len()
            )));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// COCO Constants
// ─────────────────────────────────────────────────────────────────────────────

/// Default COCO sigmas for the 17 COCO body keypoints.
///
/// Order: nose, left_eye, right_eye, left_ear, right_ear,
/// left_shoulder, right_shoulder, left_elbow, right_elbow,
/// left_wrist, right_wrist, left_hip, right_hip,
/// left_knee, right_knee, left_ankle, right_ankle.
pub fn coco_sigmas() -> Vec<f64> {
    vec![
        0.026, // nose
        0.025, // left_eye
        0.025, // right_eye
        0.035, // left_ear
        0.035, // right_ear
        0.079, // left_shoulder
        0.079, // right_shoulder
        0.072, // left_elbow
        0.072, // right_elbow
        0.062, // left_wrist
        0.062, // right_wrist
        0.107, // left_hip
        0.107, // right_hip
        0.087, // left_knee
        0.087, // right_knee
        0.089, // left_ankle
        0.089, // right_ankle
    ]
}

// ─────────────────────────────────────────────────────────────────────────────
// OKS
// ─────────────────────────────────────────────────────────────────────────────

/// Object Keypoint Similarity (OKS) — COCO standard.
///
/// ```text
/// OKS = Σ_i [exp(−d_i² / (2 s² k_i²)) * δ(v_i > 0)] / Σ_i [δ(v_i > 0)]
/// ```
///
/// where `d_i` is the Euclidean distance between predicted and GT keypoint `i`,
/// `s` is the object scale (square root of area), and `k_i` is the per-keypoint
/// constant from `sigmas`.
///
/// # Arguments
/// * `predicted`    — predicted annotation
/// * `ground_truth` — ground-truth annotation
/// * `sigmas`       — per-keypoint sigma constants (same length as keypoints)
pub fn object_keypoint_similarity(
    predicted: &KeypointAnnotation,
    ground_truth: &KeypointAnnotation,
    sigmas: &[f64],
) -> Result<f64> {
    predicted.validate()?;
    ground_truth.validate()?;

    let n = predicted.keypoints.len();
    if n == 0 {
        return Err(MetricsError::InvalidInput(
            "keypoint annotations must have at least one keypoint".to_string(),
        ));
    }
    if ground_truth.keypoints.len() != n {
        return Err(MetricsError::DimensionMismatch(format!(
            "predicted has {n} keypoints but GT has {}",
            ground_truth.keypoints.len()
        )));
    }
    if sigmas.len() != n {
        return Err(MetricsError::DimensionMismatch(format!(
            "sigmas len {} != keypoints len {n}",
            sigmas.len()
        )));
    }

    let s = ground_truth.scale;
    if s <= 0.0 {
        return Err(MetricsError::InvalidInput(
            "object scale must be positive".to_string(),
        ));
    }

    let mut numerator = 0.0_f64;
    let mut denominator = 0.0_f64;

    for i in 0..n {
        let v_gt = ground_truth.visibility[i];
        if v_gt == 0 {
            // Keypoint absent in GT — skip.
            continue;
        }
        denominator += 1.0;

        let [px, py] = predicted.keypoints[i];
        let [gx, gy] = ground_truth.keypoints[i];
        let d_sq = (px - gx).powi(2) + (py - gy).powi(2);
        let ki = sigmas[i];
        let e = -d_sq / (2.0 * s * s * ki * ki);
        numerator += e.exp();
    }

    if denominator == 0.0 {
        return Ok(0.0);
    }
    Ok(numerator / denominator)
}

// ─────────────────────────────────────────────────────────────────────────────
// PCK / PCKh
// ─────────────────────────────────────────────────────────────────────────────

/// Percentage of Correct Keypoints (PCK) at threshold `t`.
///
/// A keypoint is "correct" when the predicted distance to GT is
/// `< threshold_fraction * reference_distance`.
///
/// Only visible keypoints (`visibility[i] > 0`) are evaluated.
pub fn pck(
    predicted: &[[f64; 2]],
    ground_truth: &[[f64; 2]],
    visibility: &[u8],
    threshold_fraction: f64,
    reference_distance: f64,
) -> Result<f64> {
    let n = predicted.len();
    if n == 0 {
        return Err(MetricsError::InvalidInput(
            "predicted keypoints must not be empty".to_string(),
        ));
    }
    if ground_truth.len() != n || visibility.len() != n {
        return Err(MetricsError::DimensionMismatch(format!(
            "predicted, ground_truth and visibility must all have length {n}"
        )));
    }
    if threshold_fraction <= 0.0 || reference_distance <= 0.0 {
        return Err(MetricsError::InvalidInput(
            "threshold_fraction and reference_distance must be positive".to_string(),
        ));
    }

    let threshold = threshold_fraction * reference_distance;
    let mut correct = 0usize;
    let mut total = 0usize;

    for i in 0..n {
        if visibility[i] == 0 {
            continue;
        }
        total += 1;
        let [px, py] = predicted[i];
        let [gx, gy] = ground_truth[i];
        let dist = ((px - gx).powi(2) + (py - gy).powi(2)).sqrt();
        if dist < threshold {
            correct += 1;
        }
    }

    if total == 0 {
        return Ok(0.0);
    }
    Ok(correct as f64 / total as f64)
}

/// PCKh: PCK with threshold relative to head size.
///
/// A keypoint is correct when predicted distance < `threshold_fraction * head_size`.
pub fn pckh(
    predicted: &[[f64; 2]],
    ground_truth: &[[f64; 2]],
    visibility: &[u8],
    head_size: f64,
    threshold_fraction: f64,
) -> Result<f64> {
    pck(
        predicted,
        ground_truth,
        visibility,
        threshold_fraction,
        head_size,
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Dataset-level metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Mean OKS over a dataset.
///
/// Computes OKS for each (prediction, GT) pair and returns the mean.
pub fn mean_oks(
    predictions: &[KeypointAnnotation],
    ground_truths: &[KeypointAnnotation],
    sigmas: &[f64],
) -> Result<f64> {
    if predictions.is_empty() {
        return Err(MetricsError::InvalidInput(
            "predictions must not be empty".to_string(),
        ));
    }
    if predictions.len() != ground_truths.len() {
        return Err(MetricsError::DimensionMismatch(format!(
            "predictions len {} != ground_truths len {}",
            predictions.len(),
            ground_truths.len()
        )));
    }
    let total: f64 = predictions
        .iter()
        .zip(ground_truths)
        .map(|(pred, gt)| object_keypoint_similarity(pred, gt, sigmas))
        .sum::<Result<f64>>()?;
    Ok(total / predictions.len() as f64)
}

/// Mean Euclidean keypoint error for visible keypoints.
///
/// Returns the average distance between predicted and GT keypoints
/// where `visibility[i] > 0`.
pub fn mean_keypoint_error(
    predicted: &[[f64; 2]],
    ground_truth: &[[f64; 2]],
    visibility: &[u8],
) -> Result<f64> {
    let n = predicted.len();
    if n == 0 {
        return Err(MetricsError::InvalidInput(
            "predicted keypoints must not be empty".to_string(),
        ));
    }
    if ground_truth.len() != n || visibility.len() != n {
        return Err(MetricsError::DimensionMismatch(format!(
            "predicted, ground_truth and visibility must all have length {n}"
        )));
    }

    let mut total_dist = 0.0_f64;
    let mut count = 0usize;

    for i in 0..n {
        if visibility[i] == 0 {
            continue;
        }
        let [px, py] = predicted[i];
        let [gx, gy] = ground_truth[i];
        total_dist += ((px - gx).powi(2) + (py - gy).powi(2)).sqrt();
        count += 1;
    }

    if count == 0 {
        return Ok(0.0);
    }
    Ok(total_dist / count as f64)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_annotation(kps: Vec<[f64; 2]>, vis: Vec<u8>, scale: f64) -> KeypointAnnotation {
        KeypointAnnotation {
            keypoints: kps,
            visibility: vis,
            scale,
        }
    }

    #[test]
    fn test_oks_perfect_prediction() {
        let kps = vec![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]];
        let vis = vec![2, 2, 2];
        let sigmas = vec![0.05, 0.05, 0.05];
        let pred = make_annotation(kps.clone(), vis.clone(), 50.0);
        let gt = make_annotation(kps, vis, 50.0);
        let oks = object_keypoint_similarity(&pred, &gt, &sigmas).expect("should succeed");
        assert!(
            (oks - 1.0).abs() < 1e-10,
            "perfect OKS should be 1.0, got {oks}"
        );
    }

    #[test]
    fn test_oks_large_distance_near_zero() {
        let gt_kps = vec![[0.0, 0.0], [0.0, 0.0]];
        let pred_kps = vec![[1000.0, 1000.0], [1000.0, 1000.0]];
        let vis = vec![2, 2];
        let sigmas = vec![0.05, 0.05];
        let pred = make_annotation(pred_kps, vis.clone(), 1.0);
        let gt = make_annotation(gt_kps, vis, 1.0);
        let oks = object_keypoint_similarity(&pred, &gt, &sigmas).expect("should succeed");
        assert!(
            oks < 1e-6,
            "OKS for very large error should be ~0, got {oks}"
        );
    }

    #[test]
    fn test_pck_all_correct() {
        let kps: Vec<[f64; 2]> = vec![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let vis = vec![2, 2, 2];
        let score = pck(&kps, &kps, &vis, 0.1, 100.0).expect("should succeed");
        assert!(
            (score - 1.0).abs() < 1e-12,
            "perfect PCK should be 1.0, got {score}"
        );
    }

    #[test]
    fn test_pck_none_correct() {
        let pred = vec![[0.0, 0.0], [0.0, 0.0]];
        let gt = vec![[100.0, 100.0], [200.0, 200.0]];
        let vis = vec![2, 2];
        // threshold = 0.01 * 1.0 = 0.01; distances are >> 0.01
        let score = pck(&pred, &gt, &vis, 0.01, 1.0).expect("should succeed");
        assert!((score - 0.0).abs() < 1e-12, "expected PCK=0, got {score}");
    }

    #[test]
    fn test_pckh_head_size_reference() {
        let pred = vec![[10.0, 10.0], [20.0, 20.0]];
        let gt = vec![[10.0, 10.0], [20.0, 20.0]];
        let vis = vec![2, 2];
        let score = pckh(&pred, &gt, &vis, 200.0, 0.5).expect("should succeed");
        assert!(
            (score - 1.0).abs() < 1e-12,
            "expected PCKh=1.0, got {score}"
        );
    }

    #[test]
    fn test_mean_oks_batch() {
        let kps = vec![[5.0, 5.0], [10.0, 10.0]];
        let vis = vec![2, 2];
        let sigmas = vec![0.05, 0.05];
        let ann = make_annotation(kps.clone(), vis.clone(), 20.0);
        let predictions = vec![ann.clone(), ann.clone()];
        let ground_truths = vec![ann.clone(), ann];
        let moks = mean_oks(&predictions, &ground_truths, &sigmas).expect("should succeed");
        assert!(
            (moks - 1.0).abs() < 1e-10,
            "mean OKS for perfect predictions should be 1.0, got {moks}"
        );
    }

    #[test]
    fn test_mean_keypoint_error_perfect() {
        let kps = vec![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let vis = vec![2, 2, 2];
        let err = mean_keypoint_error(&kps, &kps, &vis).expect("should succeed");
        assert!(
            err.abs() < 1e-12,
            "perfect predictions → error = 0, got {err}"
        );
    }

    #[test]
    fn test_coco_sigmas_returns_17() {
        let s = coco_sigmas();
        assert_eq!(s.len(), 17, "COCO has 17 body keypoints, got {}", s.len());
        for (i, &sigma) in s.iter().enumerate() {
            assert!(sigma > 0.0, "sigma[{i}] must be positive, got {sigma}");
        }
    }

    #[test]
    fn test_oks_invisible_keypoints_excluded() {
        // GT visibility = 0 for first keypoint → should not contribute to denominator
        let pred_kps = vec![[999.0, 999.0], [10.0, 10.0]];
        let gt_kps = vec![[0.0, 0.0], [10.0, 10.0]];
        let vis_gt = vec![0, 2]; // first invisible
        let sigmas = vec![0.05, 0.05];
        let pred = KeypointAnnotation {
            keypoints: pred_kps,
            visibility: vec![2, 2],
            scale: 50.0,
        };
        let gt = KeypointAnnotation {
            keypoints: gt_kps,
            visibility: vis_gt,
            scale: 50.0,
        };
        let oks = object_keypoint_similarity(&pred, &gt, &sigmas).expect("should succeed");
        // Only second keypoint evaluated; perfect match → OKS = 1.0
        assert!(
            (oks - 1.0).abs() < 1e-10,
            "invisible GT keypoints should be excluded, OKS={oks}"
        );
    }
}
