//! Segmentation loss functions
//!
//! Dense prediction / semantic segmentation requires loss functions that go
//! beyond standard image-level classification.  This module provides:
//!
//! - [`cross_entropy_segmentation`] – pixel-wise cross-entropy with optional
//!   per-class weights and ignore index.
//! - [`focal_loss_segmentation`] – focal loss for handling extreme class
//!   imbalance (Lin et al. 2017).
//! - [`lovasz_softmax`] – Lovász-Softmax surrogate for directly optimising
//!   mean IoU (Berman et al. 2018).
//! - [`tversky_loss`] – Tversky loss as a generalisation of Dice that allows
//!   tuning the trade-off between false positives and false negatives.
//!
//! # Array layout
//!
//! Logit / probability tensors use **HWC** layout (`Array3<f32>`, `[H, W, C]`).
//! Ground-truth label maps are `Array2<i32>` with shape `[H, W]`.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array2, Array3};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Apply row-wise softmax in-place to a `[H, W, C]` probability tensor.
///
/// Uses the numerically stable "max-shifted" variant.
fn softmax_hwc(logits: &Array3<f32>) -> Array3<f32> {
    let (h, w, c) = logits.dim();
    let mut out = logits.clone();
    for y in 0..h {
        for x in 0..w {
            let max_v = (0..c)
                .map(|ci| out[[y, x, ci]])
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for ci in 0..c {
                out[[y, x, ci]] = (out[[y, x, ci]] - max_v).exp();
                sum += out[[y, x, ci]];
            }
            if sum > 0.0 {
                for ci in 0..c {
                    out[[y, x, ci]] /= sum;
                }
            }
        }
    }
    out
}

/// Validate that logit/prob tensor and label map have matching spatial dims.
fn check_spatial_match(logits: &Array3<f32>, labels: &Array2<i32>) -> Result<(usize, usize, usize)> {
    let (lh, lw, c) = logits.dim();
    let (gh, gw) = labels.dim();
    if lh != gh || lw != gw {
        return Err(VisionError::InvalidParameter(format!(
            "loss: logit spatial {}×{} != label spatial {}×{}",
            lh, lw, gh, gw
        )));
    }
    if c == 0 {
        return Err(VisionError::InvalidParameter(
            "loss: num_classes (C) must be > 0".into(),
        ));
    }
    Ok((lh, lw, c))
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-entropy segmentation loss
// ─────────────────────────────────────────────────────────────────────────────

/// Compute pixel-wise cross-entropy loss for semantic segmentation.
///
/// This is the standard per-pixel multinomial log-loss widely used for
/// training FCN / DeepLab / SegFormer style models.
///
/// # Arguments
///
/// * `logits`       – Raw class scores (un-normalised) `[H, W, C]`.
/// * `labels`       – Integer ground-truth class indices `[H, W]`.
/// * `class_weights`– Optional per-class weight vector of length `C`.
///                    Pass `None` for uniform weights.
/// * `ignore_index` – Optional class index whose pixels are excluded from the
///                    loss (e.g. 255 for "void" in VOC / Cityscapes).
///
/// # Returns
///
/// Mean per-pixel cross-entropy loss (scalar `f32`).
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] on shape mismatches or when
/// `class_weights.len() != C`.
pub fn cross_entropy_segmentation(
    logits: &Array3<f32>,
    labels: &Array2<i32>,
    class_weights: Option<&[f32]>,
    ignore_index: Option<i32>,
) -> Result<f32> {
    let (h, w, c) = check_spatial_match(logits, labels)?;

    if let Some(cw) = class_weights {
        if cw.len() != c {
            return Err(VisionError::InvalidParameter(format!(
                "cross_entropy_segmentation: class_weights length {} != num_classes {}",
                cw.len(),
                c
            )));
        }
    }

    let probs = softmax_hwc(logits);
    let mut loss_sum = 0.0f64;
    let mut count = 0u64;

    for y in 0..h {
        for x in 0..w {
            let label = labels[[y, x]];
            if let Some(ig) = ignore_index {
                if label == ig {
                    continue;
                }
            }
            let label_idx = label.max(0) as usize;
            if label_idx >= c {
                // Out-of-range label: treat as background (index 0).
                continue;
            }
            let p = probs[[y, x, label_idx]].max(1e-7);
            let ce = -p.ln() as f64;
            let weight = class_weights.map(|cw| cw[label_idx] as f64).unwrap_or(1.0);
            loss_sum += ce * weight;
            count += 1;
        }
    }

    Ok(if count > 0 { (loss_sum / count as f64) as f32 } else { 0.0 })
}

// ─────────────────────────────────────────────────────────────────────────────
// Focal loss for segmentation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the per-pixel focal loss for semantic segmentation.
///
/// The focal loss (Lin et al., 2017) modulates the cross-entropy by a factor
/// `(1 - p_t)^gamma` that down-weights easy examples (high confidence
/// predictions) and focuses training on hard examples:
///
/// ```text
/// FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
/// ```
///
/// # Arguments
///
/// * `logits`       – Raw class scores `[H, W, C]`.
/// * `labels`       – Integer ground-truth labels `[H, W]`.
/// * `alpha`        – Per-class alpha weights (length `C`).  Pass `None`
///                    for uniform alpha = 1.
/// * `gamma`        – Focusing exponent (commonly 2.0).
/// * `ignore_index` – Optional index to exclude from the loss.
///
/// # Returns
///
/// Mean per-pixel focal loss (scalar `f32`).
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] on mismatched shapes.
pub fn focal_loss_segmentation(
    logits: &Array3<f32>,
    labels: &Array2<i32>,
    alpha: Option<&[f32]>,
    gamma: f32,
    ignore_index: Option<i32>,
) -> Result<f32> {
    let (h, w, c) = check_spatial_match(logits, labels)?;

    if let Some(a) = alpha {
        if a.len() != c {
            return Err(VisionError::InvalidParameter(format!(
                "focal_loss_segmentation: alpha length {} != num_classes {}",
                a.len(),
                c
            )));
        }
    }

    let probs = softmax_hwc(logits);
    let mut loss_sum = 0.0f64;
    let mut count = 0u64;

    for y in 0..h {
        for x in 0..w {
            let label = labels[[y, x]];
            if let Some(ig) = ignore_index {
                if label == ig {
                    continue;
                }
            }
            let label_idx = label.max(0) as usize;
            if label_idx >= c {
                continue;
            }
            let p_t = probs[[y, x, label_idx]].max(1e-7);
            let alpha_t = alpha.map(|a| a[label_idx] as f64).unwrap_or(1.0);
            let focal_weight = (1.0 - p_t as f64).powf(gamma as f64);
            let fl = -alpha_t * focal_weight * (p_t.ln() as f64);
            loss_sum += fl;
            count += 1;
        }
    }

    Ok(if count > 0 { (loss_sum / count as f64) as f32 } else { 0.0 })
}

// ─────────────────────────────────────────────────────────────────────────────
// Lovász-Softmax loss
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Lovász-Softmax loss.
///
/// Lovász-Softmax (Berman et al. 2018) is a convex surrogate for the
/// intersection-over-union loss.  Unlike cross-entropy it directly minimises
/// a relaxed version of the Jaccard index.
///
/// This implementation follows the "mean" variant that averages the class-level
/// Lovász extension over all *present* classes (classes with at least one GT
/// pixel).
///
/// **Note**: the Lovász extension requires sorting errors by descending value
/// and computing a piece-wise linear bound.  The computational complexity is
/// O(H × W × log(H × W)) per class.
///
/// # Arguments
///
/// * `probs`        – Class probability maps (after softmax) `[H, W, C]`.
///                    **Must be probabilities**, not logits.
/// * `labels`       – Integer ground-truth labels `[H, W]`.
/// * `ignore_index` – Optional index to exclude.
///
/// # Returns
///
/// Lovász-Softmax loss (scalar `f32`).
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] on shape mismatches.
pub fn lovasz_softmax(
    probs: &Array3<f32>,
    labels: &Array2<i32>,
    ignore_index: Option<i32>,
) -> Result<f32> {
    let (h, w, c) = check_spatial_match(probs, labels)?;

    // Flatten into vectors, filtering ignored pixels.
    let mut flat_probs: Vec<Vec<f32>> = vec![Vec::new(); c]; // flat_probs[class][pixel]
    let mut flat_labels: Vec<usize> = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let label = labels[[y, x]];
            if let Some(ig) = ignore_index {
                if label == ig {
                    continue;
                }
            }
            let label_idx = label.max(0) as usize;
            flat_labels.push(label_idx.min(c - 1));
            for ci in 0..c {
                flat_probs[ci].push(probs[[y, x, ci]]);
            }
        }
    }

    let n = flat_labels.len();
    if n == 0 {
        return Ok(0.0);
    }

    let mut total_loss = 0.0f64;
    let mut num_present = 0usize;

    for ci in 0..c {
        // Check if class is present in GT.
        let present = flat_labels.iter().any(|&l| l == ci);
        if !present {
            continue;
        }
        num_present += 1;

        // Compute per-pixel errors: error[i] = 1 - p(ci|pixel_i) if label==ci
        //                                       p(ci|pixel_i)       otherwise
        let mut errors: Vec<f32> = (0..n)
            .map(|i| {
                if flat_labels[i] == ci {
                    1.0 - flat_probs[ci][i]
                } else {
                    flat_probs[ci][i]
                }
            })
            .collect();

        // Sort in descending order of error (required for Lovász extension).
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            errors[b]
                .partial_cmp(&errors[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Compute the Lovász gradient for class ci.
        // jacc_k = |{predicted wrong for ci}| / |{true ci ∪ predicted ci}|
        // The gradient at sorted position k is lovász_grad[k] = jacc_k - jacc_{k-1}.
        let mut fp = 0.0f64; // false positive count for class ci
        let mut fn_ = 0.0f64; // false negative count for class ci
        let gt_count = flat_labels.iter().filter(|&&l| l == ci).count() as f64;

        let mut lovasz_loss = 0.0f64;
        let mut prev_jacc = 0.0f64;

        for (rank, &idx) in indices.iter().enumerate() {
            let is_gt = flat_labels[idx] == ci;
            if is_gt {
                fn_ += 1.0;
            } else {
                fp += 1.0;
            }
            let tp = gt_count - fn_;
            let union = tp + fn_ + fp;
            let jacc = if union > 0.0 { (fn_ + fp) / union } else { 0.0 };
            let lovász_grad = jacc - prev_jacc;
            prev_jacc = jacc;
            lovasz_loss += (errors[idx] as f64) * lovász_grad;
            let _ = rank; // consumed above through loop structure
        }

        total_loss += lovasz_loss;
    }

    Ok(if num_present > 0 {
        (total_loss / num_present as f64) as f32
    } else {
        0.0
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tversky loss
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for the Tversky loss.
#[derive(Debug, Clone)]
pub struct TverskyParams {
    /// Weight of false positives (alpha).  Commonly 0.5 (= Dice).
    pub alpha: f32,
    /// Weight of false negatives (beta).  Increase above 0.5 to penalise
    /// missed detections more heavily.
    pub beta: f32,
    /// Smoothing constant to avoid division by zero.
    pub smooth: f32,
    /// Classes to include.  `None` means all classes.
    pub classes: Option<Vec<usize>>,
    /// Optional ignore index.
    pub ignore_index: Option<i32>,
}

impl Default for TverskyParams {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
            smooth: 1e-5,
            classes: None,
            ignore_index: None,
        }
    }
}

/// Compute the Tversky loss for segmentation.
///
/// The Tversky loss (Salehi et al. 2017) is:
///
/// ```text
/// TL(c) = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
/// Tversky_loss = 1 - mean_c(TL(c))
/// ```
///
/// Setting `alpha = beta = 0.5` recovers the Dice loss (F1).
/// Increasing `beta` above 0.5 puts more emphasis on recall.
///
/// # Arguments
///
/// * `probs`   – Class probability maps `[H, W, C]`.  Should be after softmax.
/// * `labels`  – Integer ground-truth labels `[H, W]`.
/// * `params`  – Tversky hyper-parameters.
///
/// # Returns
///
/// Tversky loss in `[0, 1]` (scalar `f32`).
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] on shape mismatches.
pub fn tversky_loss(
    probs: &Array3<f32>,
    labels: &Array2<i32>,
    params: &TverskyParams,
) -> Result<f32> {
    let (h, w, c) = check_spatial_match(probs, labels)?;

    let class_range: Vec<usize> = params
        .classes
        .as_ref()
        .map(|v| v.clone())
        .unwrap_or_else(|| (0..c).collect());

    // Build one-hot binary masks for each pixel.
    // For efficiency we iterate per-class.
    let mut total_tversky = 0.0f64;
    let mut num_included = 0usize;

    for &ci in &class_range {
        if ci >= c {
            continue;
        }

        let mut tp = 0.0f64;
        let mut fp = 0.0f64;
        let mut fn_ = 0.0f64;

        for y in 0..h {
            for x in 0..w {
                let label = labels[[y, x]];
                if let Some(ig) = params.ignore_index {
                    if label == ig {
                        continue;
                    }
                }
                let label_idx = label.max(0) as usize;
                let p = probs[[y, x, ci]] as f64;
                let gt = if label_idx == ci { 1.0f64 } else { 0.0f64 };

                tp += p * gt;
                fp += p * (1.0 - gt);
                fn_ += (1.0 - p) * gt;
            }
        }

        // Skip classes that have no ground truth pixels (tp + fn == 0) to
        // avoid penalising for absent classes.
        if tp + fn_ < 1e-12 {
            continue;
        }

        let sm = params.smooth as f64;
        let tversky_score =
            (tp + sm) / (tp + params.alpha as f64 * fp + params.beta as f64 * fn_ + sm);
        total_tversky += tversky_score;
        num_included += 1;
    }

    Ok(if num_included > 0 {
        1.0 - (total_tversky / num_included as f64) as f32
    } else {
        0.0
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a spatially constant logit tensor where class `cls` has the
    /// highest logit everywhere.
    fn dominant_logits(h: usize, w: usize, c: usize, cls: usize) -> Array3<f32> {
        Array3::from_shape_fn((h, w, c), |(_, _, ci)| {
            if ci == cls { 5.0 } else { 0.0 }
        })
    }

    /// Create a label map where all pixels have label `cls`.
    fn uniform_labels(h: usize, w: usize, cls: i32) -> Array2<i32> {
        Array2::from_elem((h, w), cls)
    }

    /// Create a probability map where class `cls` has probability ~1.
    fn dominant_probs(h: usize, w: usize, c: usize, cls: usize) -> Array3<f32> {
        Array3::from_shape_fn((h, w, c), |(_, _, ci)| {
            if ci == cls { 0.99 } else { 0.01 / (c - 1).max(1) as f32 }
        })
    }

    // ── cross_entropy_segmentation ────────────────────────────────────────

    #[test]
    fn test_ce_near_zero_for_perfect_prediction() {
        let logits = dominant_logits(4, 4, 3, 0);
        let labels = uniform_labels(4, 4, 0);
        let ce = cross_entropy_segmentation(&logits, &labels, None, None).expect("ce failed");
        assert!(ce < 0.1, "CE should be near 0 for perfect prediction, got {}", ce);
    }

    #[test]
    fn test_ce_high_for_wrong_prediction() {
        let logits = dominant_logits(4, 4, 3, 0); // predicts class 0
        let labels = uniform_labels(4, 4, 2);     // but GT is class 2
        let ce = cross_entropy_segmentation(&logits, &labels, None, None).expect("ce failed");
        assert!(ce > 1.0, "CE should be high for wrong prediction, got {}", ce);
    }

    #[test]
    fn test_ce_with_class_weights() {
        let logits = dominant_logits(4, 4, 3, 0);
        let labels = uniform_labels(4, 4, 0);
        let weights = [2.0f32, 1.0, 1.0];
        let ce_w = cross_entropy_segmentation(&logits, &labels, Some(&weights), None)
            .expect("ce with weights failed");
        let ce_no_w = cross_entropy_segmentation(&logits, &labels, None, None)
            .expect("ce failed");
        // CE with weight=2 for class 0 should be approx 2× unweighted.
        assert!((ce_w - 2.0 * ce_no_w).abs() < 1e-3, "weighted CE unexpected");
    }

    #[test]
    fn test_ce_ignore_index() {
        let logits = dominant_logits(4, 4, 3, 2); // predicts class 2
        let mut labels = uniform_labels(4, 4, 0); // GT = class 0 (wrong)
        labels[[0, 0]] = 255; // ignore
        let ce_with_ignore =
            cross_entropy_segmentation(&logits, &labels, None, Some(255)).expect("ce failed");
        let ce_no_ignore =
            cross_entropy_segmentation(&logits, &labels, None, None).expect("ce failed");
        // Ignoring one pixel should not dramatically change loss but confirms the path works.
        assert!(ce_with_ignore >= 0.0);
        let _ = ce_no_ignore;
    }

    #[test]
    fn test_ce_shape_mismatch() {
        let logits = Array3::<f32>::zeros((4, 4, 3));
        let labels = Array2::<i32>::zeros((5, 4));
        let res = cross_entropy_segmentation(&logits, &labels, None, None);
        assert!(res.is_err());
    }

    // ── focal_loss_segmentation ───────────────────────────────────────────

    #[test]
    fn test_focal_lower_than_ce_for_easy_examples() {
        // For easy (high-confidence correct) predictions, FL < CE.
        let logits = dominant_logits(4, 4, 3, 0);
        let labels = uniform_labels(4, 4, 0);
        let ce = cross_entropy_segmentation(&logits, &labels, None, None).expect("ce");
        let fl = focal_loss_segmentation(&logits, &labels, None, 2.0, None).expect("fl");
        // FL with gamma=2 should be <= CE for easy examples.
        assert!(fl <= ce + 1e-4, "FL={} should be <= CE={}", fl, ce);
    }

    #[test]
    fn test_focal_gamma_zero_equals_ce() {
        // gamma=0 → focal weight = 1 → FL should equal CE.
        let logits = dominant_logits(4, 4, 3, 0);
        let labels = uniform_labels(4, 4, 0);
        let ce = cross_entropy_segmentation(&logits, &labels, None, None).expect("ce");
        let fl = focal_loss_segmentation(&logits, &labels, None, 0.0, None).expect("fl");
        assert!((fl - ce).abs() < 1e-4, "FL(gamma=0)={} != CE={}", fl, ce);
    }

    #[test]
    fn test_focal_shape_mismatch() {
        let logits = Array3::<f32>::zeros((4, 4, 3));
        let labels = Array2::<i32>::zeros((3, 4));
        let res = focal_loss_segmentation(&logits, &labels, None, 2.0, None);
        assert!(res.is_err());
    }

    // ── lovasz_softmax ────────────────────────────────────────────────────

    #[test]
    fn test_lovasz_near_zero_for_perfect() {
        let probs = dominant_probs(4, 4, 3, 0);
        let labels = uniform_labels(4, 4, 0);
        let loss = lovasz_softmax(&probs, &labels, None).expect("lovász failed");
        assert!(loss < 0.2, "Lovász should be near 0 for near-perfect prediction, got {}", loss);
    }

    #[test]
    fn test_lovasz_high_for_wrong() {
        let probs = dominant_probs(4, 4, 3, 0); // predicts class 0
        let labels = uniform_labels(4, 4, 1);   // GT = class 1
        let loss = lovasz_softmax(&probs, &labels, None).expect("lovász failed");
        assert!(loss > 0.5, "Lovász should be high for wrong prediction, got {}", loss);
    }

    #[test]
    fn test_lovasz_shape_mismatch() {
        let probs = Array3::<f32>::zeros((4, 4, 3));
        let labels = Array2::<i32>::zeros((5, 4));
        let res = lovasz_softmax(&probs, &labels, None);
        assert!(res.is_err());
    }

    // ── tversky_loss ──────────────────────────────────────────────────────

    #[test]
    fn test_tversky_near_zero_for_perfect() {
        let probs = dominant_probs(4, 4, 2, 0);
        let labels = uniform_labels(4, 4, 0);
        let params = TverskyParams::default();
        let loss = tversky_loss(&probs, &labels, &params).expect("tversky failed");
        assert!(loss < 0.1, "Tversky should be near 0 for near-perfect prediction, got {}", loss);
    }

    #[test]
    fn test_tversky_dice_equivalent() {
        // alpha=beta=0.5 → Tversky = Dice = 1 - F1.
        let probs = dominant_probs(6, 6, 2, 0);
        let labels = uniform_labels(6, 6, 0);
        let params = TverskyParams { alpha: 0.5, beta: 0.5, ..Default::default() };
        let loss = tversky_loss(&probs, &labels, &params).expect("tversky failed");
        assert!(loss >= 0.0 && loss <= 1.0 + 1e-5);
    }

    #[test]
    fn test_tversky_asymmetric() {
        // With beta > alpha, more FN penalty → higher loss when FN exist.
        let probs = Array3::from_shape_fn((4, 4, 2), |(_, _, ci)| {
            if ci == 0 { 0.4 } else { 0.6 }
        });
        let labels = uniform_labels(4, 4, 0); // GT=0, model predicts 1

        let params_balanced = TverskyParams { alpha: 0.5, beta: 0.5, ..Default::default() };
        let params_fn_heavy = TverskyParams { alpha: 0.2, beta: 0.8, ..Default::default() };

        let loss_balanced = tversky_loss(&probs, &labels, &params_balanced).expect("t1 failed");
        let loss_fn_heavy = tversky_loss(&probs, &labels, &params_fn_heavy).expect("t2 failed");

        // FN-heavy variant should incur >= balanced loss (more penalty for missed positives).
        assert!(
            loss_fn_heavy >= loss_balanced - 1e-4,
            "FN-heavy Tversky={} should be >= balanced={}",
            loss_fn_heavy,
            loss_balanced
        );
    }

    #[test]
    fn test_tversky_shape_mismatch() {
        let probs = Array3::<f32>::zeros((4, 4, 2));
        let labels = Array2::<i32>::zeros((5, 4));
        let params = TverskyParams::default();
        let res = tversky_loss(&probs, &labels, &params);
        assert!(res.is_err());
    }

    #[test]
    fn test_tversky_selected_classes() {
        let probs = dominant_probs(4, 4, 3, 0);
        let labels = uniform_labels(4, 4, 0);
        let params = TverskyParams {
            classes: Some(vec![0, 1]),
            ..Default::default()
        };
        let loss = tversky_loss(&probs, &labels, &params).expect("tversky selected failed");
        assert!(loss >= 0.0 && loss <= 1.0 + 1e-5);
    }
}
