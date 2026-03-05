//! Extended loss functions for specialised training scenarios
//!
//! This module provides functional-style loss functions that complement the
//! struct-based losses in `scirs2_neural::losses`. All functions follow the
//! signature convention `fn loss(preds, targets, …) -> Result<f64>`.
//!
//! # Available functions
//! | Function | Use-case |
//! |---|---|
//! | [`focal_loss`]       | Class-imbalanced classification |
//! | [`dice_loss`]        | Image segmentation |
//! | [`tversky_loss`]     | Segmentation with tunable FP/FN trade-off |
//! | [`contrastive_loss`] | Metric learning with positive/negative pairs |
//! | [`triplet_loss`]     | Metric learning with anchor/positive/negative |
//!
//! All functions operate on **`ndarray::ArrayView`** slices so the caller
//! owns the data and there are no hidden copies.

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, IxDyn};

// ---------------------------------------------------------------------------
// Focal Loss (functional)
// ---------------------------------------------------------------------------

/// Focal loss for binary / multi-label classification with class imbalance.
///
/// For each element `(p, y)` in `preds` / `targets`:
/// ```text
/// FL = -α · (1 - p_t)^γ · log(p_t)
/// ```
/// where `p_t = p` when `y == 1`, `p_t = 1 - p` otherwise.
///
/// The loss is **averaged** over all elements.
///
/// # Arguments
/// * `preds`   – Predicted probabilities in `[0, 1]`, any shape (will be flattened)
/// * `targets` – Binary targets (0 or 1), same shape as `preds`
/// * `alpha`   – Scalar weighting factor (e.g. 0.25)
/// * `gamma`   – Focusing parameter (e.g. 2.0)
///
/// # Examples
/// ```
/// use scirs2_neural::losses::extended::focal_loss;
/// use scirs2_core::ndarray::array;
///
/// let preds   = array![0.9, 0.6, 0.2, 0.1].into_dyn();
/// let targets = array![1.0, 1.0, 0.0, 0.0].into_dyn();
/// let loss = focal_loss(&preds, &targets, 0.25, 2.0).expect("focal_loss failed");
/// assert!(loss >= 0.0);
/// ```
pub fn focal_loss(
    preds: &Array<f64, IxDyn>,
    targets: &Array<f64, IxDyn>,
    alpha: f64,
    gamma: f64,
) -> Result<f64> {
    if preds.shape() != targets.shape() {
        return Err(NeuralError::InferenceError(format!(
            "focal_loss: shape mismatch {:?} vs {:?}",
            preds.shape(),
            targets.shape()
        )));
    }
    let eps = 1e-12;
    let n = preds.len() as f64;
    if n == 0.0 {
        return Err(NeuralError::InferenceError(
            "focal_loss: empty tensors".to_string(),
        ));
    }
    let loss: f64 = preds
        .iter()
        .zip(targets.iter())
        .map(|(&p, &y)| {
            let p_clamped = p.max(eps).min(1.0 - eps);
            let p_t = if y > 0.5 { p_clamped } else { 1.0 - p_clamped };
            let alpha_t = if y > 0.5 { alpha } else { 1.0 - alpha };
            -alpha_t * (1.0 - p_t).powf(gamma) * p_t.ln()
        })
        .sum();
    Ok(loss / n)
}

// ---------------------------------------------------------------------------
// Dice Loss (functional)
// ---------------------------------------------------------------------------

/// Soft Dice coefficient loss for segmentation tasks.
///
/// ```text
/// Dice = 1 - (2 · Σ(p · y) + ε) / (Σp + Σy + ε)
/// ```
///
/// # Arguments
/// * `preds`   – Predicted probabilities in `[0, 1]`
/// * `targets` – Binary ground-truth mask (0 or 1)
///
/// # Examples
/// ```
/// use scirs2_neural::losses::extended::dice_loss;
/// use scirs2_core::ndarray::array;
///
/// let preds   = array![0.9, 0.8, 0.1, 0.2].into_dyn();
/// let targets = array![1.0, 1.0, 0.0, 0.0].into_dyn();
/// let loss = dice_loss(&preds, &targets).expect("dice_loss failed");
/// assert!(loss >= 0.0 && loss <= 1.0);
/// ```
pub fn dice_loss(preds: &Array<f64, IxDyn>, targets: &Array<f64, IxDyn>) -> Result<f64> {
    if preds.shape() != targets.shape() {
        return Err(NeuralError::InferenceError(format!(
            "dice_loss: shape mismatch {:?} vs {:?}",
            preds.shape(),
            targets.shape()
        )));
    }
    if preds.is_empty() {
        return Err(NeuralError::InferenceError(
            "dice_loss: empty tensors".to_string(),
        ));
    }
    let eps = 1e-8;
    let intersection: f64 = preds.iter().zip(targets.iter()).map(|(&p, &y)| p * y).sum();
    let sum_p: f64 = preds.iter().sum();
    let sum_y: f64 = targets.iter().sum();
    let dice = (2.0 * intersection + eps) / (sum_p + sum_y + eps);
    Ok(1.0 - dice)
}

// ---------------------------------------------------------------------------
// Tversky Loss (functional)
// ---------------------------------------------------------------------------

/// Tversky loss – a generalisation of Dice that separately weights false
/// positives (FP) and false negatives (FN).
///
/// ```text
/// Tversky = 1 - (TP + ε) / (TP + α·FP + β·FN + ε)
/// ```
///
/// Setting `alpha = beta = 0.5` recovers the Dice loss.
///
/// # Arguments
/// * `preds`   – Predicted probabilities in `[0, 1]`
/// * `targets` – Binary ground-truth mask
/// * `alpha`   – Weight for false positives
/// * `beta`    – Weight for false negatives
///
/// # Examples
/// ```
/// use scirs2_neural::losses::extended::tversky_loss;
/// use scirs2_core::ndarray::array;
///
/// let preds   = array![0.9, 0.8, 0.1, 0.05].into_dyn();
/// let targets = array![1.0, 1.0, 0.0, 0.0].into_dyn();
/// let loss = tversky_loss(&preds, &targets, 0.3, 0.7).expect("tversky_loss failed");
/// assert!(loss >= 0.0 && loss <= 1.0);
/// ```
pub fn tversky_loss(
    preds: &Array<f64, IxDyn>,
    targets: &Array<f64, IxDyn>,
    alpha: f64,
    beta: f64,
) -> Result<f64> {
    if preds.shape() != targets.shape() {
        return Err(NeuralError::InferenceError(format!(
            "tversky_loss: shape mismatch {:?} vs {:?}",
            preds.shape(),
            targets.shape()
        )));
    }
    if preds.is_empty() {
        return Err(NeuralError::InferenceError(
            "tversky_loss: empty tensors".to_string(),
        ));
    }
    let eps = 1e-8;
    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;
    let mut fn_ = 0.0_f64;
    for (&p, &y) in preds.iter().zip(targets.iter()) {
        tp += p * y;
        fp += p * (1.0 - y);
        fn_ += (1.0 - p) * y;
    }
    let tversky = (tp + eps) / (tp + alpha * fp + beta * fn_ + eps);
    Ok(1.0 - tversky)
}

// ---------------------------------------------------------------------------
// Contrastive Loss (functional)
// ---------------------------------------------------------------------------

/// Contrastive loss for metric learning.
///
/// For each pair `(e1_i, e2_i)` with similarity label `y_i ∈ {0, 1}`:
/// ```text
/// L = y · d² + (1 - y) · max(0, margin - d)²
/// ```
/// where `d = ‖e1 − e2‖₂`.
///
/// # Arguments
/// * `embeddings1` – First set of embeddings, shape `[batch, dim]`
/// * `embeddings2` – Second set of embeddings, shape `[batch, dim]`
/// * `labels`      – Similarity labels: 1 = similar, 0 = dissimilar, shape `[batch]`
/// * `margin`      – Minimum distance for dissimilar pairs
///
/// # Examples
/// ```
/// use scirs2_neural::losses::extended::contrastive_loss;
/// use scirs2_core::ndarray::{array, arr2};
///
/// let e1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();
/// let e2 = arr2(&[[1.0, 0.0], [1.0, 0.0]]).into_dyn();
/// let labels = array![1.0, 0.0].into_dyn(); // first similar, second dissimilar
/// let loss = contrastive_loss(&e1, &e2, &labels, 1.0).expect("contrastive_loss failed");
/// assert!(loss >= 0.0);
/// ```
pub fn contrastive_loss(
    embeddings1: &Array<f64, IxDyn>,
    embeddings2: &Array<f64, IxDyn>,
    labels: &Array<f64, IxDyn>,
    margin: f64,
) -> Result<f64> {
    let s1 = embeddings1.shape();
    let s2 = embeddings2.shape();
    let sl = labels.shape();
    if s1.len() != 2 || s2.len() != 2 {
        return Err(NeuralError::InferenceError(
            "contrastive_loss: embeddings must be 2-D [batch, dim]".to_string(),
        ));
    }
    if s1 != s2 {
        return Err(NeuralError::InferenceError(format!(
            "contrastive_loss: embedding shape mismatch {:?} vs {:?}",
            s1, s2
        )));
    }
    let batch = s1[0];
    if sl != &[batch] {
        return Err(NeuralError::InferenceError(format!(
            "contrastive_loss: labels shape {:?} does not match batch size {}",
            sl, batch
        )));
    }
    let dim = s1[1];
    let n = batch as f64;
    let mut total = 0.0_f64;
    for i in 0..batch {
        let mut dist_sq = 0.0_f64;
        for j in 0..dim {
            let d = embeddings1[[i, j]] - embeddings2[[i, j]];
            dist_sq += d * d;
        }
        let dist = dist_sq.sqrt();
        let y = labels[i];
        total += y * dist_sq + (1.0 - y) * (margin - dist).max(0.0).powi(2);
    }
    Ok(total / n)
}

// ---------------------------------------------------------------------------
// Triplet Loss (functional)
// ---------------------------------------------------------------------------

/// Triplet loss for metric learning.
///
/// For each triplet `(anchor_i, positive_i, negative_i)`:
/// ```text
/// L = max(0, d(a, p)² − d(a, n)² + margin)
/// ```
/// where `d` is the Euclidean distance.
///
/// # Arguments
/// * `anchor`   – Anchor embeddings, shape `[batch, dim]`
/// * `positive` – Positive embeddings (same class as anchor), shape `[batch, dim]`
/// * `negative` – Negative embeddings (different class), shape `[batch, dim]`
/// * `margin`   – Minimum margin between positive and negative distances
///
/// # Examples
/// ```
/// use scirs2_neural::losses::extended::triplet_loss;
/// use scirs2_core::ndarray::arr2;
///
/// let a = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();
/// let p = arr2(&[[0.9, 0.1], [0.1, 0.9]]).into_dyn();
/// let n = arr2(&[[0.0, 1.0], [1.0, 0.0]]).into_dyn();
/// let loss = triplet_loss(&a, &p, &n, 0.5).expect("triplet_loss failed");
/// assert!(loss >= 0.0);
/// ```
pub fn triplet_loss(
    anchor: &Array<f64, IxDyn>,
    positive: &Array<f64, IxDyn>,
    negative: &Array<f64, IxDyn>,
    margin: f64,
) -> Result<f64> {
    let sa = anchor.shape();
    let sp = positive.shape();
    let sn = negative.shape();
    if sa.len() != 2 {
        return Err(NeuralError::InferenceError(
            "triplet_loss: inputs must be 2-D [batch, dim]".to_string(),
        ));
    }
    if sa != sp || sa != sn {
        return Err(NeuralError::InferenceError(format!(
            "triplet_loss: shape mismatch anchor {:?}, positive {:?}, negative {:?}",
            sa, sp, sn
        )));
    }
    let batch = sa[0];
    let dim = sa[1];
    let n = batch as f64;
    let mut total = 0.0_f64;
    for i in 0..batch {
        let mut d_ap = 0.0_f64;
        let mut d_an = 0.0_f64;
        for j in 0..dim {
            let dap = anchor[[i, j]] - positive[[i, j]];
            let dan = anchor[[i, j]] - negative[[i, j]];
            d_ap += dap * dap;
            d_an += dan * dan;
        }
        total += (d_ap - d_an + margin).max(0.0);
    }
    Ok(total / n)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, arr2};

    // ---- focal_loss ----

    #[test]
    fn test_focal_loss_perfect_predictions() {
        // Perfect predictions → very low focal loss
        let preds = array![0.9999, 0.9999, 0.0001, 0.0001].into_dyn();
        let targets = array![1.0, 1.0, 0.0, 0.0].into_dyn();
        let loss = focal_loss(&preds, &targets, 0.25, 2.0).expect("focal_loss failed");
        assert!(loss < 1e-3, "Perfect predictions should have near-zero focal loss, got {}", loss);
    }

    #[test]
    fn test_focal_loss_hard_examples() {
        // Mis-classified examples → higher focal loss than standard cross-entropy
        let preds = array![0.1, 0.1].into_dyn();
        let targets = array![1.0, 1.0].into_dyn();
        let loss = focal_loss(&preds, &targets, 1.0, 0.0).expect("focal_loss failed");
        // gamma=0 recovers standard CE
        assert!(loss > 0.0, "Wrong predictions should have positive loss");
    }

    #[test]
    fn test_focal_loss_shape_mismatch() {
        let preds = array![0.5, 0.5, 0.5].into_dyn();
        let targets = array![1.0, 0.0].into_dyn();
        assert!(focal_loss(&preds, &targets, 0.25, 2.0).is_err());
    }

    // ---- dice_loss ----

    #[test]
    fn test_dice_loss_perfect() {
        let preds = array![1.0, 1.0, 0.0, 0.0].into_dyn();
        let targets = array![1.0, 1.0, 0.0, 0.0].into_dyn();
        let loss = dice_loss(&preds, &targets).expect("dice_loss failed");
        assert!(loss < 1e-6, "Perfect dice should be ~0, got {}", loss);
    }

    #[test]
    fn test_dice_loss_worst() {
        // Completely wrong: predict all 0 for all-1 target
        let preds = array![0.0, 0.0].into_dyn();
        let targets = array![1.0, 1.0].into_dyn();
        let loss = dice_loss(&preds, &targets).expect("dice_loss failed");
        assert!(loss > 0.9, "Worst dice should be near 1, got {}", loss);
    }

    #[test]
    fn test_dice_loss_range() {
        let preds = array![0.7, 0.3, 0.2, 0.8].into_dyn();
        let targets = array![1.0, 0.0, 0.0, 1.0].into_dyn();
        let loss = dice_loss(&preds, &targets).expect("dice_loss failed");
        assert!((0.0..=1.0).contains(&loss), "Dice loss must be in [0, 1], got {}", loss);
    }

    // ---- tversky_loss ----

    #[test]
    fn test_tversky_loss_symmetric_equals_dice() {
        let preds = array![0.8, 0.9, 0.1, 0.2].into_dyn();
        let targets = array![1.0, 1.0, 0.0, 0.0].into_dyn();
        let tversky = tversky_loss(&preds, &targets, 0.5, 0.5).expect("tversky failed");
        let dice = dice_loss(&preds, &targets).expect("dice failed");
        assert!((tversky - dice).abs() < 1e-6, "Tversky(0.5,0.5) should equal Dice");
    }

    #[test]
    fn test_tversky_loss_range() {
        let preds = array![0.6, 0.4, 0.3, 0.7].into_dyn();
        let targets = array![1.0, 1.0, 0.0, 0.0].into_dyn();
        let loss = tversky_loss(&preds, &targets, 0.3, 0.7).expect("tversky failed");
        assert!((0.0..=1.0).contains(&loss), "Tversky must be in [0, 1], got {}", loss);
    }

    #[test]
    fn test_tversky_loss_shape_mismatch() {
        let preds = array![0.5, 0.5].into_dyn();
        let targets = array![1.0, 0.0, 0.0].into_dyn();
        assert!(tversky_loss(&preds, &targets, 0.5, 0.5).is_err());
    }

    // ---- contrastive_loss ----

    #[test]
    fn test_contrastive_loss_identical_similar() {
        // Identical embeddings marked as similar → d=0 → loss=0
        let e1 = arr2(&[[1.0, 0.0]]).into_dyn();
        let e2 = arr2(&[[1.0, 0.0]]).into_dyn();
        let labels = array![1.0].into_dyn();
        let loss = contrastive_loss(&e1, &e2, &labels, 1.0).expect("contrastive failed");
        assert!(loss.abs() < 1e-10, "Identical similar pair should have ~0 loss, got {}", loss);
    }

    #[test]
    fn test_contrastive_loss_far_dissimilar() {
        // Embeddings far apart, marked dissimilar → low loss (margin satisfied)
        let e1 = arr2(&[[10.0, 0.0]]).into_dyn();
        let e2 = arr2(&[[0.0, 0.0]]).into_dyn();
        let labels = array![0.0].into_dyn();
        // margin = 1.0; d = 10.0 > margin → max(0, 1 - 10)^2 = 0
        let loss = contrastive_loss(&e1, &e2, &labels, 1.0).expect("contrastive failed");
        assert!(loss.abs() < 1e-10, "Far dissimilar pair should have ~0 loss, got {}", loss);
    }

    #[test]
    fn test_contrastive_loss_shape_error() {
        let e1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]).into_dyn();
        let e2 = arr2(&[[1.0, 0.0]]).into_dyn(); // wrong batch
        let labels = array![1.0, 0.0].into_dyn();
        assert!(contrastive_loss(&e1, &e2, &labels, 1.0).is_err());
    }

    // ---- triplet_loss ----

    #[test]
    fn test_triplet_loss_positive_closer() {
        // positive much closer to anchor than negative → loss = 0 (margin satisfied)
        let anchor = arr2(&[[0.0, 0.0]]).into_dyn();
        let positive = arr2(&[[0.01, 0.0]]).into_dyn();
        let negative = arr2(&[[10.0, 0.0]]).into_dyn();
        let loss = triplet_loss(&anchor, &positive, &negative, 0.5)
            .expect("triplet failed");
        assert!(loss.abs() < 1e-6, "Easy triplet should have ~0 loss, got {}", loss);
    }

    #[test]
    fn test_triplet_loss_violation() {
        // negative is as close as positive → large loss
        let anchor = arr2(&[[0.0, 0.0]]).into_dyn();
        let positive = arr2(&[[1.0, 0.0]]).into_dyn();
        let negative = arr2(&[[1.0, 0.0]]).into_dyn();
        // d_ap == d_an → loss = max(0, 0 + margin) = margin
        let margin = 0.5;
        let loss = triplet_loss(&anchor, &positive, &negative, margin)
            .expect("triplet failed");
        assert!((loss - margin).abs() < 1e-6, "Expected loss = margin = {}, got {}", margin, loss);
    }

    #[test]
    fn test_triplet_loss_non_negative() {
        let anchor = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).into_dyn();
        let positive = arr2(&[[1.1, 2.0, 3.0], [4.1, 5.0, 6.0]]).into_dyn();
        let negative = arr2(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).into_dyn();
        let loss = triplet_loss(&anchor, &positive, &negative, 1.0)
            .expect("triplet failed");
        assert!(loss >= 0.0, "Triplet loss must be non-negative, got {}", loss);
    }

    #[test]
    fn test_triplet_loss_shape_error() {
        let anchor = arr2(&[[0.0, 0.0]]).into_dyn();
        let positive = arr2(&[[0.1, 0.0], [0.2, 0.0]]).into_dyn(); // wrong batch
        let negative = arr2(&[[1.0, 0.0]]).into_dyn();
        assert!(triplet_loss(&anchor, &positive, &negative, 1.0).is_err());
    }
}
