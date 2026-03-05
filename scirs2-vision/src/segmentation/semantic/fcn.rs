//! FCN-lite: Fully Convolutional Network concepts for dense prediction
//!
//! This module implements key building blocks inspired by the seminal FCN paper
//! (Long et al. 2015) using pure-Rust ndarray operations:
//!
//! - [`bilinear_upsample_mask`] – bilinear interpolation for upsampling logit masks.
//! - [`compute_segmentation_metrics`] – per-class and mean IoU, pixel accuracy, Dice.
//! - [`dense_crf_post_process`] – simplified CRF-like bilateral / spatial smoothing.
//! - [`FCNConfig`] / [`FCNOutput`] – configuration and output containers.
//!
//! # Array layout
//!
//! All single-image tensors use **HWC** layout (`Array3<f32>`, shape `[H, W, C]`).
//! Batch tensors use **NHWC** layout (`Array4<f32>`, shape `[N, H, W, C]`).

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array2, Array3};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Backbone family for an FCN-style network.
///
/// At inference time these are conceptual labels; actual weights are not
/// bundled in this crate.  The backbone choice influences upsampling stride
/// defaults in [`FCNConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FCNBackbone {
    /// Lightweight 8-layer VGG-style backbone (stride-32 output).
    VGGLite,
    /// ResNet-18 style backbone (stride-32 output).
    ResNet18,
    /// MobileNet-v2 style backbone (stride-16 output).
    MobileNetV2,
}

/// Configuration for an FCN-lite model.
#[derive(Debug, Clone)]
pub struct FCNConfig {
    /// Number of semantic classes (including background).
    pub num_classes: usize,
    /// Number of input image channels (1 = gray, 3 = RGB).
    pub input_channels: usize,
    /// Backbone architecture.
    pub backbone: FCNBackbone,
    /// Overall spatial downsampling factor of the backbone.
    /// The upsampling stage will undo this factor.
    pub stride: usize,
    /// Whether to apply softmax to produce probability maps.
    pub apply_softmax: bool,
}

impl Default for FCNConfig {
    fn default() -> Self {
        Self {
            num_classes: 21,
            input_channels: 3,
            backbone: FCNBackbone::ResNet18,
            stride: 32,
            apply_softmax: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Output container
// ─────────────────────────────────────────────────────────────────────────────

/// Dense segmentation output from an FCN-style model.
#[derive(Debug, Clone)]
pub struct FCNOutput {
    /// Per-pixel class logits or probabilities with shape `[H, W, num_classes]`.
    pub logits: Array3<f32>,
    /// Argmax class label per pixel with shape `[H, W]`.
    pub class_map: Array2<usize>,
    /// Number of semantic classes.
    pub num_classes: usize,
}

impl FCNOutput {
    /// Build an `FCNOutput` from a raw logit tensor `[H, W, C]`.
    ///
    /// `argmax` is computed automatically.
    pub fn from_logits(logits: Array3<f32>) -> Result<Self> {
        let (h, w, c) = logits.dim();
        if c == 0 {
            return Err(VisionError::InvalidParameter(
                "logits must have at least one class channel".into(),
            ));
        }
        let mut class_map = Array2::zeros((h, w));
        for y in 0..h {
            for x in 0..w {
                let mut best_c = 0usize;
                let mut best_v = logits[[y, x, 0]];
                for ci in 1..c {
                    let v = logits[[y, x, ci]];
                    if v > best_v {
                        best_v = v;
                        best_c = ci;
                    }
                }
                class_map[[y, x]] = best_c;
            }
        }
        Ok(Self {
            logits,
            class_map,
            num_classes: c,
        })
    }

    /// Applies softmax along the class axis (in-place) and returns `self`.
    pub fn with_softmax(mut self) -> Self {
        let (h, w, c) = self.logits.dim();
        for y in 0..h {
            for x in 0..w {
                // Numerically stable softmax: subtract max first.
                let max_val = (0..c)
                    .map(|ci| self.logits[[y, x, ci]])
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for ci in 0..c {
                    let v = (self.logits[[y, x, ci]] - max_val).exp();
                    self.logits[[y, x, ci]] = v;
                    sum += v;
                }
                if sum > 0.0 {
                    for ci in 0..c {
                        self.logits[[y, x, ci]] /= sum;
                    }
                }
            }
        }
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilinear upsampling
// ─────────────────────────────────────────────────────────────────────────────

/// Bilinear upsampling for segmentation masks.
///
/// Upsamples a logit / probability tensor with shape `[H_in, W_in, C]` to
/// `[H_out, W_out, C]` using standard bilinear interpolation.  This is the
/// key operation that turns coarse feature-map predictions into full-resolution
/// segmentation maps in FCN-style architectures.
///
/// # Arguments
///
/// * `mask`   – Input tensor `[H_in, W_in, C]`.
/// * `out_h`  – Target height.
/// * `out_w`  – Target width.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when `out_h == 0` or `out_w == 0`.
pub fn bilinear_upsample_mask(
    mask: &Array3<f32>,
    out_h: usize,
    out_w: usize,
) -> Result<Array3<f32>> {
    let (in_h, in_w, c) = mask.dim();
    if out_h == 0 || out_w == 0 {
        return Err(VisionError::InvalidParameter(
            "bilinear_upsample_mask: output dimensions must be > 0".into(),
        ));
    }
    if in_h == 0 || in_w == 0 {
        return Err(VisionError::InvalidParameter(
            "bilinear_upsample_mask: input dimensions must be > 0".into(),
        ));
    }

    let mut out = Array3::<f32>::zeros((out_h, out_w, c));

    // Scale factors: map output pixel centre to input space.
    let scale_y = in_h as f32 / out_h as f32;
    let scale_x = in_w as f32 / out_w as f32;

    for oy in 0..out_h {
        // Centre-aligned mapping.
        let src_y = (oy as f32 + 0.5) * scale_y - 0.5;
        let y0 = (src_y.floor() as isize).max(0) as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let dy = src_y - src_y.floor();

        for ox in 0..out_w {
            let src_x = (ox as f32 + 0.5) * scale_x - 0.5;
            let x0 = (src_x.floor() as isize).max(0) as usize;
            let x1 = (x0 + 1).min(in_w - 1);
            let dx = src_x - src_x.floor();

            // Bilinear weights.
            let w00 = (1.0 - dy) * (1.0 - dx);
            let w01 = (1.0 - dy) * dx;
            let w10 = dy * (1.0 - dx);
            let w11 = dy * dx;

            for ci in 0..c {
                out[[oy, ox, ci]] = w00 * mask[[y0, x0, ci]]
                    + w01 * mask[[y0, x1, ci]]
                    + w10 * mask[[y1, x0, ci]]
                    + w11 * mask[[y1, x1, ci]];
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Segmentation metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Per-class and aggregate segmentation quality metrics.
#[derive(Debug, Clone)]
pub struct SegmentationMetrics {
    /// Intersection-over-Union per class.  Index `c` corresponds to class `c`.
    pub per_class_iou: Vec<f64>,
    /// Mean IoU over all classes present in ground truth.
    pub mean_iou: f64,
    /// Global pixel-accuracy: `correct / total`.
    pub pixel_accuracy: f64,
    /// Dice coefficient per class.
    pub per_class_dice: Vec<f64>,
    /// Mean Dice coefficient.
    pub mean_dice: f64,
}

/// Compute standard segmentation evaluation metrics.
///
/// # Arguments
///
/// * `pred`        – Predicted class-label map `[H, W]` (integer indices).
/// * `gt`          – Ground-truth label map `[H, W]`.
/// * `num_classes` – Total number of classes (used to size confusion matrix).
/// * `ignore_index`– Optional class index to exclude from metrics (e.g. 255 for
///   "void" / "ignore" labels in Cityscapes / PASCAL VOC).
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when dimensions of `pred` and `gt`
/// differ, or when `num_classes == 0`.
pub fn compute_segmentation_metrics(
    pred: &Array2<usize>,
    gt: &Array2<usize>,
    num_classes: usize,
    ignore_index: Option<usize>,
) -> Result<SegmentationMetrics> {
    let (h, w) = pred.dim();
    if pred.dim() != gt.dim() {
        return Err(VisionError::InvalidParameter(
            "compute_segmentation_metrics: pred and gt must have the same shape".into(),
        ));
    }
    if num_classes == 0 {
        return Err(VisionError::InvalidParameter(
            "compute_segmentation_metrics: num_classes must be > 0".into(),
        ));
    }

    // Build flat confusion matrix: conf[true_c * num_classes + pred_c].
    let mut conf = vec![0u64; num_classes * num_classes];
    let mut total_valid = 0u64;
    let mut correct = 0u64;

    for y in 0..h {
        for x in 0..w {
            let gt_c = gt[[y, x]];
            let pr_c = pred[[y, x]];
            if let Some(ig) = ignore_index {
                if gt_c == ig {
                    continue;
                }
            }
            // Clamp out-of-range predictions to a safe index.
            let safe_pr = pr_c.min(num_classes - 1);
            let safe_gt = gt_c.min(num_classes - 1);
            conf[safe_gt * num_classes + safe_pr] += 1;
            total_valid += 1;
            if safe_pr == safe_gt {
                correct += 1;
            }
        }
    }

    let pixel_accuracy = if total_valid > 0 {
        correct as f64 / total_valid as f64
    } else {
        1.0
    };

    // Per-class TP / FP / FN from the confusion matrix.
    let mut per_class_iou = vec![0.0f64; num_classes];
    let mut per_class_dice = vec![0.0f64; num_classes];
    let mut valid_count = 0usize;
    let mut iou_sum = 0.0f64;
    let mut dice_sum = 0.0f64;

    for c in 0..num_classes {
        let tp = conf[c * num_classes + c] as f64;
        // FP: predicted as c but not c.
        let fp: f64 = (0..num_classes)
            .filter(|&r| r != c)
            .map(|r| conf[r * num_classes + c] as f64)
            .sum();
        // FN: is c but predicted as something else.
        let fn_: f64 = (0..num_classes)
            .filter(|&p| p != c)
            .map(|p| conf[c * num_classes + p] as f64)
            .sum();

        let denom_iou = tp + fp + fn_;
        let denom_dice = 2.0 * tp + fp + fn_;

        if denom_iou > 0.0 {
            let iou = tp / denom_iou;
            per_class_iou[c] = iou;
            iou_sum += iou;
            valid_count += 1;
        }
        if denom_dice > 0.0 {
            per_class_dice[c] = 2.0 * tp / denom_dice;
        }
        dice_sum += per_class_dice[c];
    }

    let mean_iou = if valid_count > 0 {
        iou_sum / valid_count as f64
    } else {
        0.0
    };
    let mean_dice = if num_classes > 0 {
        dice_sum / num_classes as f64
    } else {
        0.0
    };

    Ok(SegmentationMetrics {
        per_class_iou,
        mean_iou,
        pixel_accuracy,
        per_class_dice,
        mean_dice,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense CRF post-processing
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for the simplified dense CRF smoothing pass.
#[derive(Debug, Clone)]
pub struct DenseCRFParams {
    /// Number of mean-field iterations.
    pub iterations: usize,
    /// Spatial Gaussian bandwidth (in pixels).
    pub spatial_sigma: f32,
    /// Appearance (colour) Gaussian bandwidth.
    /// Set to `None` to disable appearance term.
    pub appearance_sigma: Option<f32>,
    /// Weight of the spatial (smoothness) term.
    pub spatial_weight: f32,
    /// Weight of the appearance (bilateral) term.
    pub appearance_weight: f32,
}

impl Default for DenseCRFParams {
    fn default() -> Self {
        Self {
            iterations: 5,
            spatial_sigma: 3.0,
            appearance_sigma: Some(10.0),
            spatial_weight: 3.0,
            appearance_weight: 10.0,
        }
    }
}

/// Apply a simplified dense CRF-like spatial smoothing to probability maps.
///
/// This is an efficient bilateral-filter-inspired approximation of the full
/// dense CRF inference used by DeepLab.  It alternates between:
///
/// 1. **Spatial smoothing** – convolve probability maps with a Gaussian kernel
///    defined by `params.spatial_sigma`.
/// 2. **Appearance gating** – use RGB colour proximity (when `appearance_sigma`
///    is set and `rgb_image` is supplied) to downweight cross-class diffusion
///    across strong colour boundaries.
/// 3. **Compatibility transform + re-normalization** – subtract the current
///    prediction from the message to encode the Potts compatibility matrix and
///    re-normalise to a valid probability simplex.
///
/// # Arguments
///
/// * `prob_map`  – Per-pixel class probabilities `[H, W, C]`.  Each pixel's
///   values should sum to approximately 1.
/// * `rgb_image` – Optional RGB image `[H, W, 3]` used for appearance term.
/// * `params`    – CRF hyper-parameters.
///
/// # Returns
///
/// Refined probability map with the same shape as `prob_map`.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] on dimension mismatches.
pub fn dense_crf_post_process(
    prob_map: &Array3<f32>,
    rgb_image: Option<&Array3<f32>>,
    params: &DenseCRFParams,
) -> Result<Array3<f32>> {
    let (h, w, c) = prob_map.dim();
    if c == 0 {
        return Err(VisionError::InvalidParameter(
            "dense_crf_post_process: num_classes must be > 0".into(),
        ));
    }
    if let Some(rgb) = rgb_image {
        let (rh, rw, rc) = rgb.dim();
        if rh != h || rw != w {
            return Err(VisionError::InvalidParameter(
                "dense_crf_post_process: rgb_image spatial dimensions must match prob_map".into(),
            ));
        }
        if rc != 3 {
            return Err(VisionError::InvalidParameter(
                "dense_crf_post_process: rgb_image must have 3 channels".into(),
            ));
        }
    }

    let mut q = prob_map.clone();

    // Precompute spatial Gaussian kernel (1-D, applied separably).
    let sigma_s = params.spatial_sigma.max(0.1);
    let radius_s = (3.0 * sigma_s).ceil() as isize;
    let spatial_kernel: Vec<f32> = (-radius_s..=radius_s)
        .map(|d| (-(d as f32 * d as f32) / (2.0 * sigma_s * sigma_s)).exp())
        .collect();
    let kernel_sum: f32 = spatial_kernel.iter().sum();
    let spatial_kernel: Vec<f32> = spatial_kernel.iter().map(|v| v / kernel_sum).collect();

    for _iter in 0..params.iterations {
        // ── 1. Spatial Gaussian message passing ──────────────────────────────
        let mut msg_spatial = Array3::<f32>::zeros((h, w, c));

        // Horizontal pass.
        let mut tmp = Array3::<f32>::zeros((h, w, c));
        for y in 0..h {
            for x in 0..w {
                for (ki, &kv) in spatial_kernel.iter().enumerate() {
                    let dx = ki as isize - radius_s;
                    let nx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    for ci in 0..c {
                        tmp[[y, x, ci]] += kv * q[[y, nx, ci]];
                    }
                }
            }
        }
        // Vertical pass on tmp -> msg_spatial.
        for y in 0..h {
            for x in 0..w {
                for (ki, &kv) in spatial_kernel.iter().enumerate() {
                    let dy = ki as isize - radius_s;
                    let ny = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                    for ci in 0..c {
                        msg_spatial[[y, x, ci]] += kv * tmp[[ny, x, ci]];
                    }
                }
            }
        }

        // ── 2. Optional appearance (bilateral) term ───────────────────────────
        let msg_appearance =
            if let (Some(rgb), Some(sigma_a)) = (rgb_image, params.appearance_sigma) {
                let mut app = Array3::<f32>::zeros((h, w, c));
                let sigma_a2 = 2.0 * sigma_a * sigma_a;
                // Simplified: only integrate 3×3 colour-weighted neighbourhood.
                let r = 3usize;
                for y in 0..h {
                    for x in 0..w {
                        let mut w_sum = 0.0f32;
                        let mut acc = vec![0.0f32; c];
                        let y0 = y.saturating_sub(r);
                        let y1 = (y + r + 1).min(h);
                        let x0 = x.saturating_sub(r);
                        let x1 = (x + r + 1).min(w);
                        for ny in y0..y1 {
                            for nx in x0..x1 {
                                let dr = rgb[[y, x, 0]] - rgb[[ny, nx, 0]];
                                let dg = rgb[[y, x, 1]] - rgb[[ny, nx, 1]];
                                let db = rgb[[y, x, 2]] - rgb[[ny, nx, 2]];
                                let colour_dist2 = dr * dr + dg * dg + db * db;
                                let dy = (y as f32 - ny as f32).powi(2);
                                let dx2 = (x as f32 - nx as f32).powi(2);
                                let spatial_dist2 = dy + dx2;
                                let w = (-colour_dist2 / sigma_a2
                                    - spatial_dist2 / (2.0 * sigma_s * sigma_s))
                                    .exp();
                                for ci in 0..c {
                                    acc[ci] += w * q[[ny, nx, ci]];
                                }
                                w_sum += w;
                            }
                        }
                        if w_sum > 0.0 {
                            for ci in 0..c {
                                app[[y, x, ci]] = acc[ci] / w_sum;
                            }
                        }
                    }
                }
                Some(app)
            } else {
                None
            };

        // ── 3. Combine messages and apply Potts compatibility ─────────────────
        let mut new_q = Array3::<f32>::zeros((h, w, c));
        for y in 0..h {
            for x in 0..w {
                for ci in 0..c {
                    let spatial_msg = msg_spatial[[y, x, ci]];
                    let app_msg = msg_appearance
                        .as_ref()
                        .map(|a| a[[y, x, ci]])
                        .unwrap_or(0.0);

                    // Weighted combination.
                    let combined =
                        params.spatial_weight * spatial_msg + params.appearance_weight * app_msg;

                    // Potts compatibility: subtract current prediction ("self-inhibition").
                    let compat = combined - q[[y, x, ci]];

                    // Unary + compatibility message.
                    new_q[[y, x, ci]] = prob_map[[y, x, ci]] * (-compat).exp();
                }

                // ── 4. Re-normalise to simplex ─────────────────────────────────
                let sum: f32 = (0..c).map(|ci| new_q[[y, x, ci]]).sum();
                if sum > 1e-8 {
                    for ci in 0..c {
                        new_q[[y, x, ci]] /= sum;
                    }
                } else {
                    // Fallback: uniform distribution.
                    let uniform = 1.0 / c as f32;
                    for ci in 0..c {
                        new_q[[y, x, ci]] = uniform;
                    }
                }
            }
        }

        q = new_q;
    }

    Ok(q)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_probs(h: usize, w: usize, c: usize) -> Array3<f32> {
        Array3::from_elem((h, w, c), 1.0 / c as f32)
    }

    // ── FCNConfig / FCNOutput ──────────────────────────────────────────────

    #[test]
    fn test_fcn_config_default() {
        let cfg = FCNConfig::default();
        assert_eq!(cfg.num_classes, 21);
        assert_eq!(cfg.input_channels, 3);
        assert_eq!(cfg.stride, 32);
    }

    #[test]
    fn test_fcnoutput_from_logits() {
        // 4×4 image, 3 classes.
        let logits = Array3::from_shape_fn((4, 4, 3), |(y, x, c)| {
            if c == 0 {
                1.0
            } else if y > 1 && c == 1 {
                2.0
            } else {
                0.0
            }
        });
        let out = FCNOutput::from_logits(logits).expect("FCNOutput::from_logits failed");
        assert_eq!(out.class_map[[0, 0]], 0);
        assert_eq!(out.class_map[[2, 0]], 1);
    }

    #[test]
    fn test_fcnoutput_softmax() {
        let logits = Array3::from_elem((2, 2, 2), 1.0f32);
        let out = FCNOutput::from_logits(logits)
            .expect("from_logits")
            .with_softmax();
        for y in 0..2 {
            for x in 0..2 {
                let s: f32 = (0..2).map(|c| out.logits[[y, x, c]]).sum();
                assert!((s - 1.0).abs() < 1e-5, "softmax sum={}", s);
            }
        }
    }

    // ── bilinear_upsample_mask ────────────────────────────────────────────

    #[test]
    fn test_bilinear_upsample_trivial() {
        // Upsampling a constant map should preserve values.
        let mask = Array3::from_elem((4, 4, 3), 0.5f32);
        let up = bilinear_upsample_mask(&mask, 8, 8).expect("upsample failed");
        assert_eq!(up.dim(), (8, 8, 3));
        for v in up.iter() {
            assert!((*v - 0.5).abs() < 1e-4, "unexpected value {}", v);
        }
    }

    #[test]
    fn test_bilinear_upsample_noop() {
        // Same-size "upsampling" should approximate identity.
        let mask = Array3::from_shape_fn((3, 3, 2), |(y, x, c)| (y + x + c) as f32 * 0.1);
        let up = bilinear_upsample_mask(&mask, 3, 3).expect("upsample failed");
        assert_eq!(up.dim(), (3, 3, 2));
    }

    #[test]
    fn test_bilinear_upsample_invalid() {
        let mask = Array3::from_elem((4, 4, 2), 0.5f32);
        let res = bilinear_upsample_mask(&mask, 0, 8);
        assert!(res.is_err());
    }

    // ── compute_segmentation_metrics ─────────────────────────────────────

    #[test]
    fn test_metrics_perfect_prediction() {
        let gt = Array2::from_shape_fn((4, 4), |(y, x)| if y < 2 { 0 } else { 1 });
        let metrics =
            compute_segmentation_metrics(&gt.clone(), &gt, 2, None).expect("metrics failed");
        assert!((metrics.mean_iou - 1.0).abs() < 1e-9, "mIoU should be 1.0");
        assert!((metrics.pixel_accuracy - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_all_wrong() {
        let gt = Array2::from_shape_fn((4, 4), |(_, _)| 0usize);
        let pred = Array2::from_shape_fn((4, 4), |(_, _)| 1usize);
        let metrics = compute_segmentation_metrics(&pred, &gt, 2, None).expect("metrics failed");
        assert!(
            (metrics.pixel_accuracy).abs() < 1e-9,
            "accuracy should be 0.0"
        );
        // Class 0: tp=0, fn=16, fp=0 → iou = 0
        assert!((metrics.per_class_iou[0]).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_ignore_index() {
        let mut gt = Array2::from_shape_fn((4, 4), |(_, _)| 0usize);
        gt[[0, 0]] = 255; // ignore
        let pred = Array2::zeros((4, 4));
        let metrics =
            compute_segmentation_metrics(&pred, &gt, 2, Some(255)).expect("metrics failed");
        assert!((metrics.pixel_accuracy - 1.0).abs() < 1e-9);
    }

    // ── dense_crf_post_process ────────────────────────────────────────────

    #[test]
    fn test_dense_crf_shape_preserved() {
        let prob = make_uniform_probs(8, 8, 4);
        let params = DenseCRFParams {
            iterations: 2,
            ..Default::default()
        };
        let refined = dense_crf_post_process(&prob, None, &params).expect("crf failed");
        assert_eq!(refined.dim(), (8, 8, 4));
    }

    #[test]
    fn test_dense_crf_rows_sum_to_one() {
        let prob = make_uniform_probs(6, 6, 3);
        let params = DenseCRFParams {
            iterations: 3,
            ..Default::default()
        };
        let refined = dense_crf_post_process(&prob, None, &params).expect("crf failed");
        for y in 0..6 {
            for x in 0..6 {
                let s: f32 = (0..3).map(|c| refined[[y, x, c]]).sum();
                assert!((s - 1.0).abs() < 1e-4, "prob sum={} at ({},{})", s, y, x);
            }
        }
    }

    #[test]
    fn test_dense_crf_with_rgb() {
        let prob = make_uniform_probs(6, 6, 2);
        let rgb = Array3::<f32>::from_elem((6, 6, 3), 128.0);
        let params = DenseCRFParams {
            iterations: 2,
            appearance_sigma: Some(10.0),
            ..Default::default()
        };
        let refined =
            dense_crf_post_process(&prob, Some(&rgb), &params).expect("crf with rgb failed");
        assert_eq!(refined.dim(), (6, 6, 2));
    }

    #[test]
    fn test_dense_crf_invalid_rgb_channels() {
        let prob = make_uniform_probs(4, 4, 2);
        let bad_rgb = Array3::<f32>::zeros((4, 4, 1));
        let params = DenseCRFParams::default();
        let res = dense_crf_post_process(&prob, Some(&bad_rgb), &params);
        assert!(res.is_err());
    }
}
