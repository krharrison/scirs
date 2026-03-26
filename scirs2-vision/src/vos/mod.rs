//! Video Object Segmentation (VOS) foundations.
//!
//! Provides semi-supervised VOS: given a binary object mask in frame 0,
//! propagate it to subsequent frames using a matching-based attention mechanism
//! inspired by STM / STCN.
//!
//! ## Overview
//!
//! 1. **Feature extraction** – downsample the frame to a coarser resolution
//!    and compute lightweight colour + spatial-pyramid features per cell.
//! 2. **Mask propagation** – compute soft attention between query and memory
//!    cell features, accumulate weighted memory masks, upsample to the original
//!    resolution.
//! 3. **Evaluation** – threshold to binary, compute mask IoU.

use crate::error::{Result, VisionError};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for video object segmentation.
#[derive(Debug, Clone)]
pub struct VosConfig {
    /// Maximum number of memory frames to keep.
    pub n_memory_frames: usize,
    /// Spatial downsampling factor (feature map has H/stride × W/stride cells).
    pub feature_stride: usize,
    /// Softmax temperature for attention computation.
    pub similarity_temperature: f64,
}

impl Default for VosConfig {
    fn default() -> Self {
        Self {
            n_memory_frames: 3,
            feature_stride: 4,
            similarity_temperature: 0.1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/// Binary object mask for a single frame.
#[derive(Debug, Clone)]
pub struct FrameMask {
    /// Index of the frame in the video sequence.
    pub frame_idx: usize,
    /// Binary mask: `mask[row][col]` is `true` if the pixel belongs to the
    /// foreground object.
    pub mask: Vec<Vec<bool>>,
}

impl FrameMask {
    /// Create a new `FrameMask` with the given frame index and mask data.
    pub fn new(frame_idx: usize, mask: Vec<Vec<bool>>) -> Self {
        Self { frame_idx, mask }
    }

    /// Return (height, width) of the mask.
    pub fn shape(&self) -> (usize, usize) {
        let h = self.mask.len();
        let w = if h > 0 { self.mask[0].len() } else { 0 };
        (h, w)
    }
}

/// Feature representation of a single frame at reduced resolution.
///
/// Shape: `(H/stride) × (W/stride) × feature_dim`.
#[derive(Debug, Clone)]
pub struct FrameFeatures {
    /// Index of the source frame.
    pub frame_idx: usize,
    /// Feature tensor stored as a nested Vec: `features[row][col]` is the
    /// feature vector for that cell.
    pub features: Vec<Vec<Vec<f64>>>,
}

impl FrameFeatures {
    /// Return `(feat_height, feat_width, feature_dim)`.
    pub fn shape(&self) -> (usize, usize, usize) {
        let fh = self.features.len();
        let fw = if fh > 0 { self.features[0].len() } else { 0 };
        let fd = if fh > 0 && fw > 0 {
            self.features[0][0].len()
        } else {
            0
        };
        (fh, fw, fd)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Feature extraction
// ─────────────────────────────────────────────────────────────────────────────

/// Extract mask-conditioned features from a video frame.
///
/// The frame is downsampled by `stride` in both spatial dimensions.  For each
/// cell at `(row, col)` in the feature map the corresponding `stride × stride`
/// patch is examined:
///
/// - **Mean colour** `[mean_r, mean_g, mean_b]` of the masked pixels inside the
///   patch (or the mean of all pixels if no foreground pixel is present).
/// - **2 × 2 spatial pyramid** – the patch is further divided into four
///   quadrants; each quadrant contributes the fraction of masked pixels in that
///   quadrant, yielding 4 values.
///
/// The resulting feature vector has length 7 (= 3 + 4).
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] if `frame` is empty or `stride`
/// is zero.
pub fn extract_mask_features(
    frame: &[Vec<[f64; 3]>],
    mask: &[Vec<bool>],
    stride: usize,
) -> Result<FrameFeatures> {
    if frame.is_empty() {
        return Err(VisionError::InvalidParameter(
            "frame must not be empty".into(),
        ));
    }
    if stride == 0 {
        return Err(VisionError::InvalidParameter("stride must be > 0".into()));
    }

    let h = frame.len();
    let w = frame[0].len();
    let fh = h.div_ceil(stride);
    let fw = w.div_ceil(stride);

    let mut feat_map: Vec<Vec<Vec<f64>>> = Vec::with_capacity(fh);

    for fr in 0..fh {
        let mut row_feats: Vec<Vec<f64>> = Vec::with_capacity(fw);
        for fc in 0..fw {
            let pr_start = fr * stride;
            let pr_end = (pr_start + stride).min(h);
            let pc_start = fc * stride;
            let pc_end = (pc_start + stride).min(w);

            // Collect mean colour of masked pixels in patch
            let mut sum_r = 0.0_f64;
            let mut sum_g = 0.0_f64;
            let mut sum_b = 0.0_f64;
            let mut mask_count = 0usize;
            let mut total_count = 0usize;

            // 2×2 quadrant counts
            let mid_r = pr_start + (pr_end - pr_start) / 2;
            let mid_c = pc_start + (pc_end - pc_start) / 2;
            let mut quad_mask = [0usize; 4];
            let mut quad_total = [0usize; 4];

            for (r, frame_row) in frame.iter().enumerate().take(pr_end).skip(pr_start) {
                for (c, pixel) in frame_row.iter().enumerate().take(pc_end).skip(pc_start) {
                    let is_masked = mask
                        .get(r)
                        .and_then(|row| row.get(c))
                        .copied()
                        .unwrap_or(false);
                    total_count += 1;

                    let q = match (r < mid_r, c < mid_c) {
                        (true, true) => 0,
                        (true, false) => 1,
                        (false, true) => 2,
                        (false, false) => 3,
                    };
                    quad_total[q] += 1;

                    if is_masked {
                        sum_r += pixel[0];
                        sum_g += pixel[1];
                        sum_b += pixel[2];
                        mask_count += 1;
                        quad_mask[q] += 1;
                    }
                }
            }

            let (mean_r, mean_g, mean_b) = if mask_count > 0 {
                let n = mask_count as f64;
                (sum_r / n, sum_g / n, sum_b / n)
            } else if total_count > 0 {
                // Fall back to mean of all pixels
                let n = total_count as f64;
                let (ar, ag, ab) = frame[pr_start..pr_end]
                    .iter()
                    .flat_map(|r| r[pc_start..pc_end].iter())
                    .fold((0.0_f64, 0.0_f64, 0.0_f64), |(ar, ag, ab), px| {
                        (ar + px[0], ag + px[1], ab + px[2])
                    });
                (ar / n, ag / n, ab / n)
            } else {
                (0.0, 0.0, 0.0)
            };

            // Spatial pyramid fractions
            let pyramid: Vec<f64> = (0..4)
                .map(|q| {
                    if quad_total[q] > 0 {
                        quad_mask[q] as f64 / quad_total[q] as f64
                    } else {
                        0.0
                    }
                })
                .collect();

            let mut fv = vec![mean_r, mean_g, mean_b];
            fv.extend_from_slice(&pyramid);
            row_feats.push(fv);
        }
        feat_map.push(row_feats);
    }

    Ok(FrameFeatures {
        frame_idx: 0,
        features: feat_map,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Dot product for feature vectors
// ─────────────────────────────────────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Mask propagation
// ─────────────────────────────────────────────────────────────────────────────

/// Propagate memory masks to the query frame using soft attention.
///
/// For each query cell `q_i` and each memory cell `m_j` the attention weight
/// is:
///
/// ```text
/// A[i, j] = exp(q_i · m_j / T) / Σ_k exp(q_i · m_k / T)
/// ```
///
/// The soft mask at query cell `i` is `Σ_j A[i,j] * mask_value(m_j)`.
/// The mask value of a memory cell is the fraction of foreground pixels in the
/// corresponding patch of the full-resolution binary mask.
///
/// The soft feature-map mask is then upsampled to the original frame
/// resolution using nearest-neighbour interpolation.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] if `memory_frames` and
/// `memory_features` are empty or of inconsistent length.
pub fn propagate_mask(
    memory_frames: &[FrameMask],
    memory_features: &[FrameFeatures],
    query_features: &FrameFeatures,
    config: &VosConfig,
) -> Result<Vec<Vec<f64>>> {
    if memory_frames.is_empty() || memory_features.is_empty() {
        return Err(VisionError::InvalidParameter(
            "memory_frames and memory_features must not be empty".into(),
        ));
    }
    if memory_frames.len() != memory_features.len() {
        return Err(VisionError::InvalidParameter(
            "memory_frames and memory_features must have the same length".into(),
        ));
    }

    let (qfh, qfw, _) = query_features.shape();
    let temp = config.similarity_temperature;

    // Build flattened memory cells: (feature_vec, mask_value)
    // mask_value = fraction of foreground pixels in the cell's patch.
    let mut mem_cells: Vec<(Vec<f64>, f64)> = Vec::new();

    for (mf, mfeat) in memory_frames.iter().zip(memory_features.iter()) {
        let (mh, mw) = mf.shape();
        let (fh, fw, _) = mfeat.shape();

        let stride_r = if fh > 0 { mh.div_ceil(fh) } else { 1 };
        let stride_c = if fw > 0 { mw.div_ceil(fw) } else { 1 };

        for fr in 0..fh {
            for fc in 0..fw {
                let fv = mfeat.features[fr][fc].clone();

                // Fraction of foreground pixels in the corresponding patch
                let pr_start = fr * stride_r;
                let pr_end = (pr_start + stride_r).min(mh);
                let pc_start = fc * stride_c;
                let pc_end = (pc_start + stride_c).min(mw);

                let mut fg_count = 0usize;
                let mut total = 0usize;
                for r in pr_start..pr_end {
                    for c in pc_start..pc_end {
                        total += 1;
                        if mf
                            .mask
                            .get(r)
                            .and_then(|row| row.get(c))
                            .copied()
                            .unwrap_or(false)
                        {
                            fg_count += 1;
                        }
                    }
                }
                let mask_val = if total > 0 {
                    fg_count as f64 / total as f64
                } else {
                    0.0
                };
                mem_cells.push((fv, mask_val));
            }
        }
    }

    if mem_cells.is_empty() {
        return Err(VisionError::InvalidParameter(
            "No memory cells available".into(),
        ));
    }

    // Compute soft mask at feature resolution (qfh × qfw)
    let mut soft_feat: Vec<Vec<f64>> = vec![vec![0.0; qfw]; qfh];

    for (qr, soft_feat_row) in soft_feat.iter_mut().enumerate().take(qfh) {
        for (qc, soft_feat_val) in soft_feat_row.iter_mut().enumerate().take(qfw) {
            let q = &query_features.features[qr][qc];

            // Attention weights (numerically stable softmax)
            let logits: Vec<f64> = mem_cells.iter().map(|(mv, _)| dot(q, mv) / temp).collect();
            let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum_exp: f64 = exps.iter().sum();

            let soft = if sum_exp > 0.0 {
                exps.iter()
                    .zip(mem_cells.iter())
                    .map(|(&e, (_, mv))| (e / sum_exp) * mv)
                    .sum::<f64>()
            } else {
                0.0_f64
            };
            *soft_feat_val = soft.clamp(0.0, 1.0);
        }
    }

    // Upsample to original resolution using nearest-neighbour
    // We don't know the original H×W exactly; infer from memory mask shape.
    let (orig_h, orig_w) = memory_frames[0].shape();
    let out_h = if orig_h > 0 { orig_h } else { qfh };
    let out_w = if orig_w > 0 { orig_w } else { qfw };
    let stride_h = config.feature_stride.max(1);
    let stride_w = config.feature_stride.max(1);

    let mut soft_mask: Vec<Vec<f64>> = vec![vec![0.0; out_w]; out_h];
    for (r, mask_row) in soft_mask.iter_mut().enumerate().take(out_h) {
        for (c, mask_val) in mask_row.iter_mut().enumerate().take(out_w) {
            let fr = (r / stride_h).min(qfh.saturating_sub(1));
            let fc = (c / stride_w).min(qfw.saturating_sub(1));
            *mask_val = soft_feat[fr][fc];
        }
    }

    Ok(soft_mask)
}

// ─────────────────────────────────────────────────────────────────────────────
// Thresholding and evaluation
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a soft probability mask to a binary mask by thresholding.
///
/// A pixel is foreground if `soft_mask[r][c] >= threshold`.
pub fn threshold_mask(soft_mask: &[Vec<f64>], threshold: f64) -> Vec<Vec<bool>> {
    soft_mask
        .iter()
        .map(|row| row.iter().map(|&v| v >= threshold).collect())
        .collect()
}

/// Compute the Intersection over Union between two binary masks.
///
/// Returns a value in `[0, 1]`.  Returns `1.0` if both masks are entirely
/// background (no positive pixels).
pub fn mask_iou(pred: &[Vec<bool>], gt: &[Vec<bool>]) -> f64 {
    let mut intersection = 0usize;
    let mut union_ = 0usize;

    let h = pred.len().min(gt.len());
    for r in 0..h {
        let w = pred[r].len().min(gt[r].len());
        for c in 0..w {
            let p = pred[r][c];
            let g = gt[r][c];
            if p && g {
                intersection += 1;
            }
            if p || g {
                union_ += 1;
            }
        }
    }

    if union_ == 0 {
        1.0
    } else {
        intersection as f64 / union_ as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(h: usize, w: usize) -> Vec<Vec<[f64; 3]>> {
        (0..h)
            .map(|r| {
                (0..w)
                    .map(|c| [(r as f64) / (h as f64), (c as f64) / (w as f64), 0.5])
                    .collect()
            })
            .collect()
    }

    fn make_mask(h: usize, w: usize, fg_rows: std::ops::Range<usize>) -> Vec<Vec<bool>> {
        (0..h)
            .map(|r| (0..w).map(|_| fg_rows.contains(&r)).collect())
            .collect()
    }

    #[test]
    fn test_extract_mask_features_shape() {
        let h = 16;
        let w = 20;
        let stride = 4;
        let frame = make_frame(h, w);
        let mask = make_mask(h, w, 4..12);
        let ff = extract_mask_features(&frame, &mask, stride).expect("extraction failed");
        let (fh, fw, fd) = ff.shape();
        // ceil(16/4)=4, ceil(20/4)=5
        assert_eq!(fh, 4, "fh mismatch");
        assert_eq!(fw, 5, "fw mismatch");
        assert_eq!(fd, 7, "fd mismatch: expected 3 colour + 4 spatial = 7");
    }

    #[test]
    fn test_extract_mask_features_zero_stride_error() {
        let frame = make_frame(8, 8);
        let mask = make_mask(8, 8, 0..4);
        assert!(extract_mask_features(&frame, &mask, 0).is_err());
    }

    #[test]
    fn test_extract_mask_features_empty_error() {
        let mask: Vec<Vec<bool>> = Vec::new();
        let frame: Vec<Vec<[f64; 3]>> = Vec::new();
        assert!(extract_mask_features(&frame, &mask, 4).is_err());
    }

    #[test]
    fn test_propagate_mask_soft_in_range() {
        let h = 8;
        let w = 8;
        let stride = 2;
        let frame = make_frame(h, w);
        let mask_data = make_mask(h, w, 2..6);

        let ff = extract_mask_features(&frame, &mask_data, stride).expect("extract failed");
        let fm = FrameMask::new(0, mask_data);

        let config = VosConfig {
            n_memory_frames: 1,
            feature_stride: stride,
            similarity_temperature: 0.1,
        };

        let soft = propagate_mask(&[fm], std::slice::from_ref(&ff), &ff, &config)
            .expect("propagation failed");

        for row in &soft {
            for &v in row {
                assert!((0.0..=1.0).contains(&v), "soft value out of range: {v}");
            }
        }
    }

    #[test]
    fn test_mask_iou_identical() {
        let mask = make_mask(8, 8, 2..6);
        let iou = mask_iou(&mask, &mask);
        assert!((iou - 1.0).abs() < 1e-9, "iou = {iou}");
    }

    #[test]
    fn test_mask_iou_disjoint() {
        let pred = make_mask(8, 8, 0..4);
        let gt = make_mask(8, 8, 4..8);
        let iou = mask_iou(&pred, &gt);
        assert_eq!(iou, 0.0, "iou = {iou}");
    }

    #[test]
    fn test_mask_iou_all_background() {
        let h = 4;
        let w = 4;
        let pred: Vec<Vec<bool>> = vec![vec![false; w]; h];
        let gt: Vec<Vec<bool>> = vec![vec![false; w]; h];
        let iou = mask_iou(&pred, &gt);
        assert_eq!(iou, 1.0);
    }

    #[test]
    fn test_threshold_mask() {
        let soft: Vec<Vec<f64>> = vec![vec![0.3, 0.7], vec![0.5, 0.2]];
        let binary = threshold_mask(&soft, 0.5);
        assert!(!binary[0][0]);
        assert!(binary[0][1]);
        assert!(binary[1][0]);
        assert!(!binary[1][1]);
    }

    #[test]
    fn test_frame_mask_shape() {
        let mask = make_mask(10, 12, 0..5);
        let fm = FrameMask::new(3, mask);
        assert_eq!(fm.shape(), (10, 12));
        assert_eq!(fm.frame_idx, 3);
    }
}
