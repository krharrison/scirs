//! Mask decoder: fuses image features and prompt embeddings to produce
//! segmentation masks, IoU predictions and stability scores.
//!
//! The decoder applies simplified cross-attention (prompt tokens attend to
//! image features), self-attention among tokens, and then up-samples through
//! transposed convolutions to produce masks at the original image resolution.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::Array2;

use super::prompt_encoder::PromptEmbedding;
use super::types::{SAMConfig, SegmentationResult};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Simplified single-head attention: `softmax(Q K^T / sqrt(d)) V`.
///
/// * `queries` – `[n_q, d]`
/// * `keys`    – `[n_k, d]`
/// * `values`  – `[n_k, d_v]`
///
/// Returns `[n_q, d_v]`.
fn simple_attention(
    queries: &Array2<f64>,
    keys: &Array2<f64>,
    values: &Array2<f64>,
) -> Result<Array2<f64>> {
    let (n_q, d) = queries.dim();
    let (n_k, kd) = keys.dim();
    let (n_v, d_v) = values.dim();

    if d != kd {
        return Err(VisionError::DimensionMismatch(format!(
            "attention: query dim {d} != key dim {kd}"
        )));
    }
    if n_k != n_v {
        return Err(VisionError::DimensionMismatch(format!(
            "attention: key count {n_k} != value count {n_v}"
        )));
    }

    let scale = 1.0 / (d as f64).sqrt();

    let mut out = Array2::zeros((n_q, d_v));
    for q in 0..n_q {
        // Compute attention scores.
        let mut scores = vec![0.0f64; n_k];
        let mut max_score = f64::NEG_INFINITY;
        for k in 0..n_k {
            let mut dot = 0.0f64;
            for i in 0..d {
                dot += queries[[q, i]] * keys[[k, i]];
            }
            scores[k] = dot * scale;
            if scores[k] > max_score {
                max_score = scores[k];
            }
        }

        // Softmax.
        let mut sum_exp = 0.0f64;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        if sum_exp > 0.0 {
            for s in &mut scores {
                *s /= sum_exp;
            }
        }

        // Weighted sum of values.
        for k in 0..n_k {
            for v in 0..d_v {
                out[[q, v]] += scores[k] * values[[k, v]];
            }
        }
    }
    Ok(out)
}

/// Simple MLP: `ReLU(x W1 + b1) W2 + b2`.
///
/// Uses deterministic pseudo-random initialisation.
fn mlp_forward(input: &Array2<f64>, hidden: usize, out_dim: usize) -> Array2<f64> {
    let (n, in_dim) = input.dim();

    // Generate pseudo-random weights.
    let gen = |total: usize, seed_start: u64, fan_in: usize| -> Vec<f64> {
        let std_dev = (2.0 / fan_in as f64).sqrt();
        let mut v = Vec::with_capacity(total);
        let mut s = seed_start;
        for _ in 0..total {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u = (s >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
            v.push(u * std_dev);
        }
        v
    };

    let w1_data = gen(in_dim * hidden, 123, in_dim);
    let w2_data = gen(hidden * out_dim, 456, hidden);

    // First layer.
    let mut h = Array2::zeros((n, hidden));
    for i in 0..n {
        for j in 0..hidden {
            let mut val = 0.0f64;
            for k in 0..in_dim {
                val += input[[i, k]] * w1_data[k * hidden + j];
            }
            h[[i, j]] = val.max(0.0); // ReLU
        }
    }

    // Second layer.
    let mut out = Array2::zeros((n, out_dim));
    for i in 0..n {
        for j in 0..out_dim {
            let mut val = 0.0f64;
            for k in 0..hidden {
                val += h[[i, k]] * w2_data[k * out_dim + j];
            }
            out[[i, j]] = val;
        }
    }
    out
}

/// Bilinear 2x up-sample a feature map stored as `[H*W, C]`.
///
/// Returns `(2H, 2W, Array2<f64>[4*H*W, C])`.
fn upsample_2x(input: &Array2<f64>, h: usize, w: usize) -> Result<(usize, usize, Array2<f64>)> {
    let (n, c) = input.dim();
    if n != h * w {
        return Err(VisionError::DimensionMismatch(format!(
            "upsample: n={n} != h*w={}",
            h * w
        )));
    }

    let oh = h * 2;
    let ow = w * 2;
    let mut out = Array2::zeros((oh * ow, c));

    for oy in 0..oh {
        for ox in 0..ow {
            // Source coordinate (fractional).
            let sy = (oy as f64) / 2.0;
            let sx = (ox as f64) / 2.0;

            let y0 = (sy.floor() as usize).min(h.saturating_sub(1));
            let x0 = (sx.floor() as usize).min(w.saturating_sub(1));
            let y1 = (y0 + 1).min(h.saturating_sub(1));
            let x1 = (x0 + 1).min(w.saturating_sub(1));

            let fy = sy - y0 as f64;
            let fx = sx - x0 as f64;

            let w00 = (1.0 - fy) * (1.0 - fx);
            let w01 = (1.0 - fy) * fx;
            let w10 = fy * (1.0 - fx);
            let w11 = fy * fx;

            let i00 = y0 * w + x0;
            let i01 = y0 * w + x1;
            let i10 = y1 * w + x0;
            let i11 = y1 * w + x1;
            let oi = oy * ow + ox;

            for ch in 0..c {
                out[[oi, ch]] = w00 * input[[i00, ch]]
                    + w01 * input[[i01, ch]]
                    + w10 * input[[i10, ch]]
                    + w11 * input[[i11, ch]];
            }
        }
    }
    Ok((oh, ow, out))
}

// ---------------------------------------------------------------------------
// MaskDecoder
// ---------------------------------------------------------------------------

/// Decodes image features + prompt embeddings into segmentation masks.
#[derive(Debug, Clone)]
pub struct MaskDecoder {
    /// Pipeline configuration.
    config: SAMConfig,
}

impl MaskDecoder {
    /// Create a new mask decoder.
    pub fn new(config: &SAMConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Decode masks from image features and prompt embeddings.
    ///
    /// # Arguments
    ///
    /// * `image_features`  - Encoder output `[spatial_tokens, channels]`.
    /// * `prompt_embedding` - Output of the prompt encoder.
    /// * `image_size`      - Original image `(height, width)`.
    ///
    /// # Returns
    ///
    /// A [`SegmentationResult`] with `num_mask_outputs` candidate masks at the
    /// original image resolution, together with IoU predictions and stability
    /// scores.
    pub fn decode(
        &self,
        image_features: &Array2<f64>,
        prompt_embedding: &PromptEmbedding,
        image_size: (usize, usize),
    ) -> Result<SegmentationResult> {
        let (img_h, img_w) = image_size;
        if img_h == 0 || img_w == 0 {
            return Err(VisionError::InvalidParameter(
                "mask_decoder: image_size must be non-zero".into(),
            ));
        }

        let embed_dim = self.config.embed_dim;
        let n_masks = self.config.num_mask_outputs;

        // --- Prepare tokens ------------------------------------------------
        // Combine sparse prompt tokens with learnable mask output tokens.
        let n_sparse = prompt_embedding.sparse_embeddings.dim().0;
        let total_tokens = n_sparse + n_masks;

        let mut tokens = Array2::zeros((total_tokens, embed_dim));

        // Copy sparse embeddings.
        for i in 0..n_sparse {
            let src_cols = prompt_embedding.sparse_embeddings.dim().1.min(embed_dim);
            for j in 0..src_cols {
                tokens[[i, j]] = prompt_embedding.sparse_embeddings[[i, j]];
            }
        }

        // Initialise mask output tokens (small distinct values).
        for m in 0..n_masks {
            let row = n_sparse + m;
            for j in 0..embed_dim {
                tokens[[row, j]] = 0.01 * ((m * embed_dim + j) as f64 * 0.07).sin();
            }
        }

        // --- Cross-attention: tokens attend to image features ---------------
        let feat_cols = image_features.dim().1;
        let proj_features = if feat_cols != embed_dim {
            // Simple linear projection to embed_dim.
            project_features(image_features, embed_dim)
        } else {
            image_features.clone()
        };

        let tokens = simple_attention(&tokens, &proj_features, &proj_features)?;

        // --- Self-attention among tokens ------------------------------------
        let tokens = simple_attention(&tokens, &tokens, &tokens)?;

        // --- Combine with dense embeddings if present -----------------------
        let n_dense = prompt_embedding.dense_embeddings.dim().0;
        let combined = if n_dense > 0 {
            // Average dense embeddings and add to each token.
            let mut avg = vec![0.0f64; embed_dim];
            let dense_cols = prompt_embedding.dense_embeddings.dim().1.min(embed_dim);
            for i in 0..n_dense {
                for (j, avg_val) in avg.iter_mut().enumerate().take(dense_cols) {
                    *avg_val += prompt_embedding.dense_embeddings[[i, j]];
                }
            }
            if n_dense > 0 {
                for v in &mut avg {
                    *v /= n_dense as f64;
                }
            }
            let mut combined = tokens.clone();
            for i in 0..total_tokens {
                for j in 0..embed_dim {
                    combined[[i, j]] += avg[j];
                }
            }
            combined
        } else {
            tokens
        };

        // --- Extract mask tokens and run IoU head ---------------------------
        let mask_token_start = n_sparse;
        let mut mask_tokens = Array2::zeros((n_masks, embed_dim));
        for m in 0..n_masks {
            let src_row = (mask_token_start + m).min(combined.dim().0.saturating_sub(1));
            for j in 0..embed_dim {
                mask_tokens[[m, j]] = combined[[src_row, j]];
            }
        }

        // IoU prediction head: MLP per mask token -> scalar.
        let iou_hidden = self.config.iou_head_hidden;
        let iou_raw = mlp_forward(&mask_tokens, iou_hidden, 1);
        let iou_predictions: Vec<f64> = (0..n_masks).map(|m| sigmoid(iou_raw[[m, 0]])).collect();

        // --- Generate masks at encoder resolution then up-sample ------------
        // Compute per-mask logit map from mask tokens and image features.
        let n_feat = proj_features.dim().0;
        let feat_side = (n_feat as f64).sqrt().ceil() as usize;
        let feat_h = feat_side;
        let feat_w = if feat_h > 0 {
            n_feat.div_ceil(feat_h)
        } else {
            0
        };

        let mut masks = Vec::with_capacity(n_masks);
        let mut stability_scores = Vec::with_capacity(n_masks);

        for m in 0..n_masks {
            // Dot product between mask token and each spatial feature.
            let mut logit_map = Array2::zeros((feat_h, feat_w));
            for fy in 0..feat_h {
                for fx in 0..feat_w {
                    let fi = fy * feat_w + fx;
                    if fi < n_feat {
                        let mut dot = 0.0f64;
                        for d in 0..embed_dim {
                            dot += mask_tokens[[m, d]] * proj_features[[fi, d]];
                        }
                        logit_map[[fy, fx]] = dot;
                    }
                }
            }

            // Up-sample to original image size using iterative 2x up-sampling.
            let full_mask = upsample_to_size(&logit_map, img_h, img_w)?;

            let stab = compute_stability_score(&full_mask, 0.0, -1.0);
            stability_scores.push(stab);
            masks.push(full_mask);
        }

        Ok(SegmentationResult {
            masks,
            iou_predictions,
            stability_scores,
        })
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Sigmoid activation.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Linearly project features from `in_dim` channels to `out_dim`.
fn project_features(features: &Array2<f64>, out_dim: usize) -> Array2<f64> {
    let (n, in_dim) = features.dim();
    let mut out = Array2::zeros((n, out_dim));
    let copy_dim = in_dim.min(out_dim);
    for i in 0..n {
        for j in 0..copy_dim {
            out[[i, j]] = features[[i, j]];
        }
    }
    out
}

/// Up-sample a 2-D map to `(target_h, target_w)` via bilinear interpolation.
fn upsample_to_size(map: &Array2<f64>, target_h: usize, target_w: usize) -> Result<Array2<f64>> {
    let (src_h, src_w) = map.dim();
    if src_h == 0 || src_w == 0 {
        return Ok(Array2::zeros((target_h, target_w)));
    }

    let mut out = Array2::zeros((target_h, target_w));
    for ty in 0..target_h {
        for tx in 0..target_w {
            let sy = ty as f64 * (src_h as f64) / target_h.max(1) as f64;
            let sx = tx as f64 * (src_w as f64) / target_w.max(1) as f64;

            let y0 = (sy.floor() as usize).min(src_h.saturating_sub(1));
            let x0 = (sx.floor() as usize).min(src_w.saturating_sub(1));
            let y1 = (y0 + 1).min(src_h.saturating_sub(1));
            let x1 = (x0 + 1).min(src_w.saturating_sub(1));

            let fy = sy - y0 as f64;
            let fx = sx - x0 as f64;

            let val = (1.0 - fy) * (1.0 - fx) * map[[y0, x0]]
                + (1.0 - fy) * fx * map[[y0, x1]]
                + fy * (1.0 - fx) * map[[y1, x0]]
                + fy * fx * map[[y1, x1]];

            out[[ty, tx]] = val;
        }
    }
    Ok(out)
}

/// Compute the stability score of a mask.
///
/// Defined as the IoU between the mask binarised at a high threshold and the
/// mask binarised at a low threshold. A score close to 1.0 indicates a stable
/// prediction.
pub fn compute_stability_score(mask: &Array2<f64>, threshold_high: f64, threshold_low: f64) -> f64 {
    let (h, w) = mask.dim();
    let mut intersection = 0usize;
    let mut union = 0usize;

    for r in 0..h {
        for c in 0..w {
            let v = mask[[r, c]];
            let in_high = v > threshold_high;
            let in_low = v > threshold_low;
            if in_high || in_low {
                union += 1;
            }
            if in_high && in_low {
                intersection += 1;
            }
        }
    }

    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt_segmentation::prompt_encoder::PromptEmbedding;
    use crate::prompt_segmentation::types::SAMConfig;
    use scirs2_core::ndarray::Array2;

    fn small_config() -> SAMConfig {
        SAMConfig {
            image_size: 16,
            embed_dim: 8,
            num_mask_outputs: 3,
            iou_head_hidden: 8,
            encoder_stages: 2,
        }
    }

    #[test]
    fn test_simple_attention_identity() {
        let q = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expect("shape");
        let k = q.clone();
        let v = q.clone();
        let out = simple_attention(&q, &k, &v).expect("attention");
        assert_eq!(out.dim(), (2, 3));
    }

    #[test]
    fn test_upsample_2x() {
        let input = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("shape");
        let (oh, ow, out) = upsample_2x(&input, 2, 2).expect("upsample");
        assert_eq!(oh, 4);
        assert_eq!(ow, 4);
        assert_eq!(out.dim(), (16, 2));
    }

    #[test]
    fn test_stability_score_all_above() {
        let mask = Array2::from_elem((4, 4), 5.0);
        let s = compute_stability_score(&mask, 0.0, -1.0);
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stability_score_mixed() {
        let mut mask = Array2::zeros((4, 4));
        // Half above high threshold, all above low threshold.
        for r in 0..2 {
            for c in 0..4 {
                mask[[r, c]] = 1.0;
            }
        }
        for r in 2..4 {
            for c in 0..4 {
                mask[[r, c]] = -0.5;
            }
        }
        let s = compute_stability_score(&mask, 0.0, -1.0);
        // intersection = 8 (top rows above both), union = 16 (all above low)
        assert!((s - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_mask_decoder_smoke() {
        let cfg = small_config();
        let decoder = MaskDecoder::new(&cfg);

        let image_features = Array2::from_elem((16, 8), 0.1);
        let prompt_emb = PromptEmbedding {
            sparse_embeddings: Array2::from_elem((1, 8), 0.5),
            dense_embeddings: Array2::zeros((0, 8)),
        };

        let result = decoder
            .decode(&image_features, &prompt_emb, (16, 16))
            .expect("decode");

        assert_eq!(result.masks.len(), 3);
        assert_eq!(result.iou_predictions.len(), 3);
        assert_eq!(result.stability_scores.len(), 3);

        for mask in &result.masks {
            assert_eq!(mask.dim(), (16, 16));
        }
        for &iou in &result.iou_predictions {
            assert!((0.0..=1.0).contains(&iou));
        }
    }

    #[test]
    fn test_mask_decoder_with_dense() {
        let cfg = small_config();
        let decoder = MaskDecoder::new(&cfg);

        let image_features = Array2::from_elem((16, 8), 0.1);
        let prompt_emb = PromptEmbedding {
            sparse_embeddings: Array2::zeros((0, 8)),
            dense_embeddings: Array2::from_elem((4, 8), 0.3),
        };

        let result = decoder
            .decode(&image_features, &prompt_emb, (8, 8))
            .expect("decode with dense");

        assert_eq!(result.masks.len(), 3);
        for mask in &result.masks {
            assert_eq!(mask.dim(), (8, 8));
        }
    }

    #[test]
    fn test_mask_decoder_zero_image_err() {
        let cfg = small_config();
        let decoder = MaskDecoder::new(&cfg);
        let prompt_emb = PromptEmbedding {
            sparse_embeddings: Array2::zeros((1, 8)),
            dense_embeddings: Array2::zeros((0, 8)),
        };
        assert!(decoder
            .decode(&Array2::zeros((4, 8)), &prompt_emb, (0, 0))
            .is_err());
    }
}
