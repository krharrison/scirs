//! Prompt encoder: converts user prompts into dense/sparse embeddings.
//!
//! Supports point, bounding-box, mask and multi-point prompts using
//! sinusoidal positional encoding and lightweight convolutions.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array1, Array2};

use super::types::{PromptType, SAMConfig, SegmentationPrompt};

// ---------------------------------------------------------------------------
// Positional encoding helpers
// ---------------------------------------------------------------------------

/// Number of frequency bands used in the sinusoidal encoding.
const NUM_FREQUENCIES: usize = 64;

/// Sinusoidal positional encoding for a 2-D coordinate normalised to `[0, 1]`.
///
/// Returns a vector of length `4 * NUM_FREQUENCIES` (sin+cos for each of x, y
/// at `NUM_FREQUENCIES` frequencies).
fn positional_encoding_2d(x_norm: f64, y_norm: f64) -> Array1<f64> {
    let dim = 4 * NUM_FREQUENCIES;
    let mut enc = Array1::zeros(dim);
    for i in 0..NUM_FREQUENCIES {
        let freq = std::f64::consts::PI * 2.0_f64.powi(i as i32);
        enc[4 * i] = (x_norm * freq).sin();
        enc[4 * i + 1] = (x_norm * freq).cos();
        enc[4 * i + 2] = (y_norm * freq).sin();
        enc[4 * i + 3] = (y_norm * freq).cos();
    }
    enc
}

/// Project a positional encoding vector to `embed_dim` via a simple linear
/// layer (truncate or zero-pad).
fn project_to_embed_dim(enc: &Array1<f64>, embed_dim: usize) -> Array1<f64> {
    let src_len = enc.len();
    let mut out = Array1::zeros(embed_dim);
    let copy_len = src_len.min(embed_dim);
    for i in 0..copy_len {
        out[i] = enc[i];
    }
    out
}

// ---------------------------------------------------------------------------
// PromptEmbedding
// ---------------------------------------------------------------------------

/// The output of the prompt encoder, split into sparse and dense components.
#[derive(Debug, Clone)]
pub struct PromptEmbedding {
    /// Sparse embeddings from point / box prompts.
    /// Shape: `[num_tokens, embed_dim]`.
    pub sparse_embeddings: Array2<f64>,
    /// Dense embeddings from mask prompts (or zeros if no mask prompt).
    /// Shape: `[spatial_tokens, embed_dim]`.
    pub dense_embeddings: Array2<f64>,
}

// ---------------------------------------------------------------------------
// PromptEncoder
// ---------------------------------------------------------------------------

/// Encodes user-supplied prompts into embeddings that can be consumed by
/// the mask decoder.
#[derive(Debug, Clone)]
pub struct PromptEncoder {
    /// Pipeline configuration.
    config: SAMConfig,
    /// Learned foreground token (simulated).
    fg_token: Array1<f64>,
    /// Learned background token (simulated).
    bg_token: Array1<f64>,
}

impl PromptEncoder {
    /// Build a new prompt encoder from the given config.
    pub fn new(config: &SAMConfig) -> Self {
        // Initialise foreground / background tokens with small distinct values.
        let mut fg = Array1::zeros(config.embed_dim);
        let mut bg = Array1::zeros(config.embed_dim);
        for i in 0..config.embed_dim {
            fg[i] = 0.1 * ((i as f64 * 0.1).sin());
            bg[i] = -0.1 * ((i as f64 * 0.1).cos());
        }
        Self {
            config: config.clone(),
            fg_token: fg,
            bg_token: bg,
        }
    }

    /// Encode a segmentation prompt.
    ///
    /// # Arguments
    ///
    /// * `prompt`     - The segmentation prompt to encode.
    /// * `image_size` - `(height, width)` of the input image.
    ///
    /// # Returns
    ///
    /// A [`PromptEmbedding`] containing sparse and dense embeddings.
    pub fn encode(
        &self,
        prompt: &SegmentationPrompt,
        image_size: (usize, usize),
    ) -> Result<PromptEmbedding> {
        let (img_h, img_w) = image_size;
        if img_h == 0 || img_w == 0 {
            return Err(VisionError::InvalidParameter(
                "prompt_encoder: image_size must be non-zero".into(),
            ));
        }

        #[allow(unreachable_patterns)]
        match &prompt.prompt_type {
            PromptType::Point {
                x,
                y,
                is_foreground,
            } => self.encode_point(*x, *y, *is_foreground, img_h, img_w),
            PromptType::BoundingBox { x1, y1, x2, y2 } => {
                self.encode_box(*x1, *y1, *x2, *y2, img_h, img_w)
            }
            PromptType::MaskPrompt { mask } => self.encode_mask(mask, img_h, img_w),
            PromptType::MultiPoint { points } => self.encode_multi_point(points, img_h, img_w),
            _ => Err(VisionError::InvalidParameter(
                "prompt_encoder: unknown prompt type variant".into(),
            )),
        }
    }

    // -- Point prompt -------------------------------------------------------

    fn encode_point(
        &self,
        x: usize,
        y: usize,
        is_foreground: bool,
        img_h: usize,
        img_w: usize,
    ) -> Result<PromptEmbedding> {
        let x_norm = x as f64 / img_w.max(1) as f64;
        let y_norm = y as f64 / img_h.max(1) as f64;

        let pos_enc = positional_encoding_2d(x_norm, y_norm);
        let mut token = project_to_embed_dim(&pos_enc, self.config.embed_dim);

        // Add foreground / background token.
        let label_token = if is_foreground {
            &self.fg_token
        } else {
            &self.bg_token
        };
        for i in 0..self.config.embed_dim {
            token[i] += label_token[i];
        }

        let sparse = Array2::from_shape_vec((1, self.config.embed_dim), token.to_vec())
            .map_err(|e| VisionError::OperationError(format!("sparse reshape: {e}")))?;

        Ok(PromptEmbedding {
            sparse_embeddings: sparse,
            dense_embeddings: Array2::zeros((0, self.config.embed_dim)),
        })
    }

    // -- Bounding-box prompt ------------------------------------------------

    fn encode_box(
        &self,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
        img_h: usize,
        img_w: usize,
    ) -> Result<PromptEmbedding> {
        // Encode as two corner points: top-left (foreground), bottom-right (background).
        let tl = self.point_token(x1, y1, true, img_h, img_w);
        let br = self.point_token(x2, y2, false, img_h, img_w);

        let mut data = Vec::with_capacity(2 * self.config.embed_dim);
        data.extend(tl.iter());
        data.extend(br.iter());

        let sparse = Array2::from_shape_vec((2, self.config.embed_dim), data)
            .map_err(|e| VisionError::OperationError(format!("box sparse reshape: {e}")))?;

        Ok(PromptEmbedding {
            sparse_embeddings: sparse,
            dense_embeddings: Array2::zeros((0, self.config.embed_dim)),
        })
    }

    // -- Mask prompt --------------------------------------------------------

    fn encode_mask(
        &self,
        mask: &Array2<f64>,
        img_h: usize,
        img_w: usize,
    ) -> Result<PromptEmbedding> {
        let (mh, mw) = mask.dim();
        if mh == 0 || mw == 0 {
            return Err(VisionError::InvalidParameter(
                "prompt_encoder: mask must be non-empty".into(),
            ));
        }

        // Down-sample mask by 4x using simple average pooling.
        let ds_h = img_h.div_ceil(4);
        let ds_w = img_w.div_ceil(4);
        let num_spatial = ds_h * ds_w;

        let mut dense = Array2::zeros((num_spatial, self.config.embed_dim));

        for dy in 0..ds_h {
            for dx in 0..ds_w {
                // Average over the 4x4 source patch.
                let sy = dy * 4;
                let sx = dx * 4;
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for ky in 0..4 {
                    for kx in 0..4 {
                        let my = sy + ky;
                        let mx = sx + kx;
                        if my < mh && mx < mw {
                            sum += mask[[my, mx]];
                            count += 1;
                        }
                    }
                }
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };

                // Fill the embedding with the average value scaled by a
                // positional encoding of the spatial location.
                let x_norm = dx as f64 / ds_w.max(1) as f64;
                let y_norm = dy as f64 / ds_h.max(1) as f64;
                let pos = positional_encoding_2d(x_norm, y_norm);
                let proj = project_to_embed_dim(&pos, self.config.embed_dim);
                let idx = dy * ds_w + dx;
                for c in 0..self.config.embed_dim {
                    dense[[idx, c]] = avg * proj[c];
                }
            }
        }

        Ok(PromptEmbedding {
            sparse_embeddings: Array2::zeros((0, self.config.embed_dim)),
            dense_embeddings: dense,
        })
    }

    // -- Multi-point prompt -------------------------------------------------

    fn encode_multi_point(
        &self,
        points: &[(usize, usize, bool)],
        img_h: usize,
        img_w: usize,
    ) -> Result<PromptEmbedding> {
        if points.is_empty() {
            return Err(VisionError::InvalidParameter(
                "prompt_encoder: MultiPoint must have at least one point".into(),
            ));
        }

        let n = points.len();
        let mut data = Vec::with_capacity(n * self.config.embed_dim);
        for &(x, y, is_fg) in points {
            let tok = self.point_token(x, y, is_fg, img_h, img_w);
            data.extend(tok.iter());
        }

        let sparse = Array2::from_shape_vec((n, self.config.embed_dim), data)
            .map_err(|e| VisionError::OperationError(format!("multi-point reshape: {e}")))?;

        Ok(PromptEmbedding {
            sparse_embeddings: sparse,
            dense_embeddings: Array2::zeros((0, self.config.embed_dim)),
        })
    }

    // -- Helpers ------------------------------------------------------------

    fn point_token(
        &self,
        x: usize,
        y: usize,
        is_fg: bool,
        img_h: usize,
        img_w: usize,
    ) -> Array1<f64> {
        let x_norm = x as f64 / img_w.max(1) as f64;
        let y_norm = y as f64 / img_h.max(1) as f64;
        let pos = positional_encoding_2d(x_norm, y_norm);
        let mut tok = project_to_embed_dim(&pos, self.config.embed_dim);
        let label = if is_fg {
            &self.fg_token
        } else {
            &self.bg_token
        };
        for i in 0..self.config.embed_dim {
            tok[i] += label[i];
        }
        tok
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt_segmentation::types::{PromptType, SAMConfig, SegmentationPrompt};
    use scirs2_core::ndarray::Array2;

    fn small_config() -> SAMConfig {
        SAMConfig {
            image_size: 32,
            embed_dim: 16,
            num_mask_outputs: 3,
            iou_head_hidden: 16,
            encoder_stages: 2,
        }
    }

    #[test]
    fn test_encode_point() {
        let cfg = small_config();
        let enc = PromptEncoder::new(&cfg);
        let prompt = SegmentationPrompt::new(PromptType::Point {
            x: 5,
            y: 10,
            is_foreground: true,
        });
        let emb = enc.encode(&prompt, (32, 32)).expect("encode point");
        assert_eq!(emb.sparse_embeddings.dim(), (1, 16));
        assert_eq!(emb.dense_embeddings.dim().0, 0);
    }

    #[test]
    fn test_encode_box() {
        let cfg = small_config();
        let enc = PromptEncoder::new(&cfg);
        let prompt = SegmentationPrompt::new(PromptType::BoundingBox {
            x1: 2,
            y1: 3,
            x2: 20,
            y2: 25,
        });
        let emb = enc.encode(&prompt, (32, 32)).expect("encode box");
        // Box produces 2 tokens (top-left, bottom-right).
        assert_eq!(emb.sparse_embeddings.dim(), (2, 16));
    }

    #[test]
    fn test_encode_mask() {
        let cfg = small_config();
        let enc = PromptEncoder::new(&cfg);
        let mask = Array2::from_elem((32, 32), 1.0);
        let prompt = SegmentationPrompt::new(PromptType::MaskPrompt { mask });
        let emb = enc.encode(&prompt, (32, 32)).expect("encode mask");
        // Dense embeddings should have (32/4)*(32/4) = 64 spatial tokens.
        assert_eq!(emb.dense_embeddings.dim(), (64, 16));
        assert_eq!(emb.sparse_embeddings.dim().0, 0);
    }

    #[test]
    fn test_encode_multipoint() {
        let cfg = small_config();
        let enc = PromptEncoder::new(&cfg);
        let pts = vec![(1, 2, true), (10, 15, false), (5, 5, true)];
        let prompt = SegmentationPrompt::new(PromptType::MultiPoint { points: pts });
        let emb = enc.encode(&prompt, (32, 32)).expect("encode multipoint");
        assert_eq!(emb.sparse_embeddings.dim(), (3, 16));
    }

    #[test]
    fn test_encode_empty_multipoint_err() {
        let cfg = small_config();
        let enc = PromptEncoder::new(&cfg);
        let prompt = SegmentationPrompt::new(PromptType::MultiPoint { points: vec![] });
        assert!(enc.encode(&prompt, (32, 32)).is_err());
    }

    #[test]
    fn test_encode_zero_image_size_err() {
        let cfg = small_config();
        let enc = PromptEncoder::new(&cfg);
        let prompt = SegmentationPrompt::new(PromptType::Point {
            x: 0,
            y: 0,
            is_foreground: true,
        });
        assert!(enc.encode(&prompt, (0, 0)).is_err());
    }
}
