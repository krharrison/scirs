//! Lightweight multi-scale CNN image encoder.
//!
//! Unlike the original SAM which uses a heavy ViT backbone, this encoder
//! provides a practical pure-Rust alternative using a 3-stage convolutional
//! network with residual connections.  Each stage halves the spatial resolution
//! while increasing the channel count, producing feature maps at 1/4, 1/8 and
//! 1/16 of the input size.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array1, Array2};

use super::types::SAMConfig;

// ---------------------------------------------------------------------------
// Patch embedding (stride-2 convolution)
// ---------------------------------------------------------------------------

/// A single convolutional "patch embedding" that halves spatial resolution.
///
/// Conceptually equivalent to `Conv2D(in_ch, out_ch, kernel=3, stride=2, pad=1)`.
/// Weights are stored as a flattened `[out_ch, in_ch * k * k]` matrix.
#[derive(Debug, Clone)]
pub struct PatchEmbedding {
    /// Weight matrix `[out_ch, in_ch * kernel * kernel]`.
    weights: Array2<f64>,
    /// Bias vector `[out_ch]`.
    bias: Array1<f64>,
    /// Input channel count.
    in_channels: usize,
    /// Output channel count.
    out_channels: usize,
    /// Kernel size (square).
    kernel_size: usize,
    /// Stride (always 2 for down-sampling).
    stride: usize,
}

impl PatchEmbedding {
    /// Create a new patch embedding with He-initialised weights.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
        let fan_in = in_channels * kernel_size * kernel_size;
        let std_dev = (2.0 / fan_in as f64).sqrt();

        // Deterministic pseudo-random initialisation (good enough for inference
        // placeholders; real training would replace these).
        let total = out_channels * fan_in;
        let mut weights_vec = Vec::with_capacity(total);
        let mut seed: u64 = 42;
        for _ in 0..total {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u = (seed >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
            weights_vec.push(u * std_dev);
        }

        let weights = Array2::from_shape_vec((out_channels, fan_in), weights_vec)
            .unwrap_or_else(|_| Array2::zeros((out_channels, fan_in)));
        let bias = Array1::zeros(out_channels);

        Self {
            weights,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
        }
    }

    /// Apply the convolution to a multi-channel feature map stored as
    /// `[H * W, channels]` with known `(height, width)`.
    ///
    /// Returns `(out_h, out_w, Array2<f64>[out_h * out_w, out_channels])`.
    pub fn forward(
        &self,
        input: &Array2<f64>,
        height: usize,
        width: usize,
    ) -> Result<(usize, usize, Array2<f64>)> {
        let (n_pixels, in_ch) = input.dim();
        if n_pixels != height * width {
            return Err(VisionError::InvalidParameter(format!(
                "PatchEmbedding: pixel count {n_pixels} != H*W = {}",
                height * width,
            )));
        }
        if in_ch != self.in_channels {
            return Err(VisionError::InvalidParameter(format!(
                "PatchEmbedding: expected {}, got {in_ch} input channels",
                self.in_channels,
            )));
        }

        let pad = self.kernel_size / 2;
        let out_h = (height + 2 * pad - self.kernel_size) / self.stride + 1;
        let out_w = (width + 2 * pad - self.kernel_size) / self.stride + 1;

        let mut output = Array2::zeros((out_h * out_w, self.out_channels));

        for oy in 0..out_h {
            for ox in 0..out_w {
                let iy_start = oy * self.stride;
                let ix_start = ox * self.stride;

                // Gather the patch into a flat vector.
                let patch_len = self.in_channels * self.kernel_size * self.kernel_size;
                let mut patch = vec![0.0f64; patch_len];
                let mut idx = 0;
                for c in 0..self.in_channels {
                    for ky in 0..self.kernel_size {
                        for kx in 0..self.kernel_size {
                            let iy = iy_start + ky;
                            let ix = ix_start + kx;
                            // Account for padding: pixel coords are shifted by `pad`.
                            let sy = iy as isize - pad as isize;
                            let sx = ix as isize - pad as isize;
                            if sy >= 0 && (sy as usize) < height && sx >= 0 && (sx as usize) < width
                            {
                                let pixel_idx = sy as usize * width + sx as usize;
                                patch[idx] = input[[pixel_idx, c]];
                            }
                            // else: zero-pad (already 0.0)
                            idx += 1;
                        }
                    }
                }

                // Matrix-vector multiply: output[oy*out_w+ox, :] = W @ patch + b
                let out_idx = oy * out_w + ox;
                for oc in 0..self.out_channels {
                    let mut val = self.bias[oc];
                    for (pi, &patch_val) in patch.iter().enumerate().take(patch_len) {
                        val += self.weights[[oc, pi]] * patch_val;
                    }
                    output[[out_idx, oc]] = val;
                }
            }
        }

        Ok((out_h, out_w, output))
    }
}

// ---------------------------------------------------------------------------
// Layer normalisation
// ---------------------------------------------------------------------------

/// Channel-wise layer normalisation over the last axis.
fn layer_norm(input: &mut Array2<f64>, eps: f64) {
    let (rows, cols) = input.dim();
    if cols == 0 {
        return;
    }
    for r in 0..rows {
        let mut mean = 0.0f64;
        for c in 0..cols {
            mean += input[[r, c]];
        }
        mean /= cols as f64;

        let mut var = 0.0f64;
        for c in 0..cols {
            let diff = input[[r, c]] - mean;
            var += diff * diff;
        }
        var /= cols as f64;

        let inv_std = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            input[[r, c]] = (input[[r, c]] - mean) * inv_std;
        }
    }
}

/// Element-wise ReLU activation.
fn relu_inplace(arr: &mut Array2<f64>) {
    arr.mapv_inplace(|v| v.max(0.0));
}

// ---------------------------------------------------------------------------
// Encoder stage (conv -> layernorm -> relu -> conv -> residual)
// ---------------------------------------------------------------------------

/// One down-sampling stage of the encoder.
#[derive(Debug, Clone)]
struct EncoderStage {
    /// Stride-2 down-sampling convolution.
    down_conv: PatchEmbedding,
    /// 1x1 convolution for channel projection in the residual path.
    proj_conv: PatchEmbedding,
    /// Stride-1 "refine" convolution.
    refine_conv: PatchEmbedding,
}

impl EncoderStage {
    fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            down_conv: PatchEmbedding::new(in_channels, out_channels, 3, 2),
            proj_conv: PatchEmbedding::new(in_channels, out_channels, 1, 2),
            refine_conv: PatchEmbedding::new(out_channels, out_channels, 3, 1),
        }
    }

    /// Run one stage: returns `(out_h, out_w, features)`.
    fn forward(
        &self,
        input: &Array2<f64>,
        h: usize,
        w: usize,
    ) -> Result<(usize, usize, Array2<f64>)> {
        // Main path: down_conv -> layernorm -> relu -> refine_conv
        let (h1, w1, mut main) = self.down_conv.forward(input, h, w)?;
        layer_norm(&mut main, 1e-5);
        relu_inplace(&mut main);

        let (_h2, _w2, refined) = self.refine_conv.forward(&main, h1, w1)?;

        // Residual path: 1x1 stride-2 projection
        let (_rh, _rw, residual) = self.proj_conv.forward(input, h, w)?;

        // Add residual
        let (rows, cols) = refined.dim();
        let (rrows, rcols) = residual.dim();
        let min_rows = rows.min(rrows);
        let min_cols = cols.min(rcols);
        let mut out = refined;
        for r in 0..min_rows {
            for c in 0..min_cols {
                out[[r, c]] += residual[[r, c]];
            }
        }

        Ok((h1, w1, out))
    }
}

// ---------------------------------------------------------------------------
// SimpleImageEncoder
// ---------------------------------------------------------------------------

/// A lightweight multi-scale CNN encoder that produces feature maps at three
/// spatial scales (1/4, 1/8, 1/16 of the input).
///
/// This replaces SAM's ViT-H/ViT-L backbone with a practical pure-Rust
/// convolutional architecture suitable for CPU inference.
#[derive(Debug, Clone)]
pub struct SimpleImageEncoder {
    /// Initial patch embedding (stride-2 conv, input -> embed_dim/4).
    initial_embed: PatchEmbedding,
    /// Per-stage down-sampling blocks.
    stages: Vec<EncoderStage>,
    /// Configuration.
    config: SAMConfig,
}

impl SimpleImageEncoder {
    /// Build a new encoder from the given configuration.
    pub fn new(config: &SAMConfig) -> Self {
        let base_ch = config.embed_dim / 4; // e.g. 64

        // Initial embedding: 1 -> base_ch, stride 2 (halves resolution once)
        let initial_embed = PatchEmbedding::new(1, base_ch, 3, 2);

        // Build encoder stages. Each stage doubles channels and halves resolution.
        let mut stages = Vec::with_capacity(config.encoder_stages);
        let mut ch = base_ch;
        for _ in 0..config.encoder_stages {
            let next_ch = (ch * 2).min(config.embed_dim);
            stages.push(EncoderStage::new(ch, next_ch));
            ch = next_ch;
        }

        Self {
            initial_embed,
            stages,
            config: config.clone(),
        }
    }

    /// Encode a single-channel image into multi-scale feature maps.
    ///
    /// # Arguments
    ///
    /// * `image` – Grayscale image `[H, W]` with values in `[0, 1]`.
    ///
    /// # Returns
    ///
    /// A `Vec` of feature maps, one per encoder stage. Each feature map is
    /// `[h_i * w_i, channels_i]` (flattened spatial dims). The first entry
    /// corresponds to the coarsest (highest-level) features.
    pub fn encode(&self, image: &Array2<f64>) -> Result<Vec<Array2<f64>>> {
        let (img_h, img_w) = image.dim();
        if img_h == 0 || img_w == 0 {
            return Err(VisionError::InvalidParameter(
                "image_encoder: image must have non-zero dimensions".into(),
            ));
        }

        // Reshape image to [H*W, 1] for the first convolution.
        let flat_len = img_h * img_w;
        let mut flat = Array2::zeros((flat_len, 1));
        for r in 0..img_h {
            for c in 0..img_w {
                flat[[r * img_w + c, 0]] = image[[r, c]];
            }
        }

        // Initial embedding (stride 2 -> resolution / 2).
        let (mut h, mut w, mut features) = self.initial_embed.forward(&flat, img_h, img_w)?;
        layer_norm(&mut features, 1e-5);
        relu_inplace(&mut features);

        // Run each stage, collecting multi-scale features.
        let mut multi_scale: Vec<Array2<f64>> = Vec::with_capacity(self.config.encoder_stages);
        for stage in &self.stages {
            let (nh, nw, new_feat) = stage.forward(&features, h, w)?;
            multi_scale.push(new_feat.clone());
            h = nh;
            w = nw;
            features = new_feat;
        }

        Ok(multi_scale)
    }

    /// Return the expected number of output channels at each stage.
    pub fn stage_channels(&self) -> Vec<usize> {
        let base_ch = self.config.embed_dim / 4;
        let mut channels = Vec::with_capacity(self.config.encoder_stages);
        let mut ch = base_ch;
        for _ in 0..self.config.encoder_stages {
            ch = (ch * 2).min(self.config.embed_dim);
            channels.push(ch);
        }
        channels
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_patch_embedding_forward() {
        let pe = PatchEmbedding::new(1, 4, 3, 2);
        let input = Array2::ones((16, 1)); // 4x4 image, 1 channel
        let (oh, ow, out) = pe.forward(&input, 4, 4).expect("forward failed");
        assert_eq!(oh, 2);
        assert_eq!(ow, 2);
        assert_eq!(out.dim(), (4, 4));
    }

    #[test]
    fn test_layer_norm() {
        let mut arr = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("shape");
        layer_norm(&mut arr, 1e-5);
        // After layer norm each row should have ~zero mean.
        let row_mean: f64 = (0..4).map(|c| arr[[0, c]]).sum::<f64>() / 4.0;
        assert!(row_mean.abs() < 1e-6);
    }

    #[test]
    fn test_simple_image_encoder_smoke() {
        let cfg = SAMConfig {
            image_size: 32,
            embed_dim: 16,
            num_mask_outputs: 3,
            iou_head_hidden: 16,
            encoder_stages: 2,
        };
        let encoder = SimpleImageEncoder::new(&cfg);
        let image = Array2::from_elem((8, 8), 0.5);
        let features = encoder.encode(&image).expect("encode failed");
        assert_eq!(features.len(), 2);
        // Each feature map should have > 0 rows.
        for f in &features {
            assert!(f.dim().0 > 0);
            assert!(f.dim().1 > 0);
        }
    }

    #[test]
    fn test_encoder_stage_channels() {
        let cfg = SAMConfig {
            embed_dim: 64,
            encoder_stages: 3,
            ..SAMConfig::default()
        };
        let enc = SimpleImageEncoder::new(&cfg);
        let chs = enc.stage_channels();
        assert_eq!(chs, vec![32, 64, 64]);
    }

    #[test]
    fn test_patch_embedding_channel_mismatch() {
        let pe = PatchEmbedding::new(3, 8, 3, 2);
        let input = Array2::ones((16, 1)); // wrong channel count
        let err = pe.forward(&input, 4, 4);
        assert!(err.is_err());
    }

    #[test]
    fn test_encoder_empty_image() {
        let cfg = SAMConfig::default();
        let enc = SimpleImageEncoder::new(&cfg);
        let img = Array2::<f64>::zeros((0, 0));
        assert!(enc.encode(&img).is_err());
    }
}
