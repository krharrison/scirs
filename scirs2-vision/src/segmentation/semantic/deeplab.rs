//! DeepLab-lite: Atrous (dilated) convolution and ASPP concepts
//!
//! This module provides pure-Rust ndarray implementations of the key
//! architectural ideas introduced by the DeepLab family (Chen et al. 2014–2018):
//!
//! - [`AtrousConv2D`] – dilated 2-D convolution (single channel group).
//! - [`ASPP`] – Atrous Spatial Pyramid Pooling with configurable dilation rates.
//! - [`global_average_pooling`] – image-level global average pooling branch.
//! - [`aspp_forward`] – full ASPP forward pass combining all branches.
//!
//! # Array layout
//!
//! Single-image feature maps use **HWC** layout (`Array3<f32>` with shape
//! `[H, W, C]`).  Convolution weights use `[K_H, K_W, C_in, C_out]` layout.

use crate::error::{Result, VisionError};
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4};

// ─────────────────────────────────────────────────────────────────────────────
// Atrous (dilated) 2-D convolution
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a single atrous (dilated) 2-D convolution layer.
#[derive(Debug, Clone)]
pub struct AtrousConv2DConfig {
    /// Height of the convolution kernel.
    pub kernel_h: usize,
    /// Width of the convolution kernel.
    pub kernel_w: usize,
    /// Number of input feature channels.
    pub in_channels: usize,
    /// Number of output feature channels.
    pub out_channels: usize,
    /// Dilation rate (atrous rate).  1 = standard convolution.
    pub dilation: usize,
    /// Padding mode: `true` → use "same" padding to preserve spatial size.
    pub same_padding: bool,
    /// Whether to add a bias term.
    pub use_bias: bool,
}

impl Default for AtrousConv2DConfig {
    fn default() -> Self {
        Self {
            kernel_h: 3,
            kernel_w: 3,
            in_channels: 1,
            out_channels: 1,
            dilation: 1,
            same_padding: true,
            use_bias: false,
        }
    }
}

/// Atrous (dilated) 2-D convolution layer.
///
/// Implements the core operation that gives DeepLab its multi-scale receptive
/// field without sacrificing spatial resolution or increasing parameters.
///
/// The weight tensor has shape `[kernel_h, kernel_w, in_channels, out_channels]`.
pub struct AtrousConv2D {
    /// Layer configuration.
    pub config: AtrousConv2DConfig,
    /// Learned weights `[K_H, K_W, C_in, C_out]`.
    pub weights: Array4<f32>,
    /// Optional bias `[C_out]`.
    pub bias: Option<Array1<f32>>,
}

impl AtrousConv2D {
    /// Construct a new `AtrousConv2D` from an explicit weight tensor.
    ///
    /// # Errors
    ///
    /// Returns [`VisionError::InvalidParameter`] when `weights` shape does not
    /// match the channel sizes in `config`.
    pub fn new(
        config: AtrousConv2DConfig,
        weights: Array4<f32>,
        bias: Option<Array1<f32>>,
    ) -> Result<Self> {
        let (kh, kw, c_in, c_out) = weights.dim();
        if kh != config.kernel_h || kw != config.kernel_w {
            return Err(VisionError::InvalidParameter(format!(
                "AtrousConv2D: weight kernel size {}×{} does not match config {}×{}",
                kh, kw, config.kernel_h, config.kernel_w
            )));
        }
        if c_in != config.in_channels {
            return Err(VisionError::InvalidParameter(format!(
                "AtrousConv2D: weight in_channels {} does not match config {}",
                c_in, config.in_channels
            )));
        }
        if c_out != config.out_channels {
            return Err(VisionError::InvalidParameter(format!(
                "AtrousConv2D: weight out_channels {} does not match config {}",
                c_out, config.out_channels
            )));
        }
        if let Some(ref b) = bias {
            if b.len() != c_out {
                return Err(VisionError::InvalidParameter(format!(
                    "AtrousConv2D: bias length {} does not match out_channels {}",
                    b.len(),
                    c_out
                )));
            }
        }
        Ok(Self {
            config,
            weights,
            bias,
        })
    }

    /// Create an identity-initialised `AtrousConv2D` (for testing / demos).
    ///
    /// Weights are zeroed except the centre kernel position is set to a scaled
    /// identity projection `weight[kh/2, kw/2, c, c] = scale`.
    pub fn identity(config: AtrousConv2DConfig, scale: f32) -> Self {
        let (kh, kw, ci, co) = (
            config.kernel_h,
            config.kernel_w,
            config.in_channels,
            config.out_channels,
        );
        let min_c = ci.min(co);
        let cy = kh / 2;
        let cx = kw / 2;
        let mut w = Array4::<f32>::zeros((kh, kw, ci, co));
        for c in 0..min_c {
            w[[cy, cx, c, c]] = scale;
        }
        Self {
            config,
            weights: w,
            bias: None,
        }
    }

    /// Apply the atrous convolution to a feature map `[H, W, C_in]`.
    ///
    /// Returns an output feature map `[H_out, W_out, C_out]`.
    ///
    /// With `same_padding = true` the output spatial size equals the input spatial
    /// size regardless of dilation.
    ///
    /// # Errors
    ///
    /// Returns [`VisionError::InvalidParameter`] on channel mismatch or empty input.
    pub fn forward(&self, input: &Array3<f32>) -> Result<Array3<f32>> {
        let (h, w, c_in) = input.dim();
        if c_in != self.config.in_channels {
            return Err(VisionError::InvalidParameter(format!(
                "AtrousConv2D::forward: input channels {} != expected {}",
                c_in, self.config.in_channels
            )));
        }
        if h == 0 || w == 0 {
            return Err(VisionError::InvalidParameter(
                "AtrousConv2D::forward: spatial dimensions must be > 0".into(),
            ));
        }

        let d = self.config.dilation;
        let kh = self.config.kernel_h;
        let kw = self.config.kernel_w;
        let c_out = self.config.out_channels;

        // Effective (dilated) kernel extents.
        let eff_kh = (kh - 1) * d + 1;
        let eff_kw = (kw - 1) * d + 1;

        let (out_h, out_w, pad_top, pad_left) = if self.config.same_padding {
            // "same" padding: output size == input size.
            let ph = eff_kh.saturating_sub(1);
            let pw = eff_kw.saturating_sub(1);
            (h, w, ph / 2, pw / 2)
        } else {
            // "valid" padding: no padding.
            let oh = h.saturating_sub(eff_kh) + 1;
            let ow = w.saturating_sub(eff_kw) + 1;
            (oh, ow, 0, 0)
        };

        let mut output = Array3::<f32>::zeros((out_h, out_w, c_out));

        for oy in 0..out_h {
            for ox in 0..out_w {
                for oc in 0..c_out {
                    let mut sum = 0.0f32;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            // Position in padded input.
                            let iy = oy as isize + ky as isize * d as isize - pad_top as isize;
                            let ix = ox as isize + kx as isize * d as isize - pad_left as isize;
                            if iy < 0 || iy >= h as isize || ix < 0 || ix >= w as isize {
                                continue; // zero-padding
                            }
                            let iy = iy as usize;
                            let ix = ix as usize;
                            for ic in 0..c_in {
                                sum += input[[iy, ix, ic]] * self.weights[[ky, kx, ic, oc]];
                            }
                        }
                    }
                    if let Some(ref b) = self.bias {
                        sum += b[oc];
                    }
                    output[[oy, ox, oc]] = sum;
                }
            }
        }

        Ok(output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global Average Pooling
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the global average pooling (GAP) of a feature map.
///
/// Reduces `[H, W, C]` to `[1, 1, C]` by averaging over the spatial dimensions.
/// This corresponds to the image-level context branch used in ASPP / PSPNet.
///
/// # Errors
///
/// Returns [`VisionError::InvalidParameter`] when the input has zero spatial
/// dimensions or zero channels.
pub fn global_average_pooling(feature_map: &Array3<f32>) -> Result<Array3<f32>> {
    let (h, w, c) = feature_map.dim();
    if h == 0 || w == 0 {
        return Err(VisionError::InvalidParameter(
            "global_average_pooling: spatial dimensions must be > 0".into(),
        ));
    }
    if c == 0 {
        return Err(VisionError::InvalidParameter(
            "global_average_pooling: channel dimension must be > 0".into(),
        ));
    }

    let n = (h * w) as f32;
    let mut out = Array3::<f32>::zeros((1, 1, c));
    for y in 0..h {
        for x in 0..w {
            for ci in 0..c {
                out[[0, 0, ci]] += feature_map[[y, x, ci]];
            }
        }
    }
    for ci in 0..c {
        out[[0, 0, ci]] /= n;
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// ASPP – Atrous Spatial Pyramid Pooling
// ─────────────────────────────────────────────────────────────────────────────

/// Single branch configuration for ASPP.
#[derive(Debug, Clone)]
pub struct ASPPBranch {
    /// Dilation rate.  Rate 1 = standard 1×1 or 3×3 convolution.
    pub dilation: usize,
    /// Use a 1×1 kernel (for the pointwise branch and GAP branch).
    pub pointwise: bool,
    /// Output channels for this branch.
    pub out_channels: usize,
}

/// Configuration for the full ASPP module.
#[derive(Debug, Clone)]
pub struct ASPPConfig {
    /// Number of input feature channels.
    pub in_channels: usize,
    /// Branch configurations (one per dilation rate + optional GAP branch).
    pub branches: Vec<ASPPBranch>,
    /// Include the global average pooling image-level context branch.
    pub use_gap_branch: bool,
    /// Output channels after the final 1×1 projection.
    pub project_channels: usize,
}

impl ASPPConfig {
    /// DeepLab-v3 style configuration with standard dilation rates.
    ///
    /// Branches: 1×1 @ rate=1, 3×3 @ rate=6, 3×3 @ rate=12, 3×3 @ rate=18,
    /// plus a GAP branch.
    pub fn deeplab_v3(in_channels: usize, out_channels: usize) -> Self {
        Self {
            in_channels,
            branches: vec![
                ASPPBranch {
                    dilation: 1,
                    pointwise: true,
                    out_channels,
                },
                ASPPBranch {
                    dilation: 6,
                    pointwise: false,
                    out_channels,
                },
                ASPPBranch {
                    dilation: 12,
                    pointwise: false,
                    out_channels,
                },
                ASPPBranch {
                    dilation: 18,
                    pointwise: false,
                    out_channels,
                },
            ],
            use_gap_branch: true,
            project_channels: out_channels,
        }
    }
}

/// Atrous Spatial Pyramid Pooling module.
///
/// Applies parallel atrous convolutions at multiple scales and concatenates
/// the resulting feature maps, optionally with a global average pooling branch.
/// A final 1×1 projection collapses the concatenated channels.
pub struct ASPP {
    /// Module configuration.
    pub config: ASPPConfig,
    /// One `AtrousConv2D` per branch.
    branch_convs: Vec<AtrousConv2D>,
    /// 1×1 projection convolution applied to the concatenated branches.
    project_conv: AtrousConv2D,
}

impl ASPP {
    /// Build an `ASPP` module with identity-initialised weights (suitable for
    /// testing and as a structural placeholder before weight loading).
    ///
    /// In real usage the weights would be replaced by trained parameters.
    pub fn new_identity(config: ASPPConfig) -> Result<Self> {
        let num_branches = config.branches.len() + if config.use_gap_branch { 1 } else { 0 };

        // Verify all branches have the same out_channels (simplification).
        for b in &config.branches {
            if b.out_channels == 0 {
                return Err(VisionError::InvalidParameter(
                    "ASPP: branch out_channels must be > 0".into(),
                ));
            }
        }

        let in_ch = config.in_channels;
        let mut branch_convs = Vec::with_capacity(config.branches.len());

        for branch in &config.branches {
            let (kh, kw) = if branch.pointwise { (1, 1) } else { (3, 3) };
            let conv_cfg = AtrousConv2DConfig {
                kernel_h: kh,
                kernel_w: kw,
                in_channels: in_ch,
                out_channels: branch.out_channels,
                dilation: branch.dilation,
                same_padding: true,
                use_bias: false,
            };
            branch_convs.push(AtrousConv2D::identity(conv_cfg, 1.0));
        }

        // If GAP branch is used, we also need a 1×1 conv from in_ch → branch_out_ch.
        // We reuse the first branch's out_channels for the GAP branch.
        let gap_out_ch = config
            .branches
            .first()
            .map(|b| b.out_channels)
            .unwrap_or(in_ch);
        if config.use_gap_branch {
            let gap_cfg = AtrousConv2DConfig {
                kernel_h: 1,
                kernel_w: 1,
                in_channels: in_ch,
                out_channels: gap_out_ch,
                dilation: 1,
                same_padding: true,
                use_bias: false,
            };
            branch_convs.push(AtrousConv2D::identity(gap_cfg, 1.0));
        }

        // Total concatenated channels = num_branches × branch_out_ch.
        let concat_ch = num_branches * gap_out_ch;

        let proj_cfg = AtrousConv2DConfig {
            kernel_h: 1,
            kernel_w: 1,
            in_channels: concat_ch,
            out_channels: config.project_channels,
            dilation: 1,
            same_padding: true,
            use_bias: false,
        };
        let project_conv = AtrousConv2D::identity(proj_cfg, 1.0 / num_branches as f32);

        Ok(Self {
            config,
            branch_convs,
            project_conv,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ASPP forward pass
// ─────────────────────────────────────────────────────────────────────────────

/// Run the ASPP forward pass.
///
/// Applies each branch convolution in parallel (sequentially in this
/// implementation), concatenates along the channel axis, applies global average
/// pooling (if configured), bilinearly upsamples the GAP output to match spatial
/// size, then passes through the projection 1×1 convolution.
///
/// # Arguments
///
/// * `aspp`    – Configured [`ASPP`] module.
/// * `feature` – Input feature map `[H, W, C_in]`.
///
/// # Returns
///
/// Output feature map `[H, W, C_project]`.
///
/// # Errors
///
/// Propagates any errors from the constituent convolutions or resizing.
pub fn aspp_forward(aspp: &ASPP, feature: &Array3<f32>) -> Result<Array3<f32>> {
    let (h, w, c_in) = feature.dim();
    if c_in != aspp.config.in_channels {
        return Err(VisionError::InvalidParameter(format!(
            "aspp_forward: input channels {} != ASPP in_channels {}",
            c_in, aspp.config.in_channels
        )));
    }

    let num_regular = aspp.config.branches.len();
    let has_gap = aspp.config.use_gap_branch;
    let total_branches = num_regular + if has_gap { 1 } else { 0 };

    if aspp.branch_convs.len() != total_branches {
        return Err(VisionError::InvalidParameter(
            "aspp_forward: internal branch count mismatch".into(),
        ));
    }

    let branch_out_ch = aspp
        .config
        .branches
        .first()
        .map(|b| b.out_channels)
        .unwrap_or(c_in);

    // ── Regular dilated branches ──────────────────────────────────────────
    let mut branch_outputs: Vec<Array3<f32>> = Vec::with_capacity(total_branches);
    for i in 0..num_regular {
        let out = aspp.branch_convs[i].forward(feature)?;
        // Verify spatial dimensions are preserved.
        let (oh, ow, _) = out.dim();
        if oh != h || ow != w {
            return Err(VisionError::InvalidParameter(format!(
                "aspp_forward: branch {} output size {}×{} != input {}×{}",
                i, oh, ow, h, w
            )));
        }
        branch_outputs.push(out);
    }

    // ── Global Average Pooling branch ─────────────────────────────────────
    if has_gap {
        let gap_idx = num_regular;
        let gap_vec = global_average_pooling(feature)?; // [1, 1, C_in]
                                                        // Apply 1×1 conv to project.
        let gap_projected = aspp.branch_convs[gap_idx].forward(&gap_vec)?; // [1, 1, gap_out_ch]

        // Bilinearly upsample to [H, W, gap_out_ch].
        let upsampled = upsample_bilinear_hwc(&gap_projected, h, w)?;
        branch_outputs.push(upsampled);
    }

    // ── Concatenate along channel axis ────────────────────────────────────
    let total_ch = total_branches * branch_out_ch;
    let mut concat = Array3::<f32>::zeros((h, w, total_ch));
    for (bi, branch_out) in branch_outputs.iter().enumerate() {
        let ch_start = bi * branch_out_ch;
        let (_, _, bo_ch) = branch_out.dim();
        for y in 0..h {
            for x in 0..w {
                for ci in 0..bo_ch {
                    concat[[y, x, ch_start + ci]] = branch_out[[y, x, ci]];
                }
            }
        }
    }

    // ── Projection 1×1 conv ───────────────────────────────────────────────
    let output = aspp.project_conv.forward(&concat)?;
    Ok(output)
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helper: bilinear upsampling for HWC tensors
// ─────────────────────────────────────────────────────────────────────────────

/// Bilinear upsample a `[H_in, W_in, C]` tensor to `[H_out, W_out, C]`.
///
/// This is an internal helper used by ASPP; for the public API see
/// [`crate::segmentation::semantic::fcn::bilinear_upsample_mask`].
fn upsample_bilinear_hwc(input: &Array3<f32>, out_h: usize, out_w: usize) -> Result<Array3<f32>> {
    let (in_h, in_w, c) = input.dim();
    if out_h == 0 || out_w == 0 {
        return Err(VisionError::InvalidParameter(
            "upsample_bilinear_hwc: output dimensions must be > 0".into(),
        ));
    }

    let mut out = Array3::<f32>::zeros((out_h, out_w, c));
    let scale_y = in_h as f32 / out_h as f32;
    let scale_x = in_w as f32 / out_w as f32;

    for oy in 0..out_h {
        let src_y = (oy as f32 + 0.5) * scale_y - 0.5;
        let y0 = (src_y.floor() as isize).max(0) as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let dy = src_y - src_y.floor();

        for ox in 0..out_w {
            let src_x = (ox as f32 + 0.5) * scale_x - 0.5;
            let x0 = (src_x.floor() as isize).max(0) as usize;
            let x1 = (x0 + 1).min(in_w - 1);
            let dx = src_x - src_x.floor();

            let w00 = (1.0 - dy) * (1.0 - dx);
            let w01 = (1.0 - dy) * dx;
            let w10 = dy * (1.0 - dx);
            let w11 = dy * dx;

            for ci in 0..c {
                out[[oy, ox, ci]] = w00 * input[[y0, x0, ci]]
                    + w01 * input[[y0, x1, ci]]
                    + w10 * input[[y1, x0, ci]]
                    + w11 * input[[y1, x1, ci]];
            }
        }
    }

    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_feature(h: usize, w: usize, c: usize) -> Array3<f32> {
        Array3::from_shape_fn((h, w, c), |(y, x, ci)| (y + x + ci) as f32 * 0.1)
    }

    // ── AtrousConv2D ──────────────────────────────────────────────────────

    #[test]
    fn test_atrous_conv_identity_preserves_shape() {
        let cfg = AtrousConv2DConfig {
            kernel_h: 3,
            kernel_w: 3,
            in_channels: 4,
            out_channels: 4,
            dilation: 1,
            same_padding: true,
            use_bias: false,
        };
        let conv = AtrousConv2D::identity(cfg, 1.0);
        let feat = make_feature(8, 8, 4);
        let out = conv.forward(&feat).expect("forward failed");
        assert_eq!(out.dim(), (8, 8, 4));
    }

    #[test]
    fn test_atrous_conv_dilation_2_same_size() {
        let cfg = AtrousConv2DConfig {
            kernel_h: 3,
            kernel_w: 3,
            in_channels: 2,
            out_channels: 2,
            dilation: 2,
            same_padding: true,
            use_bias: false,
        };
        let conv = AtrousConv2D::identity(cfg, 1.0);
        let feat = make_feature(10, 10, 2);
        let out = conv.forward(&feat).expect("forward with dilation=2 failed");
        assert_eq!(out.dim(), (10, 10, 2));
    }

    #[test]
    fn test_atrous_conv_dilation_6() {
        let cfg = AtrousConv2DConfig {
            kernel_h: 3,
            kernel_w: 3,
            in_channels: 3,
            out_channels: 3,
            dilation: 6,
            same_padding: true,
            use_bias: false,
        };
        let conv = AtrousConv2D::identity(cfg, 1.0);
        let feat = make_feature(16, 16, 3);
        let out = conv.forward(&feat).expect("forward with dilation=6 failed");
        assert_eq!(out.dim(), (16, 16, 3));
    }

    #[test]
    fn test_atrous_conv_channel_mismatch() {
        let cfg = AtrousConv2DConfig {
            in_channels: 4,
            out_channels: 4,
            ..Default::default()
        };
        let conv = AtrousConv2D::identity(cfg, 1.0);
        let feat = make_feature(8, 8, 3); // wrong channels
        let res = conv.forward(&feat);
        assert!(res.is_err());
    }

    #[test]
    fn test_atrous_conv_pointwise() {
        // 1×1 conv with dilation=1 is just a projection.
        let cfg = AtrousConv2DConfig {
            kernel_h: 1,
            kernel_w: 1,
            in_channels: 8,
            out_channels: 4,
            dilation: 1,
            same_padding: true,
            use_bias: false,
        };
        let conv = AtrousConv2D::identity(cfg, 0.5);
        let feat = make_feature(6, 6, 8);
        let out = conv.forward(&feat).expect("1×1 conv failed");
        assert_eq!(out.dim(), (6, 6, 4));
    }

    #[test]
    fn test_atrous_conv_valid_padding() {
        // valid padding reduces spatial dims.
        let cfg = AtrousConv2DConfig {
            kernel_h: 3,
            kernel_w: 3,
            in_channels: 2,
            out_channels: 2,
            dilation: 1,
            same_padding: false,
            use_bias: false,
        };
        let conv = AtrousConv2D::identity(cfg, 1.0);
        let feat = make_feature(8, 8, 2);
        let out = conv.forward(&feat).expect("valid padding forward failed");
        assert_eq!(out.dim(), (6, 6, 2));
    }

    // ── global_average_pooling ────────────────────────────────────────────

    #[test]
    fn test_gap_shape() {
        let feat = make_feature(6, 8, 4);
        let gap = global_average_pooling(&feat).expect("gap failed");
        assert_eq!(gap.dim(), (1, 1, 4));
    }

    #[test]
    fn test_gap_constant_input() {
        let feat = Array3::from_elem((4, 4, 3), 2.0f32);
        let gap = global_average_pooling(&feat).expect("gap failed");
        for ci in 0..3 {
            assert!((gap[[0, 0, ci]] - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gap_invalid_zero_spatial() {
        let feat = Array3::<f32>::zeros((0, 4, 3));
        let res = global_average_pooling(&feat);
        assert!(res.is_err());
    }

    // ── ASPP ──────────────────────────────────────────────────────────────

    #[test]
    fn test_aspp_identity_init() {
        let cfg = ASPPConfig::deeplab_v3(16, 8);
        let aspp = ASPP::new_identity(cfg).expect("ASPP::new_identity failed");
        let feat = make_feature(12, 12, 16);
        let out = aspp_forward(&aspp, &feat).expect("aspp_forward failed");
        assert_eq!(out.dim(), (12, 12, 8));
    }

    #[test]
    fn test_aspp_channel_mismatch() {
        let cfg = ASPPConfig::deeplab_v3(16, 8);
        let aspp = ASPP::new_identity(cfg).expect("init failed");
        let feat = make_feature(8, 8, 4); // wrong channel count
        let res = aspp_forward(&aspp, &feat);
        assert!(res.is_err());
    }

    #[test]
    fn test_aspp_no_gap() {
        let cfg = ASPPConfig {
            in_channels: 8,
            branches: vec![
                ASPPBranch {
                    dilation: 1,
                    pointwise: true,
                    out_channels: 4,
                },
                ASPPBranch {
                    dilation: 6,
                    pointwise: false,
                    out_channels: 4,
                },
            ],
            use_gap_branch: false,
            project_channels: 4,
        };
        let aspp = ASPP::new_identity(cfg).expect("ASPP no gap failed");
        let feat = make_feature(8, 8, 8);
        let out = aspp_forward(&aspp, &feat).expect("aspp forward no gap failed");
        assert_eq!(out.dim(), (8, 8, 4));
    }

    // ── upsample_bilinear_hwc (internal) ──────────────────────────────────

    #[test]
    fn test_upsample_constant() {
        let input = Array3::from_elem((2, 2, 2), 1.0f32);
        let out = upsample_bilinear_hwc(&input, 6, 6).expect("upsample failed");
        assert_eq!(out.dim(), (6, 6, 2));
        for v in out.iter() {
            assert!((*v - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_upsample_invalid_zero_size() {
        let input = Array3::from_elem((2, 2, 2), 1.0f32);
        let res = upsample_bilinear_hwc(&input, 0, 4);
        assert!(res.is_err());
    }
}
