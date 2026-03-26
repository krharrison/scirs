//! 2D CNN backbone and feature-pyramid neck for BEV feature maps.

use scirs2_core::ndarray::Array2;

use crate::error::{Result, VisionError};

// ---------------------------------------------------------------------------
// Conv2D (lightweight implementation for this module)
// ---------------------------------------------------------------------------

/// Minimal 2D convolution layer (single-channel, square kernel).
#[derive(Debug, Clone)]
struct Conv2D {
    /// Kernel weights: (out_channels, in_channels * k * k).
    kernel: Array2<f64>,
    /// Bias per output channel.
    bias: Vec<f64>,
    /// Kernel spatial size.
    kernel_size: usize,
    /// Stride.
    stride: usize,
    /// Input channels.
    in_channels: usize,
    /// Output channels.
    out_channels: usize,
}

impl Conv2D {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize) -> Self {
        let k2 = kernel_size * kernel_size;
        let fan_in = in_channels * k2;
        let scale = (2.0 / fan_in as f64).sqrt();
        let mut kernel = Array2::zeros((out_channels, in_channels * k2));
        for i in 0..out_channels {
            for j in 0..in_channels * k2 {
                kernel[[i, j]] = ((i * 7 + j * 13 + 5) as f64).sin() * scale;
            }
        }
        Self {
            kernel,
            bias: vec![0.0; out_channels],
            kernel_size,
            stride,
            in_channels,
            out_channels,
        }
    }

    /// Apply convolution on a feature map stored as `(H * W, C)`.
    ///
    /// `h` and `w` are the spatial dimensions of the input. Returns `(H_out *
    /// W_out, out_channels)` together with the new `(h_out, w_out)`.
    fn forward(
        &self,
        input: &Array2<f64>,
        h: usize,
        w: usize,
    ) -> Result<(Array2<f64>, usize, usize)> {
        if input.nrows() != h * w {
            return Err(VisionError::DimensionMismatch(format!(
                "Conv2D: input rows {} != h*w {}",
                input.nrows(),
                h * w
            )));
        }
        if input.ncols() != self.in_channels {
            return Err(VisionError::DimensionMismatch(format!(
                "Conv2D: input cols {} != in_channels {}",
                input.ncols(),
                self.in_channels
            )));
        }

        let pad = self.kernel_size / 2;
        let h_out = (h + 2 * pad - self.kernel_size) / self.stride + 1;
        let w_out = (w + 2 * pad - self.kernel_size) / self.stride + 1;

        let mut out = Array2::zeros((h_out * w_out, self.out_channels));

        for oy in 0..h_out {
            for ox in 0..w_out {
                let out_idx = oy * w_out + ox;
                for oc in 0..self.out_channels {
                    let mut val = self.bias[oc];
                    for ky in 0..self.kernel_size {
                        for kx in 0..self.kernel_size {
                            let iy = oy * self.stride + ky;
                            let ix = ox * self.stride + kx;
                            // Subtract pad offset; clamp to zero-padding.
                            let iy_s = iy as isize - pad as isize;
                            let ix_s = ix as isize - pad as isize;
                            if iy_s < 0 || iy_s >= h as isize || ix_s < 0 || ix_s >= w as isize {
                                continue;
                            }
                            let in_idx = iy_s as usize * w + ix_s as usize;
                            for ic in 0..self.in_channels {
                                let k_idx = ic * self.kernel_size * self.kernel_size
                                    + ky * self.kernel_size
                                    + kx;
                                val += input[[in_idx, ic]] * self.kernel[[oc, k_idx]];
                            }
                        }
                    }
                    out[[out_idx, oc]] = val;
                }
            }
        }

        Ok((out, h_out, w_out))
    }
}

/// Apply ReLU in-place.
fn relu_inplace(arr: &mut Array2<f64>) {
    arr.mapv_inplace(|v| v.max(0.0));
}

// ---------------------------------------------------------------------------
// SimpleBEVBackbone
// ---------------------------------------------------------------------------

/// Three-block 2D CNN backbone operating on BEV feature maps.
///
/// Block layout:
/// 1. Conv2D(in, 64, 3, stride=2) + ReLU
/// 2. Conv2D(64, 128, 3, stride=2) + ReLU
/// 3. Conv2D(128, 256, 3, stride=2) + ReLU
#[derive(Debug, Clone)]
pub struct SimpleBEVBackbone {
    conv1: Conv2D,
    conv2: Conv2D,
    conv3: Conv2D,
}

impl SimpleBEVBackbone {
    /// Create a new backbone with `in_channels` input feature channels.
    pub fn new(in_channels: usize) -> Self {
        Self {
            conv1: Conv2D::new(in_channels, 64, 3, 2),
            conv2: Conv2D::new(64, 128, 3, 2),
            conv3: Conv2D::new(128, 256, 3, 2),
        }
    }

    /// Run the backbone. `bev_features` has shape `(H*W, C)`.
    ///
    /// Returns a list of three multi-scale feature maps together with their
    /// spatial dimensions: `[(features, h, w); 3]`.
    pub fn forward(
        &self,
        bev_features: &Array2<f64>,
        h: usize,
        w: usize,
    ) -> Result<Vec<(Array2<f64>, usize, usize)>> {
        let (mut f1, h1, w1) = self.conv1.forward(bev_features, h, w)?;
        relu_inplace(&mut f1);

        let (mut f2, h2, w2) = self.conv2.forward(&f1, h1, w1)?;
        relu_inplace(&mut f2);

        let (mut f3, h3, w3) = self.conv3.forward(&f2, h2, w2)?;
        relu_inplace(&mut f3);

        Ok(vec![(f1, h1, w1), (f2, h2, w2), (f3, h3, w3)])
    }
}

// ---------------------------------------------------------------------------
// FeaturePyramidNeck
// ---------------------------------------------------------------------------

/// Feature Pyramid Network neck: upsamples and concatenates multi-scale features
/// from the backbone into a unified feature map at the first scale.
#[derive(Debug, Clone)]
pub struct FeaturePyramidNeck {
    /// 1x1 convolutions to unify channel counts.
    lateral2: Conv2D,
    lateral3: Conv2D,
    /// Target output channels.
    out_channels: usize,
}

impl FeaturePyramidNeck {
    /// Create a new FPN neck that maps all backbone scales to `out_channels`.
    pub fn new(out_channels: usize) -> Self {
        // 1x1 convs to reduce channels to out_channels.
        Self {
            lateral2: Conv2D::new(128, out_channels, 1, 1),
            lateral3: Conv2D::new(256, out_channels, 1, 1),
            out_channels,
        }
    }

    /// Fuse multi-scale features.
    ///
    /// Takes the three feature maps from `SimpleBEVBackbone::forward` and
    /// returns a single `(H1*W1, out_channels)` feature map at the first
    /// backbone scale.
    pub fn forward(
        &self,
        scales: &[(Array2<f64>, usize, usize)],
    ) -> Result<(Array2<f64>, usize, usize)> {
        if scales.len() < 3 {
            return Err(VisionError::InvalidParameter(
                "FeaturePyramidNeck expects 3 scales".to_string(),
            ));
        }

        let (ref f1, h1, w1) = scales[0];
        let (ref f2, h2, w2) = scales[1];
        let (ref f3, h3, w3) = scales[2];

        // Lateral projections.
        let (l2, _, _) = self.lateral2.forward(f2, h2, w2)?;
        let (l3, _, _) = self.lateral3.forward(f3, h3, w3)?;

        // Upsample l3 to l2 size via nearest-neighbour.
        let l3_up = nearest_upsample(&l3, h3, w3, h2, w2);
        // Add l3_up + l2.
        let fused2 = elementwise_add(&l3_up, &l2)?;

        // Upsample fused2 to f1 size.
        let fused2_up = nearest_upsample(&fused2, h2, w2, h1, w1);

        // If f1 channels differ from out_channels, we need a lateral; for
        // simplicity, just take the first `out_channels` columns or pad.
        let n1_cols = f1.ncols();
        let mut out = Array2::zeros((h1 * w1, self.out_channels));
        let copy_cols = n1_cols.min(self.out_channels);
        for r in 0..h1 * w1 {
            for c in 0..copy_cols {
                out[[r, c]] = f1[[r, c]];
            }
            // Add upsampled fused features.
            for c in 0..self.out_channels {
                out[[r, c]] += fused2_up[[r, c]];
            }
        }

        Ok((out, h1, w1))
    }
}

/// Nearest-neighbour upsample `(h_in * w_in, C)` → `(h_out * w_out, C)`.
fn nearest_upsample(
    feat: &Array2<f64>,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) -> Array2<f64> {
    let c = feat.ncols();
    let mut out = Array2::zeros((h_out * w_out, c));
    for oy in 0..h_out {
        let iy = (oy * h_in) / h_out.max(1);
        let iy = iy.min(h_in.saturating_sub(1));
        for ox in 0..w_out {
            let ix = (ox * w_in) / w_out.max(1);
            let ix = ix.min(w_in.saturating_sub(1));
            let src = iy * w_in + ix;
            let dst = oy * w_out + ox;
            for k in 0..c {
                out[[dst, k]] = feat[[src, k]];
            }
        }
    }
    out
}

/// Element-wise addition of two arrays with identical shapes.
fn elementwise_add(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
    if a.shape() != b.shape() {
        return Err(VisionError::DimensionMismatch(format!(
            "elementwise_add: shapes {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    Ok(a + b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn conv2d_output_shape() {
        let conv = Conv2D::new(4, 8, 3, 2);
        let input = Array2::zeros((16 * 16, 4));
        let (out, h, w) = conv.forward(&input, 16, 16).expect("conv forward");
        assert_eq!(h, 8);
        assert_eq!(w, 8);
        assert_eq!(out.nrows(), 64);
        assert_eq!(out.ncols(), 8);
    }

    #[test]
    fn backbone_forward() {
        let bb = SimpleBEVBackbone::new(4);
        let input = Array2::zeros((16 * 16, 4));
        let scales = bb.forward(&input, 16, 16).expect("backbone forward");
        assert_eq!(scales.len(), 3);
        // Each scale halves spatial dims.
        assert_eq!(scales[0].1, 8);
        assert_eq!(scales[1].1, 4);
        assert_eq!(scales[2].1, 2);
    }

    #[test]
    fn fpn_forward() {
        let bb = SimpleBEVBackbone::new(4);
        let input = Array2::zeros((16 * 16, 4));
        let scales = bb.forward(&input, 16, 16).expect("backbone forward");
        let fpn = FeaturePyramidNeck::new(64);
        let (out, h, w) = fpn.forward(&scales).expect("fpn forward");
        assert_eq!(h, 8);
        assert_eq!(w, 8);
        assert_eq!(out.ncols(), 64);
    }

    #[test]
    fn nearest_upsample_identity() {
        let feat = Array2::from_elem((4, 2), 1.0);
        let up = nearest_upsample(&feat, 2, 2, 2, 2);
        assert_eq!(up.shape(), &[4, 2]);
        assert!((up[[0, 0]] - 1.0).abs() < 1e-12);
    }
}
