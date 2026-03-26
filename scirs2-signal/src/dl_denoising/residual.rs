//! Residual learning denoiser (DnCNN-inspired).
//!
//! Architecture: Conv1D(1,64,3) -> [Conv1D(64,64,3) -> BN -> ReLU] x depth -> Conv1D(64,1,3)
//!
//! The network learns the noise residual n_hat = f(x_noisy).
//! Clean estimate: x_clean = x_noisy - n_hat.
//! Dilated convolutions are used for larger receptive field.

use super::types::{Conv1DParams, DenoisingResult};

/// Batch normalization for 1D signals (per-channel).
#[derive(Debug, Clone)]
pub struct BatchNorm1D {
    /// Running mean per channel.
    pub running_mean: Vec<f64>,
    /// Running variance per channel.
    pub running_var: Vec<f64>,
    /// Learnable scale (gamma).
    pub gamma: Vec<f64>,
    /// Learnable shift (beta).
    pub beta: Vec<f64>,
    /// Number of channels.
    pub num_channels: usize,
    /// Momentum for running stats update.
    pub momentum: f64,
    /// Small constant for numerical stability.
    pub epsilon: f64,
}

impl BatchNorm1D {
    /// Create a new batch norm layer for the given number of channels.
    pub fn new(num_channels: usize) -> Self {
        Self {
            running_mean: vec![0.0; num_channels],
            running_var: vec![1.0; num_channels],
            gamma: vec![1.0; num_channels],
            beta: vec![0.0; num_channels],
            num_channels,
            momentum: 0.1,
            epsilon: 1e-5,
        }
    }

    /// Forward pass (inference mode: use running stats).
    ///
    /// `input`: `[num_channels][spatial_len]`
    pub fn forward(&mut self, input: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let effective_ch = input.len().min(self.num_channels);
        input
            .iter()
            .enumerate()
            .map(|(ch, data)| {
                if ch >= effective_ch || data.is_empty() {
                    return data.clone();
                }
                // Compute batch statistics from this channel
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

                // Update running stats
                self.running_mean[ch] =
                    (1.0 - self.momentum) * self.running_mean[ch] + self.momentum * mean;
                self.running_var[ch] =
                    (1.0 - self.momentum) * self.running_var[ch] + self.momentum * var;

                let inv_std = 1.0 / (var + self.epsilon).sqrt();
                data.iter()
                    .map(|&x| self.gamma[ch] * (x - mean) * inv_std + self.beta[ch])
                    .collect()
            })
            .collect()
    }
}

/// A residual-learning denoiser inspired by DnCNN.
#[derive(Debug, Clone)]
pub struct ResidualDenoiser {
    /// First convolution: 1 -> 64 channels.
    first_conv: Conv1DParams,
    /// Intermediate blocks: Conv(64,64,3) + BN + ReLU, with dilation.
    mid_convs: Vec<Conv1DParams>,
    /// Batch norm layers for each intermediate block.
    mid_bns: Vec<BatchNorm1D>,
    /// Dilation factors for each intermediate block.
    dilations: Vec<usize>,
    /// Final convolution: 64 -> 1 channel.
    last_conv: Conv1DParams,
    /// Network depth (number of intermediate blocks).
    depth: usize,
}

impl ResidualDenoiser {
    /// Create a new residual denoiser with the given depth.
    ///
    /// Default depth is 7 intermediate blocks.
    pub fn new(depth: usize) -> Self {
        let seed_base: u64 = 987654321;
        let depth = if depth == 0 { 7 } else { depth };

        let first_conv = Conv1DParams::new_random(1, 64, 3, 1, seed_base);

        let mut mid_convs = Vec::with_capacity(depth);
        let mut mid_bns = Vec::with_capacity(depth);
        let mut dilations = Vec::with_capacity(depth);

        for i in 0..depth {
            mid_convs.push(Conv1DParams::new_random(
                64,
                64,
                3,
                1,
                seed_base.wrapping_add(i as u64 + 1),
            ));
            mid_bns.push(BatchNorm1D::new(64));
            // Dilation pattern: 1, 2, 4, 8, 4, 2, 1 (expanding then contracting)
            let half = depth / 2;
            let dilation = if i <= half {
                1usize << i.min(4)
            } else {
                1usize << (depth - i).min(4)
            };
            dilations.push(dilation);
        }

        let last_conv =
            Conv1DParams::new_random(64, 1, 3, 1, seed_base.wrapping_add(depth as u64 + 1));

        Self {
            first_conv,
            mid_convs,
            mid_bns,
            dilations,
            last_conv,
            depth,
        }
    }

    /// Forward pass: predict the noise residual, then subtract from input.
    pub fn forward(&mut self, noisy: &[f64]) -> DenoisingResult {
        // First conv + ReLU
        let input_channels = vec![noisy.to_vec()];
        let mut x = conv1d_multi(&input_channels, &self.first_conv);
        x = apply_relu(&x);

        // Intermediate blocks: Conv -> BN -> ReLU (with dilation)
        for i in 0..self.depth {
            let dilated = dilated_conv1d_multi(&x, &self.mid_convs[i], self.dilations[i]);
            let normed = self.mid_bns[i].forward(&dilated);
            x = apply_relu(&normed);
        }

        // Final conv: predict noise residual
        let residual_multi = conv1d_multi(&x, &self.last_conv);
        let noise_hat = if residual_multi.is_empty() {
            vec![0.0; noisy.len()]
        } else {
            let mut r = residual_multi[0].clone();
            r.resize(noisy.len(), 0.0);
            r
        };

        // Clean = noisy - noise_hat
        let denoised: Vec<f64> = noisy
            .iter()
            .zip(noise_hat.iter())
            .map(|(n, nh)| n - nh)
            .collect();

        let signal_power: f64 = denoised.iter().map(|x| x * x).sum::<f64>();
        let noise_power: f64 = noise_hat.iter().map(|x| x * x).sum::<f64>();
        let snr_improvement = if noise_power > 1e-15 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f64::INFINITY
        };

        DenoisingResult {
            denoised,
            noise_estimate: noise_hat,
            snr_improvement,
        }
    }

    /// Perform one training step. Returns MSE loss.
    pub fn train_step(&mut self, noisy: &[f64], clean: &[f64], lr: f64) -> f64 {
        let result = self.forward(noisy);
        let n = result.denoised.len().min(clean.len());
        if n == 0 {
            return 0.0;
        }

        let mse: f64 = result
            .denoised
            .iter()
            .zip(clean.iter())
            .take(n)
            .map(|(p, c)| (p - c).powi(2))
            .sum::<f64>()
            / n as f64;

        // Simplified gradient update on first_conv bias
        let epsilon = 1e-5;
        for i in 0..self.first_conv.bias.len() {
            let original = self.first_conv.bias[i];

            self.first_conv.bias[i] = original + epsilon;
            let r_plus = self.forward(noisy);
            let loss_plus: f64 = r_plus
                .denoised
                .iter()
                .zip(clean.iter())
                .take(n)
                .map(|(p, c)| (p - c).powi(2))
                .sum::<f64>()
                / n as f64;

            self.first_conv.bias[i] = original - epsilon;
            let r_minus = self.forward(noisy);
            let loss_minus: f64 = r_minus
                .denoised
                .iter()
                .zip(clean.iter())
                .take(n)
                .map(|(p, c)| (p - c).powi(2))
                .sum::<f64>()
                / n as f64;

            let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
            self.first_conv.bias[i] = original - lr * grad;
        }

        mse
    }

    /// Get the network depth.
    pub fn depth(&self) -> usize {
        self.depth
    }
}

// ---- Internal helpers ----

/// Basic 1D convolution on multi-channel input (stride=1, valid padding).
fn conv1d_multi(input: &[Vec<f64>], params: &Conv1DParams) -> Vec<Vec<f64>> {
    let in_ch = input.len();
    if in_ch == 0 {
        return vec![Vec::new(); params.out_channels];
    }
    let spatial = input[0].len();
    if spatial < params.kernel_size {
        let padded: Vec<Vec<f64>> = input
            .iter()
            .map(|ch| {
                let mut p = ch.clone();
                p.resize(params.kernel_size, 0.0);
                p
            })
            .collect();
        return conv1d_multi(&padded, params);
    }

    let out_len = (spatial - params.kernel_size) / params.stride + 1;
    let effective_in_ch = in_ch.min(params.in_channels);

    (0..params.out_channels)
        .map(|oc| {
            let w = &params.weights[oc];
            let bias = params.bias[oc];
            (0..out_len)
                .map(|pos| {
                    let start = pos * params.stride;
                    let mut val = bias;
                    for ic in 0..effective_in_ch {
                        for k in 0..params.kernel_size {
                            let w_idx = ic * params.kernel_size + k;
                            if w_idx < w.len() {
                                val += w[w_idx] * input[ic][start + k];
                            }
                        }
                    }
                    val
                })
                .collect()
        })
        .collect()
}

/// Dilated 1D convolution on multi-channel input.
///
/// Dilation inserts `dilation - 1` zeros between kernel elements,
/// effectively sampling input at `start + k * dilation`.
fn dilated_conv1d_multi(
    input: &[Vec<f64>],
    params: &Conv1DParams,
    dilation: usize,
) -> Vec<Vec<f64>> {
    let in_ch = input.len();
    if in_ch == 0 {
        return vec![Vec::new(); params.out_channels];
    }
    let spatial = input[0].len();
    let effective_kernel = (params.kernel_size - 1) * dilation + 1;
    if spatial < effective_kernel {
        // Pad input
        let padded: Vec<Vec<f64>> = input
            .iter()
            .map(|ch| {
                let mut p = ch.clone();
                p.resize(effective_kernel, 0.0);
                p
            })
            .collect();
        return dilated_conv1d_multi(&padded, params, dilation);
    }

    let out_len = spatial - effective_kernel + 1;
    let effective_in_ch = in_ch.min(params.in_channels);

    (0..params.out_channels)
        .map(|oc| {
            let w = &params.weights[oc];
            let bias = params.bias[oc];
            (0..out_len)
                .map(|pos| {
                    let mut val = bias;
                    for ic in 0..effective_in_ch {
                        for k in 0..params.kernel_size {
                            let w_idx = ic * params.kernel_size + k;
                            let in_idx = pos + k * dilation;
                            if w_idx < w.len() && in_idx < spatial {
                                val += w[w_idx] * input[ic][in_idx];
                            }
                        }
                    }
                    val
                })
                .collect()
        })
        .collect()
}

/// Apply ReLU to all channels.
fn apply_relu(channels: &[Vec<f64>]) -> Vec<Vec<f64>> {
    channels
        .iter()
        .map(|ch| ch.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_norm_forward() {
        let mut bn = BatchNorm1D::new(2);
        let input = vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]];
        let out = bn.forward(&input);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 4);
        // After BN, mean should be ~0 and std ~1
        let mean: f64 = out[0].iter().sum::<f64>() / out[0].len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_residual_denoiser_forward() {
        let mut denoiser = ResidualDenoiser::new(3);
        let noisy: Vec<f64> = (0..128).map(|i| (i as f64 * 0.05).sin() + 0.1).collect();
        let result = denoiser.forward(&noisy);
        assert_eq!(result.denoised.len(), noisy.len());
        assert_eq!(result.noise_estimate.len(), noisy.len());
        assert!(result.snr_improvement.is_finite() || result.snr_improvement == f64::INFINITY);
    }

    #[test]
    fn test_residual_denoiser_train_step() {
        let mut denoiser = ResidualDenoiser::new(2);
        let clean: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let noisy: Vec<f64> = clean.iter().map(|&x| x + 0.05).collect();
        let loss = denoiser.train_step(&noisy, &clean, 0.001);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_dilated_conv1d() {
        let params = Conv1DParams {
            weights: vec![vec![1.0, 0.0, 1.0]],
            bias: vec![0.0],
            kernel_size: 3,
            stride: 1,
            in_channels: 1,
            out_channels: 1,
        };
        let input = vec![vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0]];
        // dilation=2: kernel spans positions 0,2,4 then 1,3,5 etc.
        let out = dilated_conv1d_multi(&input, &params, 2);
        assert_eq!(out.len(), 1);
        // effective kernel size = (3-1)*2+1 = 5, so out_len = 7-5+1 = 3
        assert_eq!(out[0].len(), 3);
        // out[0] = input[0]*1 + input[2]*0 + input[4]*1 = 1+3 = 4
        assert!((out[0][0] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_residual_default_depth() {
        let denoiser = ResidualDenoiser::new(0);
        assert_eq!(denoiser.depth(), 7);
    }
}
