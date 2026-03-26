//! Denoising autoencoder with U-Net style skip connections.
//!
//! Architecture:
//! - Encoder: 3 stages of Conv1D -> ReLU -> AvgPool(2), channels 1->16->32->64
//! - Bottleneck: linear projection to `latent_dim`
//! - Decoder: 3 stages of Upsample(2) -> Conv1D -> ReLU, channels 64->32->16->1
//! - Skip connections: encoder features added to corresponding decoder stage

use super::types::{Conv1DParams, DLDenoiseConfig, DenoisingResult};

/// A denoising autoencoder with U-Net skip connections.
#[derive(Debug, Clone)]
pub struct DenoisingAutoencoder {
    // Encoder layers
    enc_conv1: Conv1DParams, // 1 -> 16
    enc_conv2: Conv1DParams, // 16 -> 32
    enc_conv3: Conv1DParams, // 32 -> 64

    // Bottleneck (linear projection)
    bottleneck_down_w: Vec<Vec<f64>>, // 64 -> latent_dim
    bottleneck_down_b: Vec<f64>,
    bottleneck_up_w: Vec<Vec<f64>>, // latent_dim -> 64
    bottleneck_up_b: Vec<f64>,

    // Decoder layers (after skip connection addition, channel count doubles)
    dec_conv1: Conv1DParams, // 128 -> 32  (64 from upsample + 64 skip from enc3)
    dec_conv2: Conv1DParams, // 64 -> 16   (32 + 32 skip from enc2)
    dec_conv3: Conv1DParams, // 32 -> 1    (16 + 16 skip from enc1)

    latent_dim: usize,
}

impl DenoisingAutoencoder {
    /// Create a new denoising autoencoder from the given config.
    pub fn new(config: &DLDenoiseConfig) -> Self {
        let ld = config.latent_dim;
        let seed_base: u64 = 123456789;

        // Encoder convolutions
        let enc_conv1 = Conv1DParams::new_random(1, 16, 3, 1, seed_base);
        let enc_conv2 = Conv1DParams::new_random(16, 32, 3, 1, seed_base.wrapping_add(1));
        let enc_conv3 = Conv1DParams::new_random(32, 64, 3, 1, seed_base.wrapping_add(2));

        // Bottleneck linear layers
        let (bottleneck_down_w, bottleneck_down_b) =
            random_linear(64, ld, seed_base.wrapping_add(3));
        let (bottleneck_up_w, bottleneck_up_b) = random_linear(ld, 64, seed_base.wrapping_add(4));

        // Decoder convolutions (input channels doubled due to skip connections)
        let dec_conv1 = Conv1DParams::new_random(128, 32, 3, 1, seed_base.wrapping_add(5));
        let dec_conv2 = Conv1DParams::new_random(64, 16, 3, 1, seed_base.wrapping_add(6));
        let dec_conv3 = Conv1DParams::new_random(32, 1, 3, 1, seed_base.wrapping_add(7));

        Self {
            enc_conv1,
            enc_conv2,
            enc_conv3,
            bottleneck_down_w,
            bottleneck_down_b,
            bottleneck_up_w,
            bottleneck_up_b,
            dec_conv1,
            dec_conv2,
            dec_conv3,
            latent_dim: ld,
        }
    }

    /// Forward pass: encode then decode the noisy signal.
    pub fn forward(&self, noisy: &[f64]) -> Vec<f64> {
        // --- Encoder ---
        // Stage 1: Conv(1->16) -> ReLU -> AvgPool(2)
        let e1_conv = conv1d_forward_multi(
            &[noisy.to_vec()], // 1 channel
            &self.enc_conv1,
        );
        let e1_relu = multi_channel_relu(&e1_conv);
        let e1_pool = multi_channel_avg_pool(&e1_relu, 2);

        // Stage 2: Conv(16->32) -> ReLU -> AvgPool(2)
        let e2_conv = conv1d_forward_multi(&e1_pool, &self.enc_conv2);
        let e2_relu = multi_channel_relu(&e2_conv);
        let e2_pool = multi_channel_avg_pool(&e2_relu, 2);

        // Stage 3: Conv(32->64) -> ReLU -> AvgPool(2)
        let e3_conv = conv1d_forward_multi(&e2_pool, &self.enc_conv3);
        let e3_relu = multi_channel_relu(&e3_conv);
        let e3_pool = multi_channel_avg_pool(&e3_relu, 2);

        // --- Bottleneck ---
        // Global average each channel to get a vector of length 64
        let channel_means: Vec<f64> = e3_pool
            .iter()
            .map(|ch| {
                if ch.is_empty() {
                    0.0
                } else {
                    ch.iter().sum::<f64>() / ch.len() as f64
                }
            })
            .collect();

        let latent = linear_forward(
            &channel_means,
            &self.bottleneck_down_w,
            &self.bottleneck_down_b,
        );
        let latent_relu: Vec<f64> = latent.iter().map(|&v| relu(v)).collect();
        let expanded = linear_forward(&latent_relu, &self.bottleneck_up_w, &self.bottleneck_up_b);

        // Broadcast expanded back to spatial dimensions matching e3_pool
        let spatial_len = if e3_pool.is_empty() || e3_pool[0].is_empty() {
            1
        } else {
            e3_pool[0].len()
        };
        let bottleneck_out: Vec<Vec<f64>> =
            expanded.iter().map(|&val| vec![val; spatial_len]).collect();

        // --- Decoder ---
        // Stage 1: Upsample(2) -> Concat skip(e3_relu) -> Conv(128->32) -> ReLU
        let d1_up = multi_channel_upsample(&bottleneck_out, 2);
        let d1_cat = concat_skip(&d1_up, &e3_relu);
        let d1_conv = conv1d_forward_multi(&d1_cat, &self.dec_conv1);
        let d1_relu = multi_channel_relu(&d1_conv);

        // Stage 2: Upsample(2) -> Concat skip(e2_relu) -> Conv(64->16) -> ReLU
        let d2_up = multi_channel_upsample(&d1_relu, 2);
        let d2_cat = concat_skip(&d2_up, &e2_relu);
        let d2_conv = conv1d_forward_multi(&d2_cat, &self.dec_conv2);
        let d2_relu = multi_channel_relu(&d2_conv);

        // Stage 3: Upsample(2) -> Concat skip(e1_relu) -> Conv(32->1) -> (no ReLU on final)
        let d3_up = multi_channel_upsample(&d2_relu, 2);
        let d3_cat = concat_skip(&d3_up, &e1_relu);
        let d3_conv = conv1d_forward_multi(&d3_cat, &self.dec_conv3);

        // Output: single channel
        if d3_conv.is_empty() {
            vec![0.0; noisy.len()]
        } else {
            // Trim or pad to match input length
            let mut out = d3_conv[0].clone();
            out.resize(noisy.len(), 0.0);
            out
        }
    }

    /// Perform one training step (MSE loss, simple gradient descent).
    ///
    /// Returns the MSE loss for this step.
    pub fn train_step(&mut self, noisy: &[f64], clean: &[f64], lr: f64) -> f64 {
        let predicted = self.forward(noisy);
        let n = predicted.len().min(clean.len());
        if n == 0 {
            return 0.0;
        }

        // Compute MSE loss
        let mse: f64 = predicted
            .iter()
            .zip(clean.iter())
            .take(n)
            .map(|(p, c)| (p - c).powi(2))
            .sum::<f64>()
            / n as f64;

        // Numerical gradient descent on encoder conv1 weights (simplified)
        // We perturb each weight and update based on finite-difference gradient
        let epsilon = 1e-5;
        self.numerical_gradient_update(&self.enc_conv1.clone(), noisy, clean, lr, epsilon, 0);
        self.numerical_gradient_update(&self.enc_conv2.clone(), noisy, clean, lr, epsilon, 1);
        self.numerical_gradient_update(&self.enc_conv3.clone(), noisy, clean, lr, epsilon, 2);

        mse
    }

    /// Denoise a signal, returning a full `DenoisingResult`.
    pub fn denoise(&self, noisy: &[f64]) -> DenoisingResult {
        let denoised = self.forward(noisy);
        let noise_estimate: Vec<f64> = noisy
            .iter()
            .zip(denoised.iter())
            .map(|(n, d)| n - d)
            .collect();

        let signal_power: f64 = denoised.iter().map(|x| x * x).sum::<f64>();
        let noise_power: f64 = noise_estimate.iter().map(|x| x * x).sum::<f64>();
        let snr_improvement = if noise_power > 1e-15 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            f64::INFINITY
        };

        DenoisingResult {
            denoised,
            noise_estimate,
            snr_improvement,
        }
    }

    /// Simplified numerical gradient update for a given conv layer.
    fn numerical_gradient_update(
        &mut self,
        _params: &Conv1DParams,
        noisy: &[f64],
        clean: &[f64],
        lr: f64,
        epsilon: f64,
        layer_idx: usize,
    ) {
        let n = noisy.len().min(clean.len());
        if n == 0 {
            return;
        }

        // Get the number of biases for this layer
        let num_biases = match layer_idx {
            0 => self.enc_conv1.bias.len(),
            1 => self.enc_conv2.bias.len(),
            2 => self.enc_conv3.bias.len(),
            _ => return,
        };

        for i in 0..num_biases {
            let original = self.get_bias(layer_idx, i);

            // f(x + eps)
            self.set_bias(layer_idx, i, original + epsilon);
            let pred_plus = self.forward(noisy);
            let loss_plus: f64 = pred_plus
                .iter()
                .zip(clean.iter())
                .take(n)
                .map(|(p, c)| (p - c).powi(2))
                .sum::<f64>()
                / n as f64;

            // f(x - eps)
            self.set_bias(layer_idx, i, original - epsilon);
            let pred_minus = self.forward(noisy);
            let loss_minus: f64 = pred_minus
                .iter()
                .zip(clean.iter())
                .take(n)
                .map(|(p, c)| (p - c).powi(2))
                .sum::<f64>()
                / n as f64;

            let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
            self.set_bias(layer_idx, i, original - lr * grad);
        }
    }

    /// Helper to get a bias value by layer index and position.
    fn get_bias(&self, layer_idx: usize, i: usize) -> f64 {
        match layer_idx {
            0 => self.enc_conv1.bias[i],
            1 => self.enc_conv2.bias[i],
            2 => self.enc_conv3.bias[i],
            _ => 0.0,
        }
    }

    /// Helper to set a bias value by layer index and position.
    fn set_bias(&mut self, layer_idx: usize, i: usize, val: f64) {
        match layer_idx {
            0 => self.enc_conv1.bias[i] = val,
            1 => self.enc_conv2.bias[i] = val,
            2 => self.enc_conv3.bias[i] = val,
            _ => {}
        }
    }
}

// ---- Internal helpers ----

/// ReLU activation.
fn relu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// Apply ReLU to each channel.
fn multi_channel_relu(channels: &[Vec<f64>]) -> Vec<Vec<f64>> {
    channels
        .iter()
        .map(|ch| ch.iter().map(|&v| relu(v)).collect())
        .collect()
}

/// Average pooling with the given kernel size across each channel.
fn avg_pool_1d(input: &[f64], kernel_size: usize) -> Vec<f64> {
    if kernel_size == 0 || input.is_empty() {
        return input.to_vec();
    }
    let out_len = input.len() / kernel_size;
    (0..out_len)
        .map(|i| {
            let start = i * kernel_size;
            let end = (start + kernel_size).min(input.len());
            let sum: f64 = input[start..end].iter().sum();
            sum / (end - start) as f64
        })
        .collect()
}

/// Average pooling on each channel.
fn multi_channel_avg_pool(channels: &[Vec<f64>], kernel_size: usize) -> Vec<Vec<f64>> {
    channels
        .iter()
        .map(|ch| avg_pool_1d(ch, kernel_size))
        .collect()
}

/// Nearest-neighbour upsample by the given factor.
fn upsample_1d(input: &[f64], factor: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(input.len() * factor);
    for &v in input {
        for _ in 0..factor {
            out.push(v);
        }
    }
    out
}

/// Upsample each channel.
fn multi_channel_upsample(channels: &[Vec<f64>], factor: usize) -> Vec<Vec<f64>> {
    channels.iter().map(|ch| upsample_1d(ch, factor)).collect()
}

/// 1D convolution for multi-channel input.
///
/// `input`: `[in_channels][spatial_len]`
/// `params`: Conv1DParams with weights `[out_channels][in_channels * kernel_size]`
///
/// Returns `[out_channels][output_spatial_len]`.
fn conv1d_forward_multi(input: &[Vec<f64>], params: &Conv1DParams) -> Vec<Vec<f64>> {
    let in_ch = input.len();
    if in_ch == 0 {
        return vec![Vec::new(); params.out_channels];
    }
    let spatial = input[0].len();
    if spatial < params.kernel_size {
        // Pad input so convolution produces at least 1 output
        let padded: Vec<Vec<f64>> = input
            .iter()
            .map(|ch| {
                let mut p = ch.clone();
                p.resize(params.kernel_size, 0.0);
                p
            })
            .collect();
        return conv1d_forward_multi(&padded, params);
    }

    let out_len = (spatial - params.kernel_size) / params.stride + 1;

    // Use same-padding: we pad so output length == input length / stride
    // Actually, let's just do valid convolution and let caller deal with sizes.
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

/// Concatenate skip connection features along channel dimension.
///
/// If spatial dimensions differ, the longer one is trimmed to match the shorter.
fn concat_skip(main: &[Vec<f64>], skip: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let min_len = main
        .iter()
        .chain(skip.iter())
        .map(|ch| ch.len())
        .min()
        .unwrap_or(0);

    let mut result: Vec<Vec<f64>> = main
        .iter()
        .map(|ch| ch[..min_len.min(ch.len())].to_vec())
        .collect();
    for ch in skip {
        result.push(ch[..min_len.min(ch.len())].to_vec());
    }
    result
}

/// Linear (fully-connected) layer forward: y = Wx + b.
fn linear_forward(input: &[f64], weights: &[Vec<f64>], bias: &[f64]) -> Vec<f64> {
    weights
        .iter()
        .zip(bias.iter())
        .map(|(w_row, &b)| {
            let dot: f64 = w_row.iter().zip(input.iter()).map(|(wi, xi)| wi * xi).sum();
            dot + b
        })
        .collect()
}

/// Create a random linear layer (weight matrix + bias) using LCG.
fn random_linear(in_dim: usize, out_dim: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let scale = (2.0 / in_dim as f64).sqrt();
    let mut state = seed;
    let mut next = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 11) as f64 / (1u64 << 53) as f64;
        (bits * 2.0 - 1.0) * scale
    };

    let weights: Vec<Vec<f64>> = (0..out_dim)
        .map(|_| (0..in_dim).map(|_| next()).collect())
        .collect();
    let bias: Vec<f64> = (0..out_dim).map(|_| next() * 0.01).collect();
    (weights, bias)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu() {
        assert!((relu(1.0) - 1.0).abs() < 1e-12);
        assert!((relu(-1.0)).abs() < 1e-12);
        assert!((relu(0.0)).abs() < 1e-12);
    }

    #[test]
    fn test_avg_pool_1d() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let pooled = avg_pool_1d(&input, 2);
        assert_eq!(pooled.len(), 2);
        assert!((pooled[0] - 1.5).abs() < 1e-12);
        assert!((pooled[1] - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_upsample_1d() {
        let input = vec![1.0, 2.0];
        let up = upsample_1d(&input, 2);
        assert_eq!(up, vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn test_conv1d_forward_single_channel() {
        let params = Conv1DParams {
            weights: vec![vec![1.0, 0.0, -1.0]],
            bias: vec![0.0],
            kernel_size: 3,
            stride: 1,
            in_channels: 1,
            out_channels: 1,
        };
        let input = vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]];
        let out = conv1d_forward_multi(&input, &params);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 3);
        // out[i] = input[i]*1 + input[i+1]*0 + input[i+2]*(-1)
        assert!((out[0][0] - (-2.0)).abs() < 1e-12); // 0 - 2
        assert!((out[0][1] - (-2.0)).abs() < 1e-12); // 1 - 3
        assert!((out[0][2] - (-2.0)).abs() < 1e-12); // 2 - 4
    }

    #[test]
    fn test_autoencoder_forward_shape() {
        let config = DLDenoiseConfig::default();
        let ae = DenoisingAutoencoder::new(&config);
        let noisy: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let out = ae.forward(&noisy);
        assert_eq!(out.len(), noisy.len());
    }

    #[test]
    fn test_autoencoder_denoise() {
        let config = DLDenoiseConfig::default();
        let ae = DenoisingAutoencoder::new(&config);
        let noisy: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin() + 0.1).collect();
        let result = ae.denoise(&noisy);
        assert_eq!(result.denoised.len(), noisy.len());
        assert_eq!(result.noise_estimate.len(), noisy.len());
        // noise_estimate = noisy - denoised
        for i in 0..noisy.len() {
            let expected = noisy[i] - result.denoised[i];
            assert!((result.noise_estimate[i] - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_autoencoder_train_step() {
        let config = DLDenoiseConfig::default();
        let mut ae = DenoisingAutoencoder::new(&config);
        let clean: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let noisy: Vec<f64> = clean.iter().map(|&x| x + 0.05).collect();
        let loss = ae.train_step(&noisy, &clean, 0.001);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_linear_forward() {
        let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let bias = vec![0.5, -0.5];
        let input = vec![2.0, 3.0];
        let out = linear_forward(&input, &weights, &bias);
        assert!((out[0] - 2.5).abs() < 1e-12);
        assert!((out[1] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_concat_skip() {
        let main = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let skip = vec![vec![5.0, 6.0]];
        let cat = concat_skip(&main, &skip);
        assert_eq!(cat.len(), 3);
        assert_eq!(cat[0], vec![1.0, 2.0]);
        assert_eq!(cat[2], vec![5.0, 6.0]);
    }
}
