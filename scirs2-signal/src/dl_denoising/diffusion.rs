//! Diffusion-based denoiser (DDPM-lite).
//!
//! Forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
//!
//! A small 1D U-Net predicts the noise epsilon given (x_t, t).
//! Denoising recovers x_0 from x_t in a single step.

use super::types::{Conv1DParams, DenoisingResult};

/// A diffusion-based denoiser using a simplified DDPM approach.
#[derive(Debug, Clone)]
pub struct DiffusionDenoiser {
    /// Number of diffusion timesteps.
    num_steps: usize,
    /// Cumulative product of alphas: alpha_bar[t].
    alpha_bar: Vec<f64>,
    /// Beta schedule.
    betas: Vec<f64>,

    // Noise prediction U-Net (2 down + 2 up stages)
    // Down path
    down_conv1: Conv1DParams, // 1 -> 16
    down_conv2: Conv1DParams, // 16 -> 32

    // Timestep embedding projection (sinusoidal -> 32 channels)
    time_proj_w: Vec<Vec<f64>>,
    time_proj_b: Vec<f64>,

    // Up path
    up_conv1: Conv1DParams, // 64 -> 16 (32 + 32 skip)
    up_conv2: Conv1DParams, // 32 -> 1  (16 + 16 skip)

    /// LCG RNG state for noise generation.
    rng_state: u64,
}

impl DiffusionDenoiser {
    /// Create a new diffusion denoiser with the given number of timesteps.
    pub fn new(num_steps: usize) -> Self {
        let num_steps = if num_steps == 0 { 100 } else { num_steps };
        let (betas, alpha_bar) = Self::compute_alpha_schedule(num_steps);

        let seed_base: u64 = 555555555;
        let down_conv1 = Conv1DParams::new_random(1, 16, 3, 1, seed_base);
        let down_conv2 = Conv1DParams::new_random(16, 32, 3, 1, seed_base.wrapping_add(1));

        // Timestep embedding: sinusoidal features of dim 32 -> project to 32 channels
        let time_embed_dim = 32;
        let (time_proj_w, time_proj_b) =
            random_linear(time_embed_dim, 32, seed_base.wrapping_add(2));

        let up_conv1 = Conv1DParams::new_random(64, 16, 3, 1, seed_base.wrapping_add(3));
        let up_conv2 = Conv1DParams::new_random(32, 1, 3, 1, seed_base.wrapping_add(4));

        Self {
            num_steps,
            alpha_bar,
            betas,
            down_conv1,
            down_conv2,
            time_proj_w,
            time_proj_b,
            up_conv1,
            up_conv2,
            rng_state: seed_base.wrapping_add(100),
        }
    }

    /// Compute the linear beta schedule and cumulative alpha_bar.
    ///
    /// Returns `(betas, alpha_bar)`.
    pub fn compute_alpha_schedule(num_steps: usize) -> (Vec<f64>, Vec<f64>) {
        let beta_start = 1e-4;
        let beta_end = 0.02;
        let betas: Vec<f64> = (0..num_steps)
            .map(|i| {
                beta_start
                    + (beta_end - beta_start) * i as f64 / (num_steps.max(1) - 1).max(1) as f64
            })
            .collect();

        let mut alpha_bar = Vec::with_capacity(num_steps);
        let mut cumulative = 1.0;
        for &b in &betas {
            cumulative *= 1.0 - b;
            alpha_bar.push(cumulative);
        }

        (betas, alpha_bar)
    }

    /// Add noise to a clean signal at timestep `t`.
    ///
    /// Returns `(noisy_signal, noise)`.
    pub fn add_noise(&mut self, clean: &[f64], t: usize) -> (Vec<f64>, Vec<f64>) {
        let t_clamped = t.min(self.num_steps.saturating_sub(1));
        let ab = self.alpha_bar[t_clamped];
        let sqrt_ab = ab.sqrt();
        let sqrt_one_minus_ab = (1.0 - ab).sqrt();

        let noise: Vec<f64> = (0..clean.len()).map(|_| self.next_gaussian()).collect();
        let noisy: Vec<f64> = clean
            .iter()
            .zip(noise.iter())
            .map(|(&x, &eps)| sqrt_ab * x + sqrt_one_minus_ab * eps)
            .collect();

        (noisy, noise)
    }

    /// Predict noise from a noisy signal at timestep `t`.
    pub fn predict_noise(&self, noisy: &[f64], t: usize) -> Vec<f64> {
        let t_clamped = t.min(self.num_steps.saturating_sub(1));

        // --- Sinusoidal timestep embedding ---
        let time_embed = sinusoidal_embedding(t_clamped, 32);
        let time_features = linear_forward(&time_embed, &self.time_proj_w, &self.time_proj_b);

        // --- Down path ---
        let input_ch = vec![noisy.to_vec()];
        let d1 = conv1d_multi(&input_ch, &self.down_conv1);
        let d1_relu = apply_relu(&d1);
        let d1_pool = avg_pool_multi(&d1_relu, 2);

        let d2 = conv1d_multi(&d1_pool, &self.down_conv2);
        let d2_relu = apply_relu(&d2);

        // Add timestep features (broadcast across spatial dim)
        let d2_with_time: Vec<Vec<f64>> = d2_relu
            .iter()
            .enumerate()
            .map(|(ch, data)| {
                let t_val = if ch < time_features.len() {
                    time_features[ch]
                } else {
                    0.0
                };
                data.iter().map(|&v| v + t_val).collect()
            })
            .collect();

        let d2_pool = avg_pool_multi(&d2_with_time, 2);

        // --- Up path ---
        let u1_up = upsample_multi(&d2_pool, 2);
        let u1_cat = concat_channels(&u1_up, &d2_relu);
        let u1 = conv1d_multi(&u1_cat, &self.up_conv1);
        let u1_relu = apply_relu(&u1);

        let u2_up = upsample_multi(&u1_relu, 2);
        let u2_cat = concat_channels(&u2_up, &d1_relu);
        let u2 = conv1d_multi(&u2_cat, &self.up_conv2);

        // Output: single channel, resize to input length
        if u2.is_empty() {
            vec![0.0; noisy.len()]
        } else {
            let mut out = u2[0].clone();
            out.resize(noisy.len(), 0.0);
            out
        }
    }

    /// Denoise a noisy signal at the given estimated timestep (single-step DDPM).
    ///
    /// x_0 = (x_t - sqrt(1 - alpha_bar) * eps_hat) / sqrt(alpha_bar)
    pub fn denoise_single_step(&self, noisy: &[f64], estimated_t: usize) -> DenoisingResult {
        let t_clamped = estimated_t.min(self.num_steps.saturating_sub(1));
        let ab = self.alpha_bar[t_clamped];
        let sqrt_ab = ab.sqrt();
        let sqrt_one_minus_ab = (1.0 - ab).sqrt();

        let eps_hat = self.predict_noise(noisy, t_clamped);

        let denoised: Vec<f64> = noisy
            .iter()
            .zip(eps_hat.iter())
            .map(|(&x_t, &e)| {
                if sqrt_ab.abs() < 1e-15 {
                    x_t
                } else {
                    (x_t - sqrt_one_minus_ab * e) / sqrt_ab
                }
            })
            .collect();

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

    /// Perform one training step: sample random t, add noise, predict, compute MSE.
    ///
    /// Returns the MSE loss.
    pub fn train_step(&mut self, clean: &[f64], lr: f64) -> f64 {
        // Sample a random timestep
        let t = self.next_uniform_usize(self.num_steps);

        // Add noise
        let (noisy, true_noise) = self.add_noise(clean, t);

        // Predict noise
        let predicted_noise = self.predict_noise(&noisy, t);

        // MSE between predicted and true noise
        let n = predicted_noise.len().min(true_noise.len());
        if n == 0 {
            return 0.0;
        }

        let mse: f64 = predicted_noise
            .iter()
            .zip(true_noise.iter())
            .take(n)
            .map(|(p, tr)| (p - tr).powi(2))
            .sum::<f64>()
            / n as f64;

        // Simplified gradient update on down_conv1 bias
        let epsilon = 1e-5;
        for i in 0..self.down_conv1.bias.len() {
            let original = self.down_conv1.bias[i];

            self.down_conv1.bias[i] = original + epsilon;
            let pred_plus = self.predict_noise(&noisy, t);
            let loss_plus: f64 = pred_plus
                .iter()
                .zip(true_noise.iter())
                .take(n)
                .map(|(p, tr)| (p - tr).powi(2))
                .sum::<f64>()
                / n as f64;

            self.down_conv1.bias[i] = original - epsilon;
            let pred_minus = self.predict_noise(&noisy, t);
            let loss_minus: f64 = pred_minus
                .iter()
                .zip(true_noise.iter())
                .take(n)
                .map(|(p, tr)| (p - tr).powi(2))
                .sum::<f64>()
                / n as f64;

            let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
            self.down_conv1.bias[i] = original - lr * grad;
        }

        mse
    }

    /// Get number of timesteps.
    pub fn num_steps(&self) -> usize {
        self.num_steps
    }

    /// Get alpha_bar schedule.
    pub fn alpha_bar(&self) -> &[f64] {
        &self.alpha_bar
    }

    // ---- LCG-based RNG helpers ----

    fn next_u64(&mut self) -> u64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.rng_state
    }

    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 / (1u64 << 53) as f64
    }

    fn next_gaussian(&mut self) -> f64 {
        // Box-Muller transform
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn next_uniform_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }
}

// ---- Internal helpers ----

/// Sinusoidal positional/timestep embedding.
fn sinusoidal_embedding(t: usize, dim: usize) -> Vec<f64> {
    let half = dim / 2;
    (0..dim)
        .map(|i| {
            let freq = 1.0 / 10000f64.powf(i.min(half) as f64 / half.max(1) as f64);
            let angle = t as f64 * freq;
            if i < half {
                angle.sin()
            } else {
                angle.cos()
            }
        })
        .collect()
}

/// Linear layer forward.
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

/// Create random linear layer using LCG.
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

/// Multi-channel 1D convolution (valid padding, stride 1).
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

/// Apply ReLU to all channels.
fn apply_relu(channels: &[Vec<f64>]) -> Vec<Vec<f64>> {
    channels
        .iter()
        .map(|ch| ch.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect())
        .collect()
}

/// Average pool each channel.
fn avg_pool_multi(channels: &[Vec<f64>], kernel: usize) -> Vec<Vec<f64>> {
    channels
        .iter()
        .map(|ch| {
            let out_len = ch.len() / kernel;
            (0..out_len)
                .map(|i| {
                    let start = i * kernel;
                    let end = (start + kernel).min(ch.len());
                    ch[start..end].iter().sum::<f64>() / (end - start) as f64
                })
                .collect()
        })
        .collect()
}

/// Nearest-neighbour upsample each channel.
fn upsample_multi(channels: &[Vec<f64>], factor: usize) -> Vec<Vec<f64>> {
    channels
        .iter()
        .map(|ch| {
            let mut out = Vec::with_capacity(ch.len() * factor);
            for &v in ch {
                for _ in 0..factor {
                    out.push(v);
                }
            }
            out
        })
        .collect()
}

/// Concatenate two multi-channel tensors along the channel dimension.
/// Trims spatial dims to the minimum.
fn concat_channels(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let min_len = a
        .iter()
        .chain(b.iter())
        .map(|ch| ch.len())
        .min()
        .unwrap_or(0);

    let mut result: Vec<Vec<f64>> = a
        .iter()
        .map(|ch| ch[..min_len.min(ch.len())].to_vec())
        .collect();
    for ch in b {
        result.push(ch[..min_len.min(ch.len())].to_vec());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_schedule() {
        let (betas, alpha_bar) = DiffusionDenoiser::compute_alpha_schedule(100);
        assert_eq!(betas.len(), 100);
        assert_eq!(alpha_bar.len(), 100);
        // alpha_bar should be decreasing
        for i in 1..alpha_bar.len() {
            assert!(alpha_bar[i] <= alpha_bar[i - 1] + 1e-12);
        }
        // alpha_bar[0] should be close to 1
        assert!(alpha_bar[0] > 0.99);
        // alpha_bar[last] should be small but positive
        assert!(alpha_bar[99] > 0.0);
        assert!(alpha_bar[99] < 1.0);
    }

    #[test]
    fn test_sinusoidal_embedding() {
        let emb = sinusoidal_embedding(10, 32);
        assert_eq!(emb.len(), 32);
        // Values should be in [-1, 1]
        for &v in &emb {
            assert!(v >= -1.0 - 1e-10 && v <= 1.0 + 1e-10);
        }
    }

    #[test]
    fn test_add_noise() {
        let mut dd = DiffusionDenoiser::new(100);
        let clean: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let (noisy, noise) = dd.add_noise(&clean, 50);
        assert_eq!(noisy.len(), clean.len());
        assert_eq!(noise.len(), clean.len());
    }

    #[test]
    fn test_predict_noise() {
        let dd = DiffusionDenoiser::new(50);
        let noisy: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin() + 0.1).collect();
        let predicted = dd.predict_noise(&noisy, 25);
        assert_eq!(predicted.len(), noisy.len());
    }

    #[test]
    fn test_denoise_single_step() {
        let dd = DiffusionDenoiser::new(50);
        let noisy: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin() + 0.1).collect();
        let result = dd.denoise_single_step(&noisy, 25);
        assert_eq!(result.denoised.len(), noisy.len());
        assert_eq!(result.noise_estimate.len(), noisy.len());
    }

    #[test]
    fn test_train_step() {
        let mut dd = DiffusionDenoiser::new(20);
        let clean: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        let loss = dd.train_step(&clean, 0.001);
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_diffusion_roundtrip_low_noise() {
        let mut dd = DiffusionDenoiser::new(100);
        let clean: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).sin()).collect();
        // At t=0, alpha_bar is very close to 1, so noisy ~= clean
        let (noisy, _) = dd.add_noise(&clean, 0);
        let max_diff: f64 = clean
            .iter()
            .zip(noisy.iter())
            .map(|(c, n)| (c - n).abs())
            .fold(0.0, f64::max);
        // At t=0, noise level should be very small
        assert!(
            max_diff < 1.0,
            "max_diff at t=0 should be small: {max_diff}"
        );
    }
}
