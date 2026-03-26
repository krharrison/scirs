//! Types and configuration for deep learning denoising.

/// Method selection for deep learning denoising.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DenoisingMethod {
    /// Denoising autoencoder with skip connections (U-Net style).
    Autoencoder,
    /// Residual learning: learn the noise, subtract from input (DnCNN-inspired).
    ResidualLearning,
    /// Diffusion-based denoising (DDPM-lite single-step).
    DiffusionDenoise,
    /// Wave-U-Net style architecture for audio denoising.
    WaveUNet,
}

/// Top-level configuration for deep learning denoising.
#[derive(Debug, Clone)]
pub struct DLDenoiseConfig {
    /// Which denoising method to use.
    pub method: DenoisingMethod,
    /// Learning rate for training.
    pub learning_rate: f64,
    /// Number of training epochs.
    pub epochs: usize,
    /// Mini-batch size.
    pub batch_size: usize,
    /// Latent dimension for autoencoder bottleneck.
    pub latent_dim: usize,
}

impl Default for DLDenoiseConfig {
    fn default() -> Self {
        Self {
            method: DenoisingMethod::Autoencoder,
            learning_rate: 0.001,
            epochs: 100,
            batch_size: 32,
            latent_dim: 32,
        }
    }
}

/// Result of a denoising operation.
#[derive(Debug, Clone)]
pub struct DenoisingResult {
    /// The denoised signal.
    pub denoised: Vec<f64>,
    /// Estimated noise that was removed.
    pub noise_estimate: Vec<f64>,
    /// SNR improvement in dB.
    pub snr_improvement: f64,
}

/// Training configuration for denoising networks.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate.
    pub learning_rate: f64,
    /// Maximum number of training epochs.
    pub epochs: usize,
    /// Early stopping patience (epochs without improvement).
    pub patience: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            epochs: 100,
            patience: 10,
        }
    }
}

/// Parameters for a 1D convolutional layer.
#[derive(Debug, Clone)]
pub struct Conv1DParams {
    /// Weights: outer = out_channels, inner = in_channels * kernel_size.
    pub weights: Vec<Vec<f64>>,
    /// Bias per output channel.
    pub bias: Vec<f64>,
    /// Kernel (filter) size.
    pub kernel_size: usize,
    /// Stride.
    pub stride: usize,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
}

impl Conv1DParams {
    /// Create new Conv1D parameters with random weights using an LCG RNG.
    ///
    /// `seed` drives the LCG; weights are initialized with He-like scaling.
    pub fn new_random(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        seed: u64,
    ) -> Self {
        let fan_in = in_channels * kernel_size;
        let scale = (2.0 / fan_in as f64).sqrt();
        let mut rng_state = seed;
        let mut next_f64 = || -> f64 {
            // LCG: state = (a * state + c) mod m
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map to [-1, 1]
            let bits = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
            (bits * 2.0 - 1.0) * scale
        };

        let weights_len = in_channels * kernel_size;
        let weights = (0..out_channels)
            .map(|_| (0..weights_len).map(|_| next_f64()).collect())
            .collect();
        let bias = (0..out_channels).map(|_| next_f64() * 0.01).collect();

        Self {
            weights,
            bias,
            kernel_size,
            stride,
            in_channels,
            out_channels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = DLDenoiseConfig::default();
        assert_eq!(cfg.method, DenoisingMethod::Autoencoder);
        assert!((cfg.learning_rate - 0.001).abs() < 1e-12);
        assert_eq!(cfg.epochs, 100);
        assert_eq!(cfg.batch_size, 32);
        assert_eq!(cfg.latent_dim, 32);
    }

    #[test]
    fn test_training_config_default() {
        let tc = TrainingConfig::default();
        assert!((tc.learning_rate - 0.001).abs() < 1e-12);
        assert_eq!(tc.epochs, 100);
        assert_eq!(tc.patience, 10);
    }

    #[test]
    fn test_conv1d_params_random() {
        let params = Conv1DParams::new_random(1, 16, 3, 1, 42);
        assert_eq!(params.weights.len(), 16);
        assert_eq!(params.weights[0].len(), 3);
        assert_eq!(params.bias.len(), 16);
        assert_eq!(params.kernel_size, 3);
        assert_eq!(params.stride, 1);
    }

    #[test]
    fn test_denoising_method_variants() {
        let methods = vec![
            DenoisingMethod::Autoencoder,
            DenoisingMethod::ResidualLearning,
            DenoisingMethod::DiffusionDenoise,
            DenoisingMethod::WaveUNet,
        ];
        assert_eq!(methods.len(), 4);
    }
}
