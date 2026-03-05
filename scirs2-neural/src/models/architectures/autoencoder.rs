//! Autoencoder and Variational Autoencoder (VAE) implementation
//!
//! This module provides:
//! - **Autoencoder**: Standard encoder-bottleneck-decoder architecture for unsupervised
//!   representation learning and dimensionality reduction.
//! - **VAE**: Variational autoencoder with the reparameterization trick, KL divergence
//!   regularization, and generation from the learned latent space.
//!
//! References:
//! - "Auto-Encoding Variational Bayes", Kingma & Welling (2013) <https://arxiv.org/abs/1312.6114>
//! - "An Introduction to Variational Autoencoders", Kingma & Welling (2019)

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Layer};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::SeedableRng;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for an autoencoder model
#[derive(Debug, Clone)]
pub struct AutoencoderConfig {
    /// Input dimension (flattened)
    pub input_dim: usize,
    /// Dimensions of hidden layers in the encoder (mirrored in decoder)
    pub hidden_dims: Vec<usize>,
    /// Dimension of the latent / bottleneck space
    pub latent_dim: usize,
    /// Dropout rate between hidden layers (0 to disable)
    pub dropout_rate: f64,
    /// Activation type ("relu" or "tanh")
    pub activation: String,
}

impl AutoencoderConfig {
    /// Standard MLP autoencoder
    pub fn standard(input_dim: usize, latent_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dims: vec![256, 128],
            latent_dim,
            dropout_rate: 0.0,
            activation: "relu".to_string(),
        }
    }

    /// Tiny config for testing
    pub fn tiny(input_dim: usize, latent_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dims: vec![32],
            latent_dim,
            dropout_rate: 0.0,
            activation: "relu".to_string(),
        }
    }

    /// Custom configuration
    pub fn custom(input_dim: usize, hidden_dims: Vec<usize>, latent_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dims,
            latent_dim,
            dropout_rate: 0.0,
            activation: "relu".to_string(),
        }
    }

    /// Set dropout
    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    /// Set activation
    pub fn with_activation(mut self, act: &str) -> Self {
        self.activation = act.to_string();
        self
    }
}

/// Configuration for a VAE model
#[derive(Debug, Clone)]
pub struct VAEConfig {
    /// Base autoencoder configuration
    pub base: AutoencoderConfig,
    /// Weight for KL divergence loss term (beta-VAE)
    pub kl_weight: f64,
}

impl VAEConfig {
    /// Standard VAE
    pub fn standard(input_dim: usize, latent_dim: usize) -> Self {
        Self {
            base: AutoencoderConfig::standard(input_dim, latent_dim),
            kl_weight: 1.0,
        }
    }

    /// Tiny VAE for testing
    pub fn tiny(input_dim: usize, latent_dim: usize) -> Self {
        Self {
            base: AutoencoderConfig::tiny(input_dim, latent_dim),
            kl_weight: 1.0,
        }
    }

    /// Custom VAE
    pub fn custom(input_dim: usize, hidden_dims: Vec<usize>, latent_dim: usize) -> Self {
        Self {
            base: AutoencoderConfig::custom(input_dim, hidden_dims, latent_dim),
            kl_weight: 1.0,
        }
    }

    /// Set beta (KL weight) for beta-VAE
    pub fn with_kl_weight(mut self, weight: f64) -> Self {
        self.kl_weight = weight;
        self
    }

    /// Set dropout
    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.base.dropout_rate = rate;
        self
    }
}

// ---------------------------------------------------------------------------
// Helper: apply activation
// ---------------------------------------------------------------------------

fn apply_activation<F: Float>(x: &Array<F, IxDyn>, activation: &str) -> Array<F, IxDyn> {
    match activation {
        "tanh" => x.mapv(|v| v.tanh()),
        _ => x.mapv(|v| v.max(F::zero())), // relu default
    }
}

// ---------------------------------------------------------------------------
// Standard Autoencoder
// ---------------------------------------------------------------------------

/// Standard Autoencoder: encoder -> bottleneck -> decoder
///
/// The encoder maps the input to a latent representation, and the decoder
/// reconstructs the input from the latent representation.
///
/// # Examples
///
/// ```no_run
/// use scirs2_neural::models::architectures::autoencoder::{Autoencoder, AutoencoderConfig};
///
/// let model: Autoencoder<f64> = Autoencoder::new(AutoencoderConfig::standard(784, 32))
///     .expect("Failed to create autoencoder");
/// ```
pub struct Autoencoder<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    config: AutoencoderConfig,
    /// Encoder layers
    encoder_layers: Vec<Dense<F>>,
    /// Decoder layers
    decoder_layers: Vec<Dense<F>>,
    /// Optional dropout
    dropout: Option<Dropout<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Autoencoder<F> {
    /// Create a new Autoencoder
    pub fn new(config: AutoencoderConfig) -> Result<Self> {
        if config.input_dim == 0 || config.latent_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "input_dim and latent_dim must be > 0".to_string(),
            ));
        }

        // Build encoder: input_dim -> hidden[0] -> ... -> hidden[n-1] -> latent_dim
        let mut encoder_layers = Vec::new();
        let mut in_dim = config.input_dim;
        let mut seed_counter: u8 = 70;
        for &hdim in &config.hidden_dims {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([seed_counter; 32]);
            seed_counter = seed_counter.wrapping_add(1);
            encoder_layers.push(Dense::new(in_dim, hdim, None, &mut rng)?);
            in_dim = hdim;
        }
        // Final encoder layer to latent
        {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([seed_counter; 32]);
            seed_counter = seed_counter.wrapping_add(1);
            encoder_layers.push(Dense::new(in_dim, config.latent_dim, None, &mut rng)?);
        }

        // Build decoder: latent_dim -> hidden[n-1] -> ... -> hidden[0] -> input_dim
        let mut decoder_layers = Vec::new();
        let mut in_dim = config.latent_dim;
        for &hdim in config.hidden_dims.iter().rev() {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([seed_counter; 32]);
            seed_counter = seed_counter.wrapping_add(1);
            decoder_layers.push(Dense::new(in_dim, hdim, None, &mut rng)?);
            in_dim = hdim;
        }
        // Final decoder layer to reconstruct input
        {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([seed_counter; 32]);
            let _ = seed_counter.wrapping_add(1);
            decoder_layers.push(Dense::new(in_dim, config.input_dim, None, &mut rng)?);
        }

        let dropout = if config.dropout_rate > 0.0 {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([80; 32]);
            Some(Dropout::new(config.dropout_rate, &mut rng)?)
        } else {
            None
        };

        Ok(Self {
            config,
            encoder_layers,
            decoder_layers,
            dropout,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &AutoencoderConfig {
        &self.config
    }

    /// Encode input to latent representation
    pub fn encode(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = input.clone();
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            x = layer.forward(&x)?;
            // Apply activation for all layers except the last
            if i < self.encoder_layers.len() - 1 {
                x = apply_activation(&x, &self.config.activation);
                if let Some(ref drop) = self.dropout {
                    x = drop.forward(&x)?;
                }
            }
        }
        Ok(x)
    }

    /// Decode latent representation to reconstructed input
    pub fn decode(&self, latent: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = latent.clone();
        for (i, layer) in self.decoder_layers.iter().enumerate() {
            x = layer.forward(&x)?;
            // Apply activation for all layers except the last
            if i < self.decoder_layers.len() - 1 {
                x = apply_activation(&x, &self.config.activation);
                if let Some(ref drop) = self.dropout {
                    x = drop.forward(&x)?;
                }
            }
        }
        Ok(x)
    }

    /// Total parameter count
    pub fn total_parameter_count(&self) -> usize {
        let enc: usize = self
            .encoder_layers
            .iter()
            .map(|l| l.parameter_count())
            .sum();
        let dec: usize = self
            .decoder_layers
            .iter()
            .map(|l| l.parameter_count())
            .sum();
        enc + dec
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for Autoencoder<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let latent = self.encode(input)?;
        self.decode(&latent)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        for layer in &mut self.encoder_layers {
            layer.update(learning_rate)?;
        }
        for layer in &mut self.decoder_layers {
            layer.update(learning_rate)?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = Vec::new();
        for l in &self.encoder_layers {
            p.extend(l.params());
        }
        for l in &self.decoder_layers {
            p.extend(l.params());
        }
        p
    }

    fn parameter_count(&self) -> usize {
        self.total_parameter_count()
    }

    fn layer_type(&self) -> &str {
        "Autoencoder"
    }

    fn layer_description(&self) -> String {
        format!(
            "Autoencoder(input={}, latent={}, hidden={:?}, params={})",
            self.config.input_dim,
            self.config.latent_dim,
            self.config.hidden_dims,
            self.total_parameter_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Variational Autoencoder (VAE)
// ---------------------------------------------------------------------------

/// VAE output containing reconstruction and latent statistics
#[derive(Debug)]
pub struct VAEOutput<F: Float> {
    /// Reconstructed input
    pub reconstruction: Array<F, IxDyn>,
    /// Mean of the latent distribution
    pub mu: Array<F, IxDyn>,
    /// Log-variance of the latent distribution
    pub log_var: Array<F, IxDyn>,
    /// Sampled latent vector
    pub z: Array<F, IxDyn>,
}

/// Variational Autoencoder with reparameterization trick
///
/// The encoder maps to (mu, log_var) parameters of a Gaussian latent distribution.
/// During training, z is sampled via the reparameterization trick:
///   z = mu + std * epsilon,  where epsilon ~ N(0, 1)
///
/// The loss combines reconstruction error and KL divergence:
///   L = E[||x - x_hat||^2] + beta * KL(q(z|x) || p(z))
///
/// # Examples
///
/// ```no_run
/// use scirs2_neural::models::architectures::autoencoder::{VAE, VAEConfig};
///
/// let model: VAE<f64> = VAE::new(VAEConfig::standard(784, 32))
///     .expect("Failed to create VAE");
/// ```
pub struct VAE<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    config: VAEConfig,
    /// Encoder hidden layers
    encoder_layers: Vec<Dense<F>>,
    /// Mean projection
    fc_mu: Dense<F>,
    /// Log-variance projection
    fc_log_var: Dense<F>,
    /// Decoder hidden layers
    decoder_layers: Vec<Dense<F>>,
    /// Optional dropout
    dropout: Option<Dropout<F>>,
    /// PRNG state for reparameterization
    rng_state: std::cell::Cell<u64>,
}

// VAE is Send+Sync because Cell<u64> is not Sync by default.
// We use interior mutability only for the PRNG which is deterministic per-forward.
// SAFETY: The rng_state is only modified during forward() which holds &self.
// In a multi-threaded context, each thread should use its own VAE instance.
unsafe impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Sync for VAE<F> {}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> VAE<F> {
    /// Create a new VAE
    pub fn new(config: VAEConfig) -> Result<Self> {
        let base = &config.base;
        if base.input_dim == 0 || base.latent_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "input_dim and latent_dim must be > 0".to_string(),
            ));
        }

        // Build encoder hidden layers: input_dim -> hidden[0] -> ... -> hidden[n-1]
        let mut encoder_layers = Vec::new();
        let mut in_dim = base.input_dim;
        let mut seed_counter: u8 = 90;
        for &hdim in &base.hidden_dims {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([seed_counter; 32]);
            seed_counter = seed_counter.wrapping_add(1);
            encoder_layers.push(Dense::new(in_dim, hdim, None, &mut rng)?);
            in_dim = hdim;
        }

        // Separate heads for mu and log_var
        let mut rng_mu = scirs2_core::random::rngs::SmallRng::from_seed([seed_counter; 32]);
        seed_counter = seed_counter.wrapping_add(1);
        let fc_mu = Dense::new(in_dim, base.latent_dim, None, &mut rng_mu)?;

        let mut rng_lv = scirs2_core::random::rngs::SmallRng::from_seed([seed_counter; 32]);
        seed_counter = seed_counter.wrapping_add(1);
        let fc_log_var = Dense::new(in_dim, base.latent_dim, None, &mut rng_lv)?;

        // Build decoder: latent_dim -> hidden[n-1] -> ... -> hidden[0] -> input_dim
        let mut decoder_layers = Vec::new();
        let mut dec_in = base.latent_dim;
        for &hdim in base.hidden_dims.iter().rev() {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([seed_counter; 32]);
            seed_counter = seed_counter.wrapping_add(1);
            decoder_layers.push(Dense::new(dec_in, hdim, None, &mut rng)?);
            dec_in = hdim;
        }
        {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([seed_counter; 32]);
            let _ = seed_counter.wrapping_add(1);
            decoder_layers.push(Dense::new(dec_in, base.input_dim, None, &mut rng)?);
        }

        let dropout = if base.dropout_rate > 0.0 {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([100; 32]);
            Some(Dropout::new(base.dropout_rate, &mut rng)?)
        } else {
            None
        };

        Ok(Self {
            config,
            encoder_layers,
            fc_mu,
            fc_log_var,
            decoder_layers,
            dropout,
            rng_state: std::cell::Cell::new(0xCAFE_BABE_1234_5678u64),
        })
    }

    /// Get configuration
    pub fn config(&self) -> &VAEConfig {
        &self.config
    }

    /// Encode input to (mu, log_var) parameters of the latent distribution
    pub fn encode_distribution(
        &self,
        input: &Array<F, IxDyn>,
    ) -> Result<(Array<F, IxDyn>, Array<F, IxDyn>)> {
        let mut x = input.clone();
        for layer in &self.encoder_layers {
            x = layer.forward(&x)?;
            x = apply_activation(&x, &self.config.base.activation);
            if let Some(ref drop) = self.dropout {
                x = drop.forward(&x)?;
            }
        }
        let mu = self.fc_mu.forward(&x)?;
        let log_var = self.fc_log_var.forward(&x)?;
        Ok((mu, log_var))
    }

    /// Reparameterization trick: z = mu + std * epsilon
    pub fn reparameterize(
        &self,
        mu: &Array<F, IxDyn>,
        log_var: &Array<F, IxDyn>,
    ) -> Array<F, IxDyn> {
        let half = F::from(0.5).expect("half conversion");
        let std_dev = log_var.mapv(|v| (v * half).exp());

        // Generate epsilon ~ N(0,1) using Box-Muller transform with our xorshift PRNG
        let mut state = self.rng_state.get();
        let epsilon = std_dev.mapv(|_| {
            let u1 = xorshift_f64(&mut state).max(1e-10);
            let u2 = xorshift_f64(&mut state);
            let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            F::from(normal).expect("normal conversion")
        });
        self.rng_state.set(state);

        mu + &(std_dev * epsilon)
    }

    /// Decode latent vector to reconstruction
    pub fn decode(&self, z: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = z.clone();
        for (i, layer) in self.decoder_layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.decoder_layers.len() - 1 {
                x = apply_activation(&x, &self.config.base.activation);
                if let Some(ref drop) = self.dropout {
                    x = drop.forward(&x)?;
                }
            }
        }
        Ok(x)
    }

    /// Full forward pass returning detailed output
    pub fn forward_detailed(&self, input: &Array<F, IxDyn>) -> Result<VAEOutput<F>> {
        let (mu, log_var) = self.encode_distribution(input)?;
        let z = self.reparameterize(&mu, &log_var);
        let reconstruction = self.decode(&z)?;
        Ok(VAEOutput {
            reconstruction,
            mu,
            log_var,
            z,
        })
    }

    /// Compute KL divergence loss: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    ///
    /// Returns the mean KL divergence over the batch.
    pub fn kl_divergence(mu: &Array<F, IxDyn>, log_var: &Array<F, IxDyn>) -> F {
        let half = F::from(0.5).expect("half");
        let one = F::one();
        let mut kl_sum = F::zero();
        let n = mu.len();

        for (m, lv) in mu.iter().zip(log_var.iter()) {
            // -0.5 * (1 + log_var - mu^2 - exp(log_var))
            kl_sum += one + *lv - *m * *m - lv.exp();
        }

        -half * kl_sum / F::from(n).expect("n conversion")
    }

    /// Compute ELBO loss = reconstruction_loss + kl_weight * kl_divergence
    ///
    /// Uses MSE for reconstruction loss.
    pub fn elbo_loss(&self, input: &Array<F, IxDyn>, output: &VAEOutput<F>) -> F {
        // MSE reconstruction loss
        let diff = input - &output.reconstruction;
        let mse: F = diff.mapv(|v| v * v).sum() / F::from(input.len()).expect("len");

        // KL divergence
        let kl = Self::kl_divergence(&output.mu, &output.log_var);
        let kl_weight = F::from(self.config.kl_weight).expect("kl_weight");

        mse + kl_weight * kl
    }

    /// Generate new samples by decoding random latent vectors
    ///
    /// Samples z ~ N(0, I) and decodes to data space.
    pub fn generate(&self, num_samples: usize) -> Result<Array<F, IxDyn>> {
        let latent_dim = self.config.base.latent_dim;
        let mut state = self.rng_state.get();

        // Sample z from standard normal
        let mut z = Array::zeros(IxDyn(&[num_samples, latent_dim]));
        for b in 0..num_samples {
            for d in 0..latent_dim {
                let u1 = xorshift_f64(&mut state).max(1e-10);
                let u2 = xorshift_f64(&mut state);
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z[[b, d]] = F::from(normal).expect("normal");
            }
        }
        self.rng_state.set(state);

        self.decode(&z)
    }

    /// Reconstruct input (encode then decode)
    pub fn reconstruct(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let (mu, log_var) = self.encode_distribution(input)?;
        let z = self.reparameterize(&mu, &log_var);
        self.decode(&z)
    }

    /// Total parameter count
    pub fn total_parameter_count(&self) -> usize {
        let enc: usize = self
            .encoder_layers
            .iter()
            .map(|l| l.parameter_count())
            .sum();
        let mu_params = self.fc_mu.parameter_count();
        let lv_params = self.fc_log_var.parameter_count();
        let dec: usize = self
            .decoder_layers
            .iter()
            .map(|l| l.parameter_count())
            .sum();
        enc + mu_params + lv_params + dec
    }
}

/// Simple xorshift64 returning f64 in [0, 1)
fn xorshift_f64(state: &mut u64) -> f64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    (s >> 11) as f64 / ((1u64 << 53) as f64)
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F> for VAE<F> {
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        // Standard forward returns reconstruction
        let output = self.forward_detailed(input)?;
        Ok(output.reconstruction)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        for layer in &mut self.encoder_layers {
            layer.update(learning_rate)?;
        }
        self.fc_mu.update(learning_rate)?;
        self.fc_log_var.update(learning_rate)?;
        for layer in &mut self.decoder_layers {
            layer.update(learning_rate)?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = Vec::new();
        for l in &self.encoder_layers {
            p.extend(l.params());
        }
        p.extend(self.fc_mu.params());
        p.extend(self.fc_log_var.params());
        for l in &self.decoder_layers {
            p.extend(l.params());
        }
        p
    }

    fn parameter_count(&self) -> usize {
        self.total_parameter_count()
    }

    fn layer_type(&self) -> &str {
        "VAE"
    }

    fn layer_description(&self) -> String {
        format!(
            "VAE(input={}, latent={}, hidden={:?}, kl_w={}, params={})",
            self.config.base.input_dim,
            self.config.base.latent_dim,
            self.config.base.hidden_dims,
            self.config.kl_weight,
            self.total_parameter_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Autoencoder Config Tests ----

    #[test]
    fn test_ae_config_standard() {
        let cfg = AutoencoderConfig::standard(784, 32);
        assert_eq!(cfg.input_dim, 784);
        assert_eq!(cfg.latent_dim, 32);
        assert_eq!(cfg.hidden_dims, vec![256, 128]);
    }

    #[test]
    fn test_ae_config_tiny() {
        let cfg = AutoencoderConfig::tiny(100, 10);
        assert_eq!(cfg.hidden_dims, vec![32]);
    }

    #[test]
    fn test_ae_config_custom() {
        let cfg = AutoencoderConfig::custom(50, vec![20, 10], 5);
        assert_eq!(cfg.hidden_dims, vec![20, 10]);
        assert_eq!(cfg.latent_dim, 5);
    }

    #[test]
    fn test_ae_config_builder() {
        let cfg = AutoencoderConfig::standard(784, 32)
            .with_dropout(0.3)
            .with_activation("tanh");
        assert!((cfg.dropout_rate - 0.3).abs() < 1e-10);
        assert_eq!(cfg.activation, "tanh");
    }

    // ---- Autoencoder Model Tests ----

    #[test]
    fn test_ae_creation() {
        let cfg = AutoencoderConfig::tiny(50, 10);
        let model: Autoencoder<f64> = Autoencoder::new(cfg).expect("Failed to create AE");
        assert!(model.total_parameter_count() > 0);
    }

    #[test]
    fn test_ae_forward() {
        let cfg = AutoencoderConfig::tiny(20, 5);
        let model: Autoencoder<f64> = Autoencoder::new(cfg).expect("Failed to create AE");

        let input = Array::zeros(IxDyn(&[2, 20]));
        let output = model.forward(&input).expect("Forward failed");
        assert_eq!(output.shape(), &[2, 20]);
    }

    #[test]
    fn test_ae_encode_decode() {
        let cfg = AutoencoderConfig::tiny(20, 5);
        let model: Autoencoder<f64> = Autoencoder::new(cfg).expect("Failed to create AE");

        let input = Array::zeros(IxDyn(&[1, 20]));
        let latent = model.encode(&input).expect("Encode failed");
        assert_eq!(latent.shape(), &[1, 5]);

        let recon = model.decode(&latent).expect("Decode failed");
        assert_eq!(recon.shape(), &[1, 20]);
    }

    #[test]
    fn test_ae_multi_hidden() {
        let cfg = AutoencoderConfig::custom(30, vec![20, 10], 5);
        let model: Autoencoder<f64> = Autoencoder::new(cfg).expect("Failed to create AE");

        let input = Array::zeros(IxDyn(&[3, 30]));
        let output = model.forward(&input).expect("Forward failed");
        assert_eq!(output.shape(), &[3, 30]);
    }

    #[test]
    fn test_ae_update() {
        let cfg = AutoencoderConfig::tiny(20, 5);
        let mut model: Autoencoder<f64> = Autoencoder::new(cfg).expect("Failed to create AE");
        model.update(0.001).expect("Update failed");
    }

    #[test]
    fn test_ae_layer_trait() {
        let cfg = AutoencoderConfig::tiny(20, 5);
        let model: Autoencoder<f64> = Autoencoder::new(cfg).expect("Failed to create AE");
        assert_eq!(model.layer_type(), "Autoencoder");
        let desc = model.layer_description();
        assert!(desc.contains("Autoencoder"));
        assert!(desc.contains("latent=5"));
    }

    #[test]
    fn test_ae_invalid_dims() {
        let cfg = AutoencoderConfig {
            input_dim: 0,
            hidden_dims: vec![],
            latent_dim: 5,
            dropout_rate: 0.0,
            activation: "relu".to_string(),
        };
        assert!(Autoencoder::<f64>::new(cfg).is_err());
    }

    #[test]
    fn test_ae_with_dropout() {
        let cfg = AutoencoderConfig::tiny(20, 5).with_dropout(0.5);
        let model: Autoencoder<f64> = Autoencoder::new(cfg).expect("Failed to create AE");
        assert!(model.dropout.is_some());
        let input = Array::zeros(IxDyn(&[1, 20]));
        let _output = model.forward(&input).expect("Forward with dropout failed");
    }

    #[test]
    fn test_ae_f32() {
        let cfg = AutoencoderConfig::tiny(20, 5);
        let model: Autoencoder<f32> = Autoencoder::new(cfg).expect("Failed f32 AE");
        let input = Array::zeros(IxDyn(&[1, 20]));
        let output = model.forward(&input).expect("f32 forward failed");
        assert_eq!(output.shape(), &[1, 20]);
    }

    #[test]
    fn test_ae_params() {
        let cfg = AutoencoderConfig::tiny(20, 5);
        let model: Autoencoder<f64> = Autoencoder::new(cfg).expect("Failed to create AE");
        let p = model.params();
        assert!(!p.is_empty());
    }

    // ---- VAE Config Tests ----

    #[test]
    fn test_vae_config_standard() {
        let cfg = VAEConfig::standard(784, 32);
        assert_eq!(cfg.base.input_dim, 784);
        assert!((cfg.kl_weight - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vae_config_beta() {
        let cfg = VAEConfig::standard(784, 32).with_kl_weight(0.5);
        assert!((cfg.kl_weight - 0.5).abs() < 1e-10);
    }

    // ---- VAE Model Tests ----

    #[test]
    fn test_vae_creation() {
        let cfg = VAEConfig::tiny(20, 5);
        let model: VAE<f64> = VAE::new(cfg).expect("Failed to create VAE");
        assert!(model.total_parameter_count() > 0);
    }

    #[test]
    fn test_vae_forward() {
        let cfg = VAEConfig::tiny(20, 5);
        let model: VAE<f64> = VAE::new(cfg).expect("Failed to create VAE");

        let input = Array::zeros(IxDyn(&[2, 20]));
        let output = model.forward(&input).expect("Forward failed");
        assert_eq!(output.shape(), &[2, 20]);
    }

    #[test]
    fn test_vae_forward_detailed() {
        let cfg = VAEConfig::tiny(20, 5);
        let model: VAE<f64> = VAE::new(cfg).expect("Failed to create VAE");

        let input = Array::zeros(IxDyn(&[1, 20]));
        let result = model
            .forward_detailed(&input)
            .expect("Detailed forward failed");

        assert_eq!(result.reconstruction.shape(), &[1, 20]);
        assert_eq!(result.mu.shape(), &[1, 5]);
        assert_eq!(result.log_var.shape(), &[1, 5]);
        assert_eq!(result.z.shape(), &[1, 5]);
    }

    #[test]
    fn test_vae_encode_distribution() {
        let cfg = VAEConfig::tiny(20, 5);
        let model: VAE<f64> = VAE::new(cfg).expect("Failed to create VAE");

        let input = Array::zeros(IxDyn(&[1, 20]));
        let (mu, log_var) = model
            .encode_distribution(&input)
            .expect("Encode dist failed");
        assert_eq!(mu.shape(), &[1, 5]);
        assert_eq!(log_var.shape(), &[1, 5]);
    }

    #[test]
    fn test_vae_reparameterize() {
        let cfg = VAEConfig::tiny(20, 5);
        let model: VAE<f64> = VAE::new(cfg).expect("Failed to create VAE");

        let mu = Array::zeros(IxDyn(&[1, 5]));
        let log_var = Array::zeros(IxDyn(&[1, 5]));
        let z = model.reparameterize(&mu, &log_var);
        assert_eq!(z.shape(), &[1, 5]);
    }

    #[test]
    fn test_vae_kl_divergence() {
        // KL should be 0 when mu=0, log_var=0
        let mu = Array::zeros(IxDyn(&[1, 5]));
        let log_var = Array::zeros(IxDyn(&[1, 5]));
        let kl = VAE::<f64>::kl_divergence(&mu, &log_var);
        // KL = -0.5 * (1 + 0 - 0 - 1) * 5 / 5 = 0
        assert!(
            kl.abs() < 1e-6,
            "KL should be ~0 for standard normal, got {}",
            kl
        );
    }

    #[test]
    fn test_vae_kl_divergence_nonzero() {
        let mu = Array::from_elem(IxDyn(&[1, 5]), 1.0_f64);
        let log_var = Array::zeros(IxDyn(&[1, 5]));
        let kl = VAE::<f64>::kl_divergence(&mu, &log_var);
        // KL should be positive when mu != 0
        assert!(kl > 0.0, "KL should be > 0 for mu != 0, got {}", kl);
    }

    #[test]
    fn test_vae_elbo_loss() {
        let cfg = VAEConfig::tiny(20, 5);
        let model: VAE<f64> = VAE::new(cfg).expect("Failed to create VAE");

        let input = Array::zeros(IxDyn(&[1, 20]));
        let result = model.forward_detailed(&input).expect("Forward failed");
        let loss = model.elbo_loss(&input, &result);
        assert!(loss.is_finite(), "ELBO loss should be finite");
    }

    #[test]
    fn test_vae_generate() {
        let cfg = VAEConfig::tiny(20, 5);
        let model: VAE<f64> = VAE::new(cfg).expect("Failed to create VAE");

        let samples = model.generate(3).expect("Generation failed");
        assert_eq!(samples.shape(), &[3, 20]);
    }

    #[test]
    fn test_vae_reconstruct() {
        let cfg = VAEConfig::tiny(20, 5);
        let model: VAE<f64> = VAE::new(cfg).expect("Failed to create VAE");

        let input = Array::zeros(IxDyn(&[2, 20]));
        let recon = model.reconstruct(&input).expect("Reconstruct failed");
        assert_eq!(recon.shape(), &[2, 20]);
    }

    #[test]
    fn test_vae_update() {
        let cfg = VAEConfig::tiny(20, 5);
        let mut model: VAE<f64> = VAE::new(cfg).expect("Failed to create VAE");
        model.update(0.001).expect("Update failed");
    }

    #[test]
    fn test_vae_layer_trait() {
        let cfg = VAEConfig::tiny(20, 5);
        let model: VAE<f64> = VAE::new(cfg).expect("Failed to create VAE");
        assert_eq!(model.layer_type(), "VAE");
        let desc = model.layer_description();
        assert!(desc.contains("VAE"));
        assert!(desc.contains("latent=5"));
    }

    #[test]
    fn test_vae_beta() {
        let cfg = VAEConfig::tiny(20, 5).with_kl_weight(0.1);
        let model: VAE<f64> = VAE::new(cfg).expect("Failed to create beta-VAE");
        assert!((model.config().kl_weight - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_vae_f32() {
        let cfg = VAEConfig::tiny(20, 5);
        let model: VAE<f32> = VAE::new(cfg).expect("Failed to create f32 VAE");
        let input = Array::zeros(IxDyn(&[1, 20]));
        let output = model.forward(&input).expect("f32 forward failed");
        assert_eq!(output.shape(), &[1, 20]);
    }

    #[test]
    fn test_vae_params() {
        let cfg = VAEConfig::tiny(20, 5);
        let model: VAE<f64> = VAE::new(cfg).expect("Failed to create VAE");
        let p = model.params();
        assert!(!p.is_empty());
        // Should have more params than a basic AE due to mu + log_var heads
        let ae_cfg = AutoencoderConfig::tiny(20, 5);
        let ae: Autoencoder<f64> = Autoencoder::new(ae_cfg).expect("AE");
        assert!(model.total_parameter_count() > ae.total_parameter_count());
    }

    #[test]
    fn test_xorshift_f64_range() {
        let mut state = 42u64;
        for _ in 0..100 {
            let v = xorshift_f64(&mut state);
            assert!(v >= 0.0 && v < 1.0, "xorshift out of range: {}", v);
        }
    }
}
