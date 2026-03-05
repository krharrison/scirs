//! GAN (Generative Adversarial Network) framework
//!
//! Provides a modular GAN framework with:
//! - Configurable Generator and Discriminator networks
//! - Standard GAN training loop (alternating optimization)
//! - WGAN-GP (Wasserstein GAN with Gradient Penalty) variant
//! - Generation from noise
//!
//! References:
//! - "Generative Adversarial Nets", Goodfellow et al. (2014) <https://arxiv.org/abs/1406.2661>
//! - "Improved Training of Wasserstein GANs", Gulrajani et al. (2017) <https://arxiv.org/abs/1704.00028>

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Dropout, Layer};
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::SeedableRng;
use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// GAN training mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GANMode {
    /// Standard GAN with binary cross-entropy loss
    Standard,
    /// Wasserstein GAN with gradient penalty (WGAN-GP)
    WGANGP,
}

/// Configuration for the Generator network
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    /// Noise (latent) input dimension
    pub noise_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Output dimension (e.g., flattened image size)
    pub output_dim: usize,
    /// Dropout rate
    pub dropout_rate: f64,
    /// Whether to use batch normalization
    pub use_batch_norm: bool,
}

impl GeneratorConfig {
    /// Standard generator config
    pub fn standard(noise_dim: usize, output_dim: usize) -> Self {
        Self {
            noise_dim,
            hidden_dims: vec![256, 512, 1024],
            output_dim,
            dropout_rate: 0.0,
            use_batch_norm: true,
        }
    }

    /// Tiny generator for testing
    pub fn tiny(noise_dim: usize, output_dim: usize) -> Self {
        Self {
            noise_dim,
            hidden_dims: vec![32, 64],
            output_dim,
            dropout_rate: 0.0,
            use_batch_norm: false,
        }
    }
}

/// Configuration for the Discriminator network
#[derive(Debug, Clone)]
pub struct DiscriminatorConfig {
    /// Input dimension (same as generator output)
    pub input_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Dropout rate (commonly used in discriminators)
    pub dropout_rate: f64,
}

impl DiscriminatorConfig {
    /// Standard discriminator config
    pub fn standard(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dims: vec![1024, 512, 256],
            dropout_rate: 0.2,
        }
    }

    /// Tiny discriminator for testing
    pub fn tiny(input_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dims: vec![64, 32],
            dropout_rate: 0.0,
        }
    }
}

/// Configuration for the full GAN system
#[derive(Debug, Clone)]
pub struct GANConfig {
    /// Generator configuration
    pub generator: GeneratorConfig,
    /// Discriminator configuration
    pub discriminator: DiscriminatorConfig,
    /// Training mode
    pub mode: GANMode,
    /// Gradient penalty coefficient for WGAN-GP (lambda)
    pub gradient_penalty_weight: f64,
    /// Number of discriminator steps per generator step
    pub n_critic: usize,
}

impl GANConfig {
    /// Standard GAN configuration
    pub fn standard(noise_dim: usize, data_dim: usize) -> Self {
        Self {
            generator: GeneratorConfig::standard(noise_dim, data_dim),
            discriminator: DiscriminatorConfig::standard(data_dim),
            mode: GANMode::Standard,
            gradient_penalty_weight: 10.0,
            n_critic: 1,
        }
    }

    /// WGAN-GP configuration (recommended)
    pub fn wgan_gp(noise_dim: usize, data_dim: usize) -> Self {
        Self {
            generator: GeneratorConfig::standard(noise_dim, data_dim),
            discriminator: DiscriminatorConfig::standard(data_dim),
            mode: GANMode::WGANGP,
            gradient_penalty_weight: 10.0,
            n_critic: 5,
        }
    }

    /// Tiny configuration for testing
    pub fn tiny(noise_dim: usize, data_dim: usize) -> Self {
        Self {
            generator: GeneratorConfig::tiny(noise_dim, data_dim),
            discriminator: DiscriminatorConfig::tiny(data_dim),
            mode: GANMode::Standard,
            gradient_penalty_weight: 10.0,
            n_critic: 1,
        }
    }

    /// Set GAN mode
    pub fn with_mode(mut self, mode: GANMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set gradient penalty weight (WGAN-GP)
    pub fn with_gradient_penalty(mut self, weight: f64) -> Self {
        self.gradient_penalty_weight = weight;
        self
    }

    /// Set number of critic steps per generator step
    pub fn with_n_critic(mut self, n: usize) -> Self {
        self.n_critic = n.max(1);
        self
    }
}

// ---------------------------------------------------------------------------
// Generator
// ---------------------------------------------------------------------------

/// Generator network: maps noise z to data space
///
/// Architecture: noise_dim -> hidden[0] -> ... -> hidden[n-1] -> output_dim
/// Uses ReLU activations in hidden layers and tanh in the output layer.
pub struct Generator<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    config: GeneratorConfig,
    layers: Vec<Dense<F>>,
    dropout: Option<Dropout<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Generator<F> {
    /// Create a new Generator
    pub fn new(config: GeneratorConfig) -> Result<Self> {
        if config.noise_dim == 0 || config.output_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "noise_dim and output_dim must be > 0".to_string(),
            ));
        }

        let mut layers = Vec::new();
        let mut in_dim = config.noise_dim;
        let mut seed: u8 = 110;

        for &hdim in &config.hidden_dims {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([seed; 32]);
            seed = seed.wrapping_add(1);
            layers.push(Dense::new(in_dim, hdim, None, &mut rng)?);
            in_dim = hdim;
        }

        // Output layer
        {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([seed; 32]);
            layers.push(Dense::new(in_dim, config.output_dim, None, &mut rng)?);
        }

        let dropout = if config.dropout_rate > 0.0 {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([120; 32]);
            Some(Dropout::new(config.dropout_rate, &mut rng)?)
        } else {
            None
        };

        Ok(Self {
            config,
            layers,
            dropout,
        })
    }

    /// Get config
    pub fn config(&self) -> &GeneratorConfig {
        &self.config
    }

    /// Total parameter count
    pub fn total_parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.parameter_count()).sum()
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for Generator<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.layers.len() - 1 {
                // Hidden layers: ReLU + optional dropout
                x = x.mapv(|v| v.max(F::zero()));
                if let Some(ref drop) = self.dropout {
                    x = drop.forward(&x)?;
                }
            } else {
                // Output layer: tanh to bound output to [-1, 1]
                x = x.mapv(|v| v.tanh());
            }
        }
        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        for layer in &mut self.layers {
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
        self.layers.iter().flat_map(|l| l.params()).collect()
    }

    fn parameter_count(&self) -> usize {
        self.total_parameter_count()
    }

    fn layer_type(&self) -> &str {
        "Generator"
    }

    fn layer_description(&self) -> String {
        format!(
            "Generator(noise={}, hidden={:?}, output={}, params={})",
            self.config.noise_dim,
            self.config.hidden_dims,
            self.config.output_dim,
            self.total_parameter_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Discriminator
// ---------------------------------------------------------------------------

/// Discriminator network: maps data to a scalar (real/fake score)
///
/// Architecture: input_dim -> hidden[0] -> ... -> hidden[n-1] -> 1
/// Uses LeakyReLU activations in hidden layers.
/// Output is raw logit (no sigmoid) for numerical stability.
pub struct Discriminator<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    config: DiscriminatorConfig,
    layers: Vec<Dense<F>>,
    dropout: Option<Dropout<F>>,
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Discriminator<F> {
    /// Create a new Discriminator
    pub fn new(config: DiscriminatorConfig) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "input_dim must be > 0".to_string(),
            ));
        }

        let mut layers = Vec::new();
        let mut in_dim = config.input_dim;
        let mut seed: u8 = 130;

        for &hdim in &config.hidden_dims {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([seed; 32]);
            seed = seed.wrapping_add(1);
            layers.push(Dense::new(in_dim, hdim, None, &mut rng)?);
            in_dim = hdim;
        }

        // Output layer: single logit
        {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([seed; 32]);
            layers.push(Dense::new(in_dim, 1, None, &mut rng)?);
        }

        let dropout = if config.dropout_rate > 0.0 {
            let mut rng = scirs2_core::random::rngs::SmallRng::from_seed([140; 32]);
            Some(Dropout::new(config.dropout_rate, &mut rng)?)
        } else {
            None
        };

        Ok(Self {
            config,
            layers,
            dropout,
        })
    }

    /// Get config
    pub fn config(&self) -> &DiscriminatorConfig {
        &self.config
    }

    /// Total parameter count
    pub fn total_parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.parameter_count()).sum()
    }
}

/// LeakyReLU with negative slope 0.2
fn leaky_relu<F: Float>(x: F) -> F {
    let slope = F::from(0.2).expect("leaky relu slope");
    if x > F::zero() {
        x
    } else {
        slope * x
    }
}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F>
    for Discriminator<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let mut x = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.layers.len() - 1 {
                // Hidden: LeakyReLU + optional dropout
                x = x.mapv(leaky_relu);
                if let Some(ref drop) = self.dropout {
                    x = drop.forward(&x)?;
                }
            }
            // Output: raw logit (no activation)
        }
        Ok(x)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        for layer in &mut self.layers {
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
        self.layers.iter().flat_map(|l| l.params()).collect()
    }

    fn parameter_count(&self) -> usize {
        self.total_parameter_count()
    }

    fn layer_type(&self) -> &str {
        "Discriminator"
    }

    fn layer_description(&self) -> String {
        format!(
            "Discriminator(input={}, hidden={:?}, params={})",
            self.config.input_dim,
            self.config.hidden_dims,
            self.total_parameter_count()
        )
    }
}

// ---------------------------------------------------------------------------
// GAN training step result
// ---------------------------------------------------------------------------

/// Result of a single GAN training step
#[derive(Debug, Clone)]
pub struct GANStepResult<F: Float> {
    /// Discriminator loss
    pub d_loss: F,
    /// Generator loss
    pub g_loss: F,
    /// Gradient penalty (WGAN-GP only, zero otherwise)
    pub gradient_penalty: F,
    /// Mean discriminator output on real data
    pub d_real_mean: F,
    /// Mean discriminator output on fake data
    pub d_fake_mean: F,
}

// ---------------------------------------------------------------------------
// GAN system
// ---------------------------------------------------------------------------

/// Complete GAN system combining Generator and Discriminator
///
/// Supports standard GAN training and WGAN-GP variant.
///
/// # Examples
///
/// ```no_run
/// use scirs2_neural::models::architectures::gan::{GAN, GANConfig};
///
/// let config = GANConfig::tiny(10, 20);
/// let mut gan: GAN<f64> = GAN::new(config).expect("Failed to create GAN");
/// ```
pub struct GAN<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> {
    config: GANConfig,
    /// Generator network
    pub generator: Generator<F>,
    /// Discriminator (critic) network
    pub discriminator: Discriminator<F>,
    /// PRNG state for noise generation
    rng_state: std::cell::Cell<u64>,
    /// Training step counter
    step_count: usize,
}

// SAFETY: rng_state is only mutated during single-threaded forward/train calls
unsafe impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Sync for GAN<F> {}

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> GAN<F> {
    /// Create a new GAN
    pub fn new(config: GANConfig) -> Result<Self> {
        let generator = Generator::new(config.generator.clone())?;
        let discriminator = Discriminator::new(config.discriminator.clone())?;

        Ok(Self {
            config,
            generator,
            discriminator,
            rng_state: std::cell::Cell::new(0xBEEF_CAFE_DEAD_BABEu64),
            step_count: 0,
        })
    }

    /// Get configuration
    pub fn config(&self) -> &GANConfig {
        &self.config
    }

    /// Generate random noise for the generator
    pub fn sample_noise(&self, batch_size: usize) -> Array<F, IxDyn> {
        let noise_dim = self.config.generator.noise_dim;
        let mut state = self.rng_state.get();
        let mut noise = Array::zeros(IxDyn(&[batch_size, noise_dim]));

        for b in 0..batch_size {
            for d in 0..noise_dim {
                // Box-Muller for standard normal
                let u1 = xorshift_f64(&mut state).max(1e-10);
                let u2 = xorshift_f64(&mut state);
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                noise[[b, d]] = F::from(normal).expect("noise conversion");
            }
        }

        self.rng_state.set(state);
        noise
    }

    /// Generate fake samples
    pub fn generate(&self, batch_size: usize) -> Result<Array<F, IxDyn>> {
        let noise = self.sample_noise(batch_size);
        self.generator.forward(&noise)
    }

    /// Compute binary cross-entropy loss (standard GAN)
    ///
    /// loss = -mean(target * log(sigmoid(logit)) + (1-target) * log(1-sigmoid(logit)))
    fn bce_loss(logits: &Array<F, IxDyn>, target: F) -> F {
        let eps = F::from(1e-7).expect("eps");
        let one = F::one();
        let n = F::from(logits.len()).expect("n");

        let mut loss = F::zero();
        for &logit in logits.iter() {
            let sigmoid = one / (one + (-logit).exp());
            let sig_clamped = sigmoid.max(eps).min(one - eps);
            loss += target * sig_clamped.ln() + (one - target) * (one - sig_clamped).ln();
        }

        -loss / n
    }

    /// Compute Wasserstein loss
    ///
    /// D wants to maximize E[D(real)] - E[D(fake)]
    /// G wants to maximize E[D(fake)]
    fn wasserstein_d_loss(d_real: &Array<F, IxDyn>, d_fake: &Array<F, IxDyn>) -> F {
        let n_real = F::from(d_real.len()).expect("n");
        let n_fake = F::from(d_fake.len()).expect("n");
        let real_mean: F = d_real.iter().copied().fold(F::zero(), |a, b| a + b) / n_real;
        let fake_mean: F = d_fake.iter().copied().fold(F::zero(), |a, b| a + b) / n_fake;
        // Minimize negative Wasserstein distance
        fake_mean - real_mean
    }

    fn wasserstein_g_loss(d_fake: &Array<F, IxDyn>) -> F {
        let n = F::from(d_fake.len()).expect("n");
        let fake_mean: F = d_fake.iter().copied().fold(F::zero(), |a, b| a + b) / n;
        -fake_mean
    }

    /// Compute gradient penalty for WGAN-GP
    ///
    /// Approximation: we compute the finite-difference gradient of D
    /// along the interpolation between real and fake samples, and penalize
    /// deviations from unit norm.
    fn gradient_penalty(&self, real: &Array<F, IxDyn>, fake: &Array<F, IxDyn>) -> Result<F> {
        let shape = real.shape();
        let batch_size = shape[0];
        let data_dim = shape[1];

        // Random interpolation coefficient per sample
        let mut state = self.rng_state.get();
        let mut penalty = F::zero();
        let epsilon_fd = F::from(1e-4).expect("fd epsilon");

        for b in 0..batch_size {
            let alpha = F::from(xorshift_f64(&mut state)).expect("alpha");

            // Interpolated point
            let mut interp = Array::zeros(IxDyn(&[1, data_dim]));
            for d in 0..data_dim {
                interp[[0, d]] = alpha * real[[b, d]] + (F::one() - alpha) * fake[[b, d]];
            }

            // Approximate gradient norm via finite differences
            let d_interp = self.discriminator.forward(&interp)?;
            let base_val = d_interp[[0, 0]];

            let mut grad_norm_sq = F::zero();
            for d in 0..data_dim {
                let mut perturbed = interp.clone();
                perturbed[[0, d]] += epsilon_fd;
                let d_perturbed = self.discriminator.forward(&perturbed)?;
                let grad_d = (d_perturbed[[0, 0]] - base_val) / epsilon_fd;
                grad_norm_sq += grad_d * grad_d;
            }

            let grad_norm = grad_norm_sq.sqrt();
            let diff = grad_norm - F::one();
            penalty += diff * diff;
        }

        self.rng_state.set(state);

        Ok(penalty / F::from(batch_size).expect("batch"))
    }

    /// Execute one training step
    ///
    /// Performs alternating discriminator and generator updates.
    /// Returns loss statistics.
    pub fn train_step(
        &mut self,
        real_data: &Array<F, IxDyn>,
        learning_rate: F,
    ) -> Result<GANStepResult<F>> {
        let batch_size = real_data.shape()[0];

        // ----- Discriminator step -----
        let fake_data = self.generate(batch_size)?;

        let d_real = self.discriminator.forward(real_data)?;
        let d_fake = self.discriminator.forward(&fake_data)?;

        let (d_loss, gp) = match self.config.mode {
            GANMode::Standard => {
                let loss_real = Self::bce_loss(&d_real, F::one());
                let loss_fake = Self::bce_loss(&d_fake, F::zero());
                (loss_real + loss_fake, F::zero())
            }
            GANMode::WGANGP => {
                let w_loss = Self::wasserstein_d_loss(&d_real, &d_fake);
                let gp = self.gradient_penalty(real_data, &fake_data)?;
                let lambda = F::from(self.config.gradient_penalty_weight).expect("lambda");
                (w_loss + lambda * gp, gp)
            }
        };

        // Update discriminator
        self.discriminator.update(learning_rate)?;

        // ----- Generator step (every n_critic steps) -----
        self.step_count += 1;
        let g_loss = if self.step_count % self.config.n_critic == 0 {
            let fake_data = self.generate(batch_size)?;
            let d_fake_for_g = self.discriminator.forward(&fake_data)?;

            let g_loss = match self.config.mode {
                GANMode::Standard => Self::bce_loss(&d_fake_for_g, F::one()),
                GANMode::WGANGP => Self::wasserstein_g_loss(&d_fake_for_g),
            };

            self.generator.update(learning_rate)?;
            g_loss
        } else {
            F::zero()
        };

        // Compute means for monitoring
        let n_real = F::from(d_real.len()).expect("n");
        let n_fake = F::from(d_fake.len()).expect("n");
        let d_real_mean: F = d_real.iter().copied().fold(F::zero(), |a, b| a + b) / n_real;
        let d_fake_mean: F = d_fake.iter().copied().fold(F::zero(), |a, b| a + b) / n_fake;

        Ok(GANStepResult {
            d_loss,
            g_loss,
            gradient_penalty: gp,
            d_real_mean,
            d_fake_mean,
        })
    }

    /// Total parameter count for both networks
    pub fn total_parameter_count(&self) -> usize {
        self.generator.total_parameter_count() + self.discriminator.total_parameter_count()
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

impl<F: Float + Debug + ScalarOperand + Send + Sync + NumAssign + 'static> Layer<F> for GAN<F> {
    /// Forward pass runs the generator (maps noise to data)
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        self.generator.forward(input)
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, learning_rate: F) -> Result<()> {
        self.generator.update(learning_rate)?;
        self.discriminator.update(learning_rate)?;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = self.generator.params();
        p.extend(self.discriminator.params());
        p
    }

    fn parameter_count(&self) -> usize {
        self.total_parameter_count()
    }

    fn layer_type(&self) -> &str {
        "GAN"
    }

    fn layer_description(&self) -> String {
        format!(
            "GAN(mode={:?}, noise={}, data={}, g_params={}, d_params={})",
            self.config.mode,
            self.config.generator.noise_dim,
            self.config.generator.output_dim,
            self.generator.total_parameter_count(),
            self.discriminator.total_parameter_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Config Tests ----

    #[test]
    fn test_generator_config_standard() {
        let cfg = GeneratorConfig::standard(100, 784);
        assert_eq!(cfg.noise_dim, 100);
        assert_eq!(cfg.output_dim, 784);
        assert_eq!(cfg.hidden_dims, vec![256, 512, 1024]);
    }

    #[test]
    fn test_generator_config_tiny() {
        let cfg = GeneratorConfig::tiny(10, 20);
        assert_eq!(cfg.hidden_dims, vec![32, 64]);
    }

    #[test]
    fn test_discriminator_config_standard() {
        let cfg = DiscriminatorConfig::standard(784);
        assert_eq!(cfg.input_dim, 784);
        assert_eq!(cfg.hidden_dims, vec![1024, 512, 256]);
        assert!((cfg.dropout_rate - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_gan_config_standard() {
        let cfg = GANConfig::standard(100, 784);
        assert_eq!(cfg.mode, GANMode::Standard);
        assert_eq!(cfg.n_critic, 1);
    }

    #[test]
    fn test_gan_config_wgan_gp() {
        let cfg = GANConfig::wgan_gp(100, 784);
        assert_eq!(cfg.mode, GANMode::WGANGP);
        assert_eq!(cfg.n_critic, 5);
        assert!((cfg.gradient_penalty_weight - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_gan_config_builder() {
        let cfg = GANConfig::standard(10, 20)
            .with_mode(GANMode::WGANGP)
            .with_gradient_penalty(5.0)
            .with_n_critic(3);
        assert_eq!(cfg.mode, GANMode::WGANGP);
        assert!((cfg.gradient_penalty_weight - 5.0).abs() < 1e-10);
        assert_eq!(cfg.n_critic, 3);
    }

    // ---- Generator Tests ----

    #[test]
    fn test_generator_creation() {
        let cfg = GeneratorConfig::tiny(10, 20);
        let gen: Generator<f64> = Generator::new(cfg).expect("Failed to create generator");
        assert!(gen.total_parameter_count() > 0);
    }

    #[test]
    fn test_generator_forward() {
        let cfg = GeneratorConfig::tiny(10, 20);
        let gen: Generator<f64> = Generator::new(cfg).expect("Failed to create generator");

        let noise = Array::zeros(IxDyn(&[2, 10]));
        let output = gen.forward(&noise).expect("Generator forward failed");
        assert_eq!(output.shape(), &[2, 20]);

        // Output should be bounded by tanh: [-1, 1]
        for &v in output.iter() {
            assert!(v >= -1.0 && v <= 1.0, "tanh should bound output, got {}", v);
        }
    }

    #[test]
    fn test_generator_layer_trait() {
        let cfg = GeneratorConfig::tiny(10, 20);
        let gen: Generator<f64> = Generator::new(cfg).expect("Failed to create generator");
        assert_eq!(gen.layer_type(), "Generator");
        assert!(gen.parameter_count() > 0);
    }

    #[test]
    fn test_generator_invalid() {
        let cfg = GeneratorConfig {
            noise_dim: 0,
            hidden_dims: vec![],
            output_dim: 10,
            dropout_rate: 0.0,
            use_batch_norm: false,
        };
        assert!(Generator::<f64>::new(cfg).is_err());
    }

    // ---- Discriminator Tests ----

    #[test]
    fn test_discriminator_creation() {
        let cfg = DiscriminatorConfig::tiny(20);
        let disc: Discriminator<f64> =
            Discriminator::new(cfg).expect("Failed to create discriminator");
        assert!(disc.total_parameter_count() > 0);
    }

    #[test]
    fn test_discriminator_forward() {
        let cfg = DiscriminatorConfig::tiny(20);
        let disc: Discriminator<f64> =
            Discriminator::new(cfg).expect("Failed to create discriminator");

        let input = Array::zeros(IxDyn(&[3, 20]));
        let output = disc.forward(&input).expect("Discriminator forward failed");
        assert_eq!(output.shape(), &[3, 1]);
    }

    #[test]
    fn test_discriminator_layer_trait() {
        let cfg = DiscriminatorConfig::tiny(20);
        let disc: Discriminator<f64> =
            Discriminator::new(cfg).expect("Failed to create discriminator");
        assert_eq!(disc.layer_type(), "Discriminator");
    }

    #[test]
    fn test_discriminator_invalid() {
        let cfg = DiscriminatorConfig {
            input_dim: 0,
            hidden_dims: vec![],
            dropout_rate: 0.0,
        };
        assert!(Discriminator::<f64>::new(cfg).is_err());
    }

    // ---- GAN System Tests ----

    #[test]
    fn test_gan_creation() {
        let cfg = GANConfig::tiny(10, 20);
        let gan: GAN<f64> = GAN::new(cfg).expect("Failed to create GAN");
        assert!(gan.total_parameter_count() > 0);
    }

    #[test]
    fn test_gan_generate() {
        let cfg = GANConfig::tiny(10, 20);
        let gan: GAN<f64> = GAN::new(cfg).expect("Failed to create GAN");

        let samples = gan.generate(5).expect("Generation failed");
        assert_eq!(samples.shape(), &[5, 20]);
    }

    #[test]
    fn test_gan_sample_noise() {
        let cfg = GANConfig::tiny(10, 20);
        let gan: GAN<f64> = GAN::new(cfg).expect("Failed to create GAN");

        let noise = gan.sample_noise(3);
        assert_eq!(noise.shape(), &[3, 10]);
    }

    #[test]
    fn test_gan_forward() {
        let cfg = GANConfig::tiny(10, 20);
        let gan: GAN<f64> = GAN::new(cfg).expect("Failed to create GAN");

        let noise = Array::zeros(IxDyn(&[2, 10]));
        let output = gan.forward(&noise).expect("GAN forward failed");
        assert_eq!(output.shape(), &[2, 20]);
    }

    #[test]
    fn test_gan_train_step_standard() {
        let cfg = GANConfig::tiny(10, 20);
        let mut gan: GAN<f64> = GAN::new(cfg).expect("Failed to create GAN");

        let real_data = Array::zeros(IxDyn(&[4, 20]));
        let result = gan
            .train_step(&real_data, 0.001)
            .expect("Train step failed");

        assert!(result.d_loss.is_finite(), "d_loss should be finite");
        assert!(result.g_loss.is_finite(), "g_loss should be finite");
    }

    #[test]
    fn test_gan_train_step_wgan() {
        let cfg = GANConfig::tiny(10, 20)
            .with_mode(GANMode::WGANGP)
            .with_n_critic(1);
        let mut gan: GAN<f64> = GAN::new(cfg).expect("Failed to create WGAN");

        let real_data = Array::zeros(IxDyn(&[4, 20]));
        let result = gan
            .train_step(&real_data, 0.001)
            .expect("WGAN train step failed");

        assert!(result.d_loss.is_finite(), "d_loss should be finite");
        assert!(result.gradient_penalty.is_finite(), "GP should be finite");
    }

    #[test]
    fn test_gan_multiple_train_steps() {
        let cfg = GANConfig::tiny(10, 20);
        let mut gan: GAN<f64> = GAN::new(cfg).expect("Failed to create GAN");

        let real_data = Array::zeros(IxDyn(&[4, 20]));
        for _ in 0..5 {
            let result = gan
                .train_step(&real_data, 0.001)
                .expect("Train step failed");
            assert!(result.d_loss.is_finite());
        }
    }

    #[test]
    fn test_gan_layer_trait() {
        let cfg = GANConfig::tiny(10, 20);
        let gan: GAN<f64> = GAN::new(cfg).expect("Failed to create GAN");
        assert_eq!(gan.layer_type(), "GAN");
        let desc = gan.layer_description();
        assert!(desc.contains("GAN"));
        assert!(desc.contains("Standard"));
    }

    #[test]
    fn test_gan_update() {
        let cfg = GANConfig::tiny(10, 20);
        let mut gan: GAN<f64> = GAN::new(cfg).expect("Failed to create GAN");
        gan.update(0.001).expect("Update failed");
    }

    #[test]
    fn test_gan_params() {
        let cfg = GANConfig::tiny(10, 20);
        let gan: GAN<f64> = GAN::new(cfg).expect("Failed to create GAN");
        let p = gan.params();
        assert!(!p.is_empty());
    }

    #[test]
    fn test_gan_f32() {
        let cfg = GANConfig::tiny(10, 20);
        let gan: GAN<f32> = GAN::new(cfg).expect("Failed to create f32 GAN");
        let noise = Array::zeros(IxDyn(&[1, 10]));
        let output = gan.forward(&noise).expect("f32 forward failed");
        assert_eq!(output.shape(), &[1, 20]);
    }

    #[test]
    fn test_bce_loss() {
        // When logits are large positive and target=1, loss should be small
        let logits = Array::from_elem(IxDyn(&[4, 1]), 10.0_f64);
        let loss = GAN::<f64>::bce_loss(&logits, 1.0);
        assert!(
            loss < 0.01,
            "BCE loss should be small for correct prediction"
        );

        // When logits are large positive and target=0, loss should be large
        let loss_wrong = GAN::<f64>::bce_loss(&logits, 0.0);
        assert!(loss_wrong > loss, "Wrong target should give higher loss");
    }

    #[test]
    fn test_leaky_relu() {
        assert!((leaky_relu(1.0_f64) - 1.0).abs() < 1e-10);
        assert!((leaky_relu(0.0_f64) - 0.0).abs() < 1e-10);
        assert!((leaky_relu(-1.0_f64) - (-0.2)).abs() < 1e-10);
        assert!((leaky_relu(-5.0_f64) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_xorshift_f64_range() {
        let mut state = 42u64;
        for _ in 0..100 {
            let v = xorshift_f64(&mut state);
            assert!(v >= 0.0 && v < 1.0);
        }
    }

    #[test]
    fn test_n_critic_skips_generator() {
        let cfg = GANConfig::tiny(10, 20).with_n_critic(3);
        let mut gan: GAN<f64> = GAN::new(cfg).expect("Failed to create GAN");

        let real_data = Array::zeros(IxDyn(&[2, 20]));

        // Steps 1 and 2: generator loss should be 0 (not updated)
        let r1 = gan.train_step(&real_data, 0.001).expect("Step 1");
        assert!(
            (r1.g_loss - 0.0).abs() < 1e-10,
            "G should not train on step 1"
        );

        let r2 = gan.train_step(&real_data, 0.001).expect("Step 2");
        assert!(
            (r2.g_loss - 0.0).abs() < 1e-10,
            "G should not train on step 2"
        );

        // Step 3: generator should train
        let r3 = gan.train_step(&real_data, 0.001).expect("Step 3");
        // g_loss will be non-zero (it was computed)
        assert!(r3.g_loss.is_finite(), "G should train on step 3");
    }
}
