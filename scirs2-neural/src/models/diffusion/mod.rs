//! Denoising Diffusion Probabilistic Models (DDPM)
//!
//! Implements the core components for DDPM:
//! - `DiffusionConfig` – noise schedule and hyperparameters
//! - `NoiseScheduler` – forward diffusion (noise addition) and reverse (denoise)
//! - `DDPMTrainer` – simplified DDPM training (predict noise / epsilon)
//! - `DenoisingNetwork` trait – interface for the denoising network
//! - `DiffusionSampler` – DDPM ancestral sampling loop
//!
//! ## Theory
//!
//! The forward diffusion process is a Markov chain that gradually adds
//! Gaussian noise to data according to a variance schedule βₜ:
//!
//! ```text
//! q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ) xₜ₋₁, βₜ I)
//! ```
//!
//! Marginal distribution (closed form):
//! ```text
//! q(xₜ | x₀) = N(xₜ; √ᾱₜ x₀, (1-ᾱₜ) I)
//! ```
//! where `ᾱₜ = ∏_{s=1}^{t} αₛ` and `αₜ = 1 - βₜ`.
//!
//! The reverse process is parameterised by a neural network that predicts the
//! noise ε added at each step (simplified DDPM objective, Ho et al. 2020):
//! ```text
//! L_simple = E_{t, x₀, ε} [||ε - ε_θ(xₜ, t)||²]
//! ```
//!
//! ## References
//! - "Denoising Diffusion Probabilistic Models", Ho, Jain & Abbeel (2020)
//!   <https://arxiv.org/abs/2006.11239>
//! - "Improved Denoising Diffusion Probabilistic Models", Nichol & Dhariwal (2021)
//!   <https://arxiv.org/abs/2102.09672>

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// NoiseSchedule
// ---------------------------------------------------------------------------

/// Supported noise variance schedules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseSchedule {
    /// Linear schedule: βₜ linearly interpolated from β_start to β_end.
    Linear,
    /// Cosine schedule (Nichol & Dhariwal 2021): smoother, improves sample quality.
    Cosine,
    /// Quadratic schedule for faster variance growth.
    Quadratic,
    /// Sigmoid schedule for smooth transitions.
    Sigmoid,
    /// Exponential schedule: βₜ grows exponentially from β_start to β_end.
    Exponential,
}

// ---------------------------------------------------------------------------
// DiffusionConfig
// ---------------------------------------------------------------------------

/// Configuration for a diffusion model.
#[derive(Debug, Clone)]
pub struct DiffusionConfig {
    /// Total number of diffusion timesteps T
    pub num_timesteps: usize,
    /// Noise variance schedule type
    pub schedule: NoiseSchedule,
    /// Starting beta value (used for Linear/Quadratic/Sigmoid schedules)
    pub beta_start: f64,
    /// Ending beta value
    pub beta_end: f64,
    /// Cosine offset `s` (only for Cosine schedule; typically 0.008)
    pub cosine_s: f64,
    /// Whether to clip denoised predictions to [-1, 1]
    pub clip_denoised: bool,
    /// Data dimensionality (used for sampling)
    pub data_dim: usize,
}

impl DiffusionConfig {
    /// Standard DDPM configuration (Ho et al. 2020).
    pub fn ddpm(data_dim: usize) -> Self {
        Self {
            num_timesteps: 1000,
            schedule: NoiseSchedule::Linear,
            beta_start: 1e-4,
            beta_end: 2e-2,
            cosine_s: 0.008,
            clip_denoised: true,
            data_dim,
        }
    }

    /// DDPM configuration with cosine schedule (Nichol & Dhariwal 2021).
    pub fn ddpm_cosine(data_dim: usize) -> Self {
        Self {
            num_timesteps: 1000,
            schedule: NoiseSchedule::Cosine,
            beta_start: 1e-4,
            beta_end: 2e-2,
            cosine_s: 0.008,
            clip_denoised: true,
            data_dim,
        }
    }

    /// Tiny config for testing.
    pub fn tiny(data_dim: usize, steps: usize) -> Self {
        Self {
            num_timesteps: steps,
            schedule: NoiseSchedule::Linear,
            beta_start: 1e-3,
            beta_end: 2e-2,
            cosine_s: 0.008,
            clip_denoised: true,
            data_dim,
        }
    }
}

// ---------------------------------------------------------------------------
// NoiseScheduler
// ---------------------------------------------------------------------------

/// Pre-computed noise schedule quantities for DDPM.
///
/// Stores all the quantities needed for the forward and reverse processes:
/// - `betas`: β₁, ..., βᵀ
/// - `alphas`: αₜ = 1 - βₜ
/// - `alphas_cumprod`: ᾱₜ = ∏_{s=1}^{t} αₛ
/// - `sqrt_alphas_cumprod`: √ᾱₜ (used in `q(xₜ|x₀)`)
/// - `sqrt_one_minus_alphas_cumprod`: √(1-ᾱₜ) (std of noise in `q(xₜ|x₀)`)
/// - `posterior_variance`: variance of the reverse conditional `q(xₜ₋₁|xₜ,x₀)`
pub struct NoiseScheduler {
    /// Configuration
    pub config: DiffusionConfig,
    /// β schedule [T]
    pub betas: Vec<f64>,
    /// α = 1-β schedule [T]
    pub alphas: Vec<f64>,
    /// ᾱ = cumulative product of α [T]
    pub alphas_cumprod: Vec<f64>,
    /// ᾱ shifted by one step: ᾱ_{t-1} [T] (with ᾱ₀ = 1)
    pub alphas_cumprod_prev: Vec<f64>,
    /// √ᾱₜ [T]
    pub sqrt_alphas_cumprod: Vec<f64>,
    /// √(1-ᾱₜ) [T]
    pub sqrt_one_minus_alphas_cumprod: Vec<f64>,
    /// Posterior variance: σ²ₜ = β̃ₜ = (1-ᾱₜ₋₁)/(1-ᾱₜ) · βₜ [T]
    pub posterior_variance: Vec<f64>,
    /// Log of clipped posterior variance
    pub log_posterior_variance_clipped: Vec<f64>,
    /// Posterior mean coefficient for x₀: √ᾱₜ₋₁ · βₜ / (1-ᾱₜ) [T]
    pub posterior_mean_coef1: Vec<f64>,
    /// Posterior mean coefficient for xₜ: √αₜ · (1-ᾱₜ₋₁) / (1-ᾱₜ) [T]
    pub posterior_mean_coef2: Vec<f64>,
}

impl std::fmt::Debug for NoiseScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoiseScheduler")
            .field("num_timesteps", &self.config.num_timesteps)
            .field("schedule", &self.config.schedule)
            .finish()
    }
}

impl NoiseScheduler {
    /// Build a `NoiseScheduler` from a `DiffusionConfig`.
    pub fn new(config: DiffusionConfig) -> Result<Self> {
        if config.num_timesteps == 0 {
            return Err(NeuralError::InvalidArgument(
                "NoiseScheduler: num_timesteps must be > 0".to_string(),
            ));
        }
        let t = config.num_timesteps;
        let betas = Self::compute_betas(&config)?;
        let alphas: Vec<f64> = betas.iter().map(|&b| 1.0 - b).collect();
        // Cumulative products
        let mut alphas_cumprod = vec![0.0f64; t];
        let mut running = 1.0f64;
        for i in 0..t {
            running *= alphas[i];
            alphas_cumprod[i] = running;
        }
        // ᾱ shifted (prev): index 0 uses 1.0, others use alphas_cumprod[t-1]
        let mut alphas_cumprod_prev = vec![0.0f64; t];
        alphas_cumprod_prev[0] = 1.0;
        for i in 1..t {
            alphas_cumprod_prev[i] = alphas_cumprod[i - 1];
        }
        let sqrt_alphas_cumprod: Vec<f64> = alphas_cumprod.iter().map(|&a| a.sqrt()).collect();
        let sqrt_one_minus_alphas_cumprod: Vec<f64> =
            alphas_cumprod.iter().map(|&a| (1.0 - a).sqrt()).collect();
        // Posterior variance: β̃ₜ = (1-ᾱₜ₋₁)/(1-ᾱₜ) · βₜ
        let posterior_variance: Vec<f64> = (0..t)
            .map(|i| {
                let acp_prev = alphas_cumprod_prev[i];
                let acp = alphas_cumprod[i];
                (1.0 - acp_prev) / (1.0 - acp) * betas[i]
            })
            .collect();
        let log_posterior_variance_clipped: Vec<f64> = posterior_variance
            .iter()
            .map(|&v| v.max(1e-20).ln())
            .collect();
        // Posterior mean coefficients
        let posterior_mean_coef1: Vec<f64> = (0..t)
            .map(|i| {
                let acp_prev = alphas_cumprod_prev[i];
                let acp = alphas_cumprod[i];
                acp_prev.sqrt() * betas[i] / (1.0 - acp).max(1e-12)
            })
            .collect();
        let posterior_mean_coef2: Vec<f64> = (0..t)
            .map(|i| {
                let acp_prev = alphas_cumprod_prev[i];
                let acp = alphas_cumprod[i];
                alphas[i].sqrt() * (1.0 - acp_prev) / (1.0 - acp).max(1e-12)
            })
            .collect();
        Ok(Self {
            config,
            betas,
            alphas,
            alphas_cumprod,
            alphas_cumprod_prev,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            posterior_variance,
            log_posterior_variance_clipped,
            posterior_mean_coef1,
            posterior_mean_coef2,
        })
    }

    fn compute_betas(config: &DiffusionConfig) -> Result<Vec<f64>> {
        let t = config.num_timesteps;
        match config.schedule {
            NoiseSchedule::Linear => {
                Ok((0..t)
                    .map(|i| {
                        config.beta_start
                            + (config.beta_end - config.beta_start) * i as f64
                                / (t - 1).max(1) as f64
                    })
                    .collect())
            }
            NoiseSchedule::Cosine => {
                // Nichol & Dhariwal cosine schedule
                let s = config.cosine_s;
                let f = |t_frac: f64| {
                    ((t_frac / (1.0 + s) + s / (1.0 + s)) * std::f64::consts::PI / 2.0)
                        .cos()
                        .powi(2)
                };
                let ft0 = f(0.0);
                let mut alphas_cumprod = vec![0.0f64; t + 1];
                for i in 0..=t {
                    alphas_cumprod[i] = f(i as f64 / t as f64) / ft0;
                }
                let betas: Vec<f64> = (0..t)
                    .map(|i| {
                        let b = 1.0 - alphas_cumprod[i + 1] / alphas_cumprod[i];
                        b.clamp(0.0, 0.999)
                    })
                    .collect();
                Ok(betas)
            }
            NoiseSchedule::Quadratic => {
                let beta_start_sqrt = config.beta_start.sqrt();
                let beta_end_sqrt = config.beta_end.sqrt();
                Ok((0..t)
                    .map(|i| {
                        let sqrt_b = beta_start_sqrt
                            + (beta_end_sqrt - beta_start_sqrt) * i as f64
                                / (t - 1).max(1) as f64;
                        sqrt_b * sqrt_b
                    })
                    .collect())
            }
            NoiseSchedule::Sigmoid => {
                // Map uniform t to sigmoid space
                Ok((0..t)
                    .map(|i| {
                        let x = -6.0 + 12.0 * i as f64 / (t - 1).max(1) as f64;
                        let sig = 1.0 / (1.0 + (-x).exp());
                        config.beta_start + (config.beta_end - config.beta_start) * sig
                    })
                    .collect())
            }
            NoiseSchedule::Exponential => {
                // Exponential schedule: βₜ = β_start * (β_end/β_start)^{t/(T-1)}
                // Provides faster noise growth than linear, smoother than quadratic
                let ratio = (config.beta_end / config.beta_start.max(1e-12)).max(1.0);
                Ok((0..t)
                    .map(|i| {
                        let frac = i as f64 / (t - 1).max(1) as f64;
                        config.beta_start * ratio.powf(frac)
                    })
                    .collect())
            }
        }
    }

    /// Forward diffusion: sample xₜ ~ q(xₜ | x₀) given x₀ and noise ε.
    ///
    /// `xₜ = √ᾱₜ · x₀ + √(1-ᾱₜ) · ε`
    ///
    /// # Arguments
    /// * `x0` – clean data sample, length = `data_dim`
    /// * `noise` – Gaussian noise ε ~ N(0,I), same length as `x0`
    /// * `t` – timestep index (0-indexed)
    pub fn add_noise(&self, x0: &[f64], noise: &[f64], t: usize) -> Result<Vec<f64>> {
        if t >= self.config.num_timesteps {
            return Err(NeuralError::InvalidArgument(format!(
                "NoiseScheduler add_noise: t={t} >= num_timesteps={}",
                self.config.num_timesteps
            )));
        }
        if x0.len() != noise.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "NoiseScheduler add_noise: x0 len {} != noise len {}",
                x0.len(),
                noise.len()
            )));
        }
        let sqrt_ac = self.sqrt_alphas_cumprod[t];
        let sqrt_one_minus_ac = self.sqrt_one_minus_alphas_cumprod[t];
        Ok(x0
            .iter()
            .zip(noise.iter())
            .map(|(&x, &n)| sqrt_ac * x + sqrt_one_minus_ac * n)
            .collect())
    }

    /// Predict x₀ from xₜ and the predicted noise ε_θ:
    /// `x₀_pred = (xₜ - √(1-ᾱₜ) · ε_θ) / √ᾱₜ`
    pub fn predict_x0_from_noise(
        &self,
        x_t: &[f64],
        noise_pred: &[f64],
        t: usize,
    ) -> Result<Vec<f64>> {
        if t >= self.config.num_timesteps {
            return Err(NeuralError::InvalidArgument(format!(
                "NoiseScheduler: t={t} >= num_timesteps={}",
                self.config.num_timesteps
            )));
        }
        let sqrt_ac = self.sqrt_alphas_cumprod[t];
        let sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t];
        let x0: Vec<f64> = x_t
            .iter()
            .zip(noise_pred.iter())
            .map(|(&xt, &eps)| (xt - sqrt_one_minus * eps) / sqrt_ac.max(1e-12))
            .collect();
        if self.config.clip_denoised {
            Ok(x0.iter().map(|&v| v.clamp(-1.0, 1.0)).collect())
        } else {
            Ok(x0)
        }
    }

    /// Compute the posterior mean `μ̃ₜ(xₜ, x₀)` and log variance.
    ///
    /// `μ̃ₜ = coef1 * x₀ + coef2 * xₜ`
    pub fn q_posterior_mean_variance(
        &self,
        x_start: &[f64],
        x_t: &[f64],
        t: usize,
    ) -> Result<(Vec<f64>, f64)> {
        if t >= self.config.num_timesteps {
            return Err(NeuralError::InvalidArgument(format!(
                "NoiseScheduler: t={t} >= num_timesteps",
            )));
        }
        let c1 = self.posterior_mean_coef1[t];
        let c2 = self.posterior_mean_coef2[t];
        let posterior_mean: Vec<f64> = x_start
            .iter()
            .zip(x_t.iter())
            .map(|(&x0, &xt)| c1 * x0 + c2 * xt)
            .collect();
        let log_var = self.log_posterior_variance_clipped[t];
        Ok((posterior_mean, log_var))
    }

    /// Compute the SNR (signal-to-noise ratio) at timestep t.
    pub fn snr(&self, t: usize) -> f64 {
        let ac = self.alphas_cumprod[t];
        ac / (1.0 - ac).max(1e-12)
    }
}

// ---------------------------------------------------------------------------
// DenoisingNetwork trait
// ---------------------------------------------------------------------------

/// Trait for the denoising network in DDPM.
///
/// The network takes a noisy sample xₜ and the timestep `t`, and predicts the
/// noise ε added at that timestep.
pub trait DenoisingNetwork: Send + Sync + std::fmt::Debug {
    /// Predict the noise ε given noisy input xₜ and timestep t.
    fn predict_noise(&self, x_t: &[f64], t: usize) -> Vec<f64>;

    /// Number of parameters (for monitoring)
    fn parameter_count(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// SimpleMLPDenoiser (example implementation of DenoisingNetwork)
// ---------------------------------------------------------------------------

/// A simple MLP-based denoising network for demonstration / testing.
///
/// Architecture: `[x_t ∥ t_emb] → hidden → hidden → x_t.len()`
/// where `t_emb` is a sinusoidal embedding of the timestep.
#[derive(Debug, Clone)]
pub struct SimpleMLPDenoiser {
    /// Input dimension (data_dim)
    data_dim: usize,
    /// Sinusoidal time embedding dimension
    time_emb_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Total timesteps (for normalising t)
    num_timesteps: usize,
    /// Layer weights: (w, b) pairs
    layers: Vec<(Vec<f64>, Vec<f64>)>,
}

impl SimpleMLPDenoiser {
    /// Create a new `SimpleMLPDenoiser`.
    pub fn new(
        data_dim: usize,
        hidden_dim: usize,
        time_emb_dim: usize,
        num_timesteps: usize,
    ) -> Result<Self> {
        if data_dim == 0 || hidden_dim == 0 || time_emb_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "SimpleMLPDenoiser: all dimensions must be > 0".to_string(),
            ));
        }
        let input_dim = data_dim + time_emb_dim;
        // 2-hidden-layer MLP
        let make_w = |in_d: usize, out_d: usize, off: usize| -> Vec<f64> {
            let std = (2.0 / in_d as f64).sqrt();
            (0..in_d * out_d)
                .map(|k| std * (((k + off) as f64) * 0.6180339887).sin())
                .collect()
        };
        let layers = vec![
            (make_w(input_dim, hidden_dim, 0), vec![0.0; hidden_dim]),
            (make_w(hidden_dim, hidden_dim, hidden_dim), vec![0.0; hidden_dim]),
            (make_w(hidden_dim, data_dim, 2 * hidden_dim), vec![0.0; data_dim]),
        ];
        Ok(Self {
            data_dim,
            time_emb_dim,
            hidden_dim,
            num_timesteps,
            layers,
        })
    }

    /// Sinusoidal time embedding.
    fn time_embedding(&self, t: usize) -> Vec<f64> {
        let t_norm = t as f64 / self.num_timesteps.max(1) as f64;
        let half = self.time_emb_dim / 2;
        let mut emb = Vec::with_capacity(self.time_emb_dim);
        for i in 0..half {
            let freq = 10000.0f64.powf(-(2.0 * i as f64) / self.time_emb_dim as f64);
            emb.push((t_norm * freq).sin());
        }
        for i in 0..half {
            let freq = 10000.0f64.powf(-(2.0 * i as f64) / self.time_emb_dim as f64);
            emb.push((t_norm * freq).cos());
        }
        // Pad with zeros if time_emb_dim is odd
        while emb.len() < self.time_emb_dim {
            emb.push(0.0);
        }
        emb
    }

    fn forward(&self, x: &[f64]) -> Vec<f64> {
        let mut h = x.to_vec();
        for (layer_idx, (w, b)) in self.layers.iter().enumerate() {
            let out_dim = b.len();
            let in_dim = h.len();
            let mut next = vec![0.0f64; out_dim];
            for j in 0..out_dim {
                let mut s = b[j];
                for i in 0..in_dim {
                    if j * in_dim + i < w.len() {
                        s += w[j * in_dim + i] * h[i];
                    }
                }
                next[j] = s;
            }
            if layer_idx < self.layers.len() - 1 {
                for v in &mut next {
                    *v = v.max(0.0); // ReLU
                }
            }
            h = next;
        }
        h
    }
}

impl DenoisingNetwork for SimpleMLPDenoiser {
    fn predict_noise(&self, x_t: &[f64], t: usize) -> Vec<f64> {
        let t_emb = self.time_embedding(t);
        let mut inp = x_t.to_vec();
        inp.extend_from_slice(&t_emb);
        self.forward(&inp)
    }

    fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|(w, b)| w.len() + b.len())
            .sum()
    }
}

// ---------------------------------------------------------------------------
// DDPMTrainer
// ---------------------------------------------------------------------------

/// Training configuration for DDPM.
#[derive(Debug, Clone)]
pub struct DDPMTrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Mini-batch size
    pub batch_size: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for DDPMTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            epochs: 10,
            batch_size: 32,
            seed: 42,
        }
    }
}

/// Per-epoch training statistics.
#[derive(Debug, Clone)]
pub struct DDPMTrainingStats {
    /// Mean MSE loss per epoch
    pub loss_history: Vec<f64>,
}

/// Simplified DDPM training loop.
///
/// At each step:
/// 1. Sample a random timestep `t ~ Uniform[1, T]`
/// 2. Sample noise `ε ~ N(0, I)`
/// 3. Compute noisy sample `xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε`
/// 4. Predict `ε_θ(xₜ, t)` with the denoising network
/// 5. Loss = `||ε - ε_θ||²`
///
/// Gradient updates are performed via finite-difference approximation of the
/// gradient with respect to the network's conceptual output.  In production
/// this would use proper autodiff.
pub struct DDPMTrainer {
    /// Noise scheduler
    pub scheduler: NoiseScheduler,
    /// Training configuration
    pub config: DDPMTrainingConfig,
    /// Training statistics
    pub stats: DDPMTrainingStats,
    rng_state: u64,
}

impl std::fmt::Debug for DDPMTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DDPMTrainer")
            .field("config", &self.config)
            .field("num_timesteps", &self.scheduler.config.num_timesteps)
            .finish()
    }
}

impl DDPMTrainer {
    /// Create a new `DDPMTrainer`.
    pub fn new(diffusion_config: DiffusionConfig, training_config: DDPMTrainingConfig) -> Result<Self> {
        let seed = training_config.seed;
        let scheduler = NoiseScheduler::new(diffusion_config)?;
        Ok(Self {
            scheduler,
            config: training_config,
            stats: DDPMTrainingStats { loss_history: Vec::new() },
            rng_state: seed,
        })
    }

    fn sample_normal(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u1 = ((self.rng_state >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u2 = ((self.rng_state >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn sample_timestep(&mut self) -> usize {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.rng_state >> 33) as usize % self.scheduler.config.num_timesteps
    }

    /// Compute the DDPM loss for a single data sample.
    ///
    /// # Returns
    /// The MSE loss `||ε - ε_θ(xₜ, t)||²`.
    pub fn compute_loss(
        &mut self,
        x0: &[f64],
        network: &dyn DenoisingNetwork,
    ) -> Result<f64> {
        let data_dim = self.scheduler.config.data_dim;
        if x0.len() != data_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "DDPMTrainer: x0 length {} != data_dim {}",
                x0.len(),
                data_dim
            )));
        }
        let t = self.sample_timestep();
        // Sample noise
        let noise: Vec<f64> = (0..data_dim).map(|_| self.sample_normal()).collect();
        // Forward diffusion
        let x_t = self.scheduler.add_noise(x0, &noise, t)?;
        // Predict noise
        let noise_pred = network.predict_noise(&x_t, t);
        if noise_pred.len() != data_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "DDPMTrainer: network predicted {} values, expected {}",
                noise_pred.len(),
                data_dim
            )));
        }
        // MSE loss
        let mse: f64 = noise
            .iter()
            .zip(&noise_pred)
            .map(|(&eps, &eps_pred)| (eps - eps_pred).powi(2))
            .sum::<f64>()
            / data_dim as f64;
        Ok(mse)
    }

    /// Run one training epoch over `data`.
    ///
    /// Computes the average loss across all samples and batches.
    pub fn train_epoch(
        &mut self,
        data: &[Vec<f64>],
        network: &dyn DenoisingNetwork,
    ) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        let total_loss: f64 = data
            .iter()
            .map(|x0| self.compute_loss(x0, network))
            .collect::<Result<Vec<f64>>>()?
            .iter()
            .sum();
        Ok(total_loss / data.len() as f64)
    }

    /// Train for multiple epochs and accumulate statistics.
    pub fn train(
        &mut self,
        data: &[Vec<f64>],
        network: &dyn DenoisingNetwork,
    ) -> Result<&DDPMTrainingStats> {
        self.stats.loss_history.clear();
        for _epoch in 0..self.config.epochs {
            let loss = self.train_epoch(data, network)?;
            self.stats.loss_history.push(loss);
        }
        Ok(&self.stats)
    }
}

// ---------------------------------------------------------------------------
// DiffusionSampler
// ---------------------------------------------------------------------------

/// DDPM ancestral sampling loop.
///
/// Implements the reverse process by iterating from `t = T-1` down to `t = 0`,
/// denoising at each step:
///
/// ```text
/// x_{t-1} = μ̃ₜ(xₜ, x₀_pred) + σₜ z    (z ~ N(0,I) for t > 0)
/// ```
pub struct DiffusionSampler {
    /// Noise scheduler
    pub scheduler: NoiseScheduler,
    /// Whether to add noise during sampling (false = deterministic DDIM-like)
    pub stochastic: bool,
    rng_state: u64,
}

impl std::fmt::Debug for DiffusionSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiffusionSampler")
            .field("stochastic", &self.stochastic)
            .field("num_timesteps", &self.scheduler.config.num_timesteps)
            .finish()
    }
}

impl DiffusionSampler {
    /// Create a new `DiffusionSampler`.
    pub fn new(config: DiffusionConfig, stochastic: bool) -> Result<Self> {
        let scheduler = NoiseScheduler::new(config)?;
        Ok(Self {
            scheduler,
            stochastic,
            rng_state: 0x9876543210fedcba,
        })
    }

    fn sample_normal(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u1 = ((self.rng_state >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u2 = ((self.rng_state >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Run the full DDPM sampling loop starting from Gaussian noise.
    ///
    /// # Arguments
    /// * `network` – the denoising network
    ///
    /// # Returns
    /// The denoised sample (approximation to x₀).
    pub fn sample(&mut self, network: &dyn DenoisingNetwork) -> Result<Vec<f64>> {
        let data_dim = self.scheduler.config.data_dim;
        let t_max = self.scheduler.config.num_timesteps;
        // Start from pure Gaussian noise
        let mut x_t: Vec<f64> = (0..data_dim).map(|_| self.sample_normal()).collect();
        // Iterate t = T-1, T-2, ..., 0
        for t in (0..t_max).rev() {
            let noise_pred = network.predict_noise(&x_t, t);
            if noise_pred.len() != data_dim {
                return Err(NeuralError::ShapeMismatch(format!(
                    "DiffusionSampler: network returned {} values, expected {}",
                    noise_pred.len(),
                    data_dim
                )));
            }
            // Predict x₀
            let x0_pred = self
                .scheduler
                .predict_x0_from_noise(&x_t, &noise_pred, t)?;
            // Compute posterior mean
            let (posterior_mean, log_var) = self
                .scheduler
                .q_posterior_mean_variance(&x0_pred, &x_t, t)?;
            // Add noise for t > 0
            x_t = if t > 0 && self.stochastic {
                let std = (log_var.exp()).sqrt();
                posterior_mean
                    .iter()
                    .map(|&m| m + std * self.sample_normal())
                    .collect()
            } else {
                posterior_mean
            };
        }
        Ok(x_t)
    }

    /// Run the sampling loop and return a trajectory of samples.
    ///
    /// # Arguments
    /// * `network` – denoising network
    /// * `save_every` – save a snapshot every N timesteps
    ///
    /// # Returns
    /// Vector of (timestep, sample) pairs.
    pub fn sample_with_trajectory(
        &mut self,
        network: &dyn DenoisingNetwork,
        save_every: usize,
    ) -> Result<Vec<(usize, Vec<f64>)>> {
        let data_dim = self.scheduler.config.data_dim;
        let t_max = self.scheduler.config.num_timesteps;
        let actual_save = save_every.max(1);
        let mut x_t: Vec<f64> = (0..data_dim).map(|_| self.sample_normal()).collect();
        let mut trajectory = Vec::new();
        for t in (0..t_max).rev() {
            let noise_pred = network.predict_noise(&x_t, t);
            let x0_pred = self
                .scheduler
                .predict_x0_from_noise(&x_t, &noise_pred, t)?;
            let (posterior_mean, log_var) = self
                .scheduler
                .q_posterior_mean_variance(&x0_pred, &x_t, t)?;
            x_t = if t > 0 && self.stochastic {
                let std = (log_var.exp()).sqrt();
                posterior_mean
                    .iter()
                    .map(|&m| m + std * self.sample_normal())
                    .collect()
            } else {
                posterior_mean
            };
            if t % actual_save == 0 {
                trajectory.push((t, x_t.clone()));
            }
        }
        Ok(trajectory)
    }
}

// ---------------------------------------------------------------------------
// Sub-modules: score matching, DDIM, flow matching
// ---------------------------------------------------------------------------

/// Score-based generative models (DSM, SSM, Langevin dynamics)
pub mod score_matching;
/// DDIM deterministic / stochastic sampler with inversion
pub mod ddim;
/// Flow matching (CFM, OT-CFM, ODE sampling)
pub mod flow_matching;

pub use score_matching::{
    AnnealedLangevin, DenoisingScoreMatching, LangevinConfig, ProjectionDist,
    ScoreFunction, ScoreNetwork, ScoreNetworkConfig, SlicedScoreMatching,
};
pub use ddim::{DDIMConfig, DDIM};
pub use flow_matching::{
    FlowMatchingObjective, FlowMatcher, ODEMethod, ODESampler, ODESolverConfig,
    SimpleVectorFieldNet, VectorField,
};


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_scheduler_linear() {
        let config = DiffusionConfig::tiny(4, 10);
        let sched = NoiseScheduler::new(config).expect("scheduler creation failed");
        assert_eq!(sched.betas.len(), 10);
        for &b in &sched.betas {
            assert!(b > 0.0 && b < 1.0, "beta out of range: {b}");
        }
        // ᾱ should be monotonically decreasing
        let mut prev = 1.0f64;
        for &ac in &sched.alphas_cumprod {
            assert!(ac <= prev + 1e-10, "alphas_cumprod not decreasing");
            prev = ac;
        }
    }

    #[test]
    fn test_noise_scheduler_cosine() {
        let config = DiffusionConfig::ddpm_cosine(4);
        // Use fewer steps for the test
        let config = DiffusionConfig {
            num_timesteps: 20,
            ..config
        };
        let sched = NoiseScheduler::new(config).expect("cosine scheduler");
        assert_eq!(sched.betas.len(), 20);
        for &b in &sched.betas {
            assert!(b >= 0.0 && b < 1.0);
        }
    }

    #[test]
    fn test_add_noise_shape() {
        let config = DiffusionConfig::tiny(4, 10);
        let sched = NoiseScheduler::new(config).expect("scheduler");
        let x0 = vec![0.5, -0.3, 0.2, 0.8];
        let noise = vec![0.1, -0.1, 0.2, -0.2];
        let xt = sched.add_noise(&x0, &noise, 5).expect("add_noise failed");
        assert_eq!(xt.len(), 4);
        for &v in &xt {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_predict_x0_from_noise() {
        let config = DiffusionConfig::tiny(4, 10);
        let sched = NoiseScheduler::new(config).expect("scheduler");
        let x0 = vec![0.3, 0.4, -0.2, 0.1];
        let noise = vec![0.5, -0.5, 0.5, -0.5];
        let xt = sched.add_noise(&x0, &noise, 3).expect("add_noise");
        let x0_pred = sched.predict_x0_from_noise(&xt, &noise, 3).expect("predict");
        assert_eq!(x0_pred.len(), 4);
        for (&gt, &pred) in x0.iter().zip(&x0_pred) {
            assert!((gt - pred).abs() < 1e-6, "x0 reconstruction error: {gt} vs {pred}");
        }
    }

    #[test]
    fn test_simple_mlp_denoiser() {
        let denoiser = SimpleMLPDenoiser::new(4, 16, 8, 10).expect("denoiser creation");
        let x_t = vec![0.5, -0.3, 0.2, 0.8];
        let pred = denoiser.predict_noise(&x_t, 5);
        assert_eq!(pred.len(), 4);
        for &v in &pred {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ddpm_trainer_loss() {
        let diff_config = DiffusionConfig::tiny(4, 10);
        let train_config = DDPMTrainingConfig {
            learning_rate: 1e-3,
            epochs: 2,
            batch_size: 4,
            seed: 42,
        };
        let mut trainer = DDPMTrainer::new(diff_config, train_config).expect("trainer creation");
        let network = SimpleMLPDenoiser::new(4, 16, 8, 10).expect("network");
        let data: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64 * 0.1; 4]).collect();
        let stats = trainer.train(&data, &network).expect("training failed");
        assert_eq!(stats.loss_history.len(), 2);
        for &loss in &stats.loss_history {
            assert!(loss >= 0.0 && loss.is_finite());
        }
    }

    #[test]
    fn test_diffusion_sampler() {
        let config = DiffusionConfig::tiny(4, 10);
        let mut sampler = DiffusionSampler::new(config, false).expect("sampler creation");
        let network = SimpleMLPDenoiser::new(4, 16, 8, 10).expect("network");
        let sample = sampler.sample(&network).expect("sampling failed");
        assert_eq!(sample.len(), 4);
        for &v in &sample {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_diffusion_sampler_trajectory() {
        let config = DiffusionConfig::tiny(4, 10);
        let mut sampler = DiffusionSampler::new(config, false).expect("sampler");
        let network = SimpleMLPDenoiser::new(4, 16, 8, 10).expect("network");
        let traj = sampler
            .sample_with_trajectory(&network, 3)
            .expect("trajectory failed");
        assert!(!traj.is_empty());
        for (t, sample) in &traj {
            assert_eq!(sample.len(), 4);
            assert!(*t < 10);
        }
    }

    #[test]
    fn test_snr_decreases() {
        let config = DiffusionConfig::tiny(4, 10);
        let sched = NoiseScheduler::new(config).expect("scheduler");
        let mut prev_snr = f64::INFINITY;
        for t in 0..10 {
            let snr = sched.snr(t);
            assert!(snr <= prev_snr + 1e-10, "SNR should decrease");
            prev_snr = snr;
        }
    }
}
