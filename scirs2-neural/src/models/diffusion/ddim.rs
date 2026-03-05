//! Denoising Diffusion Implicit Models (DDIM)
//!
//! Implements the deterministic/stochastic DDIM sampler from Song, Meng & Ermon (2020).
//!
//! ## Key Idea
//! DDPM's ancestral sampling requires T=1000 steps. DDIM re-parameterises the
//! same training objective but uses a *non-Markovian* inference process that
//! allows sampling with far fewer steps (e.g. 50–200) with minimal quality
//! degradation.
//!
//! ## Generalised Reverse Step
//! Given predicted noise `ε_θ(xₜ, t)`, the DDIM update is:
//! ```text
//! x_{t-1} = √ᾱ_{t-1} · x₀_pred
//!          + √(1-ᾱ_{t-1} - σₜ²) · ε_θ(xₜ, t)
//!          + σₜ · z
//! ```
//! where `x₀_pred = (xₜ - √(1-ᾱₜ) · ε_θ) / √ᾱₜ`
//! and `σₜ = η · √((1-ᾱ_{t-1})/(1-ᾱₜ)) · √(1 - ᾱₜ/ᾱ_{t-1})`.
//!
//! Setting `η = 0` gives **deterministic** DDIM; `η = 1` recovers DDPM.
//!
//! ## References
//! - "Denoising Diffusion Implicit Models", Song, Meng & Ermon (2020)
//!   <https://arxiv.org/abs/2010.02502>

use crate::error::{NeuralError, Result};
use crate::models::diffusion::{DenoisingNetwork, DiffusionConfig, NoiseScheduler};

// ---------------------------------------------------------------------------
// DDIMConfig
// ---------------------------------------------------------------------------

/// Configuration for DDIM sampling.
#[derive(Debug, Clone)]
pub struct DDIMConfig {
    /// Number of inference steps (can be much less than T).
    pub num_inference_steps: usize,
    /// Stochasticity coefficient η ∈ [0, 1].
    /// - 0: fully deterministic (original DDIM)
    /// - 1: recovers DDPM ancestral sampling
    pub eta: f64,
    /// Random seed for the stochastic noise terms.
    pub seed: u64,
    /// Clip denoised predictions to [-1, 1].
    pub clip_denoised: bool,
}

impl DDIMConfig {
    /// Fully deterministic DDIM with 50 inference steps.
    pub fn deterministic(num_steps: usize) -> Self {
        Self {
            num_inference_steps: num_steps,
            eta: 0.0,
            seed: 0,
            clip_denoised: true,
        }
    }

    /// DDPM-equivalent (η=1) with a given step count.
    pub fn stochastic(num_steps: usize, seed: u64) -> Self {
        Self {
            num_inference_steps: num_steps,
            eta: 1.0,
            seed,
            clip_denoised: true,
        }
    }

    /// Partially stochastic (0 < η < 1).
    pub fn mixed(num_steps: usize, eta: f64, seed: u64) -> Result<Self> {
        if !(0.0..=1.0).contains(&eta) {
            return Err(NeuralError::InvalidArgument(format!(
                "DDIMConfig: eta must be in [0,1], got {eta}"
            )));
        }
        Ok(Self {
            num_inference_steps: num_steps,
            eta,
            seed,
            clip_denoised: true,
        })
    }
}

impl Default for DDIMConfig {
    fn default() -> Self {
        Self::deterministic(50)
    }
}

// ---------------------------------------------------------------------------
// DDIM sampler
// ---------------------------------------------------------------------------

/// DDIM sampler that uses a pre-trained DDPM noise predictor.
///
/// The sampler maintains its own sub-sequence of timesteps `τ₁ > τ₂ > ... > τₛ`
/// (a uniformly-spaced subsequence of the full T timesteps), enabling accelerated
/// inference without re-training.
pub struct DDIM {
    /// Noise scheduler (from the DDPM training config)
    pub scheduler: NoiseScheduler,
    /// DDIM configuration
    pub config: DDIMConfig,
    /// Selected sub-sequence of timesteps [num_inference_steps]
    pub timesteps: Vec<usize>,
    rng_state: u64,
}

impl std::fmt::Debug for DDIM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DDIM")
            .field("num_inference_steps", &self.config.num_inference_steps)
            .field("eta", &self.config.eta)
            .field("T", &self.scheduler.config.num_timesteps)
            .finish()
    }
}

impl DDIM {
    /// Build a `DDIM` sampler from a `DiffusionConfig` and a `DDIMConfig`.
    ///
    /// # Arguments
    /// * `diffusion_config` – matches the config used during DDPM training
    /// * `ddim_config` – DDIM-specific sampling parameters
    pub fn new(diffusion_config: DiffusionConfig, ddim_config: DDIMConfig) -> Result<Self> {
        let t_train = diffusion_config.num_timesteps;
        let s = ddim_config.num_inference_steps;
        if s == 0 {
            return Err(NeuralError::InvalidArgument(
                "DDIM: num_inference_steps must be > 0".to_string(),
            ));
        }
        if s > t_train {
            return Err(NeuralError::InvalidArgument(format!(
                "DDIM: num_inference_steps ({s}) > training timesteps ({t_train})"
            )));
        }
        let scheduler = NoiseScheduler::new(diffusion_config)?;
        // Build uniform sub-sequence τ: T-1, T-1-step, ..., 0 (indices into αBar)
        // stride = T / S, rounding
        let stride = (t_train as f64 / s as f64).round() as usize;
        let stride = stride.max(1);
        let mut timesteps: Vec<usize> = (0..t_train).rev().step_by(stride).collect();
        timesteps.reverse(); // ascending
        // We want descending order for sampling (T-1 down to 0)
        timesteps.sort_unstable_by(|a, b| b.cmp(a));
        // Trim to exactly `s` steps if needed
        timesteps.truncate(s);
        let seed = ddim_config.seed;
        Ok(Self {
            scheduler,
            config: ddim_config,
            timesteps,
            rng_state: seed.wrapping_add(0x1234567890abcdef),
        })
    }

    fn sample_normal(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u1 = ((self.rng_state >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u2 = ((self.rng_state >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Compute the σₜ term for a DDIM step.
    ///
    /// `σₜ = η · √((1-ᾱ_{t-1})/(1-ᾱₜ)) · √(1 - ᾱₜ/ᾱ_{t-1})`
    fn sigma_t(&self, t: usize, prev_t: usize) -> f64 {
        let ac_t = self.scheduler.alphas_cumprod[t];
        let ac_prev = if prev_t < self.scheduler.alphas_cumprod.len() {
            self.scheduler.alphas_cumprod[prev_t]
        } else {
            // prev_t beyond schedule means ᾱ → 1
            1.0
        };
        // (1-ᾱ_{t-1}) / (1-ᾱₜ)
        let ratio = (1.0 - ac_prev) / (1.0 - ac_t).max(1e-12);
        // 1 - ᾱₜ / ᾱ_{t-1}
        let snr_ratio = 1.0 - ac_t / ac_prev.max(1e-12);
        self.config.eta * (ratio * snr_ratio.max(0.0)).sqrt()
    }

    /// Perform one DDIM denoising step.
    ///
    /// # Arguments
    /// * `x_t` – current noisy sample at timestep `t`
    /// * `t` – current timestep index
    /// * `prev_t` – previous (less noisy) timestep index; use `usize::MAX` for t=0
    /// * `noise_pred` – predicted noise `ε_θ(xₜ, t)` from the denoising network
    ///
    /// # Returns
    /// The denoised sample `x_{t-1}`.
    pub fn step(
        &mut self,
        x_t: &[f64],
        t: usize,
        prev_t: Option<usize>,
        noise_pred: &[f64],
    ) -> Result<Vec<f64>> {
        let d = x_t.len();
        if noise_pred.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "DDIM step: x_t len {} != noise_pred len {}",
                d,
                noise_pred.len()
            )));
        }
        if t >= self.scheduler.config.num_timesteps {
            return Err(NeuralError::InvalidArgument(format!(
                "DDIM step: t={t} >= T={}",
                self.scheduler.config.num_timesteps
            )));
        }
        let ac_t = self.scheduler.alphas_cumprod[t];
        let ac_prev = match prev_t {
            Some(pt) if pt < self.scheduler.config.num_timesteps => {
                self.scheduler.alphas_cumprod[pt]
            }
            _ => 1.0f64, // fully denoised
        };

        let sqrt_ac_t = ac_t.sqrt();
        let sqrt_one_minus_ac_t = (1.0 - ac_t).max(0.0).sqrt();

        // Predict x₀: x₀_pred = (xₜ - √(1-ᾱₜ) · ε_θ) / √ᾱₜ
        let x0_pred: Vec<f64> = x_t
            .iter()
            .zip(noise_pred)
            .map(|(&xt, &eps)| (xt - sqrt_one_minus_ac_t * eps) / sqrt_ac_t.max(1e-12))
            .collect();
        // Optional clip
        let x0_pred: Vec<f64> = if self.config.clip_denoised {
            x0_pred.iter().map(|&v| v.clamp(-1.0, 1.0)).collect()
        } else {
            x0_pred
        };

        // DDIM noise coefficient: √(1-ᾱ_{t-1} - σₜ²)
        let sigma = match prev_t {
            Some(pt) => self.sigma_t(t, pt),
            None => 0.0,
        };
        let dir_coeff = ((1.0 - ac_prev - sigma * sigma).max(0.0)).sqrt();

        // x_{t-1} = √ᾱ_{t-1} · x₀_pred + dir_coeff · ε_θ + σₜ · z
        let sqrt_ac_prev = ac_prev.sqrt();
        let x_prev: Vec<f64> = if sigma > 0.0 {
            x0_pred
                .iter()
                .zip(noise_pred)
                .map(|(&x0, &eps)| {
                    let z = self.sample_normal();
                    sqrt_ac_prev * x0 + dir_coeff * eps + sigma * z
                })
                .collect()
        } else {
            x0_pred
                .iter()
                .zip(noise_pred)
                .map(|(&x0, &eps)| sqrt_ac_prev * x0 + dir_coeff * eps)
                .collect()
        };
        Ok(x_prev)
    }

    /// Run the full DDIM sampling loop.
    ///
    /// # Arguments
    /// * `network` – the denoising network (must match the training config)
    ///
    /// # Returns
    /// The generated sample (approximation to x₀).
    pub fn sample(&mut self, network: &dyn DenoisingNetwork) -> Result<Vec<f64>> {
        let data_dim = self.scheduler.config.data_dim;
        // Start from Gaussian noise x_T ~ N(0, I)
        let mut x: Vec<f64> = (0..data_dim).map(|_| self.sample_normal()).collect();

        let timesteps = self.timesteps.clone();
        for (step_idx, &t) in timesteps.iter().enumerate() {
            let noise_pred = network.predict_noise(&x, t);
            if noise_pred.len() != data_dim {
                return Err(NeuralError::ShapeMismatch(format!(
                    "DDIM sample: network returned {} values, expected {}",
                    noise_pred.len(),
                    data_dim
                )));
            }
            let prev_t = if step_idx + 1 < timesteps.len() {
                Some(timesteps[step_idx + 1])
            } else {
                None // final step → prev is "t=-1", treat as fully denoised
            };
            x = self.step(&x, t, prev_t, &noise_pred)?;
        }
        Ok(x)
    }

    /// Run sampling and collect intermediate results.
    ///
    /// # Returns
    /// Vector of `(timestep, sample)` pairs, one per inference step.
    pub fn sample_with_trajectory(
        &mut self,
        network: &dyn DenoisingNetwork,
    ) -> Result<Vec<(usize, Vec<f64>)>> {
        let data_dim = self.scheduler.config.data_dim;
        let mut x: Vec<f64> = (0..data_dim).map(|_| self.sample_normal()).collect();
        let mut trajectory = Vec::with_capacity(self.timesteps.len());

        let timesteps = self.timesteps.clone();
        for (step_idx, &t) in timesteps.iter().enumerate() {
            let noise_pred = network.predict_noise(&x, t);
            if noise_pred.len() != data_dim {
                return Err(NeuralError::ShapeMismatch(format!(
                    "DDIM trajectory: network returned {} values, expected {}",
                    noise_pred.len(),
                    data_dim
                )));
            }
            let prev_t = if step_idx + 1 < timesteps.len() {
                Some(timesteps[step_idx + 1])
            } else {
                None
            };
            x = self.step(&x, t, prev_t, &noise_pred)?;
            trajectory.push((t, x.clone()));
        }
        Ok(trajectory)
    }

    /// Encode a sample into the latent space at a given noise level (DDIM inversion).
    ///
    /// DDIM inversion runs the forward direction of the deterministic ODE,
    /// mapping x₀ → x_T in `num_inference_steps` steps. This enables:
    /// - Real image editing (encode → modify → decode)
    /// - Latent space interpolation
    ///
    /// # Arguments
    /// * `x0` – clean sample to encode
    /// * `network` – trained denoising network
    ///
    /// # Returns
    /// The encoded latent `x_T`.
    pub fn invert(
        &mut self,
        x0: &[f64],
        network: &dyn DenoisingNetwork,
    ) -> Result<Vec<f64>> {
        let data_dim = self.scheduler.config.data_dim;
        if x0.len() != data_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "DDIM invert: x0 len {} != data_dim {}",
                x0.len(),
                data_dim
            )));
        }
        let mut x = x0.to_vec();
        // Inversion: run timesteps in reverse (ascending noise)
        let timesteps_reversed: Vec<usize> = self.timesteps.iter().copied().rev().collect();
        for (step_idx, &t) in timesteps_reversed.iter().enumerate() {
            let noise_pred = network.predict_noise(&x, t);
            if noise_pred.len() != data_dim {
                return Err(NeuralError::ShapeMismatch(format!(
                    "DDIM invert: network returned {} values, expected {}",
                    noise_pred.len(),
                    data_dim
                )));
            }
            // next_t is the noisier timestep
            let next_t = if step_idx + 1 < timesteps_reversed.len() {
                timesteps_reversed[step_idx + 1]
            } else {
                // last step: step up to t_max - 1
                self.scheduler.config.num_timesteps - 1
            };

            let ac_t = self.scheduler.alphas_cumprod[t];
            let ac_next = self.scheduler.alphas_cumprod[next_t];
            let sqrt_ac_t = ac_t.sqrt();
            let sqrt_one_minus_ac_t = (1.0 - ac_t).max(0.0).sqrt();

            // Predict x₀ at current t
            let x0_pred: Vec<f64> = x
                .iter()
                .zip(&noise_pred)
                .map(|(&xt, &eps)| (xt - sqrt_one_minus_ac_t * eps) / sqrt_ac_t.max(1e-12))
                .collect();
            let x0_pred: Vec<f64> = if self.config.clip_denoised {
                x0_pred.iter().map(|&v| v.clamp(-1.0, 1.0)).collect()
            } else {
                x0_pred
            };
            // Forward DDIM step (deterministic, η=0):
            // x_{t+1} = √ᾱ_{t+1} · x₀_pred + √(1-ᾱ_{t+1}) · ε_θ
            let sqrt_ac_next = ac_next.sqrt();
            let sqrt_one_minus_ac_next = (1.0 - ac_next).max(0.0).sqrt();
            x = x0_pred
                .iter()
                .zip(&noise_pred)
                .map(|(&x0i, &eps)| sqrt_ac_next * x0i + sqrt_one_minus_ac_next * eps)
                .collect();
        }
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::diffusion::{DiffusionConfig, SimpleMLPDenoiser};

    #[test]
    fn test_ddim_creation() {
        let diff_cfg = DiffusionConfig::tiny(4, 20);
        let ddim_cfg = DDIMConfig::deterministic(10);
        let ddim = DDIM::new(diff_cfg, ddim_cfg).expect("DDIM creation");
        assert_eq!(ddim.timesteps.len(), 10);
        // Timesteps should be descending
        for window in ddim.timesteps.windows(2) {
            assert!(window[0] > window[1], "timesteps not descending");
        }
    }

    #[test]
    fn test_ddim_step_shape() {
        let diff_cfg = DiffusionConfig::tiny(4, 20);
        let ddim_cfg = DDIMConfig::deterministic(5);
        let mut ddim = DDIM::new(diff_cfg, ddim_cfg).expect("DDIM");
        let x_t = vec![0.5, -0.3, 0.2, 0.8];
        let noise_pred = vec![0.1, 0.2, -0.1, 0.0];
        let t = ddim.timesteps[0];
        let prev_t = ddim.timesteps.get(1).copied();
        let x_prev = ddim.step(&x_t, t, prev_t, &noise_pred).expect("DDIM step");
        assert_eq!(x_prev.len(), 4);
        for &v in &x_prev {
            assert!(v.is_finite(), "DDIM step output not finite: {v}");
        }
    }

    #[test]
    fn test_ddim_deterministic_sample() {
        let diff_cfg = DiffusionConfig::tiny(4, 20);
        let ddim_cfg = DDIMConfig::deterministic(5);
        let mut ddim = DDIM::new(diff_cfg, ddim_cfg).expect("DDIM");
        let network = SimpleMLPDenoiser::new(4, 16, 8, 20).expect("network");
        let sample = ddim.sample(&network).expect("DDIM sample");
        assert_eq!(sample.len(), 4);
        for &v in &sample {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ddim_stochastic_sample() {
        let diff_cfg = DiffusionConfig::tiny(4, 20);
        let ddim_cfg = DDIMConfig::stochastic(5, 42);
        let mut ddim = DDIM::new(diff_cfg, ddim_cfg).expect("DDIM");
        let network = SimpleMLPDenoiser::new(4, 16, 8, 20).expect("network");
        let sample = ddim.sample(&network).expect("DDIM stochastic sample");
        assert_eq!(sample.len(), 4);
    }

    #[test]
    fn test_ddim_trajectory() {
        let diff_cfg = DiffusionConfig::tiny(4, 20);
        let ddim_cfg = DDIMConfig::deterministic(5);
        let mut ddim = DDIM::new(diff_cfg, ddim_cfg).expect("DDIM");
        let network = SimpleMLPDenoiser::new(4, 16, 8, 20).expect("network");
        let traj = ddim.sample_with_trajectory(&network).expect("trajectory");
        assert_eq!(traj.len(), 5);
        for (t, sample) in &traj {
            assert_eq!(sample.len(), 4);
            assert!(*t < 20, "timestep out of range: {t}");
        }
    }

    #[test]
    fn test_ddim_inversion() {
        let diff_cfg = DiffusionConfig::tiny(4, 20);
        let ddim_cfg = DDIMConfig::deterministic(5);
        let mut ddim = DDIM::new(diff_cfg, ddim_cfg).expect("DDIM");
        let network = SimpleMLPDenoiser::new(4, 16, 8, 20).expect("network");
        let x0 = vec![0.3, -0.1, 0.4, -0.2];
        let latent = ddim.invert(&x0, &network).expect("DDIM inversion");
        assert_eq!(latent.len(), 4);
        for &v in &latent {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ddim_invalid_steps() {
        let diff_cfg = DiffusionConfig::tiny(4, 10);
        let ddim_cfg = DDIMConfig {
            num_inference_steps: 20, // > T=10
            eta: 0.0,
            seed: 0,
            clip_denoised: true,
        };
        let result = DDIM::new(diff_cfg, ddim_cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_ddim_eta_validation() {
        let result = DDIMConfig::mixed(5, 1.5, 0);
        assert!(result.is_err());
        let result = DDIMConfig::mixed(5, -0.1, 0);
        assert!(result.is_err());
        let result = DDIMConfig::mixed(5, 0.5, 0);
        assert!(result.is_ok());
    }
}
