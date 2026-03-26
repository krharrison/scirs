//! Consistency Models
//!
//! Implements Consistency Models (Song et al., 2023), which enable high-quality
//! generation in as few as one or two sampling steps by learning a self-consistent
//! mapping from any noisy point on a diffusion trajectory directly to x₀.
//!
//! ## Key Idea
//!
//! A consistency function f(xₜ, t) satisfies the self-consistency property:
//! ```text
//! f(xₜ, t) = f(xₜ', t')   for all t, t' on the same PF-ODE trajectory
//! ```
//!
//! The consistency function is parameterised as:
//! ```text
//! f_θ(xₜ, t) = c_skip(t) · xₜ + c_out(t) · F_θ(c_in(t) · xₜ, t)
//! ```
//! where:
//! - `c_skip(t) = σ_data² / (σ(t)² + σ_data²)` — skip connection scaling
//! - `c_out(t)  = σ(t) · σ_data / √(σ(t)² + σ_data²)` — output scaling
//! - `c_in(t)   = 1 / √(σ(t)² + σ_data²)` — input scaling
//!
//! Noise levels follow the Karras et al. (2022) schedule:
//! ```text
//! σᵢ = (σ_max^(1/ρ) + i/(n-1) · (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
//! ```
//!
//! ## Training Objectives
//!
//! **Consistency Training (CT)** imposes the self-consistency property directly
//! without a pre-trained teacher:
//! ```text
//! L_CT = E[d(f_θ(xₜ, t), f_θ⁻(xₜ', t'))]
//! ```
//!
//! **Consistency Distillation (CD)** uses a pre-trained diffusion model score
//! function as a teacher to generate pseudo-paired training targets.
//!
//! ## References
//! - "Consistency Models", Song et al. (2023) <https://arxiv.org/abs/2303.01469>
//! - "Elucidating the Design Space of Diffusion-Based Generative Models",
//!   Karras et al. (2022) <https://arxiv.org/abs/2206.00364>

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// ConsistencyConfig
// ---------------------------------------------------------------------------

/// Configuration for a Consistency Model.
#[derive(Debug, Clone)]
pub struct ConsistencyConfig {
    /// Data dimensionality (e.g. 784 for MNIST)
    pub data_dim: usize,
    /// Minimum noise level σ_min (boundary condition: f(x, σ_min) ≈ x)
    pub sigma_min: f64,
    /// Maximum noise level σ_max (starting noise for generation)
    pub sigma_max: f64,
    /// Data standard deviation σ_data (Karras et al. recommend 0.5)
    pub sigma_data: f64,
    /// Karras schedule exponent ρ (controls spacing; 7 is standard)
    pub rho: f64,
    /// Number of discrete noise levels in the schedule
    pub n_timesteps: usize,
    /// Hidden dimension for the internal F_θ MLP
    pub hidden_dim: usize,
    /// Random seed for weight initialisation
    pub seed: u64,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            data_dim: 784,
            sigma_min: 0.002,
            sigma_max: 80.0,
            sigma_data: 0.5,
            rho: 7.0,
            n_timesteps: 40,
            hidden_dim: 512,
            seed: 42,
        }
    }
}

impl ConsistencyConfig {
    /// Compact config for unit tests.
    pub fn tiny(data_dim: usize) -> Self {
        Self {
            data_dim,
            sigma_min: 0.002,
            sigma_max: 80.0,
            sigma_data: 0.5,
            rho: 7.0,
            n_timesteps: 10,
            hidden_dim: 32,
            seed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Preconditioning helper functions (Karras et al. 2022 / consistency models)
// ---------------------------------------------------------------------------

/// Skip-connection coefficient: c_skip(σ) = σ_data² / (σ² + σ_data²).
///
/// At σ → 0: c_skip → 1 (identity map, enforcing boundary condition).
/// At σ → ∞: c_skip → 0 (pure network output).
#[inline]
pub fn c_skip(sigma: f64, sigma_data: f64) -> f64 {
    let sd2 = sigma_data * sigma_data;
    sd2 / (sigma * sigma + sd2)
}

/// Output scaling coefficient: c_out(σ) = σ · σ_data / √(σ² + σ_data²).
///
/// At σ = 0: c_out = 0 (zero output, identity function overall).
#[inline]
pub fn c_out(sigma: f64, sigma_data: f64) -> f64 {
    let sd2 = sigma_data * sigma_data;
    sigma * sigma_data / (sigma * sigma + sd2).sqrt().max(1e-12)
}

/// Input scaling coefficient: c_in(σ) = 1 / √(σ² + σ_data²).
///
/// Normalises the network input to unit variance.
#[inline]
pub fn c_in(sigma: f64, sigma_data: f64) -> f64 {
    let sd2 = sigma_data * sigma_data;
    1.0 / (sigma * sigma + sd2).sqrt().max(1e-12)
}

// ---------------------------------------------------------------------------
// ConsistencySchedule — Karras et al. discrete noise levels
// ---------------------------------------------------------------------------

/// Pre-computed noise levels following the Karras et al. (2022) schedule.
///
/// Stores n discrete σ values spanning [σ_min, σ_max] with spacing controlled
/// by the exponent ρ.  Index 0 corresponds to σ_max (highest noise) and
/// index n-1 to σ_min (lowest noise / almost clean data).
#[derive(Debug, Clone)]
pub struct ConsistencySchedule {
    /// Ordered noise levels σ₀ > σ₁ > … > σ_{n-1}
    pub sigmas: Vec<f64>,
    /// σ_data used for preconditioning
    pub sigma_data: f64,
}

impl ConsistencySchedule {
    /// Build a Karras schedule from the given config.
    ///
    /// σᵢ = (σ_max^(1/ρ) + i/(n-1) · (σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ
    pub fn new(config: &ConsistencyConfig) -> Result<Self> {
        if config.n_timesteps < 2 {
            return Err(NeuralError::InvalidArgument(
                "ConsistencySchedule: n_timesteps must be >= 2".to_string(),
            ));
        }
        if config.sigma_min <= 0.0 || config.sigma_max <= config.sigma_min {
            return Err(NeuralError::InvalidArgument(format!(
                "ConsistencySchedule: require 0 < sigma_min ({}) < sigma_max ({})",
                config.sigma_min, config.sigma_max
            )));
        }

        let n = config.n_timesteps;
        let inv_rho = 1.0 / config.rho;
        let s_max_pow = config.sigma_max.powf(inv_rho);
        let s_min_pow = config.sigma_min.powf(inv_rho);

        let sigmas: Vec<f64> = (0..n)
            .map(|i| {
                let frac = i as f64 / (n - 1) as f64;
                // i=0 → σ_max, i=n-1 → σ_min
                (s_max_pow + frac * (s_min_pow - s_max_pow)).powf(config.rho)
            })
            .collect();

        Ok(Self {
            sigmas,
            sigma_data: config.sigma_data,
        })
    }

    /// Return the σ value at schedule index `i`.
    pub fn sigma_at(&self, i: usize) -> Result<f64> {
        self.sigmas.get(i).copied().ok_or_else(|| {
            NeuralError::InvalidArgument(format!(
                "ConsistencySchedule: index {} out of range (len={})",
                i,
                self.sigmas.len()
            ))
        })
    }

    /// Return the index of the σ value closest to the given value.
    pub fn closest_index(&self, sigma: f64) -> usize {
        self.sigmas
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| {
                (a - sigma)
                    .abs()
                    .partial_cmp(&(b - sigma).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Internal MLP — F_θ
// ---------------------------------------------------------------------------
// A lightweight 2-hidden-layer network used as the "raw" backbone F_θ.
// The final consistency function wraps F_θ with the preconditioning scalars.
//
// Architecture:
//   Input: (c_in · x_t ∥ sigma_embedding)  [data_dim + 1]
//   Layer 0: Linear(data_dim+1, hidden_dim) + ReLU
//   Layer 1: Linear(hidden_dim, hidden_dim) + ReLU
//   Layer 2: Linear(hidden_dim, data_dim)
//
// Weights are initialised deterministically with a scaled sinusoidal pattern
// so that the model is reproducible without an external RNG crate.

#[derive(Debug, Clone)]
struct ConsistencyMLP {
    data_dim: usize,
    hidden_dim: usize,
    /// Weights stored as flat row-major matrices; biases as vectors.
    /// Layer i: (weight [out × in], bias [out])
    layers: Vec<(Vec<f64>, Vec<f64>)>,
}

impl ConsistencyMLP {
    /// Create an MLP with deterministic initialisation seeded from `seed`.
    fn new(data_dim: usize, hidden_dim: usize, seed: u64) -> Result<Self> {
        if data_dim == 0 || hidden_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "ConsistencyMLP: data_dim and hidden_dim must be > 0".to_string(),
            ));
        }
        // Input dim = data_dim + 1 (raw data + scalar sigma encoding)
        let input_dim = data_dim + 1;

        // Deterministic He-like weight init using golden-ratio phased sine
        let golden = 0.618_033_988_749_895_f64;
        let make_layer = |in_d: usize, out_d: usize, offset: u64| -> (Vec<f64>, Vec<f64>) {
            let std = (2.0 / in_d as f64).sqrt();
            let weights: Vec<f64> = (0..in_d * out_d)
                .map(|k| std * ((k as f64 + offset as f64 + seed as f64) * golden).sin())
                .collect();
            let bias = vec![0.0f64; out_d];
            (weights, bias)
        };

        let layers = vec![
            make_layer(input_dim, hidden_dim, 0),
            make_layer(hidden_dim, hidden_dim, (hidden_dim * hidden_dim) as u64),
            make_layer(hidden_dim, data_dim, (2 * hidden_dim * hidden_dim) as u64),
        ];

        Ok(Self {
            data_dim,
            hidden_dim,
            layers,
        })
    }

    /// Forward pass through the MLP.
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut h = input.to_vec();
        for (layer_idx, (w, b)) in self.layers.iter().enumerate() {
            let out_dim = b.len();
            let in_dim = h.len();
            let mut next = vec![0.0f64; out_dim];
            for j in 0..out_dim {
                let mut acc = b[j];
                for i in 0..in_dim {
                    let idx = j * in_dim + i;
                    if idx < w.len() {
                        acc += w[idx] * h[i];
                    }
                }
                next[j] = acc;
            }
            // ReLU on all layers except the output layer
            if layer_idx < self.layers.len() - 1 {
                for v in &mut next {
                    *v = v.max(0.0);
                }
            }
            h = next;
        }
        h
    }
}

// ---------------------------------------------------------------------------
// ConsistencyModel
// ---------------------------------------------------------------------------

/// Consistency model that maps any (xₜ, t) pair directly to x₀.
///
/// The model is constructed so that `consistency_fn(x, σ_min) ≈ x` (boundary
/// condition) by virtue of the preconditioning coefficients c_skip/c_out.
///
/// # Example (unit test style)
/// ```ignore
/// let config = ConsistencyConfig::tiny(4);
/// let model = ConsistencyModel::new(config).unwrap();
/// let x_T: Vec<f64> = vec![1.0, -0.5, 0.3, -0.2];
/// let x0 = model.sample_single_step(&x_T).unwrap();
/// assert_eq!(x0.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct ConsistencyModel {
    /// Configuration
    pub config: ConsistencyConfig,
    /// Pre-computed noise schedule
    pub schedule: ConsistencySchedule,
    /// Internal backbone MLP F_θ
    network: ConsistencyMLP,
    /// Seeded PRNG state for noise generation in training losses
    rng_state: u64,
}

impl ConsistencyModel {
    /// Construct a new `ConsistencyModel` from the given config.
    pub fn new(config: ConsistencyConfig) -> Result<Self> {
        let schedule = ConsistencySchedule::new(&config)?;
        let network = ConsistencyMLP::new(config.data_dim, config.hidden_dim, config.seed)?;
        let rng_state = config.seed.wrapping_add(0xdeadbeef_cafebabe);
        Ok(Self {
            config,
            schedule,
            network,
            rng_state,
        })
    }

    // -----------------------------------------------------------------------
    // Internal PRNG (Box-Muller)
    // -----------------------------------------------------------------------

    fn lcg_step(&mut self) -> u64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.rng_state
    }

    fn sample_normal(&mut self) -> f64 {
        let r1 = self.lcg_step();
        let r2 = self.lcg_step();
        let u1 = ((r1 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        let u2 = ((r2 >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    // -----------------------------------------------------------------------
    // Core: consistency function f_θ(xₜ, σ)
    // -----------------------------------------------------------------------

    /// Apply the consistency function: f_θ(xₜ, σ) = c_skip·xₜ + c_out·F_θ(c_in·xₜ, σ).
    ///
    /// # Arguments
    /// * `x_t` – noisy input sample at noise level σ, length = `data_dim`
    /// * `sigma` – noise level σ > 0
    ///
    /// # Returns
    /// Estimated clean sample x̂₀, same length as `x_t`.
    pub fn consistency_fn(&self, x_t: &[f64], sigma: f64) -> Vec<f64> {
        let d = self.config.data_dim;
        let sd = self.config.sigma_data;

        let cs = c_skip(sigma, sd);
        let co = c_out(sigma, sd);
        let ci = c_in(sigma, sd);

        // Build network input: c_in · x_t with sigma appended
        let mut inp: Vec<f64> = x_t.iter().map(|&v| ci * v).collect();
        // Append a scalar sigma encoding (log-normalised for numerical stability)
        let sigma_enc = (sigma / sd).ln(); // log(σ/σ_data) — scale-invariant
        inp.push(sigma_enc);

        let f_out = self.network.forward(&inp);

        // Combine: c_skip · x_t  +  c_out · F_θ(...)
        let len = d.min(f_out.len()).min(x_t.len());
        let mut out = vec![0.0f64; d];
        for i in 0..len {
            out[i] = cs * x_t[i] + co * f_out[i];
        }
        out
    }

    // -----------------------------------------------------------------------
    // Training losses
    // -----------------------------------------------------------------------

    /// Consistency Training (CT) loss between two adjacent noise levels.
    ///
    /// Corrupts `x0` with Gaussian noise at levels `sigma` and `sigma_next`,
    /// then penalises the L2 distance between the consistency function outputs:
    ///
    /// ```text
    /// L_CT = ||f_θ(x₀ + ε·σ, σ) − stop_grad[f_θ(x₀ + ε·σ_next, σ_next)]||²
    /// ```
    ///
    /// In this implementation both branches share the same F_θ (no EMA target
    /// network) for simplicity; in production the second branch uses a
    /// momentum-updated θ⁻.
    ///
    /// # Arguments
    /// * `x0` – clean data sample
    /// * `sigma` – noise level for the primary branch
    /// * `sigma_next` – slightly lower noise level for the target branch
    ///
    /// # Returns
    /// Scalar CT loss value.
    pub fn ct_loss(&mut self, x0: &[f64], sigma: f64, sigma_next: f64) -> Result<f64> {
        let d = self.config.data_dim;
        if x0.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "ConsistencyModel ct_loss: x0 len {} != data_dim {}",
                x0.len(),
                d
            )));
        }
        if sigma <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "ConsistencyModel ct_loss: sigma must be > 0, got {sigma}"
            )));
        }
        if sigma_next <= 0.0 || sigma_next >= sigma {
            return Err(NeuralError::InvalidArgument(format!(
                "ConsistencyModel ct_loss: sigma_next ({sigma_next}) must be in (0, sigma={sigma})"
            )));
        }

        // Sample shared noise ε ~ N(0, I)
        let noise: Vec<f64> = (0..d).map(|_| self.sample_normal()).collect();

        // Primary noisy sample: x_t = x0 + ε · σ
        let x_t: Vec<f64> = x0
            .iter()
            .zip(&noise)
            .map(|(&x, &n)| x + n * sigma)
            .collect();

        // Target noisy sample: x_{t'} = x0 + ε · σ_next
        let x_t_prime: Vec<f64> = x0
            .iter()
            .zip(&noise)
            .map(|(&x, &n)| x + n * sigma_next)
            .collect();

        // Consistency function outputs
        let f1 = self.consistency_fn(&x_t, sigma);
        let f2 = self.consistency_fn(&x_t_prime, sigma_next);

        // MSE loss
        let loss: f64 = f1
            .iter()
            .zip(&f2)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;

        Ok(loss)
    }

    /// Consistency Distillation (CD) loss given a teacher model's output.
    ///
    /// The teacher (pre-trained DDPM or score model) provides a pseudo-clean
    /// target by running one ODE step from xₜ to xₜ':
    ///
    /// ```text
    /// L_CD = ||f_θ(x_t, σ) − stop_grad[f_θ⁻(teacher_output, σ_next)]||²
    /// ```
    ///
    /// # Arguments
    /// * `x0` – clean data (used to construct xₜ)
    /// * `sigma` – noise level for the primary sample
    /// * `teacher_output` – the teacher model's prediction at a slightly lower
    ///                       noise level (serves as the distillation target)
    ///
    /// # Returns
    /// Scalar CD loss value.
    pub fn cd_loss(&mut self, x0: &[f64], sigma: f64, teacher_output: &[f64]) -> Result<f64> {
        let d = self.config.data_dim;
        if x0.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "ConsistencyModel cd_loss: x0 len {} != data_dim {}",
                x0.len(),
                d
            )));
        }
        if teacher_output.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "ConsistencyModel cd_loss: teacher_output len {} != data_dim {}",
                teacher_output.len(),
                d
            )));
        }
        if sigma <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "ConsistencyModel cd_loss: sigma must be > 0, got {sigma}"
            )));
        }

        // Sample noise and construct x_t
        let noise: Vec<f64> = (0..d).map(|_| self.sample_normal()).collect();
        let x_t: Vec<f64> = x0
            .iter()
            .zip(&noise)
            .map(|(&x, &n)| x + n * sigma)
            .collect();

        // Student prediction at (x_t, sigma)
        let f_student = self.consistency_fn(&x_t, sigma);

        // Teacher target: f_θ(teacher_output, sigma_next) — use sigma_min as
        // the target noise level since teacher_output approximates x₀
        let sigma_target = self.config.sigma_min;
        let f_teacher = self.consistency_fn(teacher_output, sigma_target);

        // MSE loss between student and (detached) teacher
        let loss: f64 = f_student
            .iter()
            .zip(&f_teacher)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            / d as f64;

        Ok(loss)
    }

    // -----------------------------------------------------------------------
    // Sampling
    // -----------------------------------------------------------------------

    /// Single-step generation: x₀ ≈ f_θ(x_T, σ_max).
    ///
    /// Maps pure Gaussian noise at σ_max directly to the data manifold in one
    /// function evaluation.
    ///
    /// # Arguments
    /// * `x_T` – starting noise sample (typically x_T ~ N(0, σ_max² I)),
    ///           length = `data_dim`
    ///
    /// # Returns
    /// Estimated x₀, same length as `x_T`.
    pub fn sample_single_step(&self, x_t: &[f64]) -> Result<Vec<f64>> {
        let d = self.config.data_dim;
        if x_t.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "ConsistencyModel sample_single_step: x_T len {} != data_dim {}",
                x_t.len(),
                d
            )));
        }
        Ok(self.consistency_fn(x_t, self.config.sigma_max))
    }

    /// Two-step refinement: x₀ ≈ f_θ(add_noise(f_θ(x_T, σ_max), σ_mid), σ_mid).
    ///
    /// Performs two consistency function evaluations for improved quality:
    /// 1. x̂₀ = f_θ(x_T, σ_max)
    /// 2. x_mid = x̂₀ + z · σ_mid  (z ~ N(0, I))
    /// 3. x₀ = f_θ(x_mid, σ_mid)
    ///
    /// The intermediate re-noising allows the model to refine its initial
    /// estimate.
    ///
    /// # Arguments
    /// * `x_T` – starting noise at σ_max, length = `data_dim`
    /// * `sigma_mid` – intermediate noise level; must satisfy sigma_min < sigma_mid < sigma_max
    ///
    /// # Returns
    /// Refined estimate of x₀.
    pub fn sample_two_step(&mut self, x_t: &[f64], sigma_mid: f64) -> Result<Vec<f64>> {
        let d = self.config.data_dim;
        if x_t.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "ConsistencyModel sample_two_step: x_T len {} != data_dim {}",
                x_t.len(),
                d
            )));
        }
        if sigma_mid <= self.config.sigma_min || sigma_mid >= self.config.sigma_max {
            return Err(NeuralError::InvalidArgument(format!(
                "ConsistencyModel sample_two_step: sigma_mid ({sigma_mid}) must be in \
                 (sigma_min={}, sigma_max={})",
                self.config.sigma_min, self.config.sigma_max
            )));
        }

        // Step 1: coarse estimate from full noise
        let x0_hat = self.consistency_fn(x_t, self.config.sigma_max);

        // Step 2: re-noise to sigma_mid
        let x_mid: Vec<f64> = x0_hat
            .iter()
            .map(|&v| {
                let z = self.sample_normal();
                v + z * sigma_mid
            })
            .collect();

        // Step 3: refine at sigma_mid
        Ok(self.consistency_fn(&x_mid, sigma_mid))
    }

    /// Multi-step generation using the full Karras noise schedule.
    ///
    /// Iterates from σ_max down to σ_min, re-noising and applying the
    /// consistency function at each level. This is equivalent to `n_timesteps`
    /// two-step applications and yields the best quality at the cost of more
    /// function evaluations.
    ///
    /// # Arguments
    /// * `x_T` – initial noise sample at σ_max
    ///
    /// # Returns
    /// Estimated x₀.
    pub fn sample_multistep(&mut self, x_t: &[f64]) -> Result<Vec<f64>> {
        let d = self.config.data_dim;
        if x_t.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "ConsistencyModel sample_multistep: x_T len {} != data_dim {}",
                x_t.len(),
                d
            )));
        }

        let sigmas = self.schedule.sigmas.clone();
        let mut x_current = x_t.to_vec();

        for (i, &sigma) in sigmas.iter().enumerate() {
            // Apply consistency function to get current x₀ estimate
            let x0_hat = self.consistency_fn(&x_current, sigma);

            // If not the last step, re-noise to the next sigma level
            if i + 1 < sigmas.len() {
                let sigma_next = sigmas[i + 1];
                x_current = x0_hat
                    .iter()
                    .map(|&v| {
                        let z = self.sample_normal();
                        v + z * sigma_next
                    })
                    .collect();
            } else {
                x_current = x0_hat;
            }
        }

        Ok(x_current)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Config & schedule -------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = ConsistencyConfig::default();
        assert_eq!(cfg.data_dim, 784);
        assert!((cfg.sigma_min - 0.002).abs() < 1e-9);
        assert!((cfg.sigma_max - 80.0).abs() < 1e-9);
        assert!((cfg.sigma_data - 0.5).abs() < 1e-9);
        assert!((cfg.rho - 7.0).abs() < 1e-9);
        assert_eq!(cfg.n_timesteps, 40);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_tiny_config() {
        let cfg = ConsistencyConfig::tiny(4);
        assert_eq!(cfg.data_dim, 4);
        assert_eq!(cfg.n_timesteps, 10);
    }

    #[test]
    fn test_schedule_length() {
        let cfg = ConsistencyConfig::tiny(4);
        let sched = ConsistencySchedule::new(&cfg).expect("schedule");
        assert_eq!(sched.sigmas.len(), cfg.n_timesteps);
    }

    #[test]
    fn test_schedule_strictly_decreasing() {
        let cfg = ConsistencyConfig::tiny(4);
        let sched = ConsistencySchedule::new(&cfg).expect("schedule");
        for window in sched.sigmas.windows(2) {
            assert!(
                window[0] > window[1],
                "schedule not strictly decreasing: {} >= {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_schedule_endpoints() {
        let cfg = ConsistencyConfig::tiny(4);
        let sched = ConsistencySchedule::new(&cfg).expect("schedule");
        // First value ≈ sigma_max
        assert!(
            (sched.sigmas[0] - cfg.sigma_max).abs() < 1e-6,
            "first sigma should be sigma_max; got {}",
            sched.sigmas[0]
        );
        // Last value ≈ sigma_min
        let last = *sched.sigmas.last().expect("non-empty");
        assert!(
            (last - cfg.sigma_min).abs() < 1e-6,
            "last sigma should be sigma_min; got {last}"
        );
    }

    #[test]
    fn test_schedule_invalid_args() {
        let mut cfg = ConsistencyConfig::tiny(4);
        cfg.n_timesteps = 1;
        assert!(ConsistencySchedule::new(&cfg).is_err());

        cfg.n_timesteps = 10;
        cfg.sigma_min = 0.0;
        assert!(ConsistencySchedule::new(&cfg).is_err());

        cfg.sigma_min = 100.0; // > sigma_max
        assert!(ConsistencySchedule::new(&cfg).is_err());
    }

    // ---- Preconditioning scalars -------------------------------------------

    #[test]
    fn test_c_skip_at_zero_noise() {
        // At σ=0: c_skip = σ_data² / σ_data² = 1
        let sd = 0.5;
        let eps = 1e-10;
        let cs = c_skip(eps, sd);
        assert!((cs - 1.0).abs() < 1e-4, "c_skip near 0 should ≈ 1, got {cs}");
    }

    #[test]
    fn test_c_out_at_zero_noise() {
        // At σ=0: c_out = 0 · σ_data / σ_data = 0
        let sd = 0.5;
        let eps = 1e-10;
        let co = c_out(eps, sd);
        assert!(co.abs() < 1e-5, "c_out near 0 should ≈ 0, got {co}");
    }

    #[test]
    fn test_c_skip_plus_c_out_at_sigma_data() {
        // At σ = σ_data: c_skip = 0.5, c_out = σ_data/√2
        let sd = 0.5;
        let cs = c_skip(sd, sd);
        assert!(
            (cs - 0.5).abs() < 1e-10,
            "c_skip(sigma_data) should be 0.5, got {cs}"
        );
    }

    #[test]
    fn test_c_in_normalises_scale() {
        // c_in · σ = σ / √(σ² + σ_data²) < 1 always
        let sd = 0.5;
        for sigma in [0.1, 0.5, 1.0, 10.0, 80.0] {
            let ci = c_in(sigma, sd);
            assert!(ci > 0.0 && ci.is_finite(), "c_in should be finite positive");
            // ci · sigma < 1 always
            assert!(ci * sigma < 1.0 + 1e-9);
        }
    }

    // ---- ConsistencyModel --------------------------------------------------

    #[test]
    fn test_model_creation() {
        let cfg = ConsistencyConfig::tiny(4);
        let model = ConsistencyModel::new(cfg).expect("model creation");
        assert_eq!(model.config.data_dim, 4);
    }

    #[test]
    fn test_consistency_fn_output_shape() {
        let cfg = ConsistencyConfig::tiny(4);
        let model = ConsistencyModel::new(cfg).expect("model");
        let x_noisy = vec![0.5, -0.3, 0.2, 0.8];
        let out = model.consistency_fn(&x_noisy, 10.0);
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(v.is_finite(), "consistency_fn output must be finite");
        }
    }

    #[test]
    fn test_consistency_fn_boundary_condition() {
        // At σ_min: c_skip ≈ 1, c_out ≈ 0, so f(x, σ_min) ≈ x
        let cfg = ConsistencyConfig::tiny(4);
        let model = ConsistencyModel::new(cfg.clone()).expect("model");
        let x_clean = vec![0.1, 0.2, -0.3, 0.4];
        let sigma_min = cfg.sigma_min; // 0.002 — very small
        let out = model.consistency_fn(&x_clean, sigma_min);
        // c_skip ≈ 1 so the output should be close to x (within reasonable tolerance)
        for (&xi, &oi) in x_clean.iter().zip(&out) {
            let diff = (xi - oi).abs();
            assert!(
                diff < 0.5,
                "At sigma_min, f(x,t) should be close to x; diff={diff}"
            );
        }
    }

    #[test]
    fn test_ct_loss_runs() {
        let cfg = ConsistencyConfig::tiny(4);
        let mut model = ConsistencyModel::new(cfg).expect("model");
        let x0 = vec![0.3, -0.1, 0.5, -0.2];
        let loss = model.ct_loss(&x0, 10.0, 5.0).expect("ct_loss");
        assert!(loss >= 0.0 && loss.is_finite(), "CT loss must be finite non-negative");
    }

    #[test]
    fn test_ct_loss_invalid_args() {
        let cfg = ConsistencyConfig::tiny(4);
        let mut model = ConsistencyModel::new(cfg).expect("model");
        let x0 = vec![0.3, -0.1, 0.5, -0.2];

        // sigma <= 0
        assert!(model.ct_loss(&x0, 0.0, -1.0).is_err());
        // sigma_next >= sigma
        assert!(model.ct_loss(&x0, 5.0, 5.0).is_err());
        assert!(model.ct_loss(&x0, 5.0, 10.0).is_err());
        // wrong data dim
        assert!(model.ct_loss(&[1.0, 2.0], 10.0, 5.0).is_err());
    }

    #[test]
    fn test_cd_loss_runs() {
        let cfg = ConsistencyConfig::tiny(4);
        let mut model = ConsistencyModel::new(cfg).expect("model");
        let x0 = vec![0.3, -0.1, 0.5, -0.2];
        let teacher = vec![-0.1, 0.2, 0.1, 0.3];
        let loss = model.cd_loss(&x0, 10.0, &teacher).expect("cd_loss");
        assert!(loss >= 0.0 && loss.is_finite());
    }

    #[test]
    fn test_sample_single_step_shape() {
        let cfg = ConsistencyConfig::tiny(4);
        let model = ConsistencyModel::new(cfg).expect("model");
        let x_noise = vec![1.0, -0.5, 0.3, -0.2];
        let x0 = model.sample_single_step(&x_noise).expect("single-step sample");
        assert_eq!(x0.len(), 4);
        for &v in &x0 {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sample_single_step_wrong_dim() {
        let cfg = ConsistencyConfig::tiny(4);
        let model = ConsistencyModel::new(cfg).expect("model");
        let x_wrong = vec![1.0, -0.5]; // wrong dim
        assert!(model.sample_single_step(&x_wrong).is_err());
    }

    #[test]
    fn test_sample_two_step_shape() {
        let cfg = ConsistencyConfig::tiny(4);
        let mut model = ConsistencyModel::new(cfg).expect("model");
        let x_noise = vec![1.0, -0.5, 0.3, -0.2];
        let sigma_mid = 5.0; // between sigma_min=0.002 and sigma_max=80
        let x0 = model.sample_two_step(&x_noise, sigma_mid).expect("two-step sample");
        assert_eq!(x0.len(), 4);
        for &v in &x0 {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sample_two_step_invalid_sigma_mid() {
        let cfg = ConsistencyConfig::tiny(4);
        let mut model = ConsistencyModel::new(cfg.clone()).expect("model");
        let x_noise = vec![1.0, -0.5, 0.3, -0.2];

        // sigma_mid == sigma_min → invalid
        assert!(model.sample_two_step(&x_noise, cfg.sigma_min).is_err());
        // sigma_mid == sigma_max → invalid
        assert!(model.sample_two_step(&x_noise, cfg.sigma_max).is_err());
        // sigma_mid < sigma_min → invalid
        assert!(model.sample_two_step(&x_noise, 0.0001).is_err());
    }

    #[test]
    fn test_sample_multistep_shape() {
        let cfg = ConsistencyConfig::tiny(4);
        let mut model = ConsistencyModel::new(cfg).expect("model");
        let x_noise = vec![1.0, -0.5, 0.3, -0.2];
        let x0 = model.sample_multistep(&x_noise).expect("multistep");
        assert_eq!(x0.len(), 4);
        for &v in &x0 {
            assert!(v.is_finite());
        }
    }
}
