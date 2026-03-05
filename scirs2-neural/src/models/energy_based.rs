//! Energy-based models (EBMs)
//!
//! Implements energy-based generative models:
//! - `RestrictedBoltzmannMachine` (RBM) with contrastive divergence
//! - `DeepBoltzmannMachine` (DBM) with layer-wise pretraining
//! - `EnergyBasedModel` – generic EBM with Langevin dynamics MCMC sampling
//! - `ContrastiveDivergenceK` – k-step CD training algorithm
//! - `PersistentCD` – persistent contrastive divergence (PCD)
//!
//! ## References
//! - Hinton (2002) "Training products of experts by minimizing contrastive divergence"
//! - Salakhutdinov & Hinton (2009) "Deep Boltzmann Machines"
//! - LeCun et al. (2006) "A tutorial on energy-based learning"
//! - Du & Mordatch (2019) "Implicit Generation and Modeling with Energy Based Models"

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// EnergyFunction trait
// ---------------------------------------------------------------------------

/// Trait for an energy function mapping input vectors to scalar energy values.
///
/// Lower energy corresponds to higher probability density under the model:
/// `p(x) ∝ exp(-E(x))`.
pub trait EnergyFunction: Send + Sync {
    /// Compute the scalar energy for input `x`.
    fn energy(&self, x: &[f64]) -> f64;

    /// Approximate gradient of energy w.r.t. `x` via finite differences.
    ///
    /// Subclasses should override for analytic gradients when available.
    fn energy_gradient(&self, x: &[f64], eps: f64) -> Vec<f64> {
        let e0 = self.energy(x);
        let mut grad = vec![0.0f64; x.len()];
        let mut x_perturb = x.to_vec();
        for i in 0..x.len() {
            x_perturb[i] += eps;
            let e1 = self.energy(&x_perturb);
            grad[i] = (e1 - e0) / eps;
            x_perturb[i] = x[i];
        }
        grad
    }
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Sigmoid function σ(x) = 1 / (1 + exp(-x))
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Sample a Bernoulli random variable from probability `p` using a
/// linear-congruential pseudo-random stream seeded by `state`.
fn bernoulli_sample(p: f64, state: &mut u64) -> f64 {
    // LCG: cheap but sufficient for MC sampling during training
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    let u = (*state >> 33) as f64 / (u32::MAX as f64);
    if u < p { 1.0 } else { 0.0 }
}

/// Standard normal sample via Box–Muller (consumes two LCG draws).
fn normal_sample(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    let u1 = ((*state >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    let u2 = ((*state >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// RestrictedBoltzmannMachine
// ---------------------------------------------------------------------------

/// Restricted Boltzmann Machine (RBM).
///
/// A bipartite undirected graphical model with `n_visible` visible units and
/// `n_hidden` hidden units.  The energy function is:
///
/// ```text
/// E(v, h) = -v^T W h - b_v^T v - b_h^T h
/// ```
///
/// Training uses contrastive divergence (CD-k).
#[derive(Debug, Clone)]
pub struct RestrictedBoltzmannMachine {
    /// Number of visible units
    pub n_visible: usize,
    /// Number of hidden units
    pub n_hidden: usize,
    /// Weight matrix W: [n_visible, n_hidden] (row-major)
    pub weights: Vec<f64>,
    /// Visible biases b_v: [n_visible]
    pub visible_bias: Vec<f64>,
    /// Hidden biases b_h: [n_hidden]
    pub hidden_bias: Vec<f64>,
    /// Learning rate
    pub learning_rate: f64,
    /// CD steps
    pub cd_steps: usize,
    /// Pseudo-random state for sampling
    rng_state: u64,
}

impl RestrictedBoltzmannMachine {
    /// Create a new RBM with Xavier-initialized weights.
    pub fn new(n_visible: usize, n_hidden: usize, learning_rate: f64, cd_steps: usize) -> Result<Self> {
        if n_visible == 0 || n_hidden == 0 {
            return Err(NeuralError::InvalidArgument(
                "RBM: n_visible and n_hidden must be > 0".to_string(),
            ));
        }
        if cd_steps == 0 {
            return Err(NeuralError::InvalidArgument(
                "RBM: cd_steps must be >= 1".to_string(),
            ));
        }
        let std = (2.0 / (n_visible + n_hidden) as f64).sqrt();
        let weights: Vec<f64> = (0..n_visible * n_hidden)
            .map(|k| std * ((k as f64) * 0.6180339887).sin())
            .collect();
        Ok(Self {
            n_visible,
            n_hidden,
            weights,
            visible_bias: vec![0.0; n_visible],
            hidden_bias: vec![0.0; n_hidden],
            learning_rate,
            cd_steps,
            rng_state: 0xdeadbeef_cafebabe,
        })
    }

    /// Compute hidden unit probabilities p(h=1 | v).
    pub fn hidden_probs(&self, v: &[f64]) -> Vec<f64> {
        (0..self.n_hidden)
            .map(|j| {
                let s: f64 = (0..self.n_visible)
                    .map(|i| v[i] * self.weights[i * self.n_hidden + j])
                    .sum();
                sigmoid(s + self.hidden_bias[j])
            })
            .collect()
    }

    /// Compute visible unit probabilities p(v=1 | h).
    pub fn visible_probs(&self, h: &[f64]) -> Vec<f64> {
        (0..self.n_visible)
            .map(|i| {
                let s: f64 = (0..self.n_hidden)
                    .map(|j| h[j] * self.weights[i * self.n_hidden + j])
                    .sum();
                sigmoid(s + self.visible_bias[i])
            })
            .collect()
    }

    /// Sample binary hidden units given visible units.
    pub fn sample_hidden(&mut self, v: &[f64]) -> Vec<f64> {
        let probs = self.hidden_probs(v);
        probs
            .iter()
            .map(|&p| bernoulli_sample(p, &mut self.rng_state))
            .collect()
    }

    /// Sample binary visible units given hidden units.
    pub fn sample_visible(&mut self, h: &[f64]) -> Vec<f64> {
        let probs = self.visible_probs(h);
        probs
            .iter()
            .map(|&p| bernoulli_sample(p, &mut self.rng_state))
            .collect()
    }

    /// Compute the free energy of a visible vector:
    /// `F(v) = -b_v^T v - Σ_j log(1 + exp(b_hj + Σ_i v_i W_ij))`
    pub fn free_energy(&self, v: &[f64]) -> f64 {
        let bv_term: f64 = v.iter().zip(&self.visible_bias).map(|(&vi, &bi)| vi * bi).sum();
        let hidden_term: f64 = (0..self.n_hidden)
            .map(|j| {
                let s: f64 = (0..self.n_visible)
                    .map(|i| v[i] * self.weights[i * self.n_hidden + j])
                    .sum();
                let x = s + self.hidden_bias[j];
                // log(1 + exp(x)) numerically stable
                if x > 0.0 {
                    x + (1.0 + (-x).exp()).ln()
                } else {
                    (1.0 + x.exp()).ln()
                }
            })
            .sum();
        -bv_term - hidden_term
    }

    /// Perform one CD-k update step for a single data sample `v_data`.
    ///
    /// Returns the reconstruction error (mean squared difference between
    /// the data and the model's reconstruction).
    pub fn train_step(&mut self, v_data: &[f64]) -> Result<f64> {
        if v_data.len() != self.n_visible {
            return Err(NeuralError::ShapeMismatch(format!(
                "RBM train_step: expected {} visible units, got {}",
                self.n_visible,
                v_data.len()
            )));
        }
        // Positive phase: p(h|v_data)
        let h_pos_probs = self.hidden_probs(v_data);
        let h_pos: Vec<f64> = h_pos_probs
            .iter()
            .map(|&p| bernoulli_sample(p, &mut self.rng_state))
            .collect();
        // Negative phase: CD-k Gibbs sampling
        let mut v_neg = v_data.to_vec();
        let mut h_neg = h_pos.clone();
        for _ in 0..self.cd_steps {
            v_neg = self.sample_visible(&h_neg);
            h_neg = self.sample_hidden(&v_neg);
        }
        let h_neg_probs = self.hidden_probs(&v_neg);
        // Gradient updates: Δ = lr * (pos - neg)
        let lr = self.learning_rate;
        for i in 0..self.n_visible {
            for j in 0..self.n_hidden {
                let pos = v_data[i] * h_pos_probs[j];
                let neg = v_neg[i] * h_neg_probs[j];
                self.weights[i * self.n_hidden + j] += lr * (pos - neg);
            }
            self.visible_bias[i] += lr * (v_data[i] - v_neg[i]);
        }
        for j in 0..self.n_hidden {
            self.hidden_bias[j] += lr * (h_pos_probs[j] - h_neg_probs[j]);
        }
        // Reconstruction error
        let recon = self.visible_probs(&h_pos);
        let err: f64 = v_data
            .iter()
            .zip(&recon)
            .map(|(&v, &r)| (v - r).powi(2))
            .sum::<f64>()
            / self.n_visible as f64;
        Ok(err)
    }

    /// Train for one epoch over a batch of samples.
    pub fn train_epoch(&mut self, data: &[Vec<f64>]) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        let total_err: f64 = data
            .iter()
            .map(|v| self.train_step(v))
            .collect::<Result<Vec<f64>>>()?
            .iter()
            .sum();
        Ok(total_err / data.len() as f64)
    }

    /// Reconstruct a visible vector by one round of Gibbs sampling.
    pub fn reconstruct(&mut self, v: &[f64]) -> Vec<f64> {
        let h = self.sample_hidden(v);
        self.visible_probs(&h)
    }

    /// Generate a sample by running `steps` steps of Gibbs sampling from noise.
    pub fn generate(&mut self, steps: usize) -> Vec<f64> {
        // Initialise from random binary vector
        let mut v: Vec<f64> = (0..self.n_visible)
            .map(|_| bernoulli_sample(0.5, &mut self.rng_state))
            .collect();
        for _ in 0..steps {
            let h = self.sample_hidden(&v);
            v = self.sample_visible(&h);
        }
        v
    }
}

impl EnergyFunction for RestrictedBoltzmannMachine {
    fn energy(&self, x: &[f64]) -> f64 {
        // Marginalise over hidden units: use free energy
        self.free_energy(x)
    }
}

// ---------------------------------------------------------------------------
// DeepBoltzmannMachine
// ---------------------------------------------------------------------------

/// Deep Boltzmann Machine (DBM).
///
/// A stack of RBMs for layer-wise pretraining.  During greedy layer-wise
/// pretraining each pair of adjacent layers is trained as a standalone RBM.
pub struct DeepBoltzmannMachine {
    /// Stack of RBMs (one per adjacent layer pair)
    rbms: Vec<RestrictedBoltzmannMachine>,
    /// Layer sizes (visible + hidden layers)
    pub layer_sizes: Vec<usize>,
    /// Learning rate
    pub learning_rate: f64,
    /// CD steps per RBM
    pub cd_steps: usize,
}

impl std::fmt::Debug for DeepBoltzmannMachine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeepBoltzmannMachine")
            .field("layer_sizes", &self.layer_sizes)
            .field("num_rbms", &self.rbms.len())
            .finish()
    }
}

impl DeepBoltzmannMachine {
    /// Create a new DBM with the given layer sizes.
    ///
    /// `layer_sizes[0]` is the visible layer; subsequent entries are hidden layers.
    pub fn new(layer_sizes: Vec<usize>, learning_rate: f64, cd_steps: usize) -> Result<Self> {
        if layer_sizes.len() < 2 {
            return Err(NeuralError::InvalidArgument(
                "DBM requires at least 2 layer sizes (visible + 1 hidden)".to_string(),
            ));
        }
        let rbms = layer_sizes
            .windows(2)
            .map(|w| RestrictedBoltzmannMachine::new(w[0], w[1], learning_rate, cd_steps))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            rbms,
            layer_sizes,
            learning_rate,
            cd_steps,
        })
    }

    /// Greedy layer-wise pretraining.
    ///
    /// Trains each RBM in turn using the hidden representations from the
    /// previous RBM as inputs.
    ///
    /// # Arguments
    /// * `data` – training samples (each has length `layer_sizes[0]`)
    /// * `epochs_per_layer` – number of epochs to train each RBM
    pub fn pretrain(&mut self, data: &[Vec<f64>], epochs_per_layer: usize) -> Result<()> {
        let mut current_data: Vec<Vec<f64>> = data.to_vec();
        for (layer_idx, rbm) in self.rbms.iter_mut().enumerate() {
            for epoch in 0..epochs_per_layer {
                let err = rbm.train_epoch(&current_data)?;
                if epoch % 10 == 0 {
                    let _ = err; // in production: log or track
                }
                let _ = (layer_idx, epoch);
            }
            // Transform data to next layer's representation
            current_data = current_data
                .iter()
                .map(|v| rbm.hidden_probs(v))
                .collect();
        }
        Ok(())
    }

    /// Extract deep features by propagating data through all RBM layers.
    pub fn encode(&self, v: &[f64]) -> Result<Vec<f64>> {
        if v.len() != self.layer_sizes[0] {
            return Err(NeuralError::ShapeMismatch(format!(
                "DBM encode: expected {} inputs, got {}",
                self.layer_sizes[0],
                v.len()
            )));
        }
        let mut h = v.to_vec();
        for rbm in &self.rbms {
            h = rbm.hidden_probs(&h);
        }
        Ok(h)
    }

    /// Compute the total energy by summing free energies of each layer.
    pub fn total_energy(&self, v: &[f64]) -> Result<f64> {
        if v.len() != self.layer_sizes[0] {
            return Err(NeuralError::ShapeMismatch(format!(
                "DBM total_energy: expected {} inputs, got {}",
                self.layer_sizes[0],
                v.len()
            )));
        }
        let mut energy = 0.0f64;
        let mut current = v.to_vec();
        for rbm in &self.rbms {
            energy += rbm.free_energy(&current);
            current = rbm.hidden_probs(&current);
        }
        Ok(energy)
    }

    /// Number of RBM layers
    pub fn num_layers(&self) -> usize {
        self.rbms.len()
    }
}

// ---------------------------------------------------------------------------
// EnergyBasedModel (generic EBM with Langevin dynamics)
// ---------------------------------------------------------------------------

/// Configuration for Langevin dynamics MCMC sampling.
#[derive(Debug, Clone)]
pub struct LangevinConfig {
    /// Step size η
    pub step_size: f64,
    /// Number of Langevin steps per sample
    pub num_steps: usize,
    /// Noise scale (1.0 = standard Langevin, 0.0 = gradient descent)
    pub noise_scale: f64,
    /// Gradient clipping value (0.0 = no clipping)
    pub grad_clip: f64,
    /// Finite difference epsilon for gradient estimation
    pub fd_eps: f64,
}

impl Default for LangevinConfig {
    fn default() -> Self {
        Self {
            step_size: 0.01,
            num_steps: 20,
            noise_scale: 1.0,
            grad_clip: 1.0,
            fd_eps: 1e-3,
        }
    }
}

/// Generic Energy-Based Model with Langevin dynamics MCMC sampling.
///
/// Wraps any `EnergyFunction` and provides:
/// - Contrastive divergence training
/// - Langevin dynamics sampling
/// - Score function estimation
pub struct EnergyBasedModel {
    /// Parameterized energy function
    pub energy_fn: Box<dyn EnergyFunction>,
    /// Langevin dynamics configuration
    pub langevin_config: LangevinConfig,
    /// Pseudo-random state
    rng_state: u64,
}

impl std::fmt::Debug for EnergyBasedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnergyBasedModel")
            .field("langevin_config", &self.langevin_config)
            .finish()
    }
}

impl EnergyBasedModel {
    /// Create a new `EnergyBasedModel`.
    pub fn new(energy_fn: Box<dyn EnergyFunction>, langevin_config: LangevinConfig) -> Self {
        Self {
            energy_fn,
            langevin_config,
            rng_state: 0x1234567890abcdef,
        }
    }

    /// Draw a sample via Langevin dynamics starting from `x_init`.
    ///
    /// Update rule: `x_{t+1} = x_t - η ∇E(x_t) + √(2η) ε_t`
    /// where `ε_t ~ N(0, I)`.
    pub fn langevin_sample(&mut self, x_init: &[f64]) -> Vec<f64> {
        let cfg = &self.langevin_config;
        let mut x = x_init.to_vec();
        for _ in 0..cfg.num_steps {
            let grad = self.energy_fn.energy_gradient(&x, cfg.fd_eps);
            let noise_std = (2.0 * cfg.step_size).sqrt() * cfg.noise_scale;
            for i in 0..x.len() {
                let mut g = grad[i];
                if cfg.grad_clip > 0.0 {
                    g = g.clamp(-cfg.grad_clip, cfg.grad_clip);
                }
                let noise = normal_sample(&mut self.rng_state) * noise_std;
                x[i] -= cfg.step_size * g + noise;
            }
        }
        x
    }

    /// Compute the energy of `x`.
    pub fn energy(&self, x: &[f64]) -> f64 {
        self.energy_fn.energy(x)
    }

    /// Approximate log-partition function via importance sampling (not normalised).
    /// Useful for monitoring relative likelihoods.
    pub fn log_likelihood_estimate(&mut self, x: &[f64], n_samples: usize) -> f64 {
        let e_data = self.energy(x);
        // Draw model samples
        let mut noise_init: Vec<f64> = (0..x.len())
            .map(|_| normal_sample(&mut self.rng_state))
            .collect();
        let mut energies = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let sample = self.langevin_sample(&noise_init);
            energies.push(self.energy(&sample));
            noise_init = sample;
        }
        let mean_model_e = energies.iter().sum::<f64>() / n_samples as f64;
        -(e_data - mean_model_e)
    }
}

// ---------------------------------------------------------------------------
// ContrastiveDivergenceK
// ---------------------------------------------------------------------------

/// k-step Contrastive Divergence training algorithm for RBMs.
///
/// Wraps an RBM and provides structured training over datasets.
pub struct ContrastiveDivergenceK {
    /// Inner RBM
    pub rbm: RestrictedBoltzmannMachine,
    /// Number of CD steps (k)
    pub k: usize,
}

impl std::fmt::Debug for ContrastiveDivergenceK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContrastiveDivergenceK")
            .field("rbm", &format!("RBM({}, {})", self.rbm.n_visible, self.rbm.n_hidden))
            .field("k", &self.k)
            .finish()
    }
}

impl ContrastiveDivergenceK {
    /// Create a `ContrastiveDivergenceK` trainer.
    pub fn new(
        n_visible: usize,
        n_hidden: usize,
        learning_rate: f64,
        k: usize,
    ) -> Result<Self> {
        let rbm = RestrictedBoltzmannMachine::new(n_visible, n_hidden, learning_rate, k)?;
        Ok(Self { rbm, k })
    }

    /// Run one training epoch returning the mean reconstruction error.
    pub fn train_epoch(&mut self, data: &[Vec<f64>]) -> Result<f64> {
        self.rbm.train_epoch(data)
    }

    /// Train for `epochs` epochs, returning per-epoch errors.
    pub fn train(&mut self, data: &[Vec<f64>], epochs: usize) -> Result<Vec<f64>> {
        (0..epochs).map(|_| self.train_epoch(data)).collect()
    }

    /// Compute pseudo-log-likelihood (a tractable proxy for the log-likelihood).
    ///
    /// Uses the free energy difference trick:
    /// `PLL(v) ≈ -n_visible * log(sigmoid(F(v) - F(v̄_i)))`
    /// where `v̄_i` is `v` with bit `i` flipped.
    pub fn pseudo_log_likelihood(&self, v: &[f64]) -> Result<f64> {
        if v.len() != self.rbm.n_visible {
            return Err(NeuralError::ShapeMismatch(format!(
                "CD-k PLL: expected {} visible, got {}",
                self.rbm.n_visible,
                v.len()
            )));
        }
        let fe = self.rbm.free_energy(v);
        let mut pll = 0.0f64;
        let mut v_flip = v.to_vec();
        for i in 0..self.rbm.n_visible {
            v_flip[i] = 1.0 - v_flip[i];
            let fe_flip = self.rbm.free_energy(&v_flip);
            v_flip[i] = v[i];
            // cost = log σ(F(v_flip) - F(v))  [cross-entropy form]
            let diff = fe_flip - fe;
            pll += -sigmoid(-diff).ln();
        }
        Ok(-pll)
    }
}

// ---------------------------------------------------------------------------
// PersistentCD
// ---------------------------------------------------------------------------

/// Persistent Contrastive Divergence (PCD).
///
/// Maintains a set of persistent Markov chains ("fantasy particles") that
/// are updated each training step rather than being re-initialised from
/// data.  This reduces the bias introduced by short CD chains.
///
/// Reference: Tieleman (2008) "Training Restricted Boltzmann Machines using
/// approximations to the likelihood gradient"
pub struct PersistentCD {
    /// Inner RBM
    pub rbm: RestrictedBoltzmannMachine,
    /// Number of persistent chains
    pub num_chains: usize,
    /// Persistent chain state (fantasy particles), shape [num_chains, n_visible]
    chain_state: Vec<Vec<f64>>,
    /// Number of Gibbs steps per update
    pub gibbs_steps: usize,
}

impl std::fmt::Debug for PersistentCD {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PersistentCD")
            .field("n_visible", &self.rbm.n_visible)
            .field("n_hidden", &self.rbm.n_hidden)
            .field("num_chains", &self.num_chains)
            .field("gibbs_steps", &self.gibbs_steps)
            .finish()
    }
}

impl PersistentCD {
    /// Create a new `PersistentCD` trainer.
    ///
    /// # Arguments
    /// * `n_visible`, `n_hidden` – RBM dimensions
    /// * `learning_rate` – SGD learning rate
    /// * `num_chains` – number of persistent fantasy particles
    /// * `gibbs_steps` – Gibbs steps per update
    pub fn new(
        n_visible: usize,
        n_hidden: usize,
        learning_rate: f64,
        num_chains: usize,
        gibbs_steps: usize,
    ) -> Result<Self> {
        if num_chains == 0 {
            return Err(NeuralError::InvalidArgument(
                "PersistentCD: num_chains must be >= 1".to_string(),
            ));
        }
        if gibbs_steps == 0 {
            return Err(NeuralError::InvalidArgument(
                "PersistentCD: gibbs_steps must be >= 1".to_string(),
            ));
        }
        let rbm = RestrictedBoltzmannMachine::new(n_visible, n_hidden, learning_rate, 1)?;
        // Initialise chains to uniform random binary vectors (seeded deterministically)
        let mut rng_state: u64 = 0xfeedcafedeadbeef;
        let chain_state = (0..num_chains)
            .map(|_| {
                (0..n_visible)
                    .map(|_| bernoulli_sample(0.5, &mut rng_state))
                    .collect::<Vec<f64>>()
            })
            .collect();
        Ok(Self {
            rbm,
            num_chains,
            chain_state,
            gibbs_steps,
        })
    }

    /// Perform one PCD update step for a mini-batch of data.
    ///
    /// # Returns
    /// Mean reconstruction error for the batch.
    pub fn train_step(&mut self, batch: &[Vec<f64>]) -> Result<f64> {
        if batch.is_empty() {
            return Ok(0.0);
        }
        let lr = self.rbm.learning_rate;
        let n_v = self.rbm.n_visible;
        let n_h = self.rbm.n_hidden;
        // Positive phase gradients (averaged over mini-batch)
        let mut dw_pos = vec![0.0f64; n_v * n_h];
        let mut dv_pos = vec![0.0f64; n_v];
        let mut dh_pos = vec![0.0f64; n_h];
        for v in batch.iter() {
            if v.len() != n_v {
                return Err(NeuralError::ShapeMismatch(format!(
                    "PCD train_step: expected {n_v} visible, got {}",
                    v.len()
                )));
            }
            let h_probs = self.rbm.hidden_probs(v);
            for i in 0..n_v {
                dv_pos[i] += v[i];
                for j in 0..n_h {
                    dw_pos[i * n_h + j] += v[i] * h_probs[j];
                }
            }
            for j in 0..n_h {
                dh_pos[j] += h_probs[j];
            }
        }
        let bs = batch.len() as f64;
        // Negative phase: advance persistent chains
        let mut dw_neg = vec![0.0f64; n_v * n_h];
        let mut dv_neg = vec![0.0f64; n_v];
        let mut dh_neg = vec![0.0f64; n_h];
        for c in 0..self.num_chains {
            for _ in 0..self.gibbs_steps {
                let h = self.rbm.sample_hidden(&self.chain_state[c].clone());
                self.chain_state[c] = self.rbm.sample_visible(&h);
            }
            let v_neg = &self.chain_state[c];
            let h_probs_neg = self.rbm.hidden_probs(v_neg);
            for i in 0..n_v {
                dv_neg[i] += v_neg[i];
                for j in 0..n_h {
                    dw_neg[i * n_h + j] += v_neg[i] * h_probs_neg[j];
                }
            }
            for j in 0..n_h {
                dh_neg[j] += h_probs_neg[j];
            }
        }
        let chains_f = self.num_chains as f64;
        // Apply gradients
        for i in 0..n_v {
            for j in 0..n_h {
                self.rbm.weights[i * n_h + j] +=
                    lr * (dw_pos[i * n_h + j] / bs - dw_neg[i * n_h + j] / chains_f);
            }
            self.rbm.visible_bias[i] += lr * (dv_pos[i] / bs - dv_neg[i] / chains_f);
        }
        for j in 0..n_h {
            self.rbm.hidden_bias[j] += lr * (dh_pos[j] / bs - dh_neg[j] / chains_f);
        }
        // Reconstruction error on the batch
        let recon_err: f64 = batch
            .iter()
            .map(|v| {
                let h = self.rbm.hidden_probs(v);
                let v_recon = self.rbm.visible_probs(&h);
                v.iter()
                    .zip(&v_recon)
                    .map(|(&vi, &ri)| (vi - ri).powi(2))
                    .sum::<f64>()
                    / n_v as f64
            })
            .sum::<f64>()
            / bs;
        Ok(recon_err)
    }

    /// Train for one full epoch.
    pub fn train_epoch(&mut self, data: &[Vec<f64>], batch_size: usize) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        let actual_bs = batch_size.max(1);
        let mut total_err = 0.0f64;
        let mut count = 0usize;
        let mut start = 0;
        while start < data.len() {
            let end = (start + actual_bs).min(data.len());
            let batch = &data[start..end];
            total_err += self.train_step(batch)?;
            count += 1;
            start = end;
        }
        Ok(if count > 0 { total_err / count as f64 } else { 0.0 })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rbm_creation() {
        let rbm = RestrictedBoltzmannMachine::new(4, 3, 0.01, 1).expect("RBM creation failed");
        assert_eq!(rbm.n_visible, 4);
        assert_eq!(rbm.n_hidden, 3);
    }

    #[test]
    fn test_rbm_hidden_probs_range() {
        let rbm = RestrictedBoltzmannMachine::new(4, 3, 0.01, 1).expect("RBM creation");
        let v = vec![1.0, 0.0, 1.0, 0.0];
        let probs = rbm.hidden_probs(&v);
        assert_eq!(probs.len(), 3);
        for &p in &probs {
            assert!(p >= 0.0 && p <= 1.0, "prob {p} out of range");
        }
    }

    #[test]
    fn test_rbm_train_step() {
        let mut rbm = RestrictedBoltzmannMachine::new(4, 3, 0.01, 1).expect("RBM creation");
        let v = vec![1.0, 0.0, 1.0, 0.0];
        let err = rbm.train_step(&v).expect("train step failed");
        assert!(err >= 0.0, "reconstruction error must be non-negative");
    }

    #[test]
    fn test_rbm_free_energy() {
        let rbm = RestrictedBoltzmannMachine::new(4, 3, 0.01, 1).expect("RBM creation");
        let v = vec![0.5, 0.5, 0.5, 0.5];
        let fe = rbm.free_energy(&v);
        assert!(fe.is_finite(), "free energy must be finite");
    }

    #[test]
    fn test_dbm_creation_and_pretrain() {
        let mut dbm =
            DeepBoltzmannMachine::new(vec![4, 3, 2], 0.01, 1).expect("DBM creation failed");
        let data = vec![
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
        ];
        dbm.pretrain(&data, 2).expect("pretraining failed");
        let enc = dbm.encode(&data[0]).expect("encode failed");
        assert_eq!(enc.len(), 2);
    }

    #[test]
    fn test_cd_k_pseudo_log_likelihood() {
        let cdk = ContrastiveDivergenceK::new(4, 3, 0.01, 1).expect("CDk creation");
        let v = vec![1.0, 0.0, 1.0, 0.0];
        let pll = cdk.pseudo_log_likelihood(&v).expect("pll failed");
        assert!(pll.is_finite(), "PLL should be finite");
    }

    #[test]
    fn test_persistent_cd_train() {
        let mut pcd = PersistentCD::new(4, 3, 0.01, 4, 1).expect("PCD creation");
        let data = vec![
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
        ];
        let err = pcd.train_epoch(&data, 2).expect("PCD epoch failed");
        assert!(err >= 0.0);
    }

    struct QuadraticEnergy {
        center: Vec<f64>,
    }

    impl EnergyFunction for QuadraticEnergy {
        fn energy(&self, x: &[f64]) -> f64 {
            x.iter()
                .zip(&self.center)
                .map(|(&xi, &ci)| (xi - ci).powi(2))
                .sum()
        }
    }

    #[test]
    fn test_ebm_langevin_converges() {
        let energy_fn = Box::new(QuadraticEnergy {
            center: vec![1.0, 2.0],
        });
        let cfg = LangevinConfig {
            step_size: 0.05,
            num_steps: 100,
            noise_scale: 0.01,
            grad_clip: 5.0,
            fd_eps: 1e-3,
        };
        let mut ebm = EnergyBasedModel::new(energy_fn, cfg);
        let x_init = vec![0.0, 0.0];
        let sample = ebm.langevin_sample(&x_init);
        // With low noise the sample should move toward the center
        assert!(
            (sample[0] - 1.0).abs() < 0.5,
            "Langevin should converge: x[0]={}", sample[0]
        );
        assert!(
            (sample[1] - 2.0).abs() < 0.5,
            "Langevin should converge: x[1]={}", sample[1]
        );
    }
}
