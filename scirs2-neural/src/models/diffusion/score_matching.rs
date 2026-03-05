//! Score Matching for generative modelling
//!
//! Implements two principled objectives for training score-based models:
//!
//! ## Denoising Score Matching (DSM)
//! Vincent (2011) shows that minimising
//! ```text
//! J_DSM(θ) = E_{q_σ(x̃|x) p_data(x)} [ ||s_θ(x̃, σ) + (x̃ - x)/σ² ||² ]
//! ```
//! is equivalent to minimising the Fisher divergence between the model score
//! and the data score, making it a tractable proxy for explicit score matching.
//!
//! ## Sliced Score Matching (SSM)
//! Song et al. (2019) further reduce the cost of Hutchinson-trace estimation:
//! ```text
//! J_SSM(θ) = E_{p_v} E_{p_data} [ v^T ∇_x s_θ(x) v + ½ ||s_θ(x)||² ]
//! ```
//! where `v` is a random projection vector (Rademacher or Gaussian).
//!
//! ## References
//! - "A Connection Between Score Matching and Denoising Autoencoders", Vincent (2011)
//!   <https://arxiv.org/abs/1206.3699>
//! - "Sliced Score Matching: A Scalable Approach to Density and Score Estimation",
//!   Song, Garg, Shi & Ermon (2019) <https://arxiv.org/abs/1905.07088>
//! - "Score-Based Generative Modeling through Stochastic Differential Equations",
//!   Song, Sohl-Dickstein, Kingma, Kumar, Ermon & Poole (2021)
//!   <https://arxiv.org/abs/2011.13456>

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// ScoreNetworkConfig
// ---------------------------------------------------------------------------

/// Configuration for the score network and score matching objectives.
#[derive(Debug, Clone)]
pub struct ScoreNetworkConfig {
    /// Dimensionality of the input data.
    pub data_dim: usize,
    /// Hidden layer width.
    pub hidden_dim: usize,
    /// Number of hidden layers.
    pub num_layers: usize,
    /// Number of noise levels σ in the annealed multi-scale schedule.
    pub num_noise_levels: usize,
    /// Minimum noise standard deviation σ_min.
    pub sigma_min: f64,
    /// Maximum noise standard deviation σ_max.
    pub sigma_max: f64,
    /// Random seed for deterministic weight init.
    pub seed: u64,
}

impl ScoreNetworkConfig {
    /// Default config suitable for low-dimensional data.
    pub fn default_config(data_dim: usize) -> Self {
        Self {
            data_dim,
            hidden_dim: 128,
            num_layers: 3,
            num_noise_levels: 10,
            sigma_min: 0.01,
            sigma_max: 1.0,
            seed: 42,
        }
    }

    /// Minimal config for unit testing.
    pub fn tiny(data_dim: usize) -> Self {
        Self {
            data_dim,
            hidden_dim: 16,
            num_layers: 2,
            num_noise_levels: 5,
            sigma_min: 0.05,
            sigma_max: 0.5,
            seed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ScoreFunction trait
// ---------------------------------------------------------------------------

/// Trait for score function estimators: `s_θ(x, σ) ≈ ∇_x log p_σ(x)`.
pub trait ScoreFunction: Send + Sync + std::fmt::Debug {
    /// Evaluate the score `s_θ(x, σ)`.
    ///
    /// Returns a vector of the same length as `x`.
    fn score(&self, x: &[f64], sigma: f64) -> Result<Vec<f64>>;

    /// Number of trainable parameters.
    fn parameter_count(&self) -> usize;
}

// ---------------------------------------------------------------------------
// ScoreNetwork — MLP-based score function estimator
// ---------------------------------------------------------------------------

/// MLP-based score network `s_θ(x, σ)`.
///
/// Architecture:
/// - Input: `[x; log(σ)]` of length `data_dim + 1`
/// - Hidden layers: fully-connected with SiLU activation
/// - Output: `data_dim` (same shape as input)
/// - Output is normalised by `1/σ` to satisfy the equivariance property
///
/// The network is initialised with deterministic Xavier-like weights derived
/// from a simple LCG for reproducibility without external RNG dependencies.
#[derive(Debug, Clone)]
pub struct ScoreNetwork {
    /// Configuration
    pub config: ScoreNetworkConfig,
    /// Layer weights: each entry is (weight_matrix, bias_vector).
    /// Stored row-major: weight[j * in_dim + i] = W[j,i].
    layers: Vec<(Vec<f64>, Vec<f64>)>,
}

impl ScoreNetwork {
    /// Create a new `ScoreNetwork` with deterministic weight initialisation.
    pub fn new(config: ScoreNetworkConfig) -> Result<Self> {
        if config.data_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "ScoreNetwork: data_dim must be > 0".to_string(),
            ));
        }
        if config.hidden_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "ScoreNetwork: hidden_dim must be > 0".to_string(),
            ));
        }
        if config.num_layers == 0 {
            return Err(NeuralError::InvalidArgument(
                "ScoreNetwork: num_layers must be > 0".to_string(),
            ));
        }
        // +1 for log(σ) conditioning
        let in_dim = config.data_dim + 1;
        let layers = Self::init_layers(in_dim, config.hidden_dim, config.data_dim, config.num_layers, config.seed);
        Ok(Self { config, layers })
    }

    /// Xavier-uniform initialisation via LCG PRNG.
    fn lcg_sample(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Map to (-1, 1)
        let bits = *state >> 11;
        (bits as f64) / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    fn init_layers(
        in_dim: usize,
        hidden: usize,
        out_dim: usize,
        num_layers: usize,
        seed: u64,
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let mut rng = seed.wrapping_add(0xdeadbeef);
        let mut layers = Vec::with_capacity(num_layers + 1);

        // First layer: in_dim -> hidden
        let limit = (6.0 / (in_dim + hidden) as f64).sqrt();
        let w: Vec<f64> = (0..in_dim * hidden)
            .map(|_| Self::lcg_sample(&mut rng) * limit)
            .collect();
        layers.push((w, vec![0.0f64; hidden]));

        // Intermediate hidden layers: hidden -> hidden
        for _ in 1..num_layers {
            let limit = (6.0 / (hidden + hidden) as f64).sqrt();
            let w: Vec<f64> = (0..hidden * hidden)
                .map(|_| Self::lcg_sample(&mut rng) * limit)
                .collect();
            layers.push((w, vec![0.0f64; hidden]));
        }

        // Output layer: hidden -> out_dim (zero-init bias)
        let limit = (6.0 / (hidden + out_dim) as f64).sqrt();
        let w: Vec<f64> = (0..hidden * out_dim)
            .map(|_| Self::lcg_sample(&mut rng) * limit)
            .collect();
        layers.push((w, vec![0.0f64; out_dim]));

        layers
    }

    /// SiLU (swish) activation: x * sigmoid(x).
    fn silu(x: f64) -> f64 {
        x / (1.0 + (-x).exp())
    }

    /// Forward pass of the MLP (without σ normalisation).
    fn mlp_forward(&self, inp: &[f64]) -> Vec<f64> {
        let mut h = inp.to_vec();
        let n = self.layers.len();
        for (idx, (w, b)) in self.layers.iter().enumerate() {
            let out_dim = b.len();
            let in_dim = h.len();
            let mut next = vec![0.0f64; out_dim];
            for j in 0..out_dim {
                let mut s = b[j];
                for i in 0..in_dim {
                    let wi = j * in_dim + i;
                    if wi < w.len() {
                        s += w[wi] * h[i];
                    }
                }
                next[j] = s;
            }
            // SiLU on all but last layer
            if idx < n - 1 {
                for v in &mut next {
                    *v = Self::silu(*v);
                }
            }
            h = next;
        }
        h
    }

    /// Finite-difference Jacobian-vector product approximation.
    ///
    /// Computes `J^T v ≈ (s(x + eps*v, σ) - s(x - eps*v, σ)) / (2*eps)`,
    /// used in the SSM divergence term.
    pub fn jvp_approx(&self, x: &[f64], v: &[f64], sigma: f64) -> Result<Vec<f64>> {
        const EPS: f64 = 1e-4;
        let d = x.len();
        if v.len() != d {
            return Err(NeuralError::ShapeMismatch(format!(
                "ScoreNetwork jvp_approx: x len {} != v len {}",
                d,
                v.len()
            )));
        }
        let x_plus: Vec<f64> = x.iter().zip(v).map(|(&xi, &vi)| xi + EPS * vi).collect();
        let x_minus: Vec<f64> = x.iter().zip(v).map(|(&xi, &vi)| xi - EPS * vi).collect();
        let s_plus = self.score(&x_plus, sigma)?;
        let s_minus = self.score(&x_minus, sigma)?;
        // v^T (J^T v) = (s_+ - s_-) / (2*eps) dotted with v
        let jvp: Vec<f64> = s_plus
            .iter()
            .zip(&s_minus)
            .map(|(&sp, &sm)| (sp - sm) / (2.0 * EPS))
            .collect();
        Ok(jvp)
    }

    /// Geometric noise level sequence: σ_i = σ_min * (σ_max/σ_min)^{i/(L-1)}.
    pub fn noise_levels(&self) -> Vec<f64> {
        let l = self.config.num_noise_levels.max(1);
        if l == 1 {
            return vec![self.config.sigma_max];
        }
        let ratio = self.config.sigma_max / self.config.sigma_min.max(1e-12);
        (0..l)
            .map(|i| self.config.sigma_min * ratio.powf(i as f64 / (l - 1) as f64))
            .collect()
    }
}

impl ScoreFunction for ScoreNetwork {
    /// Evaluate `s_θ(x, σ) = MLP([x; log(σ)]) / σ`.
    fn score(&self, x: &[f64], sigma: f64) -> Result<Vec<f64>> {
        if x.len() != self.config.data_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "ScoreNetwork: input dim {} != data_dim {}",
                x.len(),
                self.config.data_dim
            )));
        }
        if sigma <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "ScoreNetwork: sigma must be > 0, got {sigma}"
            )));
        }
        let mut inp = x.to_vec();
        inp.push(sigma.ln());
        let raw = self.mlp_forward(&inp);
        // Normalise by σ so score has correct dimensions
        Ok(raw.iter().map(|&v| v / sigma).collect())
    }

    fn parameter_count(&self) -> usize {
        self.layers.iter().map(|(w, b)| w.len() + b.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// DenoisingScoreMatching (DSM)
// ---------------------------------------------------------------------------

/// Denoising Score Matching objective.
///
/// Given a data sample `x` and a noise level `σ`, we:
/// 1. Sample `x̃ = x + σ * ε` where `ε ~ N(0,I)`
/// 2. The optimal denoising score is `s*(x̃, σ) = -(x̃ - x) / σ²`
/// 3. The DSM loss is `||s_θ(x̃, σ) - s*(x̃, σ)||² = ||s_θ(x̃, σ) + (x̃ - x)/σ²||²`
///
/// Training over multiple noise levels (annealed DSM) yields a single network
/// that estimates the score at all scales, enabling MCMC-based generation
/// via Langevin dynamics.
#[derive(Debug)]
pub struct DenoisingScoreMatching {
    rng_state: u64,
}

impl DenoisingScoreMatching {
    /// Create a new DSM trainer with the given random seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng_state: seed.wrapping_add(0xfeedface),
        }
    }

    fn sample_normal(&mut self) -> f64 {
        // Box-Muller transform with LCG
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

    /// Compute the DSM loss for a single data point `x` at noise level `σ`.
    ///
    /// Returns `(loss, x_tilde)` where `x_tilde` is the noisy sample used.
    pub fn compute_loss(
        &mut self,
        x: &[f64],
        sigma: f64,
        score_fn: &dyn ScoreFunction,
    ) -> Result<(f64, Vec<f64>)> {
        let d = x.len();
        if sigma <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "DSM: sigma must be > 0, got {sigma}"
            )));
        }
        // Sample noise ε ~ N(0,I)
        let eps: Vec<f64> = (0..d).map(|_| self.sample_normal()).collect();
        // Perturbed sample x̃ = x + σ ε
        let x_tilde: Vec<f64> = x.iter().zip(&eps).map(|(&xi, &ei)| xi + sigma * ei).collect();
        // Optimal score: -(x̃ - x)/σ² = -ε/σ
        // Network score s_θ(x̃, σ)
        let s_pred = score_fn.score(&x_tilde, sigma)?;
        // DSM loss: ||s_θ + (x̃-x)/σ²||² = ||s_θ + ε/σ||²
        let loss: f64 = s_pred
            .iter()
            .zip(&eps)
            .map(|(&s, &e)| {
                let residual = s + e / sigma;
                residual * residual
            })
            .sum::<f64>()
            / d as f64;
        Ok((loss, x_tilde))
    }

    /// Compute the annealed DSM loss over all noise levels.
    ///
    /// Samples one random noise level per call and returns the weighted loss.
    /// The weighting `λ(σ) = σ²` (used in Song & Ermon 2019) balances contributions
    /// from different scales.
    pub fn annealed_loss(
        &mut self,
        x: &[f64],
        score_net: &ScoreNetwork,
    ) -> Result<f64> {
        let sigmas = score_net.noise_levels();
        let l = sigmas.len();
        if l == 0 {
            return Ok(0.0);
        }
        // Pick a random noise level
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let idx = (self.rng_state >> 33) as usize % l;
        let sigma = sigmas[idx];
        let (loss, _x_tilde) = self.compute_loss(x, sigma, score_net)?;
        // Weight by σ² (matches NCSN objective weighting)
        Ok(loss * sigma * sigma)
    }
}

// ---------------------------------------------------------------------------
// SlicedScoreMatching (SSM)
// ---------------------------------------------------------------------------

/// Sliced Score Matching objective.
///
/// Estimates the Fisher divergence without requiring the Jacobian trace by
/// projecting onto random directions:
/// ```text
/// J_SSM(θ) = E_v E_x [ v^T ∇_x s_θ(x) v + ½ ||s_θ(x)||² ]
/// ```
///
/// The divergence term `v^T ∇_x s_θ(x) v` is computed via the JVP approximation
/// (finite differences), so this is fully autodiff-free.
///
/// Two projection distributions are supported:
/// - **Rademacher**: `v_i ∈ {-1, +1}` with equal probability (variance-optimal)
/// - **Gaussian**: `v ~ N(0, I)` (unbiased, slightly higher variance)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjectionDist {
    /// Rademacher ±1 projections (variance-optimal for SSM).
    Rademacher,
    /// Standard Gaussian projections.
    Gaussian,
}

/// Sliced Score Matching trainer.
#[derive(Debug)]
pub struct SlicedScoreMatching {
    /// Number of projection directions per sample.
    pub num_projections: usize,
    /// Projection distribution.
    pub proj_dist: ProjectionDist,
    rng_state: u64,
}

impl SlicedScoreMatching {
    /// Create a new SSM trainer.
    pub fn new(num_projections: usize, proj_dist: ProjectionDist, seed: u64) -> Result<Self> {
        if num_projections == 0 {
            return Err(NeuralError::InvalidArgument(
                "SSM: num_projections must be > 0".to_string(),
            ));
        }
        Ok(Self {
            num_projections,
            proj_dist,
            rng_state: seed.wrapping_add(0xc0ffee42),
        })
    }

    fn sample_rademacher(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        if self.rng_state >> 63 == 0 { 1.0 } else { -1.0 }
    }

    fn sample_gaussian(&mut self) -> f64 {
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

    fn sample_projection(&mut self, dim: usize) -> Vec<f64> {
        match self.proj_dist {
            ProjectionDist::Rademacher => (0..dim).map(|_| self.sample_rademacher()).collect(),
            ProjectionDist::Gaussian => (0..dim).map(|_| self.sample_gaussian()).collect(),
        }
    }

    /// Compute the SSM loss for a single data point `x`.
    ///
    /// `J_SSM = E_v [ v^T ∇_x s_θ(x) v + ½ ||s_θ(x)||² ]`
    ///
    /// The divergence term is estimated via finite-difference JVP.
    pub fn compute_loss(
        &mut self,
        x: &[f64],
        sigma: f64,
        score_net: &ScoreNetwork,
    ) -> Result<f64> {
        let d = x.len();
        let s = score_net.score(x, sigma)?;
        // ½ ||s_θ||²
        let half_sq_norm: f64 = s.iter().map(|&si| si * si).sum::<f64>() / (2.0 * d as f64);

        // Average v^T ∇s v over projections
        let mut div_term = 0.0f64;
        for _ in 0..self.num_projections {
            let v = self.sample_projection(d);
            // JVP: ∇_x s(x) v via finite difference
            let jvp = score_net.jvp_approx(x, &v, sigma)?;
            // v^T (∇_x s v) = sum_i v_i * jvp_i
            let vt_jvp: f64 = v.iter().zip(&jvp).map(|(&vi, &ji)| vi * ji).sum();
            div_term += vt_jvp;
        }
        div_term /= self.num_projections as f64;

        Ok(div_term + half_sq_norm)
    }

    /// Run SSM over a dataset for one epoch, returning mean loss.
    pub fn train_epoch(
        &mut self,
        data: &[Vec<f64>],
        sigma: f64,
        score_net: &ScoreNetwork,
    ) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        let total: f64 = data
            .iter()
            .map(|x| self.compute_loss(x, sigma, score_net))
            .collect::<Result<Vec<f64>>>()?
            .iter()
            .sum();
        Ok(total / data.len() as f64)
    }
}

// ---------------------------------------------------------------------------
// Langevin dynamics sampler (for score-based generation)
// ---------------------------------------------------------------------------

/// Configuration for Langevin dynamics sampling.
#[derive(Debug, Clone)]
pub struct LangevinConfig {
    /// Number of Langevin steps per noise level.
    pub steps_per_level: usize,
    /// Step-size coefficient ε (actual step size = ε * σ²/σ_max²).
    pub step_size_coeff: f64,
    /// Whether to add Langevin noise (false = gradient ascent only).
    pub add_noise: bool,
    /// Random seed.
    pub seed: u64,
}

impl Default for LangevinConfig {
    fn default() -> Self {
        Self {
            steps_per_level: 100,
            step_size_coeff: 1e-5,
            add_noise: true,
            seed: 12345,
        }
    }
}

/// Annealed Langevin dynamics sampler.
///
/// Runs MCMC sampling at each noise level σ_i following the schedule
/// from noisiest to cleanest:
/// ```text
/// x_{k+1} = x_k + α_i s_θ(x_k, σ_i) + √(2αᵢ) z_k
/// ```
/// where `α_i = ε * σ_i² / σ_L²` and `z_k ~ N(0,I)`.
#[derive(Debug)]
pub struct AnnealedLangevin {
    /// Config
    pub config: LangevinConfig,
    rng_state: u64,
}

impl AnnealedLangevin {
    /// Create a new sampler.
    pub fn new(config: LangevinConfig) -> Self {
        let rng = config.seed.wrapping_add(0xabcdef01);
        Self { config, rng_state: rng }
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

    /// Run annealed Langevin dynamics to produce a sample.
    ///
    /// # Arguments
    /// * `x_init` – initial sample (typically pure Gaussian noise)
    /// * `score_net` – trained score network
    ///
    /// # Returns
    /// The final sample after all annealing steps.
    pub fn sample(
        &mut self,
        x_init: &[f64],
        score_net: &ScoreNetwork,
    ) -> Result<Vec<f64>> {
        let sigmas = score_net.noise_levels();
        if sigmas.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "AnnealedLangevin: no noise levels".to_string(),
            ));
        }
        let sigma_max = sigmas.last().copied().unwrap_or(1.0);
        let sigma_max_sq = sigma_max * sigma_max;
        let d = x_init.len();
        let mut x = x_init.to_vec();

        // Anneal from high σ to low σ (sigmas already in ascending order, so reverse)
        for &sigma in sigmas.iter().rev() {
            let alpha = self.config.step_size_coeff * sigma * sigma / sigma_max_sq.max(1e-12);
            let noise_std = (2.0 * alpha).sqrt();
            for _ in 0..self.config.steps_per_level {
                let score = score_net.score(&x, sigma)?;
                // Gradient ascent step
                let mut x_new: Vec<f64> = x
                    .iter()
                    .zip(&score)
                    .map(|(&xi, &si)| xi + alpha * si)
                    .collect();
                // Add Langevin noise
                if self.config.add_noise {
                    for xi in x_new.iter_mut() {
                        *xi += noise_std * self.sample_normal();
                    }
                }
                // Reflect boundary check (optional; keeps x bounded)
                for i in 0..d {
                    if !x_new[i].is_finite() {
                        x_new[i] = x[i];
                    }
                }
                x = x_new;
            }
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

    #[test]
    fn test_score_network_creation() {
        let cfg = ScoreNetworkConfig::tiny(4);
        let net = ScoreNetwork::new(cfg).expect("score network creation");
        assert!(net.parameter_count() > 0);
    }

    #[test]
    fn test_score_network_output_shape() {
        let cfg = ScoreNetworkConfig::tiny(4);
        let net = ScoreNetwork::new(cfg).expect("network");
        let x = vec![0.1, -0.2, 0.3, -0.4];
        let s = net.score(&x, 0.1).expect("score evaluation");
        assert_eq!(s.len(), 4);
        for &v in &s {
            assert!(v.is_finite(), "score not finite: {v}");
        }
    }

    #[test]
    fn test_score_network_sigma_scaling() {
        // score(x, σ) should scale with 1/σ relative to MLP output
        let cfg = ScoreNetworkConfig::tiny(4);
        let net = ScoreNetwork::new(cfg).expect("network");
        let x = vec![0.1, -0.2, 0.3, -0.4];
        let s1 = net.score(&x, 0.1).expect("score at σ=0.1");
        let s2 = net.score(&x, 0.2).expect("score at σ=0.2");
        // The log(σ) conditioning changes the MLP output, so we just verify
        // that outputs are finite and different across scales
        assert_ne!(s1[0], s2[0]);
        for &v in s1.iter().chain(s2.iter()) {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_dsm_loss() {
        let cfg = ScoreNetworkConfig::tiny(4);
        let net = ScoreNetwork::new(cfg).expect("network");
        let mut dsm = DenoisingScoreMatching::new(42);
        let x = vec![0.5, -0.3, 0.2, 0.8];
        let (loss, x_tilde) = dsm.compute_loss(&x, 0.1, &net).expect("DSM loss");
        assert!(loss >= 0.0 && loss.is_finite(), "DSM loss invalid: {loss}");
        assert_eq!(x_tilde.len(), 4);
    }

    #[test]
    fn test_dsm_annealed() {
        let cfg = ScoreNetworkConfig::tiny(4);
        let net = ScoreNetwork::new(cfg).expect("network");
        let mut dsm = DenoisingScoreMatching::new(0);
        let x = vec![0.5, -0.3, 0.2, 0.8];
        let loss = dsm.annealed_loss(&x, &net).expect("annealed loss");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_ssm_loss() {
        let cfg = ScoreNetworkConfig::tiny(4);
        let net = ScoreNetwork::new(cfg).expect("network");
        let mut ssm = SlicedScoreMatching::new(4, ProjectionDist::Rademacher, 99)
            .expect("SSM creation");
        let x = vec![0.5, -0.3, 0.2, 0.8];
        let loss = ssm.compute_loss(&x, 0.1, &net).expect("SSM loss");
        assert!(loss.is_finite(), "SSM loss not finite: {loss}");
    }

    #[test]
    fn test_ssm_gaussian_projections() {
        let cfg = ScoreNetworkConfig::tiny(4);
        let net = ScoreNetwork::new(cfg).expect("network");
        let mut ssm = SlicedScoreMatching::new(2, ProjectionDist::Gaussian, 7)
            .expect("SSM");
        let data: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 * 0.1; 4]).collect();
        let loss = ssm.train_epoch(&data, 0.2, &net).expect("epoch");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_noise_levels_geometric() {
        let cfg = ScoreNetworkConfig {
            num_noise_levels: 5,
            sigma_min: 0.1,
            sigma_max: 1.0,
            ..ScoreNetworkConfig::tiny(4)
        };
        let net = ScoreNetwork::new(cfg).expect("network");
        let levels = net.noise_levels();
        assert_eq!(levels.len(), 5);
        assert!((levels[0] - 0.1).abs() < 1e-9);
        assert!((levels[4] - 1.0).abs() < 1e-9);
        // Geometrically increasing
        for i in 1..5 {
            assert!(levels[i] > levels[i - 1], "noise levels not increasing");
        }
    }

    #[test]
    fn test_annealed_langevin() {
        let cfg = ScoreNetworkConfig::tiny(4);
        let net = ScoreNetwork::new(cfg).expect("network");
        let langevin_cfg = LangevinConfig {
            steps_per_level: 3,
            step_size_coeff: 1e-5,
            add_noise: true,
            seed: 0,
        };
        let mut sampler = AnnealedLangevin::new(langevin_cfg);
        let x_init = vec![0.0; 4];
        let sample = sampler.sample(&x_init, &net).expect("langevin sample");
        assert_eq!(sample.len(), 4);
        for &v in &sample {
            assert!(v.is_finite(), "sample not finite: {v}");
        }
    }

    #[test]
    fn test_ssm_zero_projection_error() {
        let result = SlicedScoreMatching::new(0, ProjectionDist::Rademacher, 42);
        assert!(result.is_err());
    }
}
