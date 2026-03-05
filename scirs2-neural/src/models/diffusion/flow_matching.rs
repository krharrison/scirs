//! Flow Matching for generative modelling
//!
//! Implements Conditional Flow Matching (CFM) and Optimal Transport (OT) Flow Matching,
//! both training a time-dependent vector field `v_t(x)` that transports a source
//! distribution p₀ (typically Gaussian) to the target data distribution p₁.
//!
//! ## Core Idea
//! Unlike diffusion models, flow matching trains a continuous normalizing flow
//! defined by the ODE:
//! ```text
//! dx/dt = v_t(x),  x(0) ~ N(0,I),  x(1) ~ p_data
//! ```
//! by regressing `v_t` onto the **conditional** vector field that generates each
//! data point, avoiding the intractable marginal flow:
//! ```text
//! L_CFM = E_{t,x₀,x₁} [ ||v_θ(x_t, t) - (x₁ - (1-σ)x₀) / (1 - (1-σ)t)||² ]
//! ```
//! where `x_t = (1 - (1-σ)t) x₀ + t x₁` is the linear interpolant.
//!
//! ## Optimal Transport Flow Matching
//! OT-CFM (Tong et al. 2023) uses mini-batch OT couplings between samples from p₀
//! and p₁ to construct near-straight conditional flows, significantly reducing
//! the number of NFE (neural function evaluations) needed for sampling.
//!
//! ## References
//! - "Flow Matching for Generative Modeling", Lipman et al. (2022)
//!   <https://arxiv.org/abs/2210.02747>
//! - "Improving and Generalizing Flow-Matching", Tong et al. (2023)
//!   <https://arxiv.org/abs/2302.00482>
//! - "Conditional Flow Matching: Simulation-Free Dynamic Optimal Transport",
//!   Albergo, Vanden-Eijnden (2022) <https://arxiv.org/abs/2210.02747>

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// VectorField trait
// ---------------------------------------------------------------------------

/// Trait for a time-conditioned vector field `v_t(x)`.
///
/// The vector field defines the ODE `dx/dt = v_t(x)` that transforms
/// the source distribution into the target.
pub trait VectorField: Send + Sync + std::fmt::Debug {
    /// Evaluate the vector field at sample `x` and time `t ∈ [0,1]`.
    ///
    /// Returns a vector of the same length as `x`.
    fn forward(&self, x: &[f64], t: f64) -> Result<Vec<f64>>;

    /// Number of trainable parameters.
    fn parameter_count(&self) -> usize;
}

// ---------------------------------------------------------------------------
// SimpleVectorFieldNet — MLP-based vector field estimator
// ---------------------------------------------------------------------------

/// MLP-based vector field network `v_θ(x, t)`.
///
/// Architecture:
/// - Input: `[x; sin(2π·t); cos(2π·t)]` of length `data_dim + 2`
///   (Fourier time embedding improves frequency coverage)
/// - Hidden layers: fully-connected with GELU activation
/// - Output: `data_dim`
#[derive(Debug, Clone)]
pub struct SimpleVectorFieldNet {
    /// Data dimensionality
    pub data_dim: usize,
    /// Layer weights (W, b) in row-major order
    layers: Vec<(Vec<f64>, Vec<f64>)>,
}

impl SimpleVectorFieldNet {
    /// Create a new `SimpleVectorFieldNet`.
    pub fn new(data_dim: usize, hidden_dim: usize, num_layers: usize, seed: u64) -> Result<Self> {
        if data_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "VectorFieldNet: data_dim must be > 0".to_string(),
            ));
        }
        if hidden_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "VectorFieldNet: hidden_dim must be > 0".to_string(),
            ));
        }
        if num_layers == 0 {
            return Err(NeuralError::InvalidArgument(
                "VectorFieldNet: num_layers must be > 0".to_string(),
            ));
        }
        // +2 for sin/cos time embedding
        let in_dim = data_dim + 2;
        let layers = Self::init_layers(in_dim, hidden_dim, data_dim, num_layers, seed);
        Ok(Self { data_dim, layers })
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
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
        let mut rng = seed.wrapping_add(0xcafebabe);
        let mut layers = Vec::with_capacity(num_layers + 1);
        // First: in_dim -> hidden
        let lim = (2.0 / in_dim as f64).sqrt();
        let w: Vec<f64> = (0..in_dim * hidden).map(|_| Self::lcg(&mut rng) * lim).collect();
        layers.push((w, vec![0.0f64; hidden]));
        // Intermediate: hidden -> hidden
        for _ in 1..num_layers {
            let lim = (2.0 / hidden as f64).sqrt();
            let w: Vec<f64> = (0..hidden * hidden).map(|_| Self::lcg(&mut rng) * lim).collect();
            layers.push((w, vec![0.0f64; hidden]));
        }
        // Output: hidden -> out_dim (zero init for stable training start)
        let lim = (2.0 / (hidden + out_dim) as f64).sqrt();
        let w: Vec<f64> = (0..hidden * out_dim).map(|_| Self::lcg(&mut rng) * lim * 0.01).collect();
        layers.push((w, vec![0.0f64; out_dim]));
        layers
    }

    /// GELU activation: x * Φ(x), approximated via tanh.
    fn gelu(x: f64) -> f64 {
        // Fast GELU: x * 0.5 * (1 + tanh(√(2/π)(x + 0.044715 x³)))
        let inner = (2.0f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }

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
            if idx < n - 1 {
                for v in &mut next {
                    *v = Self::gelu(*v);
                }
            }
            h = next;
        }
        h
    }
}

impl VectorField for SimpleVectorFieldNet {
    fn forward(&self, x: &[f64], t: f64) -> Result<Vec<f64>> {
        if x.len() != self.data_dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "VectorFieldNet: input len {} != data_dim {}",
                x.len(),
                self.data_dim
            )));
        }
        if !(0.0..=1.0).contains(&t) {
            return Err(NeuralError::InvalidArgument(format!(
                "VectorFieldNet: t must be in [0,1], got {t}"
            )));
        }
        let mut inp = x.to_vec();
        inp.push((2.0 * std::f64::consts::PI * t).sin());
        inp.push((2.0 * std::f64::consts::PI * t).cos());
        Ok(self.mlp_forward(&inp))
    }

    fn parameter_count(&self) -> usize {
        self.layers.iter().map(|(w, b)| w.len() + b.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// FlowMatchingObjective enum
// ---------------------------------------------------------------------------

/// Which flow matching objective to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowMatchingObjective {
    /// Standard Conditional Flow Matching (linear interpolant + small noise σ).
    ConditionalFlowMatching,
    /// Optimal Transport CFM: uses uniform OT coupling via Sinkhorn on mini-batches.
    OptimalTransport,
    /// Variance-Exploding CFM: the noise schedule explodes as t→1.
    VarianceExploding,
}

// ---------------------------------------------------------------------------
// FlowMatcher
// ---------------------------------------------------------------------------

/// Flow Matching trainer.
///
/// Computes the CFM/OT-CFM training objective and provides an ODE-based sampler.
///
/// ### CFM conditional path
/// Given source `x₀ ~ N(0,I)` and target `x₁ ~ p_data`, the linear interpolant is:
/// ```text
/// x_t = (1 - (1-σ)t) · x₀ + t · x₁
/// ```
/// and the conditional vector field is:
/// ```text
/// u_t(x_t | x₀, x₁) = (x₁ - (1-σ) · x₀) / (1 - (1-σ)t)
/// ```
///
/// ### OT-CFM
/// For OT-CFM, we additionally solve a mini-batch linear assignment problem to
/// create near-optimal couplings between source and target samples, producing
/// straighter flows and fewer NFEs.
#[derive(Debug)]
pub struct FlowMatcher {
    /// Data dimensionality
    pub data_dim: usize,
    /// Noise level σ (controls how much source noise remains at t=1)
    pub sigma: f64,
    /// Which objective to use
    pub objective: FlowMatchingObjective,
    rng_state: u64,
}

impl FlowMatcher {
    /// Create a new `FlowMatcher`.
    pub fn new(data_dim: usize, sigma: f64, objective: FlowMatchingObjective) -> Result<Self> {
        if data_dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "FlowMatcher: data_dim must be > 0".to_string(),
            ));
        }
        if sigma < 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "FlowMatcher: sigma must be >= 0, got {sigma}"
            )));
        }
        Ok(Self {
            data_dim,
            sigma,
            objective,
            rng_state: 0xdeadbeef_12345678u64,
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

    fn sample_uniform_01(&mut self) -> f64 {
        self.rng_state = self
            .rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.rng_state >> 11) as f64) / (1u64 << 53) as f64
    }

    /// Compute the linear interpolant `x_t = (1 - (1-σ)t) x₀ + t x₁`.
    pub fn interpolate(&self, x0: &[f64], x1: &[f64], t: f64) -> Result<Vec<f64>> {
        if x0.len() != x1.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "FlowMatcher: x0 len {} != x1 len {}",
                x0.len(),
                x1.len()
            )));
        }
        let coeff0 = 1.0 - (1.0 - self.sigma) * t;
        Ok(x0
            .iter()
            .zip(x1)
            .map(|(&a, &b)| coeff0 * a + t * b)
            .collect())
    }

    /// Compute the conditional vector field at `(x_t, t)` given `(x₀, x₁)`.
    ///
    /// `u_t(x_t | x₀, x₁) = (x₁ - (1-σ) x₀) / (1 - (1-σ)t)`
    pub fn conditional_vector_field(
        &self,
        x0: &[f64],
        x1: &[f64],
        t: f64,
    ) -> Result<Vec<f64>> {
        if x0.len() != x1.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "FlowMatcher cfm_field: x0 len {} != x1 len {}",
                x0.len(),
                x1.len()
            )));
        }
        let denom = (1.0 - (1.0 - self.sigma) * t).max(1e-8);
        Ok(x0
            .iter()
            .zip(x1)
            .map(|(&a, &b)| (b - (1.0 - self.sigma) * a) / denom)
            .collect())
    }

    /// Compute the OT-CFM conditional vector field.
    ///
    /// For OT-CFM the path is a straight line: `x_t = x₀ + t (x₁ - x₀)`,
    /// and the conditional field is constant: `u_t = x₁ - x₀`.
    ///
    /// The OT coupling is approximated by **greedy mini-batch linear assignment**
    /// via squared Euclidean distance, which is a good proxy for the true OT.
    pub fn ot_conditional_vector_field(
        x0: &[f64],
        x1: &[f64],
    ) -> Vec<f64> {
        // For straight-line OT paths: u_t = x₁ - x₀ (constant over t)
        x1.iter().zip(x0).map(|(&b, &a)| b - a).collect()
    }

    /// Compute the VE-CFM conditional vector field.
    ///
    /// Variance-exploding schedule: `σ_t = σ_min^{1-t} σ_max^t`.
    /// The conditional field adjusts for the time-varying noise level.
    pub fn ve_conditional_vector_field(
        &self,
        x0: &[f64],
        x1: &[f64],
        t: f64,
        sigma_min: f64,
        sigma_max: f64,
    ) -> Result<Vec<f64>> {
        if x0.len() != x1.len() {
            return Err(NeuralError::ShapeMismatch(
                "FlowMatcher ve_field: shape mismatch".to_string(),
            ));
        }
        let sigma_t = sigma_min.powf(1.0 - t) * sigma_max.powf(t);
        let log_ratio = sigma_max.ln() - sigma_min.ln();
        // d(sigma_t)/dt = sigma_t * log(sigma_max/sigma_min)
        let dsigma_dt = sigma_t * log_ratio;
        // VE conditional field: (x₁ - x₀) / sigma_t_dot component + drift
        Ok(x0
            .iter()
            .zip(x1)
            .map(|(&a, &b)| dsigma_dt * (b - a) / sigma_t.max(1e-12))
            .collect())
    }

    /// Compute the CFM training loss for a batch of source-target pairs.
    ///
    /// For each pair `(x₀, x₁)`:
    /// 1. Sample `t ~ Uniform(0, 1)`
    /// 2. Compute `x_t = interpolate(x₀, x₁, t)`
    /// 3. Compute target `u_t = conditional_vector_field(x₀, x₁, t)`
    /// 4. Loss = `||v_θ(x_t, t) - u_t||²`
    pub fn compute_loss(
        &mut self,
        x0_batch: &[Vec<f64>],
        x1_batch: &[Vec<f64>],
        vf_net: &dyn VectorField,
    ) -> Result<f64> {
        if x0_batch.len() != x1_batch.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "FlowMatcher: x0_batch len {} != x1_batch len {}",
                x0_batch.len(),
                x1_batch.len()
            )));
        }
        if x0_batch.is_empty() {
            return Ok(0.0);
        }
        let mut total_loss = 0.0f64;
        for (x0, x1) in x0_batch.iter().zip(x1_batch) {
            if x0.len() != self.data_dim || x1.len() != self.data_dim {
                return Err(NeuralError::ShapeMismatch(format!(
                    "FlowMatcher: sample dim {} != data_dim {}",
                    x0.len(),
                    self.data_dim
                )));
            }
            let t = self.sample_uniform_01();
            let x_t = self.interpolate(x0, x1, t)?;
            let u_t = match self.objective {
                FlowMatchingObjective::ConditionalFlowMatching => {
                    self.conditional_vector_field(x0, x1, t)?
                }
                FlowMatchingObjective::OptimalTransport => {
                    Self::ot_conditional_vector_field(x0, x1)
                }
                FlowMatchingObjective::VarianceExploding => {
                    self.ve_conditional_vector_field(x0, x1, t, 0.01, 1.0)?
                }
            };
            let v_pred = vf_net.forward(&x_t, t)?;
            if v_pred.len() != self.data_dim {
                return Err(NeuralError::ShapeMismatch(format!(
                    "FlowMatcher: network returned {} values, expected {}",
                    v_pred.len(),
                    self.data_dim
                )));
            }
            let mse: f64 = v_pred
                .iter()
                .zip(&u_t)
                .map(|(&vp, &ut)| (vp - ut).powi(2))
                .sum::<f64>()
                / self.data_dim as f64;
            total_loss += mse;
        }
        Ok(total_loss / x0_batch.len() as f64)
    }

    /// Compute OT coupling via greedy mini-batch linear assignment.
    ///
    /// Given batches `x0_batch` and `x1_batch`, returns a permutation of indices
    /// into `x1_batch` that approximately minimises total squared distance.
    ///
    /// This is a greedy Hungarian approximation: O(n² d) complexity.
    pub fn mini_batch_ot_coupling(
        x0_batch: &[Vec<f64>],
        x1_batch: &[Vec<f64>],
    ) -> Result<Vec<usize>> {
        let n = x0_batch.len();
        let m = x1_batch.len();
        if n == 0 || m == 0 {
            return Err(NeuralError::InvalidArgument(
                "OT coupling: empty batch".to_string(),
            ));
        }
        // Build cost matrix C[i][j] = ||x0_i - x1_j||²
        let mut cost = vec![vec![0.0f64; m]; n];
        for i in 0..n {
            for j in 0..m {
                cost[i][j] = x0_batch[i]
                    .iter()
                    .zip(&x1_batch[j])
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum();
            }
        }
        // Greedy assignment: for each x0 pick the closest unassigned x1
        let mut assignment = vec![0usize; n];
        let mut used = vec![false; m];
        for i in 0..n {
            let mut best_j = 0;
            let mut best_cost = f64::INFINITY;
            for j in 0..m {
                if !used[j] && cost[i][j] < best_cost {
                    best_cost = cost[i][j];
                    best_j = j;
                }
            }
            assignment[i] = best_j;
            used[best_j] = true;
        }
        Ok(assignment)
    }

    /// Compute OT-CFM loss using mini-batch OT coupling.
    ///
    /// Source samples are drawn internally from `N(0,I)`, coupled with the provided
    /// data samples via greedy OT, then the straight-line CFM objective is applied.
    pub fn compute_ot_loss(
        &mut self,
        x1_batch: &[Vec<f64>],
        vf_net: &dyn VectorField,
    ) -> Result<f64> {
        let n = x1_batch.len();
        if n == 0 {
            return Ok(0.0);
        }
        let d = self.data_dim;
        // Sample x₀ ~ N(0,I)
        let x0_batch: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..d).map(|_| self.sample_normal()).collect())
            .collect();
        // Compute OT coupling
        let coupling = Self::mini_batch_ot_coupling(&x0_batch, x1_batch)?;
        // Apply OT-CFM loss
        let mut total_loss = 0.0f64;
        for i in 0..n {
            let x0 = &x0_batch[i];
            let x1 = &x1_batch[coupling[i]];
            let t = self.sample_uniform_01();
            // Straight-line interpolant for OT: x_t = (1-t)x₀ + t x₁
            let x_t: Vec<f64> = x0.iter().zip(x1.iter()).map(|(&a, &b)| (1.0-t)*a + t*b).collect();
            let u_t = Self::ot_conditional_vector_field(x0, x1);
            let v_pred = vf_net.forward(&x_t, t)?;
            let mse: f64 = v_pred
                .iter()
                .zip(&u_t)
                .map(|(&vp, &ut)| (vp - ut).powi(2))
                .sum::<f64>()
                / d as f64;
            total_loss += mse;
        }
        Ok(total_loss / n as f64)
    }
}

// ---------------------------------------------------------------------------
// ODESolver — Euler / RK4 integrator for sampling
// ---------------------------------------------------------------------------

/// ODE integration method for flow matching sampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ODEMethod {
    /// Forward Euler (fast but less accurate).
    Euler,
    /// Classical 4th-order Runge-Kutta (4× more network evaluations, more accurate).
    RungeKutta4,
    /// Midpoint method (2× network evaluations, intermediate accuracy).
    Midpoint,
}

/// Configuration for ODE-based flow matching sampling.
#[derive(Debug, Clone)]
pub struct ODESolverConfig {
    /// Number of integration steps.
    pub num_steps: usize,
    /// Integration method.
    pub method: ODEMethod,
    /// Integration direction: `true` = forward (0→1), `false` = backward (1→0).
    pub forward: bool,
    /// Random seed (for initialising x₀).
    pub seed: u64,
}

impl Default for ODESolverConfig {
    fn default() -> Self {
        Self {
            num_steps: 100,
            method: ODEMethod::Euler,
            forward: true,
            seed: 42,
        }
    }
}

/// ODE-based sampler for flow matching models.
///
/// Integrates `dx/dt = v_θ(x, t)` from t=0 to t=1 (forward) or t=1 to t=0 (backward).
#[derive(Debug)]
pub struct ODESampler {
    /// Configuration
    pub config: ODESolverConfig,
    rng_state: u64,
}

impl ODESampler {
    /// Create a new `ODESampler`.
    pub fn new(config: ODESolverConfig) -> Result<Self> {
        if config.num_steps == 0 {
            return Err(NeuralError::InvalidArgument(
                "ODESampler: num_steps must be > 0".to_string(),
            ));
        }
        let rng = config.seed.wrapping_add(0x9abcdef012345678u64);
        Ok(Self { config, rng_state: rng })
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

    /// Generate a sample by integrating the ODE from `x_init` (or Gaussian noise).
    ///
    /// # Arguments
    /// * `x_init` – optional initial condition; if `None`, samples `x₀ ~ N(0,I)`
    /// * `vf_net` – trained vector field network
    /// * `data_dim` – data dimensionality (only used when `x_init = None`)
    ///
    /// # Returns
    /// The final integrated sample.
    pub fn sample(
        &mut self,
        x_init: Option<&[f64]>,
        vf_net: &dyn VectorField,
        data_dim: usize,
    ) -> Result<Vec<f64>> {
        let d = if let Some(xi) = x_init { xi.len() } else { data_dim };
        let mut x: Vec<f64> = match x_init {
            Some(xi) => xi.to_vec(),
            None => (0..d).map(|_| self.sample_normal()).collect(),
        };
        let n = self.config.num_steps;
        let (t_start, t_end) = if self.config.forward {
            (0.0f64, 1.0f64)
        } else {
            (1.0f64, 0.0f64)
        };
        let dt = (t_end - t_start) / n as f64;
        let mut t = t_start;
        for _ in 0..n {
            let t_clamp = t.clamp(0.0, 1.0);
            match self.config.method {
                ODEMethod::Euler => {
                    let v = vf_net.forward(&x, t_clamp)?;
                    x = x.iter().zip(&v).map(|(&xi, &vi)| xi + dt * vi).collect();
                }
                ODEMethod::Midpoint => {
                    let v1 = vf_net.forward(&x, t_clamp)?;
                    let x_mid: Vec<f64> = x.iter().zip(&v1).map(|(&xi, &vi)| xi + 0.5 * dt * vi).collect();
                    let t_mid = (t + 0.5 * dt).clamp(0.0, 1.0);
                    let v_mid = vf_net.forward(&x_mid, t_mid)?;
                    x = x.iter().zip(&v_mid).map(|(&xi, &vi)| xi + dt * vi).collect();
                }
                ODEMethod::RungeKutta4 => {
                    let k1 = vf_net.forward(&x, t_clamp)?;
                    let x2: Vec<f64> = x.iter().zip(&k1).map(|(&xi, &ki)| xi + 0.5*dt*ki).collect();
                    let t2 = (t + 0.5 * dt).clamp(0.0, 1.0);
                    let k2 = vf_net.forward(&x2, t2)?;
                    let x3: Vec<f64> = x.iter().zip(&k2).map(|(&xi, &ki)| xi + 0.5*dt*ki).collect();
                    let k3 = vf_net.forward(&x3, t2)?;
                    let x4: Vec<f64> = x.iter().zip(&k3).map(|(&xi, &ki)| xi + dt*ki).collect();
                    let t4 = (t + dt).clamp(0.0, 1.0);
                    let k4 = vf_net.forward(&x4, t4)?;
                    x = x.iter()
                        .zip(k1.iter())
                        .zip(k2.iter())
                        .zip(k3.iter())
                        .zip(k4.iter())
                        .map(|((((xi, k1i), k2i), k3i), k4i)| {
                            xi + dt * (k1i + 2.0*k2i + 2.0*k3i + k4i) / 6.0
                        })
                        .collect();
                }
            }
            t += dt;
        }
        Ok(x)
    }

    /// Sample with trajectory recording.
    ///
    /// Returns `(times, samples)` where each entry corresponds to an integration step.
    pub fn sample_trajectory(
        &mut self,
        x_init: Option<&[f64]>,
        vf_net: &dyn VectorField,
        data_dim: usize,
        save_every: usize,
    ) -> Result<Vec<(f64, Vec<f64>)>> {
        let d = if let Some(xi) = x_init { xi.len() } else { data_dim };
        let mut x: Vec<f64> = match x_init {
            Some(xi) => xi.to_vec(),
            None => (0..d).map(|_| self.sample_normal()).collect(),
        };
        let n = self.config.num_steps;
        let actual_save = save_every.max(1);
        let (t_start, t_end) = if self.config.forward { (0.0f64, 1.0f64) } else { (1.0f64, 0.0f64) };
        let dt = (t_end - t_start) / n as f64;
        let mut t = t_start;
        let mut trajectory = Vec::new();
        for step in 0..n {
            let t_clamp = t.clamp(0.0, 1.0);
            let v = vf_net.forward(&x, t_clamp)?;
            x = x.iter().zip(&v).map(|(&xi, &vi)| xi + dt * vi).collect();
            t += dt;
            if step % actual_save == 0 {
                trajectory.push((t, x.clone()));
            }
        }
        Ok(trajectory)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_field_net_creation() {
        let net = SimpleVectorFieldNet::new(4, 16, 2, 42).expect("VF net");
        assert!(net.parameter_count() > 0);
    }

    #[test]
    fn test_vector_field_net_output() {
        let net = SimpleVectorFieldNet::new(4, 16, 2, 0).expect("VF net");
        let x = vec![0.1, -0.2, 0.3, -0.4];
        let v = net.forward(&x, 0.5).expect("VF forward");
        assert_eq!(v.len(), 4);
        for &vi in &v {
            assert!(vi.is_finite());
        }
    }

    #[test]
    fn test_flow_matcher_interpolate() {
        let fm = FlowMatcher::new(4, 0.01, FlowMatchingObjective::ConditionalFlowMatching)
            .expect("FM");
        let x0 = vec![0.0; 4];
        let x1 = vec![1.0; 4];
        // At t=0: should be x0
        let xt0 = fm.interpolate(&x0, &x1, 0.0).expect("interp t=0");
        for &v in &xt0 {
            assert!((v - 0.0).abs() < 1e-10);
        }
        // At t=1: should be x1 + σ*x0 component
        let xt1 = fm.interpolate(&x0, &x1, 1.0).expect("interp t=1");
        for &v in &xt1 {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_conditional_vector_field() {
        let fm = FlowMatcher::new(4, 0.0, FlowMatchingObjective::ConditionalFlowMatching)
            .expect("FM");
        let x0 = vec![0.0; 4];
        let x1 = vec![1.0; 4];
        // With σ=0: u_t = (x1 - x0) / (1-t) = 1/(1-t)
        let u = fm.conditional_vector_field(&x0, &x1, 0.0).expect("CFM field");
        for &v in &u {
            assert!((v - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_ot_vector_field() {
        let x0 = vec![0.0f64; 4];
        let x1 = vec![2.0f64; 4];
        let u = FlowMatcher::ot_conditional_vector_field(&x0, &x1);
        assert_eq!(u.len(), 4);
        for &v in &u {
            assert!((v - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_flow_matcher_cfm_loss() {
        let mut fm = FlowMatcher::new(4, 0.01, FlowMatchingObjective::ConditionalFlowMatching)
            .expect("FM");
        let net = SimpleVectorFieldNet::new(4, 16, 2, 1).expect("VF net");
        let x0_batch: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 * 0.1; 4]).collect();
        let x1_batch: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 * 0.2; 4]).collect();
        let loss = fm.compute_loss(&x0_batch, &x1_batch, &net).expect("CFM loss");
        assert!(loss >= 0.0 && loss.is_finite());
    }

    #[test]
    fn test_flow_matcher_ot_loss() {
        let mut fm = FlowMatcher::new(4, 0.01, FlowMatchingObjective::OptimalTransport)
            .expect("FM");
        let net = SimpleVectorFieldNet::new(4, 16, 2, 2).expect("VF net");
        let x1_batch: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 * 0.2; 4]).collect();
        let loss = fm.compute_ot_loss(&x1_batch, &net).expect("OT-CFM loss");
        assert!(loss >= 0.0 && loss.is_finite());
    }

    #[test]
    fn test_ot_coupling() {
        let x0: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let x1: Vec<Vec<f64>> = vec![
            vec![0.1, 0.0],
            vec![0.9, 0.0],
            vec![0.0, 0.9],
        ];
        let coupling = FlowMatcher::mini_batch_ot_coupling(&x0, &x1).expect("OT coupling");
        assert_eq!(coupling.len(), 3);
        // Each assignment should be unique
        let mut used = vec![false; 3];
        for &j in &coupling {
            assert!(!used[j], "duplicate assignment: {j}");
            used[j] = true;
        }
    }

    #[test]
    fn test_ode_sampler_euler() {
        let cfg = ODESolverConfig {
            num_steps: 10,
            method: ODEMethod::Euler,
            forward: true,
            seed: 42,
        };
        let mut sampler = ODESampler::new(cfg).expect("ODE sampler");
        let net = SimpleVectorFieldNet::new(4, 16, 2, 3).expect("VF net");
        let sample = sampler.sample(None, &net, 4).expect("ODE sample");
        assert_eq!(sample.len(), 4);
        for &v in &sample {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_ode_sampler_rk4() {
        let cfg = ODESolverConfig {
            num_steps: 5,
            method: ODEMethod::RungeKutta4,
            forward: true,
            seed: 7,
        };
        let mut sampler = ODESampler::new(cfg).expect("ODE sampler");
        let net = SimpleVectorFieldNet::new(4, 16, 2, 4).expect("VF net");
        let x_init = vec![0.5, -0.3, 0.2, 0.8];
        let sample = sampler.sample(Some(&x_init), &net, 4).expect("RK4 sample");
        assert_eq!(sample.len(), 4);
    }

    #[test]
    fn test_ode_sampler_midpoint() {
        let cfg = ODESolverConfig {
            num_steps: 5,
            method: ODEMethod::Midpoint,
            forward: false, // backward integration (1→0)
            seed: 99,
        };
        let mut sampler = ODESampler::new(cfg).expect("ODE sampler");
        let net = SimpleVectorFieldNet::new(4, 16, 2, 5).expect("VF net");
        let sample = sampler.sample(None, &net, 4).expect("midpoint sample");
        assert_eq!(sample.len(), 4);
    }

    #[test]
    fn test_ode_trajectory() {
        let cfg = ODESolverConfig {
            num_steps: 10,
            method: ODEMethod::Euler,
            forward: true,
            seed: 0,
        };
        let mut sampler = ODESampler::new(cfg).expect("ODE sampler");
        let net = SimpleVectorFieldNet::new(4, 16, 2, 6).expect("VF net");
        let traj = sampler.sample_trajectory(None, &net, 4, 2).expect("trajectory");
        assert!(!traj.is_empty());
        for (t, sample) in &traj {
            assert_eq!(sample.len(), 4);
            assert!(*t >= 0.0 && *t <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn test_flow_matcher_invalid_sigma() {
        let result = FlowMatcher::new(4, -0.1, FlowMatchingObjective::ConditionalFlowMatching);
        assert!(result.is_err());
    }

    #[test]
    fn test_flow_matcher_ve() {
        let fm = FlowMatcher::new(4, 0.01, FlowMatchingObjective::VarianceExploding)
            .expect("FM VE");
        let x0 = vec![0.0; 4];
        let x1 = vec![1.0; 4];
        let field = fm.ve_conditional_vector_field(&x0, &x1, 0.5, 0.01, 1.0)
            .expect("VE field");
        assert_eq!(field.len(), 4);
        for &v in &field {
            assert!(v.is_finite());
        }
    }
}
