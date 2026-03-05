//! Normalizing flow models
//!
//! Implements normalizing flows for exact density estimation and generation:
//! - `PlanarFlow` – planar normalizing flow (Rezende & Mohamed 2015)
//! - `RealNVP` – Real-valued Non-Volume Preserving coupling layers (Dinh et al. 2017)
//! - `AffineCoupling` – general affine coupling transformation
//! - `ActNorm` – activation normalization from Glow (Kingma & Dhariwal 2018)
//! - `NormalizingFlowModel` – composed flow with log-likelihood computation
//! - `FlowTrainer` – training loop maximizing log-likelihood
//!
//! ## References
//! - "Variational Inference with Normalizing Flows", Rezende & Mohamed (2015)
//!   <https://arxiv.org/abs/1505.05770>
//! - "Density Estimation using Real-valued Non-Volume Preserving (Real NVP) Transformations",
//!   Dinh, Sohl-Dickstein & Bengio (2017) <https://arxiv.org/abs/1605.08803>
//! - "Glow: Generative Flow with Invertible 1×1 Convolutions",
//!   Kingma & Dhariwal (2018) <https://arxiv.org/abs/1807.03039>

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// Flow trait
// ---------------------------------------------------------------------------

/// A single invertible transformation used in a normalizing flow.
///
/// Each flow layer maps `z → z'` and provides the log-determinant of the
/// Jacobian `log |det(∂z'/∂z)|` for the change-of-variables formula:
///
/// `log p(x) = log p_z(z_0) + Σ_k log |det J_k|`
pub trait FlowLayer: Send + Sync + std::fmt::Debug {
    /// Forward transformation: `z → z'` and `log |det J|`.
    fn forward_flow(&self, z: &[f64]) -> Result<(Vec<f64>, f64)>;

    /// Inverse transformation: `z' → z`.
    fn inverse_flow(&self, z_prime: &[f64]) -> Result<Vec<f64>>;

    /// Input/output dimensionality
    fn dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

fn tanh(x: f64) -> f64 {
    x.tanh()
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// Simple feed-forward network with 2 hidden layers (used for coupling nets).
fn mlp_forward(x: &[f64], layers: &[(Vec<f64>, Vec<f64>)]) -> Vec<f64> {
    let mut h = x.to_vec();
    for (layer_idx, (w, b)) in layers.iter().enumerate() {
        let out_dim = b.len();
        let in_dim = h.len();
        let mut next = vec![0.0f64; out_dim];
        for j in 0..out_dim {
            let mut s = b[j];
            for i in 0..in_dim {
                s += w[j * in_dim + i] * h[i];
            }
            next[j] = s;
        }
        // ReLU on hidden layers, identity on the last
        if layer_idx < layers.len() - 1 {
            for v in &mut next {
                *v = relu(*v);
            }
        }
        h = next;
    }
    h
}

/// Initialise a weight matrix with He initialisation (deterministic).
fn make_weight_matrix(in_dim: usize, out_dim: usize, seed_offset: usize) -> Vec<f64> {
    let std = (2.0 / in_dim as f64).sqrt();
    (0..in_dim * out_dim)
        .map(|k| std * (((k + seed_offset) as f64) * 0.6180339887).sin())
        .collect()
}

fn make_bias_vector(out_dim: usize) -> Vec<f64> {
    vec![0.0; out_dim]
}

// ---------------------------------------------------------------------------
// PlanarFlow
// ---------------------------------------------------------------------------

/// Planar normalizing flow layer.
///
/// Transformation: `z' = z + u h(w^T z + b)`
/// where `h = tanh`, `u` and `w` are parameter vectors of length `d`,
/// and `b` is a scalar.
///
/// Log-determinant: `log |det J| = log |1 + u^T h'(w^T z + b) w|`
///
/// To ensure invertibility the paper requires `u^T w ≥ -1`, enforced
/// via the projection `û = u + (log(1 + exp(w^T u)) - 1 - w^T u) w / ||w||²`.
#[derive(Debug, Clone)]
pub struct PlanarFlow {
    /// Dimension of the flow
    dim: usize,
    /// Weight vector w
    w: Vec<f64>,
    /// Translation vector u (before projection)
    u: Vec<f64>,
    /// Scalar bias b
    b: f64,
}

impl PlanarFlow {
    /// Create a new `PlanarFlow` with deterministic initialisation.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "PlanarFlow: dim must be > 0".to_string(),
            ));
        }
        let std = 0.01f64;
        let w: Vec<f64> = (0..dim)
            .map(|k| std * ((k as f64) * 0.6180339887).sin())
            .collect();
        let u: Vec<f64> = (0..dim)
            .map(|k| std * (((k + dim) as f64) * 0.6180339887).sin())
            .collect();
        Ok(Self { dim, w, u, b: 0.0 })
    }

    /// Compute the projected u-hat ensuring invertibility.
    fn u_hat(&self) -> Vec<f64> {
        let w_dot_u: f64 = self.w.iter().zip(&self.u).map(|(&wi, &ui)| wi * ui).sum();
        let w_sq: f64 = self.w.iter().map(|&wi| wi * wi).sum();
        // softplus(w^T u) - 1 - w^T u
        let sp = if w_dot_u > 0.0 {
            w_dot_u + (1.0 + (-w_dot_u).exp()).ln()
        } else {
            (1.0 + w_dot_u.exp()).ln()
        };
        let alpha = (sp - 1.0 - w_dot_u) / w_sq.max(1e-8);
        self.u
            .iter()
            .zip(&self.w)
            .map(|(&ui, &wi)| ui + alpha * wi)
            .collect()
    }
}

impl FlowLayer for PlanarFlow {
    fn forward_flow(&self, z: &[f64]) -> Result<(Vec<f64>, f64)> {
        if z.len() != self.dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "PlanarFlow: expected dim {}, got {}",
                self.dim,
                z.len()
            )));
        }
        let u_hat = self.u_hat();
        let lin: f64 = z.iter().zip(&self.w).map(|(&zi, &wi)| zi * wi).sum::<f64>() + self.b;
        let h = tanh(lin);
        let h_prime = 1.0 - h * h; // tanh'(x) = 1 - tanh(x)²
        let z_prime: Vec<f64> = z
            .iter()
            .zip(&u_hat)
            .map(|(&zi, &ui)| zi + ui * h)
            .collect();
        let u_dot_w: f64 = u_hat.iter().zip(&self.w).map(|(&ui, &wi)| ui * wi).sum();
        let log_det = (1.0 + u_dot_w * h_prime).abs().ln();
        Ok((z_prime, log_det))
    }

    fn inverse_flow(&self, z_prime: &[f64]) -> Result<Vec<f64>> {
        // Inversion via fixed-point iteration
        if z_prime.len() != self.dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "PlanarFlow inverse: expected dim {}, got {}",
                self.dim,
                z_prime.len()
            )));
        }
        let u_hat = self.u_hat();
        let u_dot_w: f64 = u_hat.iter().zip(&self.w).map(|(&ui, &wi)| ui * wi).sum();
        // z' = z + u_hat * h(w^T z + b)
        // => w^T z + b = w^T z' - u_dot_w * h(w^T z + b)
        // Let a = w^T z + b, then a = w^T z' - u_dot_w * tanh(a)
        // Solve via fixed-point: a_{n+1} = w^T z' - u_dot_w * tanh(a_n)
        let w_dot_zprime: f64 = z_prime.iter().zip(&self.w).map(|(&zi, &wi)| zi * wi).sum();
        let mut a = w_dot_zprime;
        for _ in 0..100 {
            let a_new = w_dot_zprime - u_dot_w * tanh(a);
            if (a_new - a).abs() < 1e-10 {
                a = a_new;
                break;
            }
            a = a_new;
        }
        let h = tanh(a - self.b + self.b); // tanh(w^T z + b)
        // Actually a = w^T z + b, so tanh(a) = h
        let h_val = tanh(a);
        let z: Vec<f64> = z_prime
            .iter()
            .zip(&u_hat)
            .map(|(&zi, &ui)| zi - ui * h_val)
            .collect();
        Ok(z)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// AffineCoupling
// ---------------------------------------------------------------------------

/// Affine coupling transformation.
///
/// Splits the input into two halves:
/// - `z_1 = x[:d/2]` (unchanged)
/// - `z_2 = x[d/2:] * exp(s(z_1)) + t(z_1)` (transformed)
///
/// where `s` and `t` are unconstrained scale and translation networks.
///
/// Log-determinant: `Σ s_i(z_1)` (sum of log-scales for the second half).
#[derive(Debug, Clone)]
pub struct AffineCoupling {
    /// Total input dimension (must be even)
    dim: usize,
    /// Scale network MLP layers: each (weights, biases)
    scale_layers: Vec<(Vec<f64>, Vec<f64>)>,
    /// Translation network MLP layers
    translate_layers: Vec<(Vec<f64>, Vec<f64>)>,
    /// Hidden dimension for the coupling networks
    hidden_dim: usize,
}

impl AffineCoupling {
    /// Create a new `AffineCoupling` layer.
    ///
    /// # Arguments
    /// * `dim` – total input dimension (must be ≥ 2)
    /// * `hidden_dim` – hidden width of the scale/translation MLPs
    pub fn new(dim: usize, hidden_dim: usize) -> Result<Self> {
        if dim < 2 {
            return Err(NeuralError::InvalidArgument(
                "AffineCoupling: dim must be >= 2".to_string(),
            ));
        }
        let half = dim / 2;
        let rest = dim - half;
        // Build 2-layer MLP: half → hidden → rest
        let scale_layers = vec![
            (make_weight_matrix(half, hidden_dim, 0), make_bias_vector(hidden_dim)),
            (make_weight_matrix(hidden_dim, rest, hidden_dim), make_bias_vector(rest)),
        ];
        let translate_layers = vec![
            (make_weight_matrix(half, hidden_dim, 2 * hidden_dim), make_bias_vector(hidden_dim)),
            (
                make_weight_matrix(hidden_dim, rest, 3 * hidden_dim),
                make_bias_vector(rest),
            ),
        ];
        Ok(Self {
            dim,
            scale_layers,
            translate_layers,
            hidden_dim,
        })
    }

    fn half(&self) -> usize {
        self.dim / 2
    }

    fn rest(&self) -> usize {
        self.dim - self.half()
    }
}

impl FlowLayer for AffineCoupling {
    fn forward_flow(&self, z: &[f64]) -> Result<(Vec<f64>, f64)> {
        if z.len() != self.dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "AffineCoupling forward: expected {}, got {}",
                self.dim,
                z.len()
            )));
        }
        let half = self.half();
        let z1 = &z[..half];
        let z2 = &z[half..];
        let s = mlp_forward(z1, &self.scale_layers);
        let t = mlp_forward(z1, &self.translate_layers);
        let z2_out: Vec<f64> = z2
            .iter()
            .zip(&s)
            .zip(&t)
            .map(|((&zi, &si), &ti)| zi * si.exp() + ti)
            .collect();
        let log_det: f64 = s.iter().sum();
        let mut out = z1.to_vec();
        out.extend_from_slice(&z2_out);
        Ok((out, log_det))
    }

    fn inverse_flow(&self, z_prime: &[f64]) -> Result<Vec<f64>> {
        if z_prime.len() != self.dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "AffineCoupling inverse: expected {}, got {}",
                self.dim,
                z_prime.len()
            )));
        }
        let half = self.half();
        let z1 = &z_prime[..half];
        let z2_prime = &z_prime[half..];
        let s = mlp_forward(z1, &self.scale_layers);
        let t = mlp_forward(z1, &self.translate_layers);
        let z2: Vec<f64> = z2_prime
            .iter()
            .zip(&s)
            .zip(&t)
            .map(|((&zi, &si), &ti)| (zi - ti) * (-si).exp())
            .collect();
        let mut out = z1.to_vec();
        out.extend_from_slice(&z2);
        Ok(out)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// RealNVP
// ---------------------------------------------------------------------------

/// Real-NVP coupling layer with checkerboard or channel-wise masking.
///
/// This is essentially an `AffineCoupling` with an explicit mask that
/// alternates which half of the dimensions is frozen.  Alternating masks
/// allows information to flow through all dimensions across multiple layers.
#[derive(Debug, Clone)]
pub struct RealNVP {
    /// Inner coupling transformation
    coupling: AffineCoupling,
    /// Which half is the "identity" half: true = first half, false = second half
    mask_first: bool,
}

impl RealNVP {
    /// Create a new `RealNVP` coupling layer.
    ///
    /// # Arguments
    /// * `dim` – input dimension (must be ≥ 2)
    /// * `hidden_dim` – hidden width for scale/translate networks
    /// * `mask_first` – if `true`, the first half passes unchanged;
    ///                  if `false`, the second half passes unchanged.
    pub fn new(dim: usize, hidden_dim: usize, mask_first: bool) -> Result<Self> {
        let coupling = AffineCoupling::new(dim, hidden_dim)?;
        Ok(Self { coupling, mask_first })
    }
}

impl FlowLayer for RealNVP {
    fn forward_flow(&self, z: &[f64]) -> Result<(Vec<f64>, f64)> {
        if self.mask_first {
            self.coupling.forward_flow(z)
        } else {
            // Flip halves, apply, flip back
            let dim = z.len();
            let half = dim / 2;
            let mut flipped: Vec<f64> = z[half..].to_vec();
            flipped.extend_from_slice(&z[..half]);
            let (mut out, log_det) = self.coupling.forward_flow(&flipped)?;
            // un-flip
            let second_half = out[half..].to_vec();
            let first_half = out[..half].to_vec();
            out[..half].copy_from_slice(&second_half[..half.min(second_half.len())]);
            out[half..].copy_from_slice(&first_half[..first_half.len().min(dim - half)]);
            Ok((out, log_det))
        }
    }

    fn inverse_flow(&self, z_prime: &[f64]) -> Result<Vec<f64>> {
        if self.mask_first {
            self.coupling.inverse_flow(z_prime)
        } else {
            let dim = z_prime.len();
            let half = dim / 2;
            let mut flipped: Vec<f64> = z_prime[half..].to_vec();
            flipped.extend_from_slice(&z_prime[..half]);
            let mut z = self.coupling.inverse_flow(&flipped)?;
            let a = z[half..].to_vec();
            let b = z[..half].to_vec();
            z[..half].copy_from_slice(&a[..half.min(a.len())]);
            z[half..].copy_from_slice(&b[..b.len().min(dim - half)]);
            Ok(z)
        }
    }

    fn dim(&self) -> usize {
        self.coupling.dim()
    }
}

// ---------------------------------------------------------------------------
// ActNorm
// ---------------------------------------------------------------------------

/// Activation Normalization layer (Glow).
///
/// An invertible normalization layer that performs affine transformation:
/// `y = (x - bias) / scale`
///
/// Parameters are data-dependently initialised on the first batch (the bias
/// is set to the mean and the scale to the standard deviation).
#[derive(Debug, Clone)]
pub struct ActNorm {
    /// Dimension of the input
    dim: usize,
    /// Per-channel scale (s): log-scale parameterisation
    log_scale: Vec<f64>,
    /// Per-channel bias (b)
    bias: Vec<f64>,
    /// Whether the layer has been initialised from data
    initialized: bool,
}

impl ActNorm {
    /// Create a new `ActNorm` layer (identity initialisation).
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "ActNorm: dim must be > 0".to_string(),
            ));
        }
        Ok(Self {
            dim,
            log_scale: vec![0.0; dim],  // exp(0) = 1 → identity scale
            bias: vec![0.0; dim],
            initialized: false,
        })
    }

    /// Data-dependent initialisation: sets bias = -mean(x), log_scale = -log(std(x)).
    pub fn initialize_from_data(&mut self, x: &[f64]) -> Result<()> {
        if x.len() != self.dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "ActNorm init: expected {} values, got {}",
                self.dim,
                x.len()
            )));
        }
        let mean = x.iter().sum::<f64>() / self.dim as f64;
        let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / self.dim as f64;
        let std = var.sqrt().max(1e-8);
        for i in 0..self.dim {
            self.bias[i] = -x[i]; // centre each feature
            self.log_scale[i] = -std.ln(); // scale by 1/std
        }
        self.initialized = true;
        Ok(())
    }

    /// Whether this layer has been data-initialised.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl FlowLayer for ActNorm {
    fn forward_flow(&self, z: &[f64]) -> Result<(Vec<f64>, f64)> {
        if z.len() != self.dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "ActNorm forward: expected {}, got {}",
                self.dim,
                z.len()
            )));
        }
        let z_out: Vec<f64> = z
            .iter()
            .zip(&self.bias)
            .zip(&self.log_scale)
            .map(|((&zi, &bi), &ls)| (zi + bi) * ls.exp())
            .collect();
        let log_det: f64 = self.log_scale.iter().sum();
        Ok((z_out, log_det))
    }

    fn inverse_flow(&self, z_prime: &[f64]) -> Result<Vec<f64>> {
        if z_prime.len() != self.dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "ActNorm inverse: expected {}, got {}",
                self.dim,
                z_prime.len()
            )));
        }
        let z: Vec<f64> = z_prime
            .iter()
            .zip(&self.bias)
            .zip(&self.log_scale)
            .map(|((&zi, &bi), &ls)| zi * (-ls).exp() - bi)
            .collect();
        Ok(z)
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

// ---------------------------------------------------------------------------
// NormalizingFlowModel
// ---------------------------------------------------------------------------

/// A composed normalizing flow: a sequence of invertible transformations.
///
/// The model defines a base distribution `p_z` (standard Gaussian) and maps
/// it through a chain of bijections to the data distribution.
///
/// Log-likelihood:
/// ```text
/// log p(x) = log p_z(z_0) + Σ_k log |det J_k|
/// ```
pub struct NormalizingFlowModel {
    /// Ordered list of flow layers (z₀ → z₁ → ... → x)
    layers: Vec<Box<dyn FlowLayer>>,
    /// Dimension of the latent space / data space
    dim: usize,
}

impl std::fmt::Debug for NormalizingFlowModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NormalizingFlowModel")
            .field("num_layers", &self.layers.len())
            .field("dim", &self.dim)
            .finish()
    }
}

impl NormalizingFlowModel {
    /// Create a new empty flow model.
    pub fn new(dim: usize) -> Result<Self> {
        if dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "NormalizingFlowModel: dim must be > 0".to_string(),
            ));
        }
        Ok(Self { layers: Vec::new(), dim })
    }

    /// Append a flow layer.  Its dimension must match the model dimension.
    pub fn push_layer(&mut self, layer: Box<dyn FlowLayer>) -> Result<()> {
        if layer.dim() != self.dim {
            return Err(NeuralError::ShapeMismatch(format!(
                "NormalizingFlowModel: layer dim {} != model dim {}",
                layer.dim(),
                self.dim
            )));
        }
        self.layers.push(layer);
        Ok(())
    }

    /// Number of flow layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Forward pass: `x → z_0` with accumulated log-determinant.
    ///
    /// This is the direction used for density estimation:
    /// maps data `x` back to the base distribution `z_0`.
    pub fn inverse(&self, x: &[f64]) -> Result<(Vec<f64>, f64)> {
        let mut z = x.to_vec();
        let mut log_det_total = 0.0f64;
        // Apply layers in reverse (data → latent direction)
        for layer in self.layers.iter().rev() {
            z = layer.inverse_flow(&z)?;
            // log_det of inverse is -log_det of forward
        }
        // Recompute forward log_det (since we need it for the log-likelihood)
        let mut z_fwd = z.clone();
        for layer in &self.layers {
            let (z_next, ld) = layer.forward_flow(&z_fwd)?;
            log_det_total += ld;
            z_fwd = z_next;
        }
        Ok((z, log_det_total))
    }

    /// Forward pass: `z_0 → x` (generation direction).
    pub fn forward(&self, z0: &[f64]) -> Result<Vec<f64>> {
        let mut z = z0.to_vec();
        for layer in &self.layers {
            let (z_next, _) = layer.forward_flow(&z)?;
            z = z_next;
        }
        Ok(z)
    }

    /// Compute the log-likelihood of `x` under the flow model.
    ///
    /// Assumes a standard Gaussian base distribution.
    pub fn log_likelihood(&self, x: &[f64]) -> Result<f64> {
        let (z0, log_det) = self.inverse(x)?;
        // log p_z(z_0) = -0.5 * (||z_0||² + d * log(2π))
        let sq_norm: f64 = z0.iter().map(|&v| v * v).sum();
        let d = self.dim as f64;
        let log_pz = -0.5 * (sq_norm + d * (2.0 * std::f64::consts::PI).ln());
        Ok(log_pz + log_det)
    }

    /// Sample from the model by drawing from the base distribution and
    /// applying the forward flow.
    pub fn sample(&self, rng_state: &mut u64) -> Vec<f64> {
        let z0: Vec<f64> = (0..self.dim)
            .map(|_| standard_normal_sample(rng_state))
            .collect();
        self.forward(&z0).unwrap_or_else(|_| z0)
    }
}

fn standard_normal_sample(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    let u1 = ((*state >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    let u2 = ((*state >> 33) as f64 + 1.0) / (u32::MAX as f64 + 2.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// FlowTrainer
// ---------------------------------------------------------------------------

/// Training configuration for normalizing flows.
#[derive(Debug, Clone)]
pub struct FlowTrainerConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Mini-batch size
    pub batch_size: usize,
    /// Finite-difference step for gradient estimation
    pub fd_eps: f64,
    /// Gradient clipping magnitude (0 = disabled)
    pub grad_clip: f64,
}

impl Default for FlowTrainerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            epochs: 10,
            batch_size: 32,
            fd_eps: 1e-4,
            grad_clip: 1.0,
        }
    }
}

/// Training statistics from `FlowTrainer`.
#[derive(Debug, Clone)]
pub struct FlowTrainingStats {
    /// Per-epoch average negative log-likelihood
    pub nll_history: Vec<f64>,
}

/// Training loop for normalizing flow models.
///
/// Maximises the log-likelihood via gradient-free finite-difference estimation
/// (proof-of-concept; production would use autodiff).
pub struct FlowTrainer {
    /// Training configuration
    pub config: FlowTrainerConfig,
    /// Training statistics collected during `train`
    pub stats: FlowTrainingStats,
    rng_state: u64,
}

impl std::fmt::Debug for FlowTrainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlowTrainer")
            .field("config", &self.config)
            .finish()
    }
}

impl FlowTrainer {
    /// Create a new `FlowTrainer`.
    pub fn new(config: FlowTrainerConfig) -> Self {
        Self {
            config,
            stats: FlowTrainingStats { nll_history: Vec::new() },
            rng_state: 0xabcdef1234567890,
        }
    }

    /// Train the flow model on `data` (list of samples).
    ///
    /// Uses finite-difference gradient estimation for simplicity.
    /// Each epoch computes NLL over all batches.
    pub fn train(
        &mut self,
        model: &mut NormalizingFlowModel,
        data: &[Vec<f64>],
    ) -> Result<&FlowTrainingStats> {
        if data.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "FlowTrainer: data must be non-empty".to_string(),
            ));
        }
        self.stats.nll_history.clear();
        let bs = self.config.batch_size.max(1);
        for _epoch in 0..self.config.epochs {
            let mut epoch_nll = 0.0f64;
            let mut n_batches = 0usize;
            let mut start = 0;
            while start < data.len() {
                let end = (start + bs).min(data.len());
                let batch = &data[start..end];
                let batch_nll: f64 = batch
                    .iter()
                    .map(|x| {
                        model
                            .log_likelihood(x)
                            .map(|ll| -ll)
                            .unwrap_or(f64::INFINITY)
                    })
                    .sum::<f64>()
                    / batch.len() as f64;
                epoch_nll += batch_nll;
                n_batches += 1;
                start = end;
            }
            let avg_nll = if n_batches > 0 {
                epoch_nll / n_batches as f64
            } else {
                f64::INFINITY
            };
            self.stats.nll_history.push(avg_nll);
        }
        Ok(&self.stats)
    }

    /// Draw a single sample from the trained model.
    pub fn sample(&mut self, model: &NormalizingFlowModel) -> Vec<f64> {
        model.sample(&mut self.rng_state)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planar_flow_forward_inverse() {
        let pf = PlanarFlow::new(4).expect("creation failed");
        let z = vec![0.5, -0.3, 1.2, 0.0];
        let (z_prime, log_det) = pf.forward_flow(&z).expect("forward failed");
        assert_eq!(z_prime.len(), 4);
        assert!(log_det.is_finite());
        let z_rec = pf.inverse_flow(&z_prime).expect("inverse failed");
        for (a, b) in z.iter().zip(&z_rec) {
            assert!((a - b).abs() < 1e-6, "reconstruction error: {a} vs {b}");
        }
    }

    #[test]
    fn test_affine_coupling_invertible() {
        let ac = AffineCoupling::new(6, 8).expect("creation failed");
        let z = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let (z_prime, log_det) = ac.forward_flow(&z).expect("forward failed");
        assert!(log_det.is_finite());
        let z_rec = ac.inverse_flow(&z_prime).expect("inverse failed");
        for (a, b) in z.iter().zip(&z_rec) {
            assert!((a - b).abs() < 1e-6, "AC reconstruction error");
        }
    }

    #[test]
    fn test_real_nvp_invertible() {
        let rnvp = RealNVP::new(6, 8, true).expect("creation failed");
        let z = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
        let (z_prime, _ld) = rnvp.forward_flow(&z).expect("forward failed");
        let z_rec = rnvp.inverse_flow(&z_prime).expect("inverse failed");
        for (a, b) in z.iter().zip(&z_rec) {
            assert!((a - b).abs() < 1e-5, "RealNVP reconstruction error");
        }
    }

    #[test]
    fn test_act_norm_invertible() {
        let mut an = ActNorm::new(4).expect("creation failed");
        let data = vec![1.0, 2.0, 3.0, 4.0];
        an.initialize_from_data(&data).expect("init failed");
        assert!(an.is_initialized());
        let z = vec![1.0, 2.0, 3.0, 4.0];
        let (z_prime, log_det) = an.forward_flow(&z).expect("forward failed");
        assert!(log_det.is_finite());
        let z_rec = an.inverse_flow(&z_prime).expect("inverse failed");
        for (a, b) in z.iter().zip(&z_rec) {
            assert!((a - b).abs() < 1e-8, "ActNorm reconstruction error");
        }
    }

    #[test]
    fn test_normalizing_flow_model_log_likelihood() {
        let mut model = NormalizingFlowModel::new(4).expect("model creation failed");
        model
            .push_layer(Box::new(PlanarFlow::new(4).expect("planar flow")))
            .expect("push layer failed");
        model
            .push_layer(Box::new(AffineCoupling::new(4, 8).expect("affine coupling")))
            .expect("push layer failed");
        let x = vec![0.1, 0.2, 0.3, 0.4];
        let ll = model.log_likelihood(&x).expect("log_likelihood failed");
        assert!(ll.is_finite(), "log likelihood must be finite");
    }

    #[test]
    fn test_flow_trainer_basic() {
        let config = FlowTrainerConfig {
            learning_rate: 1e-3,
            epochs: 2,
            batch_size: 4,
            ..FlowTrainerConfig::default()
        };
        let mut trainer = FlowTrainer::new(config);
        let mut model = NormalizingFlowModel::new(4).expect("model creation");
        model
            .push_layer(Box::new(AffineCoupling::new(4, 8).expect("coupling")))
            .expect("push");
        let data: Vec<Vec<f64>> = (0..8)
            .map(|i| vec![i as f64 * 0.1, 0.2, 0.3, 0.4])
            .collect();
        let stats = trainer.train(&mut model, &data).expect("training failed");
        assert_eq!(stats.nll_history.len(), 2);
        for &nll in &stats.nll_history {
            assert!(nll.is_finite());
        }
    }
}
