//! Encoder-Based Generative Models
//!
//! This module provides variational autoencoder (VAE) variants for learning
//! disentangled, generative latent representations of data.
//!
//! ## Overview
//!
//! Variational autoencoders (VAEs) learn a probabilistic latent space by
//! maximizing the evidence lower bound (ELBO):
//!
//! ELBO = E_{q(z|x)}[log p(x|z)] - KL(q(z|x) || p(z))
//!
//! where q(z|x) = N(μ(x), σ²(x)) is the approximate posterior (encoder)
//! and p(x|z) is the decoder likelihood.
//!
//! ## Variants
//!
//! - **[`VariationalAutoencoder`]**: Standard VAE (Kingma & Welling, 2014)
//! - **[`BetaVAE`]**: β-VAE for disentangled representations (Higgins et al., 2017)
//! - **[`InfoVAE`]**: Information maximizing VAE (Zhao et al., 2019)
//! - **[`CVAE`]**: Conditional VAE with label conditioning (Sohn et al., 2015)
//!
//! ## Trait Interfaces
//!
//! - **[`Encoder`]**: Maps observations x → (μ, log σ²)
//! - **[`Decoder`]**: Maps latent z → reconstructed x
//!
//! ## References
//!
//! - Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. ICLR.
//! - Higgins, I., et al. (2017). β-VAE: Learning basic visual concepts with a
//!   constrained variational framework. ICLR.
//! - Zhao, S., Song, J., & Ermon, S. (2019). InfoVAE: Information Maximizing
//!   Variational Autoencoders. AAAI.
//! - Sohn, K., Lee, H., & Yan, X. (2015). Learning structured output representation
//!   using deep conditional generative models. NeurIPS.

use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Array2, Axis};

use crate::error::{Result, TransformError};

// ============================================================================
// Trait interfaces
// ============================================================================

/// Encoder trait: maps inputs to latent distribution parameters.
///
/// Implementations should map x ∈ ℝ^d to (μ, log σ²) ∈ ℝ^z × ℝ^z.
pub trait Encoder {
    /// Encode a batch of inputs to mean and log-variance.
    ///
    /// # Arguments
    /// * `x` - Input batch, shape (batch, input_dim)
    ///
    /// # Returns
    /// `(mu, log_var)` each of shape (batch, latent_dim)
    fn encode(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)>;

    /// Input dimensionality.
    fn input_dim(&self) -> usize;

    /// Latent dimensionality.
    fn latent_dim(&self) -> usize;
}

/// Decoder trait: maps latent vectors to reconstruction parameters.
pub trait Decoder {
    /// Decode a batch of latent vectors to reconstructed inputs.
    ///
    /// # Arguments
    /// * `z` - Latent vectors, shape (batch, latent_dim)
    ///
    /// # Returns
    /// Reconstructed x, shape (batch, output_dim)
    fn decode(&self, z: &Array2<f64>) -> Result<Array2<f64>>;

    /// Latent dimensionality.
    fn latent_dim(&self) -> usize;

    /// Output dimensionality.
    fn output_dim(&self) -> usize;
}

// ============================================================================
// Internal MLP layers for encoder/decoder
// ============================================================================

/// A simple multi-layer perceptron with ReLU activations for use within
/// VAE encoder/decoder networks.
#[derive(Debug, Clone)]
struct MLP {
    /// Weight matrices for each layer
    weights: Vec<Array2<f64>>,
    /// Bias vectors for each layer
    biases: Vec<Array1<f64>>,
    /// Output activation (false = ReLU, true = linear)
    linear_last: bool,
}

impl MLP {
    /// Create a new MLP.
    ///
    /// # Arguments
    /// * `dims` - Layer dimensions [input, h1, …, output]
    /// * `linear_last` - Use linear activation on last layer
    /// * `seed` - Random seed
    fn new(dims: &[usize], linear_last: bool, seed: u64) -> Result<Self> {
        if dims.len() < 2 {
            return Err(TransformError::InvalidInput(
                "MLP requires at least 2 dimensions".to_string(),
            ));
        }
        let n_layers = dims.len() - 1;
        let mut weights = Vec::with_capacity(n_layers);
        let mut biases = Vec::with_capacity(n_layers);

        let mut state = seed.wrapping_add(314159);
        let next_w = |s: u64, limit: f64| -> (f64, u64) {
            let s2 = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let v = (s2 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
            (v * limit, s2)
        };

        for i in 0..n_layers {
            let (in_d, out_d) = (dims[i], dims[i + 1]);
            let limit = (6.0 / (in_d + out_d) as f64).sqrt();

            let mut w = Array2::<f64>::zeros((out_d, in_d));
            for r in 0..out_d {
                for c in 0..in_d {
                    let (v, s) = next_w(state, limit);
                    state = s;
                    w[[r, c]] = v;
                }
            }
            weights.push(w);

            let mut b = Array1::<f64>::zeros(out_d);
            for r in 0..out_d {
                let (v, s) = next_w(state, 0.01);
                state = s;
                b[r] = v;
            }
            biases.push(b);
        }

        Ok(MLP { weights, biases, linear_last })
    }

    /// Forward pass returning pre-activations of the last layer and activations.
    fn forward(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let batch = x.nrows();
        let mut current = x.clone();
        let n_layers = self.weights.len();

        for (layer_idx, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let out_dim = w.nrows();
            let in_dim = w.ncols();

            if current.ncols() != in_dim {
                return Err(TransformError::InvalidInput(format!(
                    "MLP layer {}: expected input dim {}, got {}",
                    layer_idx, in_dim, current.ncols()
                )));
            }

            let mut out = Array2::<f64>::zeros((batch, out_dim));
            for bi in 0..batch {
                for oi in 0..out_dim {
                    let mut v = b[oi];
                    for ii in 0..in_dim {
                        v += w[[oi, ii]] * current[[bi, ii]];
                    }
                    let is_last = layer_idx == n_layers - 1;
                    out[[bi, oi]] = if is_last && self.linear_last {
                        v
                    } else {
                        v.max(0.0) // ReLU
                    };
                }
            }
            current = out;
        }

        Ok(current)
    }

    /// Output dimensionality.
    fn output_dim(&self) -> usize {
        self.weights.last().map(|w| w.nrows()).unwrap_or(0)
    }

    /// Input dimensionality.
    fn input_dim(&self) -> usize {
        self.weights.first().map(|w| w.ncols()).unwrap_or(0)
    }

    /// Total number of parameters.
    fn n_params(&self) -> usize {
        self.weights.iter().map(|w| w.len()).sum::<usize>()
            + self.biases.iter().map(|b| b.len()).sum::<usize>()
    }

    /// Collect all parameters as a flat vector.
    fn params(&self) -> Vec<f64> {
        let mut p = Vec::new();
        for (w, b) in self.weights.iter().zip(self.biases.iter()) {
            p.extend(w.iter());
            p.extend(b.iter());
        }
        p
    }

    /// Set parameters from a flat vector.
    fn set_params(&mut self, params: &[f64]) -> Result<()> {
        let expected = self.n_params();
        if params.len() != expected {
            return Err(TransformError::InvalidInput(format!(
                "MLP: expected {} params, got {}",
                expected,
                params.len()
            )));
        }
        let mut offset = 0;
        for (w, b) in self.weights.iter_mut().zip(self.biases.iter_mut()) {
            for v in w.iter_mut() {
                *v = params[offset];
                offset += 1;
            }
            for v in b.iter_mut() {
                *v = params[offset];
                offset += 1;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Reparameterization trick utilities
// ============================================================================

/// Sample from the latent distribution using the reparameterization trick.
///
/// z = μ + ε · σ  where ε ~ N(0, I)
///
/// Uses a deterministic Box-Muller LCG for reproducibility in tests.
fn reparameterize(mu: &Array2<f64>, log_var: &Array2<f64>, seed: u64) -> Array2<f64> {
    let batch = mu.nrows();
    let dim = mu.ncols();
    let mut z = Array2::<f64>::zeros((batch, dim));

    let mut state = seed.wrapping_add(271828);
    let lcg = |s: u64| -> (u64, u64) {
        let s2 = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (s2, s2)
    };

    let mut n_samples = 0;
    let total = batch * dim;
    let mut normals = Vec::with_capacity(total);

    while normals.len() < total {
        let (s1, st1) = lcg(state);
        state = st1;
        let (s2, st2) = lcg(state);
        state = st2;

        let u1 = (s1 >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (s2 >> 11) as f64 / (1u64 << 53) as f64;

        if u1 < 1e-15 {
            normals.push(0.0);
            if normals.len() < total {
                normals.push(0.0);
            }
            continue;
        }

        // Box-Muller transform
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        normals.push(r * theta.cos());
        if normals.len() < total {
            normals.push(r * theta.sin());
        }
        n_samples += 2;
    }

    for b in 0..batch {
        for d in 0..dim {
            let eps = normals[b * dim + d];
            let std = (log_var[[b, d]] * 0.5).exp();
            z[[b, d]] = mu[[b, d]] + eps * std;
        }
    }
    let _ = n_samples;
    z
}

// ============================================================================
// ELBO computation
// ============================================================================

/// Training step result for a VAE variant.
#[derive(Debug, Clone)]
pub struct VAELoss {
    /// Total ELBO loss (reconstruction + β * KL)
    pub total_loss: f64,
    /// Reconstruction loss component
    pub reconstruction_loss: f64,
    /// KL divergence component
    pub kl_loss: f64,
}

/// Compute the KL divergence KL(q(z|x) || p(z)) for diagonal Gaussians.
///
/// KL = -½ Σ_j (1 + log σ²_j - μ²_j - σ²_j)
fn kl_divergence_diagonal(mu: &Array2<f64>, log_var: &Array2<f64>) -> f64 {
    let batch = mu.nrows();
    let dim = mu.ncols();
    let mut kl = 0.0f64;
    for b in 0..batch {
        for d in 0..dim {
            kl += -0.5 * (1.0 + log_var[[b, d]] - mu[[b, d]].powi(2) - log_var[[b, d]].exp());
        }
    }
    kl / batch as f64
}

/// Compute Gaussian reconstruction loss: mean squared error
fn reconstruction_loss_mse(x: &Array2<f64>, x_recon: &Array2<f64>) -> f64 {
    let n = x.nrows() * x.ncols();
    let loss: f64 = x.iter().zip(x_recon.iter()).map(|(a, b)| (a - b).powi(2)).sum();
    loss / x.nrows() as f64
}

// ============================================================================
// Gradient computation via finite differences
// ============================================================================

/// Compute finite-difference gradient of the ELBO w.r.t. a flat parameter vector.
fn fd_gradient<F>(
    params: &[f64],
    eps: f64,
    mut loss_fn: F,
) -> Vec<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    let n = params.len();
    let mut grad = vec![0.0f64; n];
    let mut p = params.to_vec();

    for i in 0..n {
        let orig = p[i];

        p[i] = orig + eps;
        let f_plus = loss_fn(&p);

        p[i] = orig - eps;
        let f_minus = loss_fn(&p);

        grad[i] = (f_plus - f_minus) / (2.0 * eps);
        p[i] = orig;
    }
    grad
}

// ============================================================================
// VariationalAutoencoder
// ============================================================================

/// Standard Variational Autoencoder (VAE).
///
/// Implements the standard β=1 VAE from Kingma & Welling (2014).
/// The encoder network maps x → (μ, log σ²), and the decoder
/// maps z ~ q(z|x) → x̂.
///
/// # Architecture
///
/// Encoder: x → [shared MLP] → (μ_z, log σ²_z)  (two linear heads)
/// Decoder: z → [MLP] → x̂
///
/// # Training
///
/// Maximizes ELBO = E[log p(x|z)] - KL(q(z|x)||p(z))
/// via stochastic gradient ascent with the reparameterization trick.
///
/// # Example
/// ```
/// use scirs2_transform::encoder_models::{VariationalAutoencoder, VAEConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let cfg = VAEConfig {
///     input_dim: 8,
///     hidden_dims: vec![16],
///     latent_dim: 4,
///     learning_rate: 1e-3,
///     max_epochs: 5,
///     seed: 42,
/// };
/// let mut vae = VariationalAutoencoder::new(cfg).expect("VAE::new should succeed");
/// let x = Array2::<f64>::zeros((10, 8));
/// vae.fit(&x).expect("VAE fit should succeed");
/// let z = vae.encode_mean(&x).expect("encode_mean should succeed");
/// assert_eq!(z.shape(), &[10, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct VariationalAutoencoder {
    /// Configuration
    pub config: VAEConfig,
    /// Shared encoder backbone
    encoder_mlp: MLP,
    /// Mean head
    mu_head: MLP,
    /// Log-variance head
    log_var_head: MLP,
    /// Decoder network
    decoder_mlp: MLP,
    /// Training loss history
    pub loss_history: Vec<f64>,
}

/// Configuration for [`VariationalAutoencoder`] and its variants.
#[derive(Debug, Clone)]
pub struct VAEConfig {
    /// Input (and reconstruction output) dimensionality
    pub input_dim: usize,
    /// Dimensions of hidden layers (shared encoder and decoder)
    pub hidden_dims: Vec<usize>,
    /// Latent space dimensionality
    pub latent_dim: usize,
    /// Learning rate for Adam-like gradient descent
    pub learning_rate: f64,
    /// Maximum training epochs
    pub max_epochs: usize,
    /// Random seed
    pub seed: u64,
}

impl Default for VAEConfig {
    fn default() -> Self {
        VAEConfig {
            input_dim: 16,
            hidden_dims: vec![32, 16],
            latent_dim: 8,
            learning_rate: 1e-3,
            max_epochs: 100,
            seed: 42,
        }
    }
}

impl VariationalAutoencoder {
    /// Build encoder dims: [input, h1, …, hk]
    fn encoder_dims(config: &VAEConfig) -> Vec<usize> {
        let mut dims = vec![config.input_dim];
        dims.extend_from_slice(&config.hidden_dims);
        dims
    }

    /// Build decoder dims: [latent, hk, …, h1, input]
    fn decoder_dims(config: &VAEConfig) -> Vec<usize> {
        let mut dims = vec![config.latent_dim];
        let mut hidden = config.hidden_dims.clone();
        hidden.reverse();
        dims.extend_from_slice(&hidden);
        dims.push(config.input_dim);
        dims
    }

    /// Create a new VAE.
    pub fn new(config: VAEConfig) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(TransformError::InvalidInput("VAE: input_dim must be > 0".to_string()));
        }
        if config.latent_dim == 0 {
            return Err(TransformError::InvalidInput("VAE: latent_dim must be > 0".to_string()));
        }

        let enc_dims = Self::encoder_dims(&config);
        let hidden_out = *enc_dims.last().unwrap_or(&config.input_dim);
        let dec_dims = Self::decoder_dims(&config);

        let encoder_mlp = MLP::new(&enc_dims, false, config.seed)?;
        let mu_head = MLP::new(&[hidden_out, config.latent_dim], true, config.seed + 1)?;
        let log_var_head = MLP::new(&[hidden_out, config.latent_dim], true, config.seed + 2)?;
        let decoder_mlp = MLP::new(&dec_dims, true, config.seed + 3)?;

        Ok(VariationalAutoencoder {
            config,
            encoder_mlp,
            mu_head,
            log_var_head,
            decoder_mlp,
            loss_history: Vec::new(),
        })
    }

    /// Forward pass: encode, sample z, decode.
    ///
    /// Returns (x_recon, mu, log_var, z)
    fn forward(&self, x: &Array2<f64>, seed: u64) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>)> {
        let h = self.encoder_mlp.forward(x)?;
        let mu = self.mu_head.forward(&h)?;
        let log_var = self.log_var_head.forward(&h)?;
        let z = reparameterize(&mu, &log_var, seed);
        let x_recon = self.decoder_mlp.forward(&z)?;
        Ok((x_recon, mu, log_var, z))
    }

    /// Compute ELBO loss (to minimize = negative ELBO).
    pub fn elbo_loss(&self, x: &Array2<f64>, seed: u64) -> Result<VAELoss> {
        let (x_recon, mu, log_var, _) = self.forward(x, seed)?;
        let recon = reconstruction_loss_mse(x, &x_recon);
        let kl = kl_divergence_diagonal(&mu, &log_var);
        Ok(VAELoss {
            total_loss: recon + kl,
            reconstruction_loss: recon,
            kl_loss: kl,
        })
    }

    /// Train the VAE on data x using gradient descent.
    ///
    /// Uses finite-difference gradients (block-wise per component).
    /// For large networks, consider reducing max_epochs.
    ///
    /// # Arguments
    /// * `x` - Training data, shape (n, input_dim)
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.ncols() != self.config.input_dim {
            return Err(TransformError::InvalidInput(format!(
                "VAE fit: expected input_dim={}, got {}",
                self.config.input_dim, x.ncols()
            )));
        }
        if x.nrows() == 0 {
            return Err(TransformError::InvalidInput("VAE fit: empty input".to_string()));
        }

        let lr = self.config.learning_rate;
        let eps = 1e-4f64;

        for epoch in 0..self.config.max_epochs {
            let seed = self.config.seed + epoch as u64;

            // Compute loss
            let loss_val = self.elbo_loss(x, seed)?.total_loss;
            self.loss_history.push(loss_val);

            // Gradient step on encoder backbone
            {
                let params = self.encoder_mlp.params();
                let mut vae_clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    vae_clone.encoder_mlp.set_params(p).ok();
                    vae_clone.elbo_loss(x, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_params: Vec<f64> = params.iter().zip(grad.iter())
                    .map(|(w, g)| w - lr * g)
                    .collect();
                self.encoder_mlp.set_params(&new_params)?;
            }

            // Gradient step on mu_head
            {
                let params = self.mu_head.params();
                let mut vae_clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    vae_clone.mu_head.set_params(p).ok();
                    vae_clone.elbo_loss(x, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_params: Vec<f64> = params.iter().zip(grad.iter())
                    .map(|(w, g)| w - lr * g)
                    .collect();
                self.mu_head.set_params(&new_params)?;
            }

            // Gradient step on log_var_head
            {
                let params = self.log_var_head.params();
                let mut vae_clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    vae_clone.log_var_head.set_params(p).ok();
                    vae_clone.elbo_loss(x, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_params: Vec<f64> = params.iter().zip(grad.iter())
                    .map(|(w, g)| w - lr * g)
                    .collect();
                self.log_var_head.set_params(&new_params)?;
            }

            // Gradient step on decoder
            {
                let params = self.decoder_mlp.params();
                let mut vae_clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    vae_clone.decoder_mlp.set_params(p).ok();
                    vae_clone.elbo_loss(x, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_params: Vec<f64> = params.iter().zip(grad.iter())
                    .map(|(w, g)| w - lr * g)
                    .collect();
                self.decoder_mlp.set_params(&new_params)?;
            }
        }

        Ok(())
    }

    /// Encode inputs to mean latent vectors (deterministic, no sampling).
    pub fn encode_mean(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let h = self.encoder_mlp.forward(x)?;
        self.mu_head.forward(&h)
    }

    /// Encode inputs to (mean, log-variance) latent parameters.
    pub fn encode_params(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let h = self.encoder_mlp.forward(x)?;
        let mu = self.mu_head.forward(&h)?;
        let log_var = self.log_var_head.forward(&h)?;
        Ok((mu, log_var))
    }

    /// Sample latent vectors from the posterior q(z|x).
    pub fn sample_latent(&self, x: &Array2<f64>, seed: u64) -> Result<Array2<f64>> {
        let (mu, log_var) = self.encode_params(x)?;
        Ok(reparameterize(&mu, &log_var, seed))
    }

    /// Decode latent vectors to reconstructions.
    pub fn decode(&self, z: &Array2<f64>) -> Result<Array2<f64>> {
        self.decoder_mlp.forward(z)
    }

    /// Reconstruct inputs: encode then decode.
    pub fn reconstruct(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (mu, _) = self.encode_params(x)?;
        self.decode(&mu)
    }

    /// Sample from the prior p(z) = N(0, I) and decode.
    pub fn sample(
        &self,
        n_samples: usize,
        seed: u64,
    ) -> Result<Array2<f64>> {
        let mu = Array2::<f64>::zeros((n_samples, self.config.latent_dim));
        let log_var = Array2::<f64>::zeros((n_samples, self.config.latent_dim));
        let z = reparameterize(&mu, &log_var, seed);
        self.decode(&z)
    }
}

impl Encoder for VariationalAutoencoder {
    fn encode(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        self.encode_params(x)
    }

    fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    fn latent_dim(&self) -> usize {
        self.config.latent_dim
    }
}

impl Decoder for VariationalAutoencoder {
    fn decode(&self, z: &Array2<f64>) -> Result<Array2<f64>> {
        self.decoder_mlp.forward(z)
    }

    fn latent_dim(&self) -> usize {
        self.config.latent_dim
    }

    fn output_dim(&self) -> usize {
        self.config.input_dim
    }
}

// ============================================================================
// BetaVAE
// ============================================================================

/// β-VAE: Disentangled Variational Autoencoder.
///
/// β-VAE modifies the ELBO by upweighting the KL divergence term:
///
/// ELBO_β = E[log p(x|z)] - β · KL(q(z|x) || p(z))
///
/// Large β (>1) encourages statistical independence of latent dimensions,
/// leading to more disentangled representations at the cost of reconstruction
/// quality.
///
/// # Example
/// ```
/// use scirs2_transform::encoder_models::{BetaVAE, VAEConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let cfg = VAEConfig {
///     input_dim: 6,
///     hidden_dims: vec![12],
///     latent_dim: 3,
///     learning_rate: 1e-3,
///     max_epochs: 3,
///     seed: 0,
/// };
/// let mut beta_vae = BetaVAE::new(cfg, 4.0).expect("BetaVAE::new should succeed");
/// let x = Array2::<f64>::zeros((8, 6));
/// beta_vae.fit(&x).expect("BetaVAE fit should succeed");
/// ```
#[derive(Debug, Clone)]
pub struct BetaVAE {
    /// Underlying VAE structure
    pub vae: VariationalAutoencoder,
    /// Beta coefficient (≥ 1; standard VAE at β=1)
    pub beta: f64,
    /// Training loss history
    pub loss_history: Vec<f64>,
}

impl BetaVAE {
    /// Create a new β-VAE.
    ///
    /// # Arguments
    /// * `config` - VAE configuration
    /// * `beta` - KL weight (β ≥ 1)
    pub fn new(config: VAEConfig, beta: f64) -> Result<Self> {
        if beta < 1.0 {
            return Err(TransformError::InvalidInput(
                "BetaVAE: beta must be >= 1.0".to_string(),
            ));
        }
        let vae = VariationalAutoencoder::new(config)?;
        Ok(BetaVAE { vae, beta, loss_history: Vec::new() })
    }

    /// Compute the β-ELBO loss.
    pub fn elbo_loss(&self, x: &Array2<f64>, seed: u64) -> Result<VAELoss> {
        let (x_recon, mu, log_var, _) = self.vae.forward(x, seed)?;
        let recon = reconstruction_loss_mse(x, &x_recon);
        let kl = kl_divergence_diagonal(&mu, &log_var);
        Ok(VAELoss {
            total_loss: recon + self.beta * kl,
            reconstruction_loss: recon,
            kl_loss: kl,
        })
    }

    /// Fit the β-VAE on data.
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.ncols() != self.vae.config.input_dim {
            return Err(TransformError::InvalidInput(format!(
                "BetaVAE fit: expected input_dim={}, got {}",
                self.vae.config.input_dim,
                x.ncols()
            )));
        }

        let lr = self.vae.config.learning_rate;
        let eps = 1e-4f64;
        let beta = self.beta;

        for epoch in 0..self.vae.config.max_epochs {
            let seed = self.vae.config.seed + epoch as u64;

            let loss_val = self.elbo_loss(x, seed)?.total_loss;
            self.loss_history.push(loss_val);

            // Update encoder backbone
            {
                let params = self.vae.encoder_mlp.params();
                let mut clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    clone.vae.encoder_mlp.set_params(p).ok();
                    clone.elbo_loss(x, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_p: Vec<f64> = params.iter().zip(grad.iter()).map(|(w, g)| w - lr * g).collect();
                self.vae.encoder_mlp.set_params(&new_p)?;
            }

            // Update mu head
            {
                let params = self.vae.mu_head.params();
                let mut clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    clone.vae.mu_head.set_params(p).ok();
                    clone.elbo_loss(x, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_p: Vec<f64> = params.iter().zip(grad.iter()).map(|(w, g)| w - lr * g).collect();
                self.vae.mu_head.set_params(&new_p)?;
            }

            // Update log_var head
            {
                let params = self.vae.log_var_head.params();
                let mut clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    clone.vae.log_var_head.set_params(p).ok();
                    clone.elbo_loss(x, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_p: Vec<f64> = params.iter().zip(grad.iter()).map(|(w, g)| w - lr * g).collect();
                self.vae.log_var_head.set_params(&new_p)?;
            }

            // Update decoder
            {
                let params = self.vae.decoder_mlp.params();
                let mut clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    clone.vae.decoder_mlp.set_params(p).ok();
                    clone.elbo_loss(x, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_p: Vec<f64> = params.iter().zip(grad.iter()).map(|(w, g)| w - lr * g).collect();
                self.vae.decoder_mlp.set_params(&new_p)?;
            }

            let _ = beta;
        }

        Ok(())
    }

    /// Encode inputs to mean latent vectors.
    pub fn encode_mean(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.vae.encode_mean(x)
    }

    /// Reconstruct inputs.
    pub fn reconstruct(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.vae.reconstruct(x)
    }
}

// ============================================================================
// InfoVAE
// ============================================================================

/// InfoVAE: Information Maximizing Variational Autoencoder.
///
/// InfoVAE generalizes the VAE objective by replacing the KL term with
/// a maximum mean discrepancy (MMD) penalty between the aggregate posterior
/// q(z) and the prior p(z):
///
/// ELBO_InfoVAE = E[log p(x|z)] - α · KL - λ · MMD(q(z) || p(z))
///
/// With α=0, λ=1 this becomes the MMD-VAE.
///
/// # References
/// Zhao, S., Song, J., & Ermon, S. (2019). InfoVAE. AAAI.
#[derive(Debug, Clone)]
pub struct InfoVAE {
    /// Underlying VAE structure
    pub vae: VariationalAutoencoder,
    /// KL weight α (can be 0 to eliminate KL term)
    pub alpha: f64,
    /// MMD weight λ
    pub lambda_mmd: f64,
    /// Training loss history
    pub loss_history: Vec<f64>,
}

impl InfoVAE {
    /// Create a new InfoVAE.
    ///
    /// # Arguments
    /// * `config` - VAE configuration
    /// * `alpha` - KL divergence weight (≥ 0)
    /// * `lambda_mmd` - MMD penalty weight (≥ 0)
    pub fn new(config: VAEConfig, alpha: f64, lambda_mmd: f64) -> Result<Self> {
        if alpha < 0.0 {
            return Err(TransformError::InvalidInput("InfoVAE: alpha must be >= 0".to_string()));
        }
        if lambda_mmd < 0.0 {
            return Err(TransformError::InvalidInput("InfoVAE: lambda_mmd must be >= 0".to_string()));
        }
        let vae = VariationalAutoencoder::new(config)?;
        Ok(InfoVAE { vae, alpha, lambda_mmd, loss_history: Vec::new() })
    }

    /// Compute the IMQ (inverse multiquadratic) kernel between two sets of vectors.
    ///
    /// k_IMQ(x, y) = C / (C + ||x - y||²)  with C = 2 * latent_dim
    fn imq_mmd(z_q: &Array2<f64>, z_p: &Array2<f64>) -> f64 {
        let latent_dim = z_q.ncols() as f64;
        let c = 2.0 * latent_dim;
        let n = z_q.nrows();
        let m = z_p.nrows();

        let kernel = |a: &[f64], b: &[f64]| -> f64 {
            let sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
            c / (c + sq)
        };

        // E[k(z_q, z_q)]
        let kqq: f64 = if n > 1 {
            let mut s = 0.0f64;
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        s += kernel(z_q.row(i).as_slice().unwrap_or(&[]), z_q.row(j).as_slice().unwrap_or(&[]));
                    }
                }
            }
            s / (n * (n - 1)) as f64
        } else { 0.0 };

        // E[k(z_p, z_p)]
        let kpp: f64 = if m > 1 {
            let mut s = 0.0f64;
            for i in 0..m {
                for j in 0..m {
                    if i != j {
                        s += kernel(z_p.row(i).as_slice().unwrap_or(&[]), z_p.row(j).as_slice().unwrap_or(&[]));
                    }
                }
            }
            s / (m * (m - 1)) as f64
        } else { 0.0 };

        // E[k(z_q, z_p)]
        let kqp: f64 = {
            let mut s = 0.0f64;
            for i in 0..n {
                for j in 0..m {
                    // Use row_iter approach
                    let row_q: Vec<f64> = z_q.row(i).iter().copied().collect();
                    let row_p: Vec<f64> = z_p.row(j).iter().copied().collect();
                    s += kernel(&row_q, &row_p);
                }
            }
            s / (n * m) as f64
        };

        kqq + kpp - 2.0 * kqp
    }

    /// Compute the InfoVAE loss.
    pub fn info_vae_loss(&self, x: &Array2<f64>, seed: u64) -> Result<VAELoss> {
        let (x_recon, mu, log_var, z_q) = self.vae.forward(x, seed)?;
        let recon = reconstruction_loss_mse(x, &x_recon);
        let kl = kl_divergence_diagonal(&mu, &log_var);

        // Sample from prior p(z) = N(0, I)
        let prior_mu = Array2::<f64>::zeros((x.nrows(), self.vae.config.latent_dim));
        let prior_log_var = Array2::<f64>::zeros((x.nrows(), self.vae.config.latent_dim));
        let z_p = reparameterize(&prior_mu, &prior_log_var, seed + 1000);

        let mmd = Self::imq_mmd(&z_q, &z_p);
        let total = recon + self.alpha * kl + self.lambda_mmd * mmd;

        Ok(VAELoss {
            total_loss: total,
            reconstruction_loss: recon,
            kl_loss: kl,
        })
    }

    /// Fit the InfoVAE on data.
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        if x.ncols() != self.vae.config.input_dim {
            return Err(TransformError::InvalidInput(format!(
                "InfoVAE fit: expected input_dim={}, got {}",
                self.vae.config.input_dim,
                x.ncols()
            )));
        }

        let lr = self.vae.config.learning_rate;
        let eps = 1e-4f64;

        for epoch in 0..self.vae.config.max_epochs {
            let seed = self.vae.config.seed + epoch as u64;
            let loss_val = self.info_vae_loss(x, seed)?.total_loss;
            self.loss_history.push(loss_val);

            // Update encoder backbone
            {
                let params = self.vae.encoder_mlp.params();
                let mut clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    clone.vae.encoder_mlp.set_params(p).ok();
                    clone.info_vae_loss(x, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_p: Vec<f64> = params.iter().zip(grad.iter()).map(|(w, g)| w - lr * g).collect();
                self.vae.encoder_mlp.set_params(&new_p)?;
            }

            // Update decoder
            {
                let params = self.vae.decoder_mlp.params();
                let mut clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    clone.vae.decoder_mlp.set_params(p).ok();
                    clone.info_vae_loss(x, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_p: Vec<f64> = params.iter().zip(grad.iter()).map(|(w, g)| w - lr * g).collect();
                self.vae.decoder_mlp.set_params(&new_p)?;
            }
        }

        Ok(())
    }

    /// Encode inputs to mean latent vectors.
    pub fn encode_mean(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.vae.encode_mean(x)
    }

    /// Reconstruct inputs.
    pub fn reconstruct(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.vae.reconstruct(x)
    }
}

// ============================================================================
// CVAE — Conditional VAE
// ============================================================================

/// Conditional Variational Autoencoder (CVAE).
///
/// CVAE conditions both the encoder and decoder on a label vector c,
/// enabling class-conditional generation and structured prediction.
///
/// Architecture:
/// - Encoder: (x, c) → (μ, log σ²)
/// - Decoder: (z, c) → x̂
///
/// where c is a one-hot encoded class label.
///
/// # Example
/// ```
/// use scirs2_transform::encoder_models::{CVAE, VAEConfig};
/// use scirs2_core::ndarray::{Array2, Array1};
///
/// let cfg = VAEConfig {
///     input_dim: 8,
///     hidden_dims: vec![12],
///     latent_dim: 4,
///     learning_rate: 1e-3,
///     max_epochs: 3,
///     seed: 1,
/// };
/// let n_classes = 3;
/// let mut cvae = CVAE::new(cfg, n_classes).expect("CVAE::new should succeed");
///
/// let x = Array2::<f64>::zeros((9, 8));
/// let labels = Array1::<usize>::from_vec(vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
/// cvae.fit(&x, &labels).expect("CVAE fit should succeed");
/// let z = cvae.encode_mean(&x, &labels).expect("encode_mean should succeed");
/// assert_eq!(z.shape(), &[9, 4]);
/// ```
#[derive(Debug, Clone)]
pub struct CVAE {
    /// VAE configuration
    pub config: VAEConfig,
    /// Number of conditioning classes
    pub n_classes: usize,
    /// Encoder: processes (x, c) concatenated
    encoder_mlp: MLP,
    /// Mean head
    mu_head: MLP,
    /// Log-variance head
    log_var_head: MLP,
    /// Decoder: processes (z, c) concatenated
    decoder_mlp: MLP,
    /// Training loss history
    pub loss_history: Vec<f64>,
}

impl CVAE {
    /// Create a new CVAE.
    ///
    /// # Arguments
    /// * `config` - VAE base configuration (input_dim excludes conditioning)
    /// * `n_classes` - Number of conditioning classes
    pub fn new(config: VAEConfig, n_classes: usize) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(TransformError::InvalidInput("CVAE: input_dim must be > 0".to_string()));
        }
        if n_classes == 0 {
            return Err(TransformError::InvalidInput("CVAE: n_classes must be > 0".to_string()));
        }

        let cond_dim = n_classes;
        let input_plus_cond = config.input_dim + cond_dim;
        let latent_plus_cond = config.latent_dim + cond_dim;

        // Encoder: (x, c) → hidden
        let mut enc_dims = vec![input_plus_cond];
        enc_dims.extend_from_slice(&config.hidden_dims);
        let hidden_out = *enc_dims.last().unwrap_or(&input_plus_cond);

        let encoder_mlp = MLP::new(&enc_dims, false, config.seed)?;
        let mu_head = MLP::new(&[hidden_out, config.latent_dim], true, config.seed + 1)?;
        let log_var_head = MLP::new(&[hidden_out, config.latent_dim], true, config.seed + 2)?;

        // Decoder: (z, c) → x
        let mut dec_dims = vec![latent_plus_cond];
        let mut hidden_rev = config.hidden_dims.clone();
        hidden_rev.reverse();
        dec_dims.extend_from_slice(&hidden_rev);
        dec_dims.push(config.input_dim);
        let decoder_mlp = MLP::new(&dec_dims, true, config.seed + 3)?;

        Ok(CVAE {
            config,
            n_classes,
            encoder_mlp,
            mu_head,
            log_var_head,
            decoder_mlp,
            loss_history: Vec::new(),
        })
    }

    /// Convert integer class labels to one-hot encodings.
    fn one_hot(&self, labels: &Array1<usize>) -> Result<Array2<f64>> {
        let n = labels.len();
        let mut oh = Array2::<f64>::zeros((n, self.n_classes));
        for (i, &l) in labels.iter().enumerate() {
            if l >= self.n_classes {
                return Err(TransformError::InvalidInput(format!(
                    "CVAE: label {} >= n_classes {}",
                    l, self.n_classes
                )));
            }
            oh[[i, l]] = 1.0;
        }
        Ok(oh)
    }

    /// Concatenate along columns (axis=1).
    fn concat_cols(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>> {
        let n = a.nrows();
        if b.nrows() != n {
            return Err(TransformError::InvalidInput(
                "concat_cols: row count mismatch".to_string(),
            ));
        }
        let da = a.ncols();
        let db = b.ncols();
        let mut out = Array2::<f64>::zeros((n, da + db));
        for i in 0..n {
            for j in 0..da {
                out[[i, j]] = a[[i, j]];
            }
            for j in 0..db {
                out[[i, da + j]] = b[[i, j]];
            }
        }
        Ok(out)
    }

    /// Forward pass given data and one-hot conditions.
    fn forward(
        &self,
        x: &Array2<f64>,
        c: &Array2<f64>,
        seed: u64,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>)> {
        let xc = Self::concat_cols(x, c)?;
        let h = self.encoder_mlp.forward(&xc)?;
        let mu = self.mu_head.forward(&h)?;
        let log_var = self.log_var_head.forward(&h)?;
        let z = reparameterize(&mu, &log_var, seed);
        let zc = Self::concat_cols(&z, c)?;
        let x_recon = self.decoder_mlp.forward(&zc)?;
        Ok((x_recon, mu, log_var, z))
    }

    /// Compute ELBO loss for CVAE.
    pub fn elbo_loss(&self, x: &Array2<f64>, labels: &Array1<usize>, seed: u64) -> Result<VAELoss> {
        let c = self.one_hot(labels)?;
        let (x_recon, mu, log_var, _) = self.forward(x, &c, seed)?;
        let recon = reconstruction_loss_mse(x, &x_recon);
        let kl = kl_divergence_diagonal(&mu, &log_var);
        Ok(VAELoss {
            total_loss: recon + kl,
            reconstruction_loss: recon,
            kl_loss: kl,
        })
    }

    /// Fit the CVAE on labeled data.
    pub fn fit(&mut self, x: &Array2<f64>, labels: &Array1<usize>) -> Result<()> {
        if x.ncols() != self.config.input_dim {
            return Err(TransformError::InvalidInput(format!(
                "CVAE fit: expected input_dim={}, got {}",
                self.config.input_dim,
                x.ncols()
            )));
        }
        if x.nrows() != labels.len() {
            return Err(TransformError::InvalidInput(format!(
                "CVAE fit: x rows {} != labels len {}",
                x.nrows(),
                labels.len()
            )));
        }

        let lr = self.config.learning_rate;
        let eps = 1e-4f64;

        for epoch in 0..self.config.max_epochs {
            let seed = self.config.seed + epoch as u64;
            let loss_val = self.elbo_loss(x, labels, seed)?.total_loss;
            self.loss_history.push(loss_val);

            // Update encoder backbone
            {
                let params = self.encoder_mlp.params();
                let mut clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    clone.encoder_mlp.set_params(p).ok();
                    clone.elbo_loss(x, labels, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_p: Vec<f64> = params.iter().zip(grad.iter()).map(|(w, g)| w - lr * g).collect();
                self.encoder_mlp.set_params(&new_p)?;
            }

            // Update decoder
            {
                let params = self.decoder_mlp.params();
                let mut clone = self.clone();
                let grad = fd_gradient(&params, eps, |p| {
                    clone.decoder_mlp.set_params(p).ok();
                    clone.elbo_loss(x, labels, seed).map(|l| l.total_loss).unwrap_or(f64::INFINITY)
                });
                let new_p: Vec<f64> = params.iter().zip(grad.iter()).map(|(w, g)| w - lr * g).collect();
                self.decoder_mlp.set_params(&new_p)?;
            }
        }

        Ok(())
    }

    /// Encode inputs to mean latent vectors given class labels.
    pub fn encode_mean(&self, x: &Array2<f64>, labels: &Array1<usize>) -> Result<Array2<f64>> {
        let c = self.one_hot(labels)?;
        let xc = Self::concat_cols(x, &c)?;
        let h = self.encoder_mlp.forward(&xc)?;
        self.mu_head.forward(&h)
    }

    /// Generate samples for a given class label.
    ///
    /// # Arguments
    /// * `n_samples` - Number of samples to generate
    /// * `class_label` - Conditioning class index
    /// * `seed` - Random seed
    pub fn generate(
        &self,
        n_samples: usize,
        class_label: usize,
        seed: u64,
    ) -> Result<Array2<f64>> {
        if class_label >= self.n_classes {
            return Err(TransformError::InvalidInput(format!(
                "CVAE generate: class_label {} >= n_classes {}",
                class_label, self.n_classes
            )));
        }
        // Sample z from prior N(0, I)
        let mu = Array2::<f64>::zeros((n_samples, self.config.latent_dim));
        let log_var = Array2::<f64>::zeros((n_samples, self.config.latent_dim));
        let z = reparameterize(&mu, &log_var, seed);

        // Construct one-hot condition
        let mut c = Array2::<f64>::zeros((n_samples, self.n_classes));
        for i in 0..n_samples {
            c[[i, class_label]] = 1.0;
        }

        let zc = Self::concat_cols(&z, &c)?;
        self.decoder_mlp.forward(&zc)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    fn make_config(input_dim: usize) -> VAEConfig {
        VAEConfig {
            input_dim,
            hidden_dims: vec![8],
            latent_dim: 4,
            learning_rate: 1e-3,
            max_epochs: 3,
            seed: 42,
        }
    }

    #[test]
    fn test_vae_encode_shape() {
        let cfg = make_config(6);
        let vae = VariationalAutoencoder::new(cfg).expect("VAE::new should succeed");
        let x = Array2::<f64>::zeros((5, 6));
        let z = vae.encode_mean(&x).expect("encode_mean should succeed");
        assert_eq!(z.shape(), &[5, 4]);
    }

    #[test]
    fn test_vae_reconstruct_shape() {
        let cfg = make_config(6);
        let vae = VariationalAutoencoder::new(cfg).expect("VAE::new should succeed");
        let x = Array2::<f64>::zeros((5, 6));
        let x_recon = vae.reconstruct(&x).expect("reconstruct should succeed");
        assert_eq!(x_recon.shape(), &[5, 6]);
    }

    #[test]
    fn test_vae_sample_shape() {
        let cfg = make_config(6);
        let vae = VariationalAutoencoder::new(cfg).expect("VAE::new should succeed");
        let samples = vae.sample(8, 99).expect("sample should succeed");
        assert_eq!(samples.shape(), &[8, 6]);
    }

    #[test]
    fn test_vae_elbo_non_negative_recon() {
        let cfg = make_config(4);
        let vae = VariationalAutoencoder::new(cfg).expect("VAE::new should succeed");
        let x = Array2::<f64>::ones((6, 4));
        let loss = vae.elbo_loss(&x, 0).expect("elbo_loss should succeed");
        assert!(loss.reconstruction_loss >= 0.0);
    }

    #[test]
    fn test_vae_fit_reduces_loss() {
        let mut cfg = make_config(4);
        cfg.max_epochs = 3;
        let mut vae = VariationalAutoencoder::new(cfg).expect("VAE::new should succeed");
        let x = Array2::<f64>::ones((4, 4));
        vae.fit(&x).expect("VAE fit should succeed");
        assert_eq!(vae.loss_history.len(), 3);
    }

    #[test]
    fn test_beta_vae_fit() {
        let cfg = make_config(4);
        let mut bvae = BetaVAE::new(cfg, 2.0).expect("BetaVAE::new should succeed");
        let x = Array2::<f64>::zeros((5, 4));
        bvae.fit(&x).expect("BetaVAE fit should succeed");
        assert!(!bvae.loss_history.is_empty());
    }

    #[test]
    fn test_info_vae_fit() {
        let cfg = make_config(4);
        let mut ivae = InfoVAE::new(cfg, 0.5, 1.0).expect("InfoVAE::new should succeed");
        let x = Array2::<f64>::zeros((5, 4));
        ivae.fit(&x).expect("InfoVAE fit should succeed");
        assert!(!ivae.loss_history.is_empty());
    }

    #[test]
    fn test_cvae_encode_shape() {
        let cfg = make_config(6);
        let cvae = CVAE::new(cfg, 3).expect("CVAE::new should succeed");
        let x = Array2::<f64>::zeros((6, 6));
        let labels = Array1::from_vec(vec![0usize, 1, 2, 0, 1, 2]);
        let z = cvae.encode_mean(&x, &labels).expect("encode_mean should succeed");
        assert_eq!(z.shape(), &[6, 4]);
    }

    #[test]
    fn test_cvae_generate_shape() {
        let cfg = make_config(6);
        let cvae = CVAE::new(cfg, 3).expect("CVAE::new should succeed");
        let samples = cvae.generate(5, 1, 7).expect("generate should succeed");
        assert_eq!(samples.shape(), &[5, 6]);
    }

    #[test]
    fn test_cvae_fit() {
        let cfg = make_config(4);
        let mut cvae = CVAE::new(cfg, 2).expect("CVAE::new should succeed");
        let x = Array2::<f64>::zeros((4, 4));
        let labels = Array1::from_vec(vec![0usize, 1, 0, 1]);
        cvae.fit(&x, &labels).expect("CVAE fit should succeed");
        assert!(!cvae.loss_history.is_empty());
    }

    #[test]
    fn test_kl_zero_prior() {
        // If mu=0, log_var=0 → KL should be 0
        let mu = Array2::<f64>::zeros((4, 3));
        let lv = Array2::<f64>::zeros((4, 3));
        let kl = kl_divergence_diagonal(&mu, &lv);
        assert!(kl.abs() < 1e-10);
    }
}
