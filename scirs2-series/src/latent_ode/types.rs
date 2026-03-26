//! Types for the Latent ODE / ODE-RNN module

/// Configuration for the Latent ODE model.
///
/// All fields are `pub` to allow direct construction.  Because this struct is
/// `#[non_exhaustive]`, downstream code *must* use `..Default::default()` to
/// fill in fields added in future versions.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct LatentOdeConfig {
    /// Dimensionality of the latent space.  Default: 16.
    pub latent_dim: usize,
    /// Number of hidden units in the ODE function MLP.  Default: 64.
    pub hidden_dim: usize,
    /// Number of hidden layers in the ODE function.  Default: 2.
    pub n_layers: usize,
    /// Number of reverse-time GRU steps used during recognition.  Default: 10.
    pub recognition_steps: usize,
    /// Relative ODE solver tolerance.  Default: 1e-3.
    pub rtol: f64,
    /// Absolute ODE solver tolerance.  Default: 1e-4.
    pub atol: f64,
    /// Learning rate for parameter updates.  Default: 1e-3.
    pub learning_rate: f64,
}

impl Default for LatentOdeConfig {
    fn default() -> Self {
        Self {
            latent_dim: 16,
            hidden_dim: 64,
            n_layers: 2,
            recognition_steps: 10,
            rtol: 1e-3,
            atol: 1e-4,
            learning_rate: 1e-3,
        }
    }
}

/// Output produced by [`crate::latent_ode::model::LatentOde`].
#[derive(Debug, Clone)]
pub struct LatentOdeResult {
    /// Query times for which predictions were made.
    pub predicted_times: Vec<f64>,
    /// Predicted observation values at each query time: `[time][dim]`.
    pub predicted_values: Vec<Vec<f64>>,
    /// Latent-space trajectory at each query time: `[time][latent_dim]`.
    pub latent_trajectory: Vec<Vec<f64>>,
    /// Reconstruction loss (MSE of decoded vs observed).
    pub reconstruction_loss: f64,
    /// KL divergence: `KL(q(z₀) || N(0, I))`.
    pub kl_divergence: f64,
}
