//! Federated Learning Primitives
//!
//! Implements core building blocks for federated learning, where a model is
//! trained collaboratively across multiple clients without sharing raw data.
//!
//! # Algorithms
//!
//! - **FedAvg**: Federated Averaging (McMahan et al., 2017)
//! - **Weighted aggregation**: Aggregate client updates proportionally to dataset size
//! - **Client selection**: Random or importance-based selection per round
//! - **Differential privacy**: Gaussian mechanism noise injection
//! - **Gradient compression**: Top-k sparsification to reduce communication
//!
//! # Example
//!
//! ```rust
//! use scirs2_neural::training::federated::{
//!     FederatedConfig, FederatedServer, ClientUpdate,
//! };
//! use scirs2_core::ndarray::{array, Array, IxDyn};
//!
//! let config = FederatedConfig::builder()
//!     .num_rounds(10)
//!     .clients_per_round(3)
//!     .build()
//!     .expect("valid config");
//!
//! // Start with global parameters
//! let global_params = vec![array![1.0_f64, 2.0, 3.0].into_dyn()];
//! let mut server = FederatedServer::new(config, global_params);
//!
//! // Simulate client updates
//! let updates = vec![
//!     ClientUpdate::new(0, vec![array![1.1, 2.1, 3.1].into_dyn()], 100),
//!     ClientUpdate::new(1, vec![array![0.9, 1.9, 2.9].into_dyn()], 200),
//! ];
//!
//! server.aggregate_round(&updates).expect("aggregation ok");
//! assert_eq!(server.current_round(), 1);
//! ```

use crate::error::{NeuralError, Result};
use scirs2_core::ndarray::{Array, ArrayD, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use scirs2_core::random::rngs::SmallRng;
use scirs2_core::random::{Rng, SeedableRng};
use std::fmt::{self, Debug, Display};

// ============================================================================
// Types
// ============================================================================

/// Strategy for selecting clients each round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientSelectionStrategy {
    /// Select clients uniformly at random.
    Random,
    /// Select clients proportionally to their dataset size.
    ImportanceBased,
    /// Select all available clients.
    All,
}

impl Display for ClientSelectionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Random => write!(f, "Random"),
            Self::ImportanceBased => write!(f, "ImportanceBased"),
            Self::All => write!(f, "All"),
        }
    }
}

/// Aggregation method for combining client updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    /// FedAvg: weighted average by number of samples.
    FedAvg,
    /// Simple (unweighted) mean of client parameters.
    SimpleMean,
    /// Median aggregation (more robust to outliers).
    Median,
}

impl Display for AggregationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FedAvg => write!(f, "FedAvg"),
            Self::SimpleMean => write!(f, "SimpleMean"),
            Self::Median => write!(f, "Median"),
        }
    }
}

// ============================================================================
// Differential Privacy
// ============================================================================

/// Configuration for differential privacy noise addition.
#[derive(Debug, Clone)]
pub struct DifferentialPrivacyConfig {
    /// Whether DP is enabled.
    pub enabled: bool,
    /// Noise multiplier (sigma). Larger = more privacy, less utility.
    pub noise_multiplier: f64,
    /// Maximum L2 norm for gradient clipping before noise addition.
    pub max_grad_norm: f64,
    /// Target delta for (epsilon, delta)-DP.
    pub delta: f64,
}

impl Default for DifferentialPrivacyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            noise_multiplier: 1.0,
            max_grad_norm: 1.0,
            delta: 1e-5,
        }
    }
}

/// Configuration for gradient compression.
#[derive(Debug, Clone)]
pub struct GradientCompressionConfig {
    /// Whether compression is enabled.
    pub enabled: bool,
    /// Fraction of top-k values to keep (0.0 to 1.0).
    pub top_k_fraction: f64,
}

impl Default for GradientCompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            top_k_fraction: 0.1,
        }
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for federated learning.
#[derive(Debug, Clone)]
pub struct FederatedConfig {
    /// Total number of communication rounds.
    pub num_rounds: usize,
    /// Number of clients to select per round.
    pub clients_per_round: usize,
    /// Client selection strategy.
    pub client_selection: ClientSelectionStrategy,
    /// Aggregation method.
    pub aggregation: AggregationMethod,
    /// Differential privacy configuration.
    pub dp_config: DifferentialPrivacyConfig,
    /// Gradient compression configuration.
    pub compression: GradientCompressionConfig,
    /// Local epochs per client per round.
    pub local_epochs: usize,
    /// Local learning rate for client training.
    pub local_lr: f64,
    /// Random seed for reproducibility.
    pub seed: Option<u64>,
}

impl Default for FederatedConfig {
    fn default() -> Self {
        Self {
            num_rounds: 100,
            clients_per_round: 10,
            client_selection: ClientSelectionStrategy::Random,
            aggregation: AggregationMethod::FedAvg,
            dp_config: DifferentialPrivacyConfig::default(),
            compression: GradientCompressionConfig::default(),
            local_epochs: 1,
            local_lr: 0.01,
            seed: None,
        }
    }
}

impl FederatedConfig {
    /// Create a builder.
    pub fn builder() -> FederatedConfigBuilder {
        FederatedConfigBuilder::default()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.num_rounds == 0 {
            return Err(NeuralError::InvalidArgument(
                "num_rounds must be > 0".into(),
            ));
        }
        if self.clients_per_round == 0 {
            return Err(NeuralError::InvalidArgument(
                "clients_per_round must be > 0".into(),
            ));
        }
        if self.local_epochs == 0 {
            return Err(NeuralError::InvalidArgument(
                "local_epochs must be > 0".into(),
            ));
        }
        if self.local_lr <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "local_lr must be positive".into(),
            ));
        }
        if self.dp_config.enabled && self.dp_config.noise_multiplier <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "noise_multiplier must be positive when DP is enabled".into(),
            ));
        }
        if self.dp_config.enabled && self.dp_config.max_grad_norm <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "max_grad_norm must be positive when DP is enabled".into(),
            ));
        }
        if self.compression.enabled && !(0.0..=1.0).contains(&self.compression.top_k_fraction) {
            return Err(NeuralError::InvalidArgument(
                "top_k_fraction must be in [0.0, 1.0]".into(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for [`FederatedConfig`].
#[derive(Debug, Clone, Default)]
pub struct FederatedConfigBuilder {
    config: FederatedConfig,
}

impl FederatedConfigBuilder {
    /// Set the number of communication rounds.
    pub fn num_rounds(mut self, n: usize) -> Self {
        self.config.num_rounds = n;
        self
    }

    /// Set the number of clients per round.
    pub fn clients_per_round(mut self, n: usize) -> Self {
        self.config.clients_per_round = n;
        self
    }

    /// Set the client selection strategy.
    pub fn client_selection(mut self, s: ClientSelectionStrategy) -> Self {
        self.config.client_selection = s;
        self
    }

    /// Set the aggregation method.
    pub fn aggregation(mut self, a: AggregationMethod) -> Self {
        self.config.aggregation = a;
        self
    }

    /// Enable differential privacy.
    pub fn differential_privacy(mut self, noise_multiplier: f64, max_grad_norm: f64) -> Self {
        self.config.dp_config.enabled = true;
        self.config.dp_config.noise_multiplier = noise_multiplier;
        self.config.dp_config.max_grad_norm = max_grad_norm;
        self
    }

    /// Set the DP delta.
    pub fn dp_delta(mut self, delta: f64) -> Self {
        self.config.dp_config.delta = delta;
        self
    }

    /// Enable gradient compression.
    pub fn gradient_compression(mut self, top_k_fraction: f64) -> Self {
        self.config.compression.enabled = true;
        self.config.compression.top_k_fraction = top_k_fraction;
        self
    }

    /// Set local epochs per client.
    pub fn local_epochs(mut self, n: usize) -> Self {
        self.config.local_epochs = n;
        self
    }

    /// Set local learning rate.
    pub fn local_lr(mut self, lr: f64) -> Self {
        self.config.local_lr = lr;
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.config.seed = Some(s);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> Result<FederatedConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

// ============================================================================
// Client Update
// ============================================================================

/// A client's update to send to the server.
#[derive(Debug, Clone)]
pub struct ClientUpdate {
    /// Client identifier.
    pub client_id: usize,
    /// Updated parameters (one array per parameter tensor).
    pub parameters: Vec<ArrayD<f64>>,
    /// Number of local training samples.
    pub num_samples: usize,
    /// Optional: local training loss (for diagnostics).
    pub local_loss: Option<f64>,
    /// Optional: local training metrics.
    pub metrics: std::collections::HashMap<String, f64>,
}

impl ClientUpdate {
    /// Create a new client update.
    pub fn new(client_id: usize, parameters: Vec<ArrayD<f64>>, num_samples: usize) -> Self {
        Self {
            client_id,
            parameters,
            num_samples,
            local_loss: None,
            metrics: std::collections::HashMap::new(),
        }
    }

    /// Set the local training loss.
    pub fn with_loss(mut self, loss: f64) -> Self {
        self.local_loss = Some(loss);
        self
    }

    /// Add a metric.
    pub fn with_metric(mut self, name: &str, value: f64) -> Self {
        self.metrics.insert(name.to_string(), value);
        self
    }
}

// ============================================================================
// Round statistics
// ============================================================================

/// Statistics for a single federated round.
#[derive(Debug, Clone)]
pub struct RoundStats {
    /// Round number (0-indexed).
    pub round: usize,
    /// Number of clients participating.
    pub num_clients: usize,
    /// Total samples across participating clients.
    pub total_samples: usize,
    /// Average local loss (if reported).
    pub avg_loss: Option<f64>,
    /// IDs of participating clients.
    pub client_ids: Vec<usize>,
}

// ============================================================================
// Federated Server
// ============================================================================

/// Federated learning server that coordinates the training process.
///
/// The server holds the global model parameters and aggregates updates
/// from participating clients each round.
#[derive(Debug, Clone)]
pub struct FederatedServer {
    /// Configuration.
    config: FederatedConfig,
    /// Current global parameters.
    global_params: Vec<ArrayD<f64>>,
    /// Current communication round.
    current_round: usize,
    /// History of round statistics.
    round_history: Vec<RoundStats>,
    /// RNG for client selection.
    rng: SmallRng,
}

impl FederatedServer {
    /// Create a new federated server with initial global parameters.
    pub fn new(config: FederatedConfig, global_params: Vec<ArrayD<f64>>) -> Self {
        let rng = match config.seed {
            Some(s) => SmallRng::seed_from_u64(s),
            None => SmallRng::seed_from_u64(42),
        };
        Self {
            config,
            global_params,
            current_round: 0,
            round_history: Vec::new(),
            rng,
        }
    }

    /// Get the current global parameters.
    pub fn global_params(&self) -> &[ArrayD<f64>] {
        &self.global_params
    }

    /// Get the current round number.
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Get the round history.
    pub fn round_history(&self) -> &[RoundStats] {
        &self.round_history
    }

    /// Whether training is complete (all rounds done).
    pub fn is_complete(&self) -> bool {
        self.current_round >= self.config.num_rounds
    }

    /// Select clients for the current round.
    ///
    /// # Arguments
    /// * `available_clients` - list of (client_id, num_samples)
    ///
    /// Returns the selected client IDs.
    pub fn select_clients(&mut self, available_clients: &[(usize, usize)]) -> Vec<usize> {
        if available_clients.is_empty() {
            return Vec::new();
        }

        let k = self.config.clients_per_round.min(available_clients.len());

        match self.config.client_selection {
            ClientSelectionStrategy::All => available_clients.iter().map(|&(id, _)| id).collect(),
            ClientSelectionStrategy::Random => {
                // Fisher-Yates partial shuffle
                let mut indices: Vec<usize> = (0..available_clients.len()).collect();
                for i in 0..k {
                    let j = i + self.rng.random_range(0..indices.len() - i);
                    indices.swap(i, j);
                }
                indices[..k]
                    .iter()
                    .map(|&i| available_clients[i].0)
                    .collect()
            }
            ClientSelectionStrategy::ImportanceBased => {
                // Select proportionally to dataset size
                let total: usize = available_clients.iter().map(|&(_, n)| n).sum();
                if total == 0 {
                    return available_clients
                        .iter()
                        .take(k)
                        .map(|&(id, _)| id)
                        .collect();
                }

                let mut selected = Vec::with_capacity(k);
                let mut used = vec![false; available_clients.len()];

                for _ in 0..k {
                    let threshold = self.rng.random_range(0..total);
                    let mut cumulative = 0usize;
                    for (idx, &(client_id, n)) in available_clients.iter().enumerate() {
                        if used[idx] {
                            continue;
                        }
                        cumulative += n;
                        if cumulative > threshold {
                            selected.push(client_id);
                            used[idx] = true;
                            break;
                        }
                    }
                    // If we didn't select anyone (edge case), pick first unused
                    if selected.len() < selected.capacity()
                        && selected.len() < k
                        && selected.len() == selected.len()
                    {
                        // already handled by break above
                    }
                }

                // Fill up if importance sampling missed some
                if selected.len() < k {
                    for (idx, &(client_id, _)) in available_clients.iter().enumerate() {
                        if selected.len() >= k {
                            break;
                        }
                        if !used[idx] {
                            selected.push(client_id);
                            used[idx] = true;
                        }
                    }
                }

                selected
            }
        }
    }

    /// Aggregate client updates for one round using the configured method.
    pub fn aggregate_round(&mut self, updates: &[ClientUpdate]) -> Result<()> {
        if updates.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "No client updates to aggregate".into(),
            ));
        }

        // Validate parameter shapes
        for update in updates {
            if update.parameters.len() != self.global_params.len() {
                return Err(NeuralError::ShapeMismatch(format!(
                    "Client {} has {} parameter tensors, expected {}",
                    update.client_id,
                    update.parameters.len(),
                    self.global_params.len()
                )));
            }
            for (i, param) in update.parameters.iter().enumerate() {
                if param.shape() != self.global_params[i].shape() {
                    return Err(NeuralError::ShapeMismatch(format!(
                        "Client {} param[{}] shape {:?} != global {:?}",
                        update.client_id,
                        i,
                        param.shape(),
                        self.global_params[i].shape()
                    )));
                }
            }
        }

        // Apply compression if enabled
        let processed_updates = if self.config.compression.enabled {
            updates
                .iter()
                .map(|u| {
                    let compressed = compress_gradients(
                        &u.parameters,
                        &self.global_params,
                        self.config.compression.top_k_fraction,
                    );
                    ClientUpdate {
                        client_id: u.client_id,
                        parameters: apply_compressed_delta(&self.global_params, &compressed),
                        num_samples: u.num_samples,
                        local_loss: u.local_loss,
                        metrics: u.metrics.clone(),
                    }
                })
                .collect::<Vec<_>>()
        } else {
            updates.to_vec()
        };

        // Aggregate
        match self.config.aggregation {
            AggregationMethod::FedAvg => self.fedavg_aggregate(&processed_updates),
            AggregationMethod::SimpleMean => self.simple_mean_aggregate(&processed_updates),
            AggregationMethod::Median => self.median_aggregate(&processed_updates),
        }?;

        // Apply differential privacy noise
        if self.config.dp_config.enabled {
            self.apply_dp_noise()?;
        }

        // Record round stats
        let total_samples: usize = updates.iter().map(|u| u.num_samples).sum();
        let avg_loss = {
            let losses: Vec<f64> = updates.iter().filter_map(|u| u.local_loss).collect();
            if losses.is_empty() {
                None
            } else {
                Some(losses.iter().sum::<f64>() / losses.len() as f64)
            }
        };

        self.round_history.push(RoundStats {
            round: self.current_round,
            num_clients: updates.len(),
            total_samples,
            avg_loss,
            client_ids: updates.iter().map(|u| u.client_id).collect(),
        });

        self.current_round += 1;
        Ok(())
    }

    /// FedAvg: weighted average by number of samples.
    fn fedavg_aggregate(&mut self, updates: &[ClientUpdate]) -> Result<()> {
        let total_samples: f64 = updates.iter().map(|u| u.num_samples as f64).sum();
        if total_samples < f64::EPSILON {
            return Err(NeuralError::ComputationError(
                "Total samples is zero".into(),
            ));
        }

        for p_idx in 0..self.global_params.len() {
            let mut aggregated = ArrayD::<f64>::zeros(self.global_params[p_idx].raw_dim());
            for update in updates {
                let weight = update.num_samples as f64 / total_samples;
                aggregated = aggregated + &update.parameters[p_idx] * weight;
            }
            self.global_params[p_idx] = aggregated;
        }
        Ok(())
    }

    /// Simple mean (unweighted).
    fn simple_mean_aggregate(&mut self, updates: &[ClientUpdate]) -> Result<()> {
        let n = updates.len() as f64;
        for p_idx in 0..self.global_params.len() {
            let mut aggregated = ArrayD::<f64>::zeros(self.global_params[p_idx].raw_dim());
            for update in updates {
                aggregated += &update.parameters[p_idx];
            }
            self.global_params[p_idx] = aggregated / n;
        }
        Ok(())
    }

    /// Median aggregation (element-wise median across clients).
    fn median_aggregate(&mut self, updates: &[ClientUpdate]) -> Result<()> {
        for p_idx in 0..self.global_params.len() {
            let shape = self.global_params[p_idx].raw_dim();
            let flat_len = self.global_params[p_idx].len();
            let mut result = ArrayD::<f64>::zeros(shape);

            for elem_idx in 0..flat_len {
                let mut values: Vec<f64> = updates
                    .iter()
                    .map(|u| {
                        u.parameters[p_idx]
                            .as_slice()
                            .map(|s| s[elem_idx])
                            .unwrap_or(0.0)
                    })
                    .collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median = if values.len().is_multiple_of(2) && values.len() >= 2 {
                    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                } else {
                    values[values.len() / 2]
                };

                if let Some(slice) = result.as_slice_mut() {
                    slice[elem_idx] = median;
                }
            }
            self.global_params[p_idx] = result;
        }
        Ok(())
    }

    /// Apply differential privacy Gaussian noise to global parameters.
    fn apply_dp_noise(&mut self) -> Result<()> {
        let sigma = self.config.dp_config.noise_multiplier * self.config.dp_config.max_grad_norm;

        for param in &mut self.global_params {
            let noise = generate_gaussian_noise(param.len(), 0.0, sigma, &mut self.rng);
            let noise_arr = ArrayD::from_shape_vec(param.raw_dim(), noise).map_err(|e| {
                NeuralError::ComputationError(format!("Failed to create noise array: {e}"))
            })?;
            *param = &*param + &noise_arr;
        }
        Ok(())
    }

    /// Generate a text summary of the federated training.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Federated Learning Summary ===\n");
        out.push_str(&format!("Aggregation: {}\n", self.config.aggregation));
        out.push_str(&format!("Selection: {}\n", self.config.client_selection));
        out.push_str(&format!(
            "Rounds: {} / {}\n",
            self.current_round, self.config.num_rounds
        ));
        out.push_str(&format!("DP enabled: {}\n", self.config.dp_config.enabled));
        out.push_str(&format!(
            "Compression enabled: {}\n",
            self.config.compression.enabled
        ));

        if let Some(last) = self.round_history.last() {
            out.push_str(&format!(
                "Last round: {} clients, {} samples",
                last.num_clients, last.total_samples
            ));
            if let Some(loss) = last.avg_loss {
                out.push_str(&format!(", avg_loss={loss:.6}"));
            }
            out.push('\n');
        }
        out
    }
}

// ============================================================================
// Gradient compression
// ============================================================================

/// Compress parameter deltas using top-k sparsification.
///
/// Computes delta = client_params - global_params, then keeps only the
/// top-k elements by absolute value.
fn compress_gradients(
    client_params: &[ArrayD<f64>],
    global_params: &[ArrayD<f64>],
    top_k_fraction: f64,
) -> Vec<Vec<(usize, f64)>> {
    let mut compressed = Vec::with_capacity(client_params.len());

    for (cp, gp) in client_params.iter().zip(global_params.iter()) {
        let delta = cp - gp;
        let flat: Vec<f64> = delta
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| delta.iter().copied().collect());

        let k = ((flat.len() as f64 * top_k_fraction).ceil() as usize)
            .max(1)
            .min(flat.len());

        // Find top-k by absolute value
        let mut indexed: Vec<(usize, f64)> = flat.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
        compressed.push(indexed);
    }

    compressed
}

/// Reconstruct parameters from compressed deltas.
fn apply_compressed_delta(
    global_params: &[ArrayD<f64>],
    compressed: &[Vec<(usize, f64)>],
) -> Vec<ArrayD<f64>> {
    let mut result = global_params.to_vec();
    for (p_idx, deltas) in compressed.iter().enumerate() {
        if let Some(slice) = result[p_idx].as_slice_mut() {
            for &(idx, val) in deltas {
                if idx < slice.len() {
                    slice[idx] += val;
                }
            }
        }
    }
    result
}

/// Clip the L2 norm of a parameter vector.
pub fn clip_l2_norm(params: &mut [ArrayD<f64>], max_norm: f64) {
    let norm_sq: f64 = params
        .iter()
        .map(|p| p.iter().map(|&x| x * x).sum::<f64>())
        .sum();
    let norm = norm_sq.sqrt();
    if norm > max_norm && norm > f64::EPSILON {
        let scale = max_norm / norm;
        for p in params.iter_mut() {
            p.mapv_inplace(|x| x * scale);
        }
    }
}

/// Generate Gaussian noise for differential privacy.
fn generate_gaussian_noise(len: usize, mean: f64, std_dev: f64, rng: &mut SmallRng) -> Vec<f64> {
    // Box-Muller transform for Gaussian noise
    let mut result = Vec::with_capacity(len);
    let mut i = 0;
    while i < len {
        let u1: f64 = rng.random_range(f64::EPSILON..1.0);
        let u2: f64 = rng.random_range(0.0..std::f64::consts::TAU);
        let r = (-2.0 * u1.ln()).sqrt();
        let z0 = r * u2.cos() * std_dev + mean;
        let z1 = r * u2.sin() * std_dev + mean;
        result.push(z0);
        i += 1;
        if i < len {
            result.push(z1);
            i += 1;
        }
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_config_defaults() {
        let config = FederatedConfig::default();
        assert_eq!(config.num_rounds, 100);
        assert_eq!(config.clients_per_round, 10);
        assert_eq!(config.aggregation, AggregationMethod::FedAvg);
    }

    #[test]
    fn test_config_builder() {
        let config = FederatedConfig::builder()
            .num_rounds(50)
            .clients_per_round(5)
            .aggregation(AggregationMethod::SimpleMean)
            .local_epochs(3)
            .local_lr(0.1)
            .seed(123)
            .build()
            .expect("valid config");

        assert_eq!(config.num_rounds, 50);
        assert_eq!(config.clients_per_round, 5);
        assert_eq!(config.local_epochs, 3);
    }

    #[test]
    fn test_config_validation_errors() {
        assert!(FederatedConfig::builder().num_rounds(0).build().is_err());
        assert!(FederatedConfig::builder()
            .clients_per_round(0)
            .build()
            .is_err());
        assert!(FederatedConfig::builder().local_epochs(0).build().is_err());
        assert!(FederatedConfig::builder().local_lr(-1.0).build().is_err());
        assert!(FederatedConfig::builder()
            .differential_privacy(0.0, 1.0)
            .build()
            .is_err());
        assert!(FederatedConfig::builder()
            .gradient_compression(-0.1)
            .build()
            .is_err());
    }

    #[test]
    fn test_fedavg_aggregation() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(2)
            .aggregation(AggregationMethod::FedAvg)
            .build()
            .expect("valid");

        let global = vec![array![0.0_f64, 0.0, 0.0].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        // Client A: 100 samples, params [1, 2, 3]
        // Client B: 300 samples, params [3, 2, 1]
        // FedAvg: (100*[1,2,3] + 300*[3,2,1]) / 400 = [2.5, 2.0, 1.5]
        let updates = vec![
            ClientUpdate::new(0, vec![array![1.0, 2.0, 3.0].into_dyn()], 100),
            ClientUpdate::new(1, vec![array![3.0, 2.0, 1.0].into_dyn()], 300),
        ];

        server.aggregate_round(&updates).expect("ok");

        let result = &server.global_params()[0];
        let slice = result.as_slice().expect("contiguous");
        assert!((slice[0] - 2.5).abs() < 1e-10);
        assert!((slice[1] - 2.0).abs() < 1e-10);
        assert!((slice[2] - 1.5).abs() < 1e-10);
        assert_eq!(server.current_round(), 1);
    }

    #[test]
    fn test_simple_mean_aggregation() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(3)
            .aggregation(AggregationMethod::SimpleMean)
            .build()
            .expect("valid");

        let global = vec![array![0.0_f64, 0.0].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        let updates = vec![
            ClientUpdate::new(0, vec![array![1.0, 4.0].into_dyn()], 10),
            ClientUpdate::new(1, vec![array![2.0, 5.0].into_dyn()], 10),
            ClientUpdate::new(2, vec![array![3.0, 6.0].into_dyn()], 10),
        ];

        server.aggregate_round(&updates).expect("ok");

        let result = &server.global_params()[0];
        let slice = result.as_slice().expect("contiguous");
        assert!((slice[0] - 2.0).abs() < 1e-10);
        assert!((slice[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_aggregation() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(3)
            .aggregation(AggregationMethod::Median)
            .build()
            .expect("valid");

        let global = vec![array![0.0_f64, 0.0].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        let updates = vec![
            ClientUpdate::new(0, vec![array![1.0, 100.0].into_dyn()], 10),
            ClientUpdate::new(1, vec![array![2.0, 5.0].into_dyn()], 10),
            ClientUpdate::new(2, vec![array![3.0, 6.0].into_dyn()], 10),
        ];

        server.aggregate_round(&updates).expect("ok");

        let result = &server.global_params()[0];
        let slice = result.as_slice().expect("contiguous");
        // Median of [1,2,3] = 2, median of [5,6,100] = 6
        assert!((slice[0] - 2.0).abs() < 1e-10);
        assert!((slice[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_updates_error() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(2)
            .build()
            .expect("valid");

        let global = vec![array![1.0_f64, 2.0].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        assert!(server.aggregate_round(&[]).is_err());
    }

    #[test]
    fn test_shape_mismatch_error() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(1)
            .build()
            .expect("valid");

        let global = vec![array![1.0_f64, 2.0].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        // Wrong number of parameter tensors
        let updates = vec![ClientUpdate::new(
            0,
            vec![array![1.0, 2.0].into_dyn(), array![3.0].into_dyn()],
            10,
        )];
        assert!(server.aggregate_round(&updates).is_err());
    }

    #[test]
    fn test_client_selection_random() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(3)
            .client_selection(ClientSelectionStrategy::Random)
            .seed(42)
            .build()
            .expect("valid");

        let global = vec![array![0.0_f64].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        let clients = vec![(0, 100), (1, 200), (2, 300), (3, 400), (4, 500)];
        let selected = server.select_clients(&clients);

        assert_eq!(selected.len(), 3);
        // All should be valid IDs
        for id in &selected {
            assert!(*id <= 4);
        }
    }

    #[test]
    fn test_client_selection_all() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(2)
            .client_selection(ClientSelectionStrategy::All)
            .build()
            .expect("valid");

        let global = vec![array![0.0_f64].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        let clients = vec![(0, 100), (1, 200), (2, 300)];
        let selected = server.select_clients(&clients);

        assert_eq!(selected.len(), 3); // All selected
    }

    #[test]
    fn test_client_selection_importance() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(2)
            .client_selection(ClientSelectionStrategy::ImportanceBased)
            .seed(42)
            .build()
            .expect("valid");

        let global = vec![array![0.0_f64].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        let clients = vec![(0, 1), (1, 1000), (2, 1)];
        let selected = server.select_clients(&clients);

        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_dp_noise_application() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(1)
            .differential_privacy(1.0, 1.0)
            .seed(42)
            .build()
            .expect("valid");

        let global = vec![array![0.0_f64, 0.0, 0.0].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        let updates = vec![ClientUpdate::new(
            0,
            vec![array![1.0, 2.0, 3.0].into_dyn()],
            100,
        )];

        server.aggregate_round(&updates).expect("ok");

        // With DP noise, the result should not be exactly [1, 2, 3]
        let result = &server.global_params()[0];
        let slice = result.as_slice().expect("contiguous");
        let any_noisy = slice[0] != 1.0 || slice[1] != 2.0 || slice[2] != 3.0;
        assert!(any_noisy, "DP noise should perturb the result");
    }

    #[test]
    fn test_gradient_compression() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(1)
            .gradient_compression(0.5)
            .build()
            .expect("valid");

        let global = vec![array![1.0_f64, 2.0, 3.0, 4.0].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        // Client has a big delta on elements 0 and 3, small on 1 and 2
        let updates = vec![ClientUpdate::new(
            0,
            vec![array![10.0, 2.1, 3.1, 14.0].into_dyn()],
            100,
        )];

        server.aggregate_round(&updates).expect("ok");

        // With top-50% compression, only the 2 largest deltas should be applied
        // Deltas: [9, 0.1, 0.1, 10] -> top-2: indices 3(10) and 0(9)
        let result = &server.global_params()[0];
        let slice = result.as_slice().expect("contiguous");
        // Elements 0 and 3 should get the delta; 1 and 2 should stay at global
        assert!((slice[0] - 10.0).abs() < 1e-10);
        assert!((slice[1] - 2.0).abs() < 1e-10); // no change
        assert!((slice[2] - 3.0).abs() < 1e-10); // no change
        assert!((slice[3] - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_l2_norm() {
        let mut params = vec![array![3.0_f64, 4.0].into_dyn()];
        // norm = 5.0
        clip_l2_norm(&mut params, 1.0);
        let slice = params[0].as_slice().expect("contiguous");
        let norm = (slice[0] * slice[0] + slice[1] * slice[1]).sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_l2_norm_no_clip_needed() {
        let mut params = vec![array![0.3_f64, 0.4].into_dyn()];
        // norm = 0.5 < 1.0
        clip_l2_norm(&mut params, 1.0);
        let slice = params[0].as_slice().expect("contiguous");
        assert!((slice[0] - 0.3).abs() < 1e-10);
        assert!((slice[1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_rounds() {
        let config = FederatedConfig::builder()
            .num_rounds(3)
            .clients_per_round(2)
            .aggregation(AggregationMethod::SimpleMean)
            .build()
            .expect("valid");

        let global = vec![array![0.0_f64, 0.0].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        for round in 0..3 {
            let v = (round + 1) as f64;
            let updates = vec![
                ClientUpdate::new(0, vec![array![v, v * 2.0].into_dyn()], 10),
                ClientUpdate::new(1, vec![array![v * 3.0, v * 4.0].into_dyn()], 10),
            ];
            server.aggregate_round(&updates).expect("ok");
        }

        assert_eq!(server.current_round(), 3);
        assert!(server.is_complete());
        assert_eq!(server.round_history().len(), 3);
    }

    #[test]
    fn test_client_update_with_metrics() {
        let update = ClientUpdate::new(0, vec![array![1.0_f64].into_dyn()], 100)
            .with_loss(0.5)
            .with_metric("accuracy", 0.95);

        assert_eq!(update.local_loss, Some(0.5));
        assert!((update.metrics["accuracy"] - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_round_stats_avg_loss() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(2)
            .build()
            .expect("valid");

        let global = vec![array![0.0_f64].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        let updates = vec![
            ClientUpdate::new(0, vec![array![1.0].into_dyn()], 10).with_loss(0.3),
            ClientUpdate::new(1, vec![array![2.0].into_dyn()], 10).with_loss(0.7),
        ];

        server.aggregate_round(&updates).expect("ok");

        let stats = &server.round_history()[0];
        assert_eq!(stats.num_clients, 2);
        assert_eq!(stats.total_samples, 20);
        assert!((stats.avg_loss.expect("has loss") - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_summary_generation() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(2)
            .build()
            .expect("valid");

        let global = vec![array![0.0_f64].into_dyn()];
        let mut server = FederatedServer::new(config, global);

        let updates = vec![ClientUpdate::new(0, vec![array![1.0].into_dyn()], 10)];
        server.aggregate_round(&updates).expect("ok");

        let summary = server.summary();
        assert!(summary.contains("Federated Learning Summary"));
        assert!(summary.contains("FedAvg"));
    }

    #[test]
    fn test_display_types() {
        assert_eq!(format!("{}", ClientSelectionStrategy::Random), "Random");
        assert_eq!(
            format!("{}", ClientSelectionStrategy::ImportanceBased),
            "ImportanceBased"
        );
        assert_eq!(format!("{}", ClientSelectionStrategy::All), "All");
        assert_eq!(format!("{}", AggregationMethod::FedAvg), "FedAvg");
        assert_eq!(format!("{}", AggregationMethod::SimpleMean), "SimpleMean");
        assert_eq!(format!("{}", AggregationMethod::Median), "Median");
    }

    #[test]
    fn test_gaussian_noise_generation() {
        let mut rng = SmallRng::seed_from_u64(42);
        let noise = generate_gaussian_noise(1000, 0.0, 1.0, &mut rng);
        assert_eq!(noise.len(), 1000);

        // Check that mean is approximately 0
        let mean = noise.iter().sum::<f64>() / noise.len() as f64;
        assert!(mean.abs() < 0.2, "mean={mean}, expected ~0");

        // Check that std is approximately 1
        let var = noise.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / noise.len() as f64;
        let std = var.sqrt();
        assert!((std - 1.0).abs() < 0.2, "std={std}, expected ~1");
    }

    #[test]
    fn test_multi_param_tensors() {
        let config = FederatedConfig::builder()
            .num_rounds(10)
            .clients_per_round(2)
            .aggregation(AggregationMethod::SimpleMean)
            .build()
            .expect("valid");

        let global = vec![
            array![1.0_f64, 2.0].into_dyn(),
            array![3.0_f64, 4.0, 5.0].into_dyn(),
        ];
        let mut server = FederatedServer::new(config, global);

        let updates = vec![
            ClientUpdate::new(
                0,
                vec![
                    array![2.0, 4.0].into_dyn(),
                    array![6.0, 8.0, 10.0].into_dyn(),
                ],
                10,
            ),
            ClientUpdate::new(
                1,
                vec![
                    array![4.0, 6.0].into_dyn(),
                    array![9.0, 12.0, 15.0].into_dyn(),
                ],
                10,
            ),
        ];

        server.aggregate_round(&updates).expect("ok");

        let p0 = server.global_params()[0].as_slice().expect("contiguous");
        assert!((p0[0] - 3.0).abs() < 1e-10);
        assert!((p0[1] - 5.0).abs() < 1e-10);

        let p1 = server.global_params()[1].as_slice().expect("contiguous");
        assert!((p1[0] - 7.5).abs() < 1e-10);
        assert!((p1[1] - 10.0).abs() < 1e-10);
        assert!((p1[2] - 12.5).abs() < 1e-10);
    }
}
