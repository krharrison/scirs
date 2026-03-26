//! GDAS: Gradient-based Architecture Search (Dong & Yang 2019)
//!
//! GDAS uses Gumbel-top1 sampling to select a single operation per edge
//! during the forward pass, enabling efficient single-path training while
//! maintaining differentiability through the Straight-Through Estimator (STE).
//!
//! Reference: "Searching for A Robust Neural Architecture in Four GPU Hours"
//! Xuanyi Dong, Yi Yang, CVPR 2019.

use crate::error::{NeuralError, Result};
use scirs2_core::random::{Rng, RngExt, SeedableRng};

/// Temperature annealing schedule for GDAS
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemperatureSchedule {
    /// Exponential decay: tau = tau_start * (tau_end/tau_start)^(epoch/n_epochs)
    Exponential,
    /// Linear decay: tau = tau_start - (tau_start - tau_end) * epoch/n_epochs
    Linear,
    /// Cosine annealing: tau = tau_end + 0.5*(tau_start-tau_end)*(1+cos(pi*epoch/n_epochs))
    Cosine,
}

/// Configuration for GDAS architecture search
#[derive(Debug, Clone)]
pub struct GdasConfig {
    /// Number of intermediate nodes in the DAG cell
    pub n_nodes: usize,
    /// Number of candidate operations per edge
    pub n_ops: usize,
    /// Initial Gumbel temperature (high = more uniform)
    pub tau_start: f64,
    /// Final Gumbel temperature (low = more peaked)
    pub tau_end: f64,
    /// Total number of search epochs
    pub n_epochs: usize,
    /// Learning rate for architecture parameters
    pub arch_lr: f64,
    /// Learning rate for model weights
    pub weight_lr: f64,
    /// Temperature annealing schedule
    pub schedule: TemperatureSchedule,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for GdasConfig {
    fn default() -> Self {
        Self {
            n_nodes: 4,
            n_ops: 8,
            tau_start: 10.0,
            tau_end: 0.1,
            n_epochs: 100,
            arch_lr: 3e-4,
            weight_lr: 1e-3,
            schedule: TemperatureSchedule::Exponential,
            seed: 42,
        }
    }
}

/// GDAS architecture search state
///
/// Maintains architecture parameters (alpha) and current temperature.
/// Uses Gumbel-top1 sampling to select a single operation per edge
/// while allowing gradient flow via the Straight-Through Estimator.
pub struct GdasSearch {
    /// Architecture weight logits: shape [n_edges][n_ops]
    pub alpha: Vec<Vec<f64>>,
    /// Current Gumbel sampling temperature
    pub temperature: f64,
    config: GdasConfig,
}

impl GdasSearch {
    /// Create a new GDAS search instance with the given configuration.
    ///
    /// Architecture parameters are initialized to zero (uniform before softmax).
    pub fn new(config: GdasConfig) -> Self {
        let n_edges = config.n_nodes * (config.n_nodes - 1) / 2;
        let alpha = vec![vec![0.0_f64; config.n_ops]; n_edges];
        let temperature = config.tau_start;
        Self {
            alpha,
            temperature,
            config,
        }
    }

    /// Number of directed edges in the DAG cell.
    ///
    /// For a cell with `n_nodes` intermediate nodes, the number of edges is
    /// `n_nodes * (n_nodes - 1) / 2` (all pairs i < j).
    pub fn n_edges(&self) -> usize {
        self.config.n_nodes * (self.config.n_nodes - 1) / 2
    }

    /// Sample a single operation using the Gumbel-top1 trick.
    ///
    /// Algorithm:
    /// 1. Sample Gumbel noise: g_i = -ln(-ln(U_i)), U_i ~ Uniform(0, 1)
    /// 2. Compute perturbed logits: y_i = (logits_i + g_i) / tau
    /// 3. Select k = argmax y_i  (hard, discrete selection)
    /// 4. Return hard one-hot vector: soft_weights[k] = 1, others = 0
    ///
    /// The one-hot output is used in the forward pass. During backpropagation,
    /// the Straight-Through Estimator (STE) passes gradients through as-if
    /// the softmax probabilities were used.
    ///
    /// # Returns
    /// `(selected_op_index, soft_weights)` where soft_weights is the one-hot vector.
    pub fn gumbel_top1_sample(
        &self,
        logits: &[f64],
        rng: &mut impl Rng,
    ) -> Result<(usize, Vec<f64>)> {
        if logits.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "logits must be non-empty for Gumbel-top1 sampling".to_string(),
            ));
        }

        let tau = self.temperature;
        if tau <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "Temperature must be positive, got {tau}"
            )));
        }

        // Sample Gumbel noise and compute perturbed logits
        let mut best_val = f64::NEG_INFINITY;
        let mut best_idx = 0usize;

        for (i, &logit) in logits.iter().enumerate() {
            // Sample U ~ Uniform(0, 1), clamp away from 0 to avoid ln(0)
            let u: f64 = rng.random();
            let u_clamped = u.max(1e-40);
            // Gumbel noise: g = -ln(-ln(U))
            let gumbel = -(-u_clamped.ln()).ln();
            let y = (logit + gumbel) / tau;
            if y > best_val {
                best_val = y;
                best_idx = i;
            }
        }

        // Hard one-hot: only the selected operation is active
        let mut soft_weights = vec![0.0_f64; logits.len()];
        soft_weights[best_idx] = 1.0;

        Ok((best_idx, soft_weights))
    }

    /// Anneal the temperature according to the configured schedule.
    ///
    /// Call this once per epoch. The temperature is clamped to [tau_end, tau_start].
    pub fn anneal_temperature(&mut self, epoch: usize) {
        let tau_start = self.config.tau_start;
        let tau_end = self.config.tau_end;
        let n_epochs = self.config.n_epochs.max(1);
        let t = (epoch as f64) / (n_epochs as f64);

        self.temperature = match self.config.schedule {
            TemperatureSchedule::Exponential => {
                // tau = tau_start * (tau_end / tau_start)^t
                let ratio = tau_end / tau_start;
                tau_start * ratio.powf(t)
            }
            TemperatureSchedule::Linear => {
                // tau = tau_start - (tau_start - tau_end) * t
                tau_start - (tau_start - tau_end) * t
            }
            TemperatureSchedule::Cosine => {
                // tau = tau_end + 0.5 * (tau_start - tau_end) * (1 + cos(pi * t))
                tau_end + 0.5 * (tau_start - tau_end) * (1.0 + (std::f64::consts::PI * t).cos())
            }
        };

        // Clamp to valid range
        self.temperature = self.temperature.clamp(
            tau_end.min(tau_start),
            tau_end.max(tau_start),
        );
    }

    /// Derive the discrete architecture by taking argmax of alpha per edge.
    ///
    /// Returns a vector of length `n_edges()` where each entry is the
    /// index of the selected operation for that edge.
    pub fn derive_architecture(&self) -> Vec<usize> {
        self.alpha
            .iter()
            .map(|edge_logits| {
                edge_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Compute softmax probabilities over architecture parameters.
    ///
    /// Returns shape [n_edges][n_ops] where each row sums to 1.
    pub fn architecture_probabilities(&self) -> Vec<Vec<f64>> {
        self.alpha
            .iter()
            .map(|edge_logits| softmax(edge_logits))
            .collect()
    }

    /// Apply a gradient update to a single architecture parameter.
    ///
    /// Performs gradient descent: `alpha[edge_idx][op_idx] -= lr * gradient`
    pub fn update_alpha(&mut self, edge_idx: usize, op_idx: usize, gradient: f64, lr: f64) {
        if let Some(edge) = self.alpha.get_mut(edge_idx) {
            if let Some(param) = edge.get_mut(op_idx) {
                *param -= lr * gradient;
            }
        }
    }

    /// Get the configuration used to create this search instance.
    pub fn config(&self) -> &GdasConfig {
        &self.config
    }
}

/// Numerically stable softmax over a slice of f64 values.
fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max_val = logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum <= 0.0 {
        let n = exps.len() as f64;
        return vec![1.0 / n; exps.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::{rngs::StdRng, SeedableRng};

    fn make_rng(seed: u64) -> StdRng {
        StdRng::seed_from_u64(seed)
    }

    #[test]
    fn test_gdas_gumbel_top1_returns_valid_index() {
        let config = GdasConfig::default();
        let search = GdasSearch::new(config.clone());
        let mut rng = make_rng(0);
        let logits = vec![1.0, 2.0, 0.5, 3.0, 1.5, 0.0, 2.5, 1.0];
        let (idx, weights) = search.gumbel_top1_sample(&logits, &mut rng).unwrap();
        assert!(idx < config.n_ops, "Selected index must be in [0, n_ops)");
        assert_eq!(weights.len(), config.n_ops);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Weights must sum to 1");
        assert_eq!(weights[idx], 1.0, "Selected operation weight must be 1");
        for (i, &w) in weights.iter().enumerate() {
            if i != idx {
                assert_eq!(w, 0.0, "Non-selected weights must be 0");
            }
        }
    }

    #[test]
    fn test_gdas_n_edges_correct() {
        let mut config = GdasConfig::default();
        config.n_nodes = 4;
        let search = GdasSearch::new(config);
        // 4 * 3 / 2 = 6 edges
        assert_eq!(search.n_edges(), 6);
        assert_eq!(search.alpha.len(), 6);
    }

    #[test]
    fn test_gdas_anneal_temperature_decreases_exponential() {
        let mut config = GdasConfig::default();
        config.schedule = TemperatureSchedule::Exponential;
        config.tau_start = 10.0;
        config.tau_end = 0.1;
        config.n_epochs = 100;
        let mut search = GdasSearch::new(config);
        let initial_temp = search.temperature;
        search.anneal_temperature(50);
        assert!(
            search.temperature < initial_temp,
            "Temperature should decrease after annealing"
        );
        search.anneal_temperature(100);
        assert!(
            (search.temperature - 0.1).abs() < 0.01,
            "Temperature should approach tau_end at end of schedule"
        );
    }

    #[test]
    fn test_gdas_anneal_temperature_decreases_linear() {
        let mut config = GdasConfig::default();
        config.schedule = TemperatureSchedule::Linear;
        config.tau_start = 10.0;
        config.tau_end = 0.1;
        config.n_epochs = 100;
        let mut search = GdasSearch::new(config);
        search.anneal_temperature(50);
        let mid_temp = search.temperature;
        assert!(
            (mid_temp - 5.05).abs() < 0.1,
            "Linear midpoint temperature should be ~5.05"
        );
    }

    #[test]
    fn test_gdas_anneal_temperature_cosine() {
        let mut config = GdasConfig::default();
        config.schedule = TemperatureSchedule::Cosine;
        config.tau_start = 10.0;
        config.tau_end = 0.1;
        config.n_epochs = 100;
        let mut search = GdasSearch::new(config);
        search.anneal_temperature(0);
        assert!(
            (search.temperature - 10.0).abs() < 0.01,
            "Cosine schedule at epoch 0 should be tau_start"
        );
    }

    #[test]
    fn test_gdas_derive_architecture_argmax() {
        let config = GdasConfig {
            n_nodes: 3,
            n_ops: 4,
            ..GdasConfig::default()
        };
        let mut search = GdasSearch::new(config);
        // Edge 0: best op is 2
        search.alpha[0] = vec![0.1, 0.2, 5.0, 0.0];
        // Edge 1: best op is 0
        search.alpha[1] = vec![3.0, 0.1, 0.2, 0.0];
        // Edge 2: best op is 3
        search.alpha[2] = vec![0.0, 0.1, 0.2, 7.0];

        let arch = search.derive_architecture();
        assert_eq!(arch.len(), 3);
        assert_eq!(arch[0], 2);
        assert_eq!(arch[1], 0);
        assert_eq!(arch[2], 3);
    }

    #[test]
    fn test_gdas_architecture_probabilities_sum_to_one() {
        let config = GdasConfig::default();
        let search = GdasSearch::new(config);
        let probs = search.architecture_probabilities();
        for edge_probs in &probs {
            let sum: f64 = edge_probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "Probabilities must sum to 1");
        }
    }

    #[test]
    fn test_gdas_update_alpha_gradient_step() {
        let config = GdasConfig::default();
        let mut search = GdasSearch::new(config);
        let initial = search.alpha[0][0];
        search.update_alpha(0, 0, 1.0, 0.01);
        let expected = initial - 0.01 * 1.0;
        assert!(
            (search.alpha[0][0] - expected).abs() < 1e-12,
            "Alpha should be updated by gradient step"
        );
    }

    #[test]
    fn test_gdas_gumbel_top1_empty_logits_error() {
        let config = GdasConfig::default();
        let search = GdasSearch::new(config);
        let mut rng = make_rng(0);
        let result = search.gumbel_top1_sample(&[], &mut rng);
        assert!(result.is_err(), "Empty logits should return an error");
    }
}
