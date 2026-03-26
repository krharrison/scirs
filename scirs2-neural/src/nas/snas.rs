//! SNAS: Stochastic Neural Architecture Search (Xie et al. 2019)
//!
//! SNAS casts architecture search as a variational inference problem.
//! Architecture parameters define a distribution over discrete operations,
//! and the ELBO (Evidence Lower BOund) is optimized using the
//! Gumbel-softmax reparameterization trick.
//!
//! The ELBO = E_q[task_loss] + lambda * KL[q(z|alpha) || Uniform(n_ops)]
//!
//! Reference: "SNAS: Stochastic Neural Architecture Search"
//! Sirui Xie et al., ICLR 2019.

use crate::error::{NeuralError, Result};
use scirs2_core::random::{Rng, RngExt, SeedableRng};

/// Configuration for SNAS architecture search
#[derive(Debug, Clone)]
pub struct SnasConfig {
    /// Number of intermediate nodes in the DAG cell
    pub n_nodes: usize,
    /// Number of candidate operations per edge
    pub n_ops: usize,
    /// Gumbel-softmax temperature (lower = more discrete)
    pub temperature: f64,
    /// KL regularization weight (lambda in the ELBO)
    pub lambda_kl: f64,
    /// Total number of search epochs
    pub n_epochs: usize,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for SnasConfig {
    fn default() -> Self {
        Self {
            n_nodes: 4,
            n_ops: 8,
            temperature: 1.0,
            lambda_kl: 0.01,
            n_epochs: 100,
            seed: 42,
        }
    }
}

/// SNAS architecture search state
///
/// Maintains architecture parameters alpha and exposes methods for
/// Gumbel-softmax sampling, KL divergence computation, and gradient updates.
pub struct SnasSearch {
    /// Architecture weight logits: shape [n_edges][n_ops]
    pub alpha: Vec<Vec<f64>>,
    config: SnasConfig,
}

impl SnasSearch {
    /// Create a new SNAS search instance with the given configuration.
    ///
    /// Architecture parameters are initialized to zero (uniform before softmax).
    pub fn new(config: SnasConfig) -> Self {
        let n_edges = config.n_nodes * (config.n_nodes - 1) / 2;
        let alpha = vec![vec![0.0_f64; config.n_ops]; n_edges];
        Self { alpha, config }
    }

    /// Number of directed edges in the DAG cell.
    pub fn n_edges(&self) -> usize {
        self.config.n_nodes * (self.config.n_nodes - 1) / 2
    }

    /// Sample architecture weights using the Gumbel-softmax trick.
    ///
    /// This provides a differentiable, continuous relaxation of the
    /// categorical distribution over operations.
    ///
    /// Algorithm:
    /// 1. Sample Gumbel noise: g_i = -ln(-ln(U_i)), U_i ~ Uniform(0, 1)
    /// 2. Compute: y_i = softmax((logits_i + g_i) / temperature)
    ///
    /// Returns soft weights that sum to 1 and can be used in a
    /// mixed-operation forward pass.
    pub fn gumbel_softmax_sample(
        &self,
        logits: &[f64],
        temperature: f64,
        rng: &mut impl Rng,
    ) -> Result<Vec<f64>> {
        if logits.is_empty() {
            return Err(NeuralError::InvalidArgument(
                "logits must be non-empty for Gumbel-softmax sampling".to_string(),
            ));
        }
        if temperature <= 0.0 {
            return Err(NeuralError::InvalidArgument(format!(
                "Temperature must be positive, got {temperature}"
            )));
        }

        let perturbed: Vec<f64> = logits
            .iter()
            .map(|&logit| {
                let u: f64 = rng.random();
                let u_clamped = u.max(1e-40);
                let gumbel = -(-u_clamped.ln()).ln();
                (logit + gumbel) / temperature
            })
            .collect();

        Ok(softmax_f64(&perturbed))
    }

    /// Compute KL divergence from the architecture distribution to Uniform(n_ops).
    ///
    /// KL[q(z|alpha_edge) || Uniform] = sum_i p_i * ln(p_i * n_ops)
    ///
    /// where p_i = softmax(alpha[edge_idx])_i
    ///
    /// This is zero when all operations are equally likely (uniform distribution)
    /// and positive when the distribution is more peaked.
    pub fn kl_divergence_from_uniform(&self, edge_idx: usize) -> Result<f64> {
        let edge_logits = self.alpha.get(edge_idx).ok_or_else(|| {
            NeuralError::InvalidArgument(format!(
                "Edge index {edge_idx} out of bounds (n_edges={})",
                self.n_edges()
            ))
        })?;

        let n_ops = self.config.n_ops as f64;
        let probs = softmax_f64(edge_logits);

        // KL[q || Uniform] = sum_i p_i * ln(p_i * n_ops)
        let kl: f64 = probs
            .iter()
            .map(|&p| {
                if p <= 1e-40 {
                    0.0
                } else {
                    p * (p * n_ops).ln()
                }
            })
            .sum();

        Ok(kl.max(0.0)) // KL divergence is always non-negative
    }

    /// Sum KL divergence across all edges.
    ///
    /// This is the total regularization term added to the ELBO.
    pub fn total_kl_loss(&self) -> f64 {
        (0..self.n_edges())
            .filter_map(|i| self.kl_divergence_from_uniform(i).ok())
            .sum()
    }

    /// Compute the combined ELBO gradient for one edge's architecture parameters.
    ///
    /// ELBO = task_loss + lambda * KL, so gradient = task_grad + lambda * kl_grad
    ///
    /// Both task_grad and kl_grad must have length n_ops.
    pub fn elbo_gradient(task_grad: &[f64], kl_grad: &[f64], lambda: f64) -> Result<Vec<f64>> {
        if task_grad.len() != kl_grad.len() {
            return Err(NeuralError::ShapeMismatch(format!(
                "task_grad length {} != kl_grad length {}",
                task_grad.len(),
                kl_grad.len()
            )));
        }
        Ok(task_grad
            .iter()
            .zip(kl_grad.iter())
            .map(|(&tg, &kg)| tg + lambda * kg)
            .collect())
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
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Compute the average entropy of the architecture distributions.
    ///
    /// H = -sum_edge sum_op p_{eo} * ln(p_{eo}) / n_edges
    ///
    /// High entropy means the distribution is still close to uniform
    /// (the search has not converged). Low entropy means the distribution
    /// is peaked on specific operations.
    pub fn architecture_entropy(&self) -> f64 {
        let n_edges = self.n_edges();
        if n_edges == 0 {
            return 0.0;
        }

        let total_entropy: f64 = self
            .alpha
            .iter()
            .map(|edge_logits| {
                let probs = softmax_f64(edge_logits);
                -probs
                    .iter()
                    .map(|&p| if p <= 1e-40 { 0.0 } else { p * p.ln() })
                    .sum::<f64>()
            })
            .sum();

        total_entropy / (n_edges as f64)
    }

    /// Apply a gradient update to all operations of one edge.
    ///
    /// Performs gradient descent: `alpha[edge_idx][op] -= lr * gradient[op]`
    pub fn update_alpha(&mut self, edge_idx: usize, gradient: &[f64], lr: f64) {
        if let Some(edge) = self.alpha.get_mut(edge_idx) {
            for (param, &grad) in edge.iter_mut().zip(gradient.iter()) {
                *param -= lr * grad;
            }
        }
    }

    /// Get the configuration used to create this search instance.
    pub fn config(&self) -> &SnasConfig {
        &self.config
    }

    /// Get softmax probabilities for one edge.
    pub fn edge_probabilities(&self, edge_idx: usize) -> Option<Vec<f64>> {
        self.alpha.get(edge_idx).map(|logits| softmax_f64(logits))
    }
}

/// Numerically stable softmax over a slice of f64 values.
fn softmax_f64(logits: &[f64]) -> Vec<f64> {
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
    fn test_snas_gumbel_softmax_sums_to_one() {
        let config = SnasConfig::default();
        let search = SnasSearch::new(config);
        let mut rng = make_rng(42);
        let logits = vec![1.0, 0.5, -1.0, 2.0, 0.0, -0.5, 1.5, 0.3];
        let weights = search
            .gumbel_softmax_sample(&logits, 1.0, &mut rng)
            .unwrap();
        assert_eq!(weights.len(), 8);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Soft weights must sum to 1");
        for &w in &weights {
            assert!(w >= 0.0, "All weights must be non-negative");
        }
    }

    #[test]
    fn test_snas_gumbel_softmax_low_temperature_peaked() {
        let config = SnasConfig::default();
        let search = SnasSearch::new(config);
        let mut rng = make_rng(0);
        // Strongly favor op 3 with logit=10, others are 0
        let logits = vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0];
        let weights = search
            .gumbel_softmax_sample(&logits, 0.01, &mut rng)
            .unwrap();
        // At very low temperature, the distribution should be very peaked on op 3
        assert!(weights[3] > 0.9, "Low temperature should peak on the dominant logit");
    }

    #[test]
    fn test_snas_kl_divergence_uniform_alpha_near_zero() {
        let config = SnasConfig::default();
        let search = SnasSearch::new(config);
        // All zeros in alpha -> softmax gives uniform distribution -> KL ~ 0
        let kl = search.kl_divergence_from_uniform(0).unwrap();
        assert!(
            kl < 1e-10,
            "KL from uniform alpha should be ~0, got {kl}"
        );
    }

    #[test]
    fn test_snas_kl_divergence_peaked_positive() {
        let config = SnasConfig {
            n_nodes: 3,
            n_ops: 4,
            ..SnasConfig::default()
        };
        let mut search = SnasSearch::new(config);
        // Make edge 0 strongly prefer op 0
        search.alpha[0] = vec![20.0, -10.0, -10.0, -10.0];
        let kl = search.kl_divergence_from_uniform(0).unwrap();
        assert!(kl > 0.1, "Peaked distribution should have positive KL divergence, got {kl}");
    }

    #[test]
    fn test_snas_kl_divergence_invalid_edge() {
        let config = SnasConfig::default();
        let search = SnasSearch::new(config);
        let result = search.kl_divergence_from_uniform(999);
        assert!(result.is_err(), "Out-of-bounds edge index should return an error");
    }

    #[test]
    fn test_snas_architecture_entropy_uniform_max() {
        let config = SnasConfig {
            n_nodes: 3,
            n_ops: 8,
            ..SnasConfig::default()
        };
        let search = SnasSearch::new(config);
        // Uniform distribution: entropy = ln(n_ops)
        let entropy = search.architecture_entropy();
        let expected = (8.0_f64).ln();
        assert!(
            (entropy - expected).abs() < 1e-6,
            "Uniform alpha should give maximum entropy ln(8)={expected:.4}, got {entropy:.4}"
        );
    }

    #[test]
    fn test_snas_architecture_entropy_decreases_when_peaked() {
        let config = SnasConfig {
            n_nodes: 3,
            n_ops: 4,
            ..SnasConfig::default()
        };
        let mut search = SnasSearch::new(config);
        let uniform_entropy = search.architecture_entropy();
        // Make all edges strongly peaked on op 0
        for edge in search.alpha.iter_mut() {
            *edge = vec![100.0, -100.0, -100.0, -100.0];
        }
        let peaked_entropy = search.architecture_entropy();
        assert!(
            peaked_entropy < uniform_entropy,
            "Peaked distribution should have lower entropy"
        );
    }

    #[test]
    fn test_snas_elbo_gradient_combines_correctly() {
        let task_grad = vec![1.0, 2.0, 3.0];
        let kl_grad = vec![0.1, 0.2, 0.3];
        let lambda = 0.5;
        let elbo_grad = SnasSearch::elbo_gradient(&task_grad, &kl_grad, lambda).unwrap();
        assert_eq!(elbo_grad.len(), 3);
        assert!((elbo_grad[0] - 1.05).abs() < 1e-10);
        assert!((elbo_grad[1] - 2.10).abs() < 1e-10);
        assert!((elbo_grad[2] - 3.15).abs() < 1e-10);
    }

    #[test]
    fn test_snas_elbo_gradient_shape_mismatch_error() {
        let result = SnasSearch::elbo_gradient(&[1.0, 2.0], &[1.0, 2.0, 3.0], 0.01);
        assert!(result.is_err(), "Mismatched gradient shapes should return an error");
    }

    #[test]
    fn test_snas_derive_architecture_argmax() {
        let config = SnasConfig {
            n_nodes: 3,
            n_ops: 4,
            ..SnasConfig::default()
        };
        let mut search = SnasSearch::new(config);
        search.alpha[0] = vec![0.1, 0.2, 5.0, 0.0];
        search.alpha[1] = vec![3.0, 0.1, 0.2, 0.0];
        search.alpha[2] = vec![0.0, 0.1, 0.2, 7.0];
        let arch = search.derive_architecture();
        assert_eq!(arch, vec![2, 0, 3]);
    }

    #[test]
    fn test_snas_update_alpha_gradient_step() {
        let config = SnasConfig {
            n_nodes: 3,
            n_ops: 4,
            ..SnasConfig::default()
        };
        let mut search = SnasSearch::new(config);
        let grad = vec![1.0, -1.0, 0.5, 0.0];
        let lr = 0.01;
        search.update_alpha(0, &grad, lr);
        assert!((search.alpha[0][0] - (-0.01)).abs() < 1e-12);
        assert!((search.alpha[0][1] - 0.01).abs() < 1e-12);
    }
}
