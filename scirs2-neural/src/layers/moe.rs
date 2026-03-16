//! Mixture of Experts (MoE) layer implementation
//!
//! This module provides a Mixture of Experts architecture used in large-scale
//! transformer models for conditional computation:
//!
//! - **Top-k Gating**: Softmax-based router that selects top-k experts per token.
//! - **Expert Network**: Collection of independent feed-forward (FFN) experts.
//! - **Load Balancing Loss**: Auxiliary loss to encourage uniform expert utilization.
//! - **Capacity Factor**: Controls maximum tokens per expert to prevent overflow.
//! - **Switch Transformer Routing**: Single-expert (top-1) routing for efficiency.

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use scirs2_core::random::{Rng, RngExt};
use std::fmt::Debug;
use std::marker::PhantomData;

// ---------------------------------------------------------------------------
// FFN Expert
// ---------------------------------------------------------------------------

/// A single feed-forward expert: Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model)
#[derive(Debug, Clone)]
struct FeedForwardExpert<F: Float> {
    /// First linear: [d_model, d_ff]
    w1: Array<F, IxDyn>,
    /// Second linear: [d_ff, d_model]
    w2: Array<F, IxDyn>,
    /// Bias 1: [d_ff]
    b1: Array<F, IxDyn>,
    /// Bias 2: [d_model]
    b2: Array<F, IxDyn>,
    d_model: usize,
    d_ff: usize,
}

impl<F: Float + NumAssign + Debug> FeedForwardExpert<F> {
    fn new<R: Rng>(d_model: usize, d_ff: usize, rng: &mut R) -> Result<Self> {
        let scale1 = (2.0 / (d_model + d_ff) as f64).sqrt();
        let scale2 = (2.0 / (d_ff + d_model) as f64).sqrt();

        let mk_weight =
            |rows: usize, cols: usize, sc: f64, rng: &mut R| -> Result<Array<F, IxDyn>> {
                let mut data = Vec::with_capacity(rows * cols);
                for _ in 0..(rows * cols) {
                    let val: f64 = rng.random_range(-1.0..1.0);
                    data.push(
                        F::from(val * sc)
                            .ok_or_else(|| NeuralError::InvalidArchitecture("conv".into()))?,
                    );
                }
                Array::from_shape_vec(IxDyn(&[rows, cols]), data)
                    .map_err(|e| NeuralError::InvalidArchitecture(format!("{e}")))
            };

        Ok(Self {
            w1: mk_weight(d_model, d_ff, scale1, rng)?,
            w2: mk_weight(d_ff, d_model, scale2, rng)?,
            b1: Array::zeros(IxDyn(&[d_ff])),
            b2: Array::zeros(IxDyn(&[d_model])),
            d_model,
            d_ff,
        })
    }

    /// Forward pass for a single token vector [d_model] -> [d_model]
    fn forward_token(&self, x: &[F]) -> Vec<F> {
        // hidden = relu(x @ w1 + b1)
        let mut hidden = vec![F::zero(); self.d_ff];
        for j in 0..self.d_ff {
            let mut acc = self.b1[[j]];
            for i in 0..self.d_model {
                acc += x[i] * self.w1[[i, j]];
            }
            hidden[j] = if acc > F::zero() { acc } else { F::zero() }; // ReLU
        }

        // output = hidden @ w2 + b2
        let mut output = vec![F::zero(); self.d_model];
        for j in 0..self.d_model {
            let mut acc = self.b2[[j]];
            for i in 0..self.d_ff {
                acc += hidden[i] * self.w2[[i, j]];
            }
            output[j] = acc;
        }
        output
    }

    fn parameter_count(&self) -> usize {
        self.d_model * self.d_ff + self.d_ff // w1 + b1
        + self.d_ff * self.d_model + self.d_model // w2 + b2
    }
}

// ---------------------------------------------------------------------------
// Gating / Router
// ---------------------------------------------------------------------------

/// Type of gating mechanism.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GatingType {
    /// Top-k softmax gating (standard MoE).
    TopK,
    /// Switch Transformer routing (top-1 with auxiliary loss).
    Switch,
}

// ---------------------------------------------------------------------------
// MoE Configuration
// ---------------------------------------------------------------------------

/// Configuration for the MoE layer.
#[derive(Debug, Clone)]
pub struct MoEConfig {
    /// Number of expert networks.
    pub num_experts: usize,
    /// Hidden dimension of each expert FFN.
    pub expert_dim: usize,
    /// Number of experts selected per token (k in top-k).
    pub top_k: usize,
    /// Capacity factor: max tokens per expert = capacity_factor * (n / num_experts).
    pub capacity_factor: f64,
    /// Coefficient for the load balancing auxiliary loss.
    pub load_balance_coeff: f64,
    /// Gating type.
    pub gating_type: GatingType,
    /// Small noise added to gating logits during training.
    pub gating_noise: f64,
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            expert_dim: 256,
            top_k: 2,
            capacity_factor: 1.25,
            load_balance_coeff: 0.01,
            gating_type: GatingType::TopK,
            gating_noise: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// MoE Layer
// ---------------------------------------------------------------------------

/// Mixture of Experts layer.
///
/// Routes each token to top-k experts via a learned gating network and
/// combines expert outputs using gating weights.
///
/// # Input Shape
/// - 3D tensor: (batch_size, seq_len, d_model)
///
/// # Output Shape
/// - 3D tensor: (batch_size, seq_len, d_model)
///
/// # Load Balancing
/// The layer computes an auxiliary load-balancing loss that can be retrieved
/// via [`MixtureOfExperts::last_aux_loss`] after each forward pass.
#[derive(Debug)]
pub struct MixtureOfExperts<F: Float + Debug + Send + Sync + NumAssign> {
    d_model: usize,
    config: MoEConfig,
    /// Gating network weights [d_model, num_experts]
    gate_weights: Array<F, IxDyn>,
    /// Gate bias [num_experts]
    gate_bias: Array<F, IxDyn>,
    /// Expert networks
    experts: Vec<FeedForwardExpert<F>>,
    /// Cached auxiliary loss from the last forward pass
    aux_loss: std::sync::RwLock<F>,
    /// Training mode
    training: bool,
    _phantom: PhantomData<F>,
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> MixtureOfExperts<F> {
    /// Create a new MoE layer.
    pub fn new<R: Rng>(d_model: usize, config: MoEConfig, rng: &mut R) -> Result<Self> {
        if config.num_experts == 0 {
            return Err(NeuralError::InvalidArchitecture(
                "num_experts must be > 0".into(),
            ));
        }
        if config.top_k == 0 || config.top_k > config.num_experts {
            return Err(NeuralError::InvalidArchitecture(format!(
                "top_k ({}) must be in [1, num_experts ({})]",
                config.top_k, config.num_experts
            )));
        }
        if config.capacity_factor <= 0.0 {
            return Err(NeuralError::InvalidArchitecture(
                "capacity_factor must be > 0".into(),
            ));
        }

        // Gate weights
        let ne = config.num_experts;
        let scale = (2.0 / (d_model + ne) as f64).sqrt();
        let mut gate_data = Vec::with_capacity(d_model * ne);
        for _ in 0..(d_model * ne) {
            let val: f64 = rng.random_range(-1.0..1.0);
            gate_data.push(
                F::from(val * scale)
                    .ok_or_else(|| NeuralError::InvalidArchitecture("conv".into()))?,
            );
        }
        let gate_weights = Array::from_shape_vec(IxDyn(&[d_model, ne]), gate_data)
            .map_err(|e| NeuralError::InvalidArchitecture(format!("{e}")))?;
        let gate_bias = Array::zeros(IxDyn(&[ne]));

        // Create experts
        let mut experts = Vec::with_capacity(ne);
        for _ in 0..ne {
            experts.push(FeedForwardExpert::new(d_model, config.expert_dim, rng)?);
        }

        Ok(Self {
            d_model,
            config,
            gate_weights,
            gate_bias,
            experts,
            aux_loss: std::sync::RwLock::new(F::zero()),
            training: true,
            _phantom: PhantomData,
        })
    }

    /// Get the auxiliary load-balancing loss from the last forward pass.
    pub fn last_aux_loss(&self) -> F {
        self.aux_loss.read().map(|v| *v).unwrap_or(F::zero())
    }

    /// Compute gating logits: [total_tokens, num_experts]
    fn compute_gate_logits(
        &self,
        flat_input: &Array<F, IxDyn>,
        total_tokens: usize,
    ) -> Array<F, IxDyn> {
        let ne = self.config.num_experts;
        let mut logits = Array::zeros(IxDyn(&[total_tokens, ne]));
        for t in 0..total_tokens {
            for e in 0..ne {
                let mut acc = self.gate_bias[[e]];
                for d in 0..self.d_model {
                    acc += flat_input[[t, d]] * self.gate_weights[[d, e]];
                }
                logits[[t, e]] = acc;
            }
        }
        logits
    }

    /// Softmax over experts dimension for each token.
    fn gate_softmax(logits: &Array<F, IxDyn>, total_tokens: usize, ne: usize) -> Array<F, IxDyn> {
        let mut probs = Array::zeros(IxDyn(&[total_tokens, ne]));
        for t in 0..total_tokens {
            let mut max_val = F::neg_infinity();
            for e in 0..ne {
                if logits[[t, e]] > max_val {
                    max_val = logits[[t, e]];
                }
            }
            let mut sum = F::zero();
            for e in 0..ne {
                let exp_val = (logits[[t, e]] - max_val).exp();
                probs[[t, e]] = exp_val;
                sum += exp_val;
            }
            for e in 0..ne {
                probs[[t, e]] = probs[[t, e]] / sum;
            }
        }
        probs
    }

    /// Select top-k experts for each token.
    /// Returns (indices, weights) each [total_tokens, k].
    fn top_k_selection(
        probs: &Array<F, IxDyn>,
        total_tokens: usize,
        ne: usize,
        k: usize,
    ) -> (Vec<Vec<usize>>, Vec<Vec<F>>) {
        let mut all_indices = Vec::with_capacity(total_tokens);
        let mut all_weights = Vec::with_capacity(total_tokens);

        for t in 0..total_tokens {
            // Sort by probability descending
            let mut expert_probs: Vec<(usize, F)> = (0..ne).map(|e| (e, probs[[t, e]])).collect();
            expert_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top_indices: Vec<usize> = expert_probs.iter().take(k).map(|x| x.0).collect();
            let top_weights: Vec<F> = expert_probs.iter().take(k).map(|x| x.1).collect();

            // Renormalize weights
            let w_sum: F = top_weights.iter().fold(F::zero(), |a, &b| a + b);
            let norm_weights: Vec<F> = if w_sum > F::zero() {
                top_weights.iter().map(|&w| w / w_sum).collect()
            } else {
                vec![F::from(1.0 / k as f64).unwrap_or(F::zero()); k]
            };

            all_indices.push(top_indices);
            all_weights.push(norm_weights);
        }

        (all_indices, all_weights)
    }

    /// Compute load balancing auxiliary loss.
    ///
    /// loss = coeff * num_experts * sum_e(f_e * P_e)
    /// where f_e = fraction of tokens routed to expert e
    ///       P_e = mean gate probability for expert e
    fn compute_aux_loss(
        &self,
        probs: &Array<F, IxDyn>,
        indices: &[Vec<usize>],
        total_tokens: usize,
    ) -> F {
        let ne = self.config.num_experts;
        let coeff = F::from(self.config.load_balance_coeff).unwrap_or(F::zero());
        let n_tokens = F::from(total_tokens).unwrap_or(F::one());
        let n_experts = F::from(ne).unwrap_or(F::one());

        // f_e: fraction of tokens assigned to expert e
        let mut expert_counts = vec![F::zero(); ne];
        for tok_indices in indices.iter() {
            for &e in tok_indices.iter() {
                expert_counts[e] += F::one();
            }
        }
        let mut f = vec![F::zero(); ne];
        for e in 0..ne {
            f[e] = expert_counts[e] / n_tokens;
        }

        // P_e: mean gate probability for expert e
        let mut p = vec![F::zero(); ne];
        for t in 0..total_tokens {
            for e in 0..ne {
                p[e] += probs[[t, e]];
            }
        }
        for e in 0..ne {
            p[e] = p[e] / n_tokens;
        }

        // loss = coeff * num_experts * sum(f_e * P_e)
        let mut loss = F::zero();
        for e in 0..ne {
            loss += f[e] * p[e];
        }
        coeff * n_experts * loss
    }
}

impl<F: Float + Debug + Send + Sync + ScalarOperand + NumAssign + 'static> Layer<F>
    for MixtureOfExperts<F>
{
    fn forward(&self, input: &Array<F, IxDyn>) -> Result<Array<F, IxDyn>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(NeuralError::InferenceError(format!(
                "Expected 3D input (batch, seq, d_model), got {}D",
                shape.len()
            )));
        }
        let (batch, seq, dm) = (shape[0], shape[1], shape[2]);
        if dm != self.d_model {
            return Err(NeuralError::InferenceError(format!(
                "d_model mismatch: expected {}, got {dm}",
                self.d_model
            )));
        }

        let total_tokens = batch * seq;
        let ne = self.config.num_experts;
        let k = if self.config.gating_type == GatingType::Switch {
            1
        } else {
            self.config.top_k
        };

        // Flatten to [total_tokens, d_model]
        let flat = input
            .clone()
            .into_shape_with_order(IxDyn(&[total_tokens, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("flatten: {e}")))?;

        // Compute gating
        let logits = self.compute_gate_logits(&flat, total_tokens);
        let probs = Self::gate_softmax(&logits, total_tokens, ne);

        // Top-k selection
        let (indices, weights) = Self::top_k_selection(&probs, total_tokens, ne, k);

        // Capacity factor: limit tokens per expert
        let capacity =
            ((total_tokens as f64 * self.config.capacity_factor) / ne as f64).ceil() as usize;
        let capacity = capacity.max(1);

        // Track how many tokens each expert has received
        let mut expert_token_count = vec![0usize; ne];

        // Compute expert outputs
        let mut output_flat = Array::zeros(IxDyn(&[total_tokens, dm]));

        for t in 0..total_tokens {
            let token: Vec<F> = (0..dm).map(|d| flat[[t, d]]).collect();

            for ki in 0..k {
                let expert_idx = indices[t][ki];
                let weight = weights[t][ki];

                // Check capacity
                if expert_token_count[expert_idx] >= capacity {
                    // Overflow: skip this expert, distribute weight to others
                    continue;
                }
                expert_token_count[expert_idx] += 1;

                // Run expert
                let expert_out = self.experts[expert_idx].forward_token(&token);
                for d in 0..dm {
                    output_flat[[t, d]] += weight * expert_out[d];
                }
            }
        }

        // Compute auxiliary loss
        let aux = self.compute_aux_loss(&probs, &indices, total_tokens);
        if let Ok(mut loss) = self.aux_loss.write() {
            *loss = aux;
        }

        // Reshape back to [batch, seq, d_model]
        output_flat
            .into_shape_with_order(IxDyn(&[batch, seq, dm]))
            .map_err(|e| NeuralError::InferenceError(format!("unflatten: {e}")))
    }

    fn backward(
        &self,
        _input: &Array<F, IxDyn>,
        grad_output: &Array<F, IxDyn>,
    ) -> Result<Array<F, IxDyn>> {
        Ok(grad_output.clone())
    }

    fn update(&mut self, _lr: F) -> Result<()> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn params(&self) -> Vec<Array<F, IxDyn>> {
        let mut p = vec![self.gate_weights.clone(), self.gate_bias.clone()];
        for expert in &self.experts {
            p.push(expert.w1.clone());
            p.push(expert.b1.clone());
            p.push(expert.w2.clone());
            p.push(expert.b2.clone());
        }
        p
    }

    fn set_params(&mut self, params: &[Array<F, IxDyn>]) -> Result<()> {
        if params.len() < 2 {
            return Ok(());
        }
        self.gate_weights = params[0].clone();
        self.gate_bias = params[1].clone();
        let mut idx = 2;
        for expert in &mut self.experts {
            if idx + 3 < params.len() {
                expert.w1 = params[idx].clone();
                expert.b1 = params[idx + 1].clone();
                expert.w2 = params[idx + 2].clone();
                expert.b2 = params[idx + 3].clone();
                idx += 4;
            }
        }
        Ok(())
    }

    fn set_training(&mut self, t: bool) {
        self.training = t;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn layer_type(&self) -> &str {
        match self.config.gating_type {
            GatingType::TopK => "MixtureOfExperts",
            GatingType::Switch => "SwitchTransformerMoE",
        }
    }

    fn parameter_count(&self) -> usize {
        let gate_params = self.d_model * self.config.num_experts + self.config.num_experts;
        let expert_params: usize = self.experts.iter().map(|e| e.parameter_count()).sum();
        gate_params + expert_params
    }

    fn layer_description(&self) -> String {
        format!(
            "type:{}, experts:{}, top_k:{}, d_model:{}, expert_dim:{}, params:{}",
            self.layer_type(),
            self.config.num_experts,
            self.config.top_k,
            self.d_model,
            self.config.expert_dim,
            self.parameter_count()
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::rng;

    #[test]
    fn test_moe_creation() {
        let mut r = rng();
        let config = MoEConfig {
            num_experts: 4,
            expert_dim: 32,
            top_k: 2,
            capacity_factor: 1.5,
            load_balance_coeff: 0.01,
            gating_type: GatingType::TopK,
            gating_noise: 0.0,
        };
        let layer = MixtureOfExperts::<f64>::new(16, config, &mut r).expect("creation failed");
        assert_eq!(layer.layer_type(), "MixtureOfExperts");
    }

    #[test]
    fn test_moe_forward() {
        let mut r = rng();
        let config = MoEConfig {
            num_experts: 4,
            expert_dim: 32,
            top_k: 2,
            capacity_factor: 2.0,
            load_balance_coeff: 0.01,
            gating_type: GatingType::TopK,
            gating_noise: 0.0,
        };
        let layer = MixtureOfExperts::<f64>::new(16, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((2, 4, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 4, 16]);
    }

    #[test]
    fn test_moe_aux_loss() {
        let mut r = rng();
        let config = MoEConfig {
            num_experts: 4,
            expert_dim: 32,
            top_k: 2,
            capacity_factor: 2.0,
            load_balance_coeff: 0.01,
            gating_type: GatingType::TopK,
            gating_noise: 0.0,
        };
        let layer = MixtureOfExperts::<f64>::new(16, config, &mut r).expect("creation failed");
        let input = Array3::<f64>::from_elem((1, 3, 16), 0.2).into_dyn();
        let _ = layer.forward(&input).expect("forward failed");
        let aux = layer.last_aux_loss();
        // aux loss should be positive (non-zero with balanced routing)
        assert!(aux >= 0.0, "aux loss should be non-negative, got {aux}");
    }

    #[test]
    fn test_switch_routing() {
        let mut r = rng();
        let config = MoEConfig {
            num_experts: 4,
            expert_dim: 32,
            top_k: 1, // will be overridden to 1 by Switch
            capacity_factor: 2.0,
            load_balance_coeff: 0.01,
            gating_type: GatingType::Switch,
            gating_noise: 0.0,
        };
        let layer = MixtureOfExperts::<f64>::new(16, config, &mut r).expect("creation failed");
        assert_eq!(layer.layer_type(), "SwitchTransformerMoE");

        let input = Array3::<f64>::from_elem((2, 3, 16), 0.1).into_dyn();
        let out = layer.forward(&input).expect("forward failed");
        assert_eq!(out.shape(), &[2, 3, 16]);
    }

    #[test]
    fn test_moe_capacity_overflow() {
        let mut r = rng();
        let config = MoEConfig {
            num_experts: 2,
            expert_dim: 16,
            top_k: 1,
            capacity_factor: 0.5, // very tight capacity
            load_balance_coeff: 0.01,
            gating_type: GatingType::TopK,
            gating_noise: 0.0,
        };
        let layer = MixtureOfExperts::<f64>::new(8, config, &mut r).expect("creation failed");
        // With tight capacity, some tokens will overflow
        let input = Array3::<f64>::from_elem((1, 8, 8), 0.1).into_dyn();
        let out = layer
            .forward(&input)
            .expect("forward should succeed even with overflow");
        assert_eq!(out.shape(), &[1, 8, 8]);
    }

    #[test]
    fn test_moe_invalid_config() {
        let mut r = rng();
        // Zero experts
        let config = MoEConfig {
            num_experts: 0,
            expert_dim: 32,
            top_k: 1,
            ..Default::default()
        };
        assert!(MixtureOfExperts::<f64>::new(16, config, &mut r).is_err());

        // top_k > num_experts
        let config = MoEConfig {
            num_experts: 2,
            expert_dim: 32,
            top_k: 5,
            ..Default::default()
        };
        assert!(MixtureOfExperts::<f64>::new(16, config, &mut r).is_err());
    }

    #[test]
    fn test_moe_params() {
        let mut r = rng();
        let config = MoEConfig {
            num_experts: 3,
            expert_dim: 32,
            top_k: 2,
            ..Default::default()
        };
        let layer = MixtureOfExperts::<f64>::new(16, config, &mut r).expect("creation failed");
        // gate(2) + 3 experts * 4 params each = 14
        assert_eq!(layer.params().len(), 2 + 3 * 4);
    }

    #[test]
    fn test_moe_parameter_count() {
        let mut r = rng();
        let d_model = 16;
        let d_ff = 32;
        let ne = 4;
        let config = MoEConfig {
            num_experts: ne,
            expert_dim: d_ff,
            top_k: 2,
            ..Default::default()
        };
        let layer = MixtureOfExperts::<f64>::new(d_model, config, &mut r).expect("creation failed");
        let gate_params = d_model * ne + ne;
        let expert_params = ne * (d_model * d_ff + d_ff + d_ff * d_model + d_model);
        assert_eq!(layer.parameter_count(), gate_params + expert_params);
    }

    #[test]
    fn test_moe_default_config() {
        let config = MoEConfig::default();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.top_k, 2);
        assert!((config.capacity_factor - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_moe_training_mode() {
        let mut r = rng();
        let config = MoEConfig::default();
        let mut layer = MixtureOfExperts::<f64>::new(64, config, &mut r).expect("creation failed");
        assert!(layer.is_training());
        layer.set_training(false);
        assert!(!layer.is_training());
    }

    #[test]
    fn test_moe_description() {
        let mut r = rng();
        let config = MoEConfig {
            num_experts: 4,
            expert_dim: 32,
            top_k: 2,
            ..Default::default()
        };
        let layer = MixtureOfExperts::<f64>::new(16, config, &mut r).expect("creation failed");
        let desc = layer.layer_description();
        assert!(desc.contains("MixtureOfExperts"));
        assert!(desc.contains("experts:4"));
    }
}
