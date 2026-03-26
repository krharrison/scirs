//! Sophia-H optimizer with Hutchinson diagonal Hessian estimation.
//!
//! Uses a stochastic diagonal Hessian estimate to precondition gradients.
//! Achieves better loss per gradient step than Adam (Liu et al., 2023).
//!
//! The Hutchinson estimator approximates the diagonal of the Hessian using
//! random Rademacher vectors z ∈ {-1, +1}^d:
//! `h ≈ (g ⊙ z)^2`  (element-wise)
//!
//! # Examples
//! ```
//! use scirs2_core::ndarray::{Array, IxDyn};
//! use scirs2_neural::optimizers::{Sophia, Optimizer};
//!
//! let mut sophia = Sophia::<f64>::default_with_lr(2e-4)
//!     .expect("optimizer creation should succeed");
//! ```

use crate::error::{NeuralError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;

/// Sophia-H optimizer configuration.
#[derive(Debug, Clone)]
pub struct SophiaConfig {
    /// Learning rate (default: 2e-4)
    pub learning_rate: f64,
    /// Momentum decay for first moment (default: 0.965)
    pub beta1: f64,
    /// EMA decay for Hessian estimate (default: 0.99)
    pub beta2: f64,
    /// Clipping threshold for preconditioned gradient (default: 0.04)
    pub rho: f64,
    /// Weight decay (default: 0.1)
    pub weight_decay: f64,
    /// How often (in steps) to update the Hessian estimate (default: 10)
    pub hutchinson_period: usize,
    /// Numerical stability epsilon (default: 1e-8)
    pub epsilon: f64,
}

impl Default for SophiaConfig {
    fn default() -> Self {
        Self {
            learning_rate: 2e-4,
            beta1: 0.965,
            beta2: 0.99,
            rho: 0.04,
            weight_decay: 0.1,
            hutchinson_period: 10,
            epsilon: 1e-8,
        }
    }
}

/// Sophia-H optimizer for neural networks.
///
/// Implements the Sophia optimizer from:
/// "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training"
/// by Liu et al. (2023).
///
/// Uses Hutchinson's method to estimate the diagonal Hessian:
/// - Every `hutchinson_period` steps, update `h ← beta2 * h + (1-beta2) * (g ⊙ z)^2`
///   where z is a Rademacher vector.
/// - Otherwise, use the cached h estimate.
///
/// The update rule is:
/// ```text
/// m = beta1 * m + (1-beta1) * g
/// clipped = clamp(m / max(rho*h, eps), -1, 1)
/// params -= lr * (clipped * max(rho*h, eps) + wd * params)
/// ```
#[derive(Debug)]
pub struct Sophia<F: Float + NumAssign + ScalarOperand + Debug> {
    /// Optimizer configuration
    config: SophiaConfig,
    /// First moment (momentum) buffers per parameter tensor
    m: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
    /// Hutchinson diagonal Hessian estimate per parameter tensor
    h: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
    /// Current step count
    t: usize,
}

impl<F: Float + NumAssign + ScalarOperand + Debug> Sophia<F> {
    /// Creates a new Sophia-H optimizer.
    pub fn new(config: SophiaConfig) -> Result<Self> {
        if config.learning_rate <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "learning_rate must be positive".to_string(),
            ));
        }
        if config.beta1 < 0.0 || config.beta1 >= 1.0 {
            return Err(NeuralError::InvalidArgument(
                "beta1 must be in [0, 1)".to_string(),
            ));
        }
        if config.beta2 < 0.0 || config.beta2 >= 1.0 {
            return Err(NeuralError::InvalidArgument(
                "beta2 must be in [0, 1)".to_string(),
            ));
        }
        if config.hutchinson_period == 0 {
            return Err(NeuralError::InvalidArgument(
                "hutchinson_period must be at least 1".to_string(),
            ));
        }
        Ok(Self {
            config,
            m: Vec::new(),
            h: Vec::new(),
            t: 0,
        })
    }

    /// Creates a Sophia-H optimizer with default hyperparameters.
    pub fn default_with_lr(learning_rate: f64) -> Result<Self> {
        let config = SophiaConfig {
            learning_rate,
            ..SophiaConfig::default()
        };
        Self::new(config)
    }

    /// Returns true if this step should update the Hessian estimate.
    ///
    /// The Hessian is updated every `hutchinson_period` steps.
    pub fn should_update_hessian(&self) -> bool {
        self.t > 0 && self.t % self.config.hutchinson_period == 0
    }

    /// Update the Hutchinson diagonal Hessian estimate for parameter at `layer_idx`.
    ///
    /// `h ← beta2 * h + (1-beta2) * (g ⊙ z)^2`
    ///
    /// # Arguments
    /// * `layer_idx` - Index of the parameter tensor
    /// * `grads` - Current gradient for this parameter
    /// * `rademacher_z` - Random ±1 Rademacher vector (same shape as grads)
    pub fn update_hessian(
        &mut self,
        layer_idx: usize,
        grads: &Array<F, scirs2_core::ndarray::IxDyn>,
        rademacher_z: &Array<F, scirs2_core::ndarray::IxDyn>,
    ) -> Result<()> {
        if layer_idx >= self.h.len() {
            return Err(NeuralError::InvalidArgument(format!(
                "layer_idx {layer_idx} out of bounds (have {} layers)",
                self.h.len()
            )));
        }
        if grads.shape() != rademacher_z.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "grads shape {:?} != rademacher_z shape {:?}",
                grads.shape(),
                rademacher_z.shape()
            )));
        }

        let beta2 = F::from(self.config.beta2).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert beta2 to F".to_string())
        })?;
        let one_minus_beta2 = F::one() - beta2;

        for ((hi, &gi), &zi) in self.h[layer_idx]
            .iter_mut()
            .zip(grads.iter())
            .zip(rademacher_z.iter())
        {
            let hz = gi * zi;
            *hi = beta2 * (*hi) + one_minus_beta2 * hz * hz;
        }

        Ok(())
    }

    /// Returns the current step count.
    pub fn get_step(&self) -> usize {
        self.t
    }

    /// Returns a reference to the Hessian estimate for a given layer.
    pub fn get_hessian(&self, layer_idx: usize) -> Option<&Array<F, scirs2_core::ndarray::IxDyn>> {
        self.h.get(layer_idx)
    }

    /// Resets internal optimizer state.
    pub fn reset_state(&mut self) {
        self.m.clear();
        self.h.clear();
        self.t = 0;
    }
}

impl<F: Float + NumAssign + ScalarOperand + Debug> Optimizer<F> for Sophia<F> {
    fn update(
        &mut self,
        params: &mut [Array<F, scirs2_core::ndarray::IxDyn>],
        grads: &[Array<F, scirs2_core::ndarray::IxDyn>],
    ) -> Result<()> {
        if params.len() != grads.len() {
            return Err(NeuralError::TrainingError(format!(
                "Number of parameter arrays ({}) does not match number of gradient arrays ({})",
                params.len(),
                grads.len()
            )));
        }

        self.t += 1;

        // Initialize moment and Hessian buffers if needed
        if self.m.len() != params.len() {
            self.m = params
                .iter()
                .map(|p| Array::zeros(p.raw_dim()))
                .collect();
            // Initialize h to small positive value to avoid division by zero initially
            self.h = params
                .iter()
                .map(|p| {
                    let mut arr = Array::zeros(p.raw_dim());
                    arr.fill(F::from(1e-10).unwrap_or(F::zero()));
                    arr
                })
                .collect();
        }

        let lr = F::from(self.config.learning_rate).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert learning_rate to F".to_string())
        })?;
        let beta1 = F::from(self.config.beta1).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert beta1 to F".to_string())
        })?;
        let rho = F::from(self.config.rho).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert rho to F".to_string())
        })?;
        let wd = F::from(self.config.weight_decay).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert weight_decay to F".to_string())
        })?;
        let eps = F::from(self.config.epsilon).ok_or_else(|| {
            NeuralError::ComputationError("Failed to convert epsilon to F".to_string())
        })?;
        let one_minus_beta1 = F::one() - beta1;

        for i in 0..params.len() {
            // Update first moment: m = beta1 * m + (1 - beta1) * g
            self.m[i] = &self.m[i] * beta1 + &(&grads[i] * one_minus_beta1);

            // Compute preconditioned update
            // For each element j:
            //   h_denom = max(rho * h[j], eps)
            //   clipped  = clamp(m[j] / h_denom, -1.0, 1.0)
            //   update   = clipped * h_denom + wd * params[j]
            //   params[j] -= lr * update
            for ((pj, &mj), &hj) in params[i]
                .iter_mut()
                .zip(self.m[i].iter())
                .zip(self.h[i].iter())
            {
                let h_denom = (rho * hj).max(eps);
                let clipped = (mj / h_denom).min(F::one()).max(-F::one());
                *pj -= lr * (clipped * h_denom + wd * (*pj));
            }
        }

        Ok(())
    }

    fn get_learning_rate(&self) -> F {
        F::from(self.config.learning_rate).unwrap_or(F::zero())
    }

    fn set_learning_rate(&mut self, lr: F) {
        self.config.learning_rate = lr.to_f64().unwrap_or(2e-4);
    }

    fn reset(&mut self) {
        self.reset_state();
    }

    fn name(&self) -> &'static str {
        "Sophia"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, IxDyn};

    fn make_param(vals: &[f64]) -> Array<f64, IxDyn> {
        Array1::from_vec(vals.to_vec()).into_dyn()
    }

    fn rademacher(size: usize, positive: bool) -> Array<f64, IxDyn> {
        // For testing, use either all +1 or alternating +1/-1
        let vals: Vec<f64> = if positive {
            vec![1.0; size]
        } else {
            (0..size).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect()
        };
        Array1::from_vec(vals).into_dyn()
    }

    #[test]
    fn test_sophia_default_config() {
        let sophia = Sophia::<f64>::default_with_lr(2e-4).expect("should succeed");
        assert!((sophia.get_learning_rate() - 2e-4).abs() < 1e-10);
        let cfg = &sophia.config;
        assert!((cfg.beta1 - 0.965).abs() < 1e-10);
        assert!((cfg.beta2 - 0.99).abs() < 1e-10);
        assert!((cfg.rho - 0.04).abs() < 1e-10);
        assert_eq!(cfg.hutchinson_period, 10);
    }

    #[test]
    fn test_sophia_hessian_update() {
        // Verify Hessian EMA formula: h = beta2 * h + (1-beta2) * (g*z)^2
        let mut sophia = Sophia::<f64>::default_with_lr(2e-4).expect("should succeed");

        // Initialize state by calling update once
        let mut params = vec![make_param(&[1.0_f64, 2.0, 3.0])];
        let grads = vec![make_param(&[0.1_f64, 0.2, 0.3])];
        sophia.update(&mut params, &grads).expect("initial update");

        // Manually compute expected h after update_hessian
        let g = make_param(&[0.5_f64, -0.5, 0.5]);
        let z = make_param(&[1.0_f64, -1.0, 1.0]);
        let beta2 = sophia.config.beta2;

        // Get current h before update
        let h_before: Vec<f64> = sophia.h[0].iter().cloned().collect();

        sophia
            .update_hessian(0, &g, &z)
            .expect("hessian update should succeed");

        let h_after: Vec<f64> = sophia.h[0].iter().cloned().collect();

        for i in 0..3 {
            let gz = g[i] * z[i];
            let expected = beta2 * h_before[i] + (1.0 - beta2) * gz * gz;
            assert!(
                (h_after[i] - expected).abs() < 1e-12,
                "h[{i}]: expected={expected}, got={}",
                h_after[i]
            );
        }
    }

    #[test]
    fn test_sophia_update_step() {
        // Params should move in negative gradient direction
        let config = SophiaConfig {
            learning_rate: 0.01,
            weight_decay: 0.0,
            ..SophiaConfig::default()
        };
        let mut sophia = Sophia::<f64>::new(config).expect("should succeed");

        let initial = vec![2.0_f64, 3.0, -1.0];
        let mut params = vec![make_param(&initial)];
        // Positive gradients → params should decrease
        let grads = vec![make_param(&[1.0_f64, 1.0, 1.0])];

        sophia.update(&mut params, &grads).expect("update should succeed");

        for (p, &p0) in params[0].iter().zip(initial.iter()) {
            assert!(
                *p < p0,
                "params should decrease with positive gradient: got {p}, was {p0}"
            );
        }
    }

    #[test]
    fn test_sophia_clipping() {
        // Very large gradient with large h → preconditioned value should be clipped to [-1,1]
        let config = SophiaConfig {
            learning_rate: 1.0,
            beta1: 0.0, // No momentum: m = g directly
            weight_decay: 0.0,
            rho: 0.04,
            epsilon: 1e-8,
            beta2: 0.99,
            hutchinson_period: 10,
        };
        let mut sophia = Sophia::<f64>::new(config).expect("should succeed");

        let mut params = vec![make_param(&[0.0_f64; 4])];
        let grads = vec![make_param(&[1000.0_f64; 4])];

        // After initialization, h starts at 1e-10; rho*h = 0.04*1e-10 is tiny
        // clipped = clamp(m / (rho*h), -1, 1) = clamp(large_val, -1, 1) = 1.0
        // update_step = 1.0 * rho*h ≈ very small
        sophia.update(&mut params, &grads).expect("update should succeed");

        // With beta1=0, m = (1-0)*g = g = 1000
        // rho*h = 0.04 * 1e-10 = 4e-12, clipped = clamp(1000 / 4e-12, -1, 1) = 1.0
        // step = lr * (1.0 * 4e-12 + 0 * params) ≈ 4e-12, very small but negative
        for p in params[0].iter() {
            assert!(p.is_finite(), "params should be finite after clipped update: {p}");
            assert!(*p <= 0.0, "params should decrease: {p}");
        }
    }

    #[test]
    fn test_sophia_weight_decay() {
        // With weight_decay > 0, large params should shrink even with zero gradient
        let config = SophiaConfig {
            learning_rate: 0.1,
            beta1: 0.0, // m = 0 when g=0
            weight_decay: 0.5,
            rho: 0.04,
            epsilon: 1e-8,
            beta2: 0.99,
            hutchinson_period: 10,
        };
        let mut sophia = Sophia::<f64>::new(config).expect("should succeed");
        let initial = vec![10.0_f64, -10.0, 5.0];
        let mut params = vec![make_param(&initial)];
        let grads = vec![make_param(&[0.0_f64; 3])]; // zero gradient

        sophia.update(&mut params, &grads).expect("update should succeed");

        // With beta1=0 and zero grad: m = 0, clipped = clamp(0 / h_denom, -1, 1) = 0
        // update = 0 * h_denom + wd * params = wd * params
        // params[j] -= lr * wd * params[j]  →  params shrink toward 0
        for (p, &p0) in params[0].iter().zip(initial.iter()) {
            let expected = p0 - 0.1 * 0.5 * p0; // p0 * (1 - lr * wd)
            assert!(
                (p - expected).abs() < 1e-6,
                "Weight decay: expected {expected}, got {p}"
            );
        }
    }

    #[test]
    fn test_sophia_hutchinson_period() {
        // should_update_hessian should be true only at multiples of hutchinson_period
        let config = SophiaConfig {
            hutchinson_period: 5,
            ..SophiaConfig::default()
        };
        let mut sophia = Sophia::<f64>::new(config).expect("should succeed");

        // At t=0 (before any steps), should_update_hessian returns false
        assert!(!sophia.should_update_hessian());

        let mut params = vec![make_param(&[1.0_f64])];
        let grads = vec![make_param(&[0.1_f64])];

        for step in 1..=15 {
            sophia.update(&mut params, &grads).expect("update should succeed");
            let should_update = sophia.should_update_hessian();
            let expected = step % 5 == 0;
            assert_eq!(
                should_update, expected,
                "At step {step}: should_update_hessian={should_update}, expected={expected}"
            );
        }
    }

    #[test]
    fn test_sophia_mismatched_lengths() {
        let mut sophia = Sophia::<f64>::default_with_lr(2e-4).expect("should succeed");
        let mut params = vec![make_param(&[1.0_f64])];
        let grads = vec![make_param(&[0.1_f64]), make_param(&[0.2_f64])];
        assert!(
            sophia.update(&mut params, &grads).is_err(),
            "Mismatched lengths should return error"
        );
    }

    #[test]
    fn test_sophia_invalid_hutchinson_period() {
        let config = SophiaConfig {
            hutchinson_period: 0,
            ..SophiaConfig::default()
        };
        assert!(
            Sophia::<f64>::new(config).is_err(),
            "hutchinson_period=0 should be invalid"
        );
    }
}
