//! LARS (Layer-wise Adaptive Rate Scaling) and LARC optimizers.
//!
//! Designed for large-batch training (You et al., 2017).
//! Computes per-layer learning rates based on weight/gradient norms.
//!
//! LARS scales the learning rate per-layer based on the ratio of the
//! parameter norm to the gradient norm. LARC is the clipped variant
//! that never exceeds the global learning rate.
//!
//! # Examples
//! ```
//! use scirs2_core::ndarray::{Array, IxDyn};
//! use scirs2_neural::optimizers::{Lars, Optimizer};
//!
//! let mut lars = Lars::<f64>::new(0.01, 0.9, 1e-4, 0.001, false)
//!     .expect("optimizer creation should succeed");
//! ```

use crate::error::{NeuralError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;

/// LARS optimizer for neural networks.
///
/// Implements Layer-wise Adaptive Rate Scaling from:
/// "Large Batch Training of Convolutional Networks" by You et al. (2017).
///
/// The per-layer local learning rate is computed as:
/// `local_lr = trust * ||w|| / (||g|| + wd * ||w|| + eps)`
///
/// When `clip = true`, this becomes LARC, which clips the local learning rate
/// to not exceed the global learning rate.
#[derive(Debug, Clone)]
pub struct Lars<F: Float + NumAssign + ScalarOperand + Debug> {
    /// Global learning rate
    learning_rate: F,
    /// Momentum coefficient
    momentum: F,
    /// Weight decay (L2 regularization)
    weight_decay: F,
    /// Trust coefficient (eta in the paper)
    trust_coefficient: F,
    /// If true, clip local lr to global lr (LARC variant)
    clip: bool,
    /// Numerical stability epsilon
    epsilon: F,
    /// Velocity (momentum) buffers per parameter tensor
    velocity: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
    /// Current timestep
    t: usize,
}

impl<F: Float + NumAssign + ScalarOperand + Debug> Lars<F> {
    /// Creates a new LARS optimizer.
    ///
    /// # Arguments
    /// * `learning_rate` - Global learning rate
    /// * `momentum` - Momentum coefficient (default 0.9)
    /// * `weight_decay` - Weight decay factor (default 1e-4)
    /// * `trust_coefficient` - Trust coefficient eta (default 0.001)
    /// * `clip` - If true, enables LARC (clip local lr to global lr)
    pub fn new(
        learning_rate: F,
        momentum: F,
        weight_decay: F,
        trust_coefficient: F,
        clip: bool,
    ) -> Result<Self> {
        if learning_rate <= F::zero() {
            return Err(NeuralError::InvalidArgument(
                "learning_rate must be positive".to_string(),
            ));
        }
        if momentum < F::zero() || momentum >= F::one() {
            return Err(NeuralError::InvalidArgument(
                "momentum must be in [0, 1)".to_string(),
            ));
        }
        let epsilon = F::from(1e-8).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 1e-8 to the floating point type".to_string(),
            )
        })?;
        Ok(Self {
            learning_rate,
            momentum,
            weight_decay,
            trust_coefficient,
            clip,
            epsilon,
            velocity: Vec::new(),
            t: 0,
        })
    }

    /// Creates a LARS optimizer with default hyperparameters.
    pub fn default_with_lr(learning_rate: F) -> Result<Self> {
        let momentum = F::from(0.9).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 0.9 to the floating point type".to_string(),
            )
        })?;
        let weight_decay = F::from(1e-4).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 1e-4 to the floating point type".to_string(),
            )
        })?;
        let trust_coefficient = F::from(0.001).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 0.001 to the floating point type".to_string(),
            )
        })?;
        Self::new(learning_rate, momentum, weight_decay, trust_coefficient, false)
    }

    /// Creates a LARC optimizer (LARS with clipping) with default hyperparameters.
    pub fn larc_with_lr(learning_rate: F) -> Result<Self> {
        let momentum = F::from(0.9).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 0.9 to the floating point type".to_string(),
            )
        })?;
        let weight_decay = F::from(1e-4).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 1e-4 to the floating point type".to_string(),
            )
        })?;
        let trust_coefficient = F::from(0.001).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 0.001 to the floating point type".to_string(),
            )
        })?;
        Self::new(learning_rate, momentum, weight_decay, trust_coefficient, true)
    }

    /// Compute the per-layer local learning rate.
    ///
    /// `local_lr = trust * ||w|| / (||g|| + wd * ||w|| + eps)`
    ///
    /// When clip=true (LARC), the result is clamped to `[0, global_lr]`.
    pub fn local_lr(&self, param_norm: F, grad_norm: F) -> F {
        let denom = grad_norm + self.weight_decay * param_norm + self.epsilon;
        if denom < self.epsilon {
            return self.learning_rate;
        }
        let local = self.trust_coefficient * param_norm / denom;
        if self.clip {
            if local < self.learning_rate {
                local
            } else {
                self.learning_rate
            }
        } else {
            local
        }
    }

    /// Returns whether clipping (LARC) is enabled.
    pub fn is_larc(&self) -> bool {
        self.clip
    }

    /// Returns the trust coefficient.
    pub fn get_trust_coefficient(&self) -> F {
        self.trust_coefficient
    }

    /// Returns the momentum coefficient.
    pub fn get_momentum(&self) -> F {
        self.momentum
    }

    /// Returns the weight decay.
    pub fn get_weight_decay(&self) -> F {
        self.weight_decay
    }

    /// Resets the velocity buffers.
    pub fn reset_state(&mut self) {
        self.velocity.clear();
        self.t = 0;
    }
}

impl<F: Float + NumAssign + ScalarOperand + Debug> Optimizer<F> for Lars<F> {
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

        // Initialize velocity buffers if needed
        if self.velocity.len() != params.len() {
            self.velocity = params
                .iter()
                .map(|p| Array::zeros(p.raw_dim()))
                .collect();
        }

        for i in 0..params.len() {
            // Compute L2 norms
            let param_norm_sq: F = params[i].iter().fold(F::zero(), |acc, &x| acc + x * x);
            let param_norm = param_norm_sq.sqrt();

            let grad_norm_sq: F = grads[i].iter().fold(F::zero(), |acc, &x| acc + x * x);
            let grad_norm = grad_norm_sq.sqrt();

            // Per-layer adaptive learning rate
            let local_lr = self.local_lr(param_norm, grad_norm);

            // Effective gradient = grad + wd * params
            let effective_grad = if self.weight_decay > F::zero() {
                &grads[i] + &(&params[i] * self.weight_decay)
            } else {
                grads[i].clone()
            };

            // velocity = momentum * velocity + local_lr * effective_grad
            self.velocity[i] = &self.velocity[i] * self.momentum
                + &(&effective_grad * local_lr);

            // params -= velocity
            params[i] = &params[i] - &self.velocity[i];
        }

        Ok(())
    }

    fn get_learning_rate(&self) -> F {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: F) {
        self.learning_rate = lr;
    }

    fn reset(&mut self) {
        self.reset_state();
    }

    fn name(&self) -> &'static str {
        "LARS"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, IxDyn};

    fn make_param(vals: &[f64]) -> Array<f64, IxDyn> {
        Array1::from_vec(vals.to_vec()).into_dyn()
    }

    #[test]
    fn test_lars_default_config() {
        let lars = Lars::<f64>::default_with_lr(0.01).expect("should succeed");
        assert!((lars.get_learning_rate() - 0.01).abs() < 1e-12);
        assert!((lars.get_momentum() - 0.9).abs() < 1e-12);
        assert!((lars.get_weight_decay() - 1e-4).abs() < 1e-12);
        assert!(!lars.is_larc());
    }

    #[test]
    fn test_lars_local_lr_formula() {
        let lars = Lars::<f64>::new(0.1, 0.9, 0.0, 0.01, false).expect("should succeed");
        // local_lr = trust * ||w|| / (||g|| + wd * ||w|| + eps)
        // = 0.01 * 3.0 / (4.0 + 0.0 * 3.0 + 1e-8) ≈ 0.0075
        let local = lars.local_lr(3.0, 4.0);
        let expected = 0.01 * 3.0 / (4.0 + 1e-8);
        assert!(
            (local - expected).abs() < 1e-6,
            "local_lr={local}, expected={expected}"
        );
    }

    #[test]
    fn test_larc_clips_lr() {
        // LARC: local_lr should never exceed global lr
        let global_lr = 0.01;
        let larc = Lars::<f64>::new(global_lr, 0.9, 0.0, 1.0, true).expect("should succeed");
        // With trust_coefficient=1.0, local_lr would be ||w||/||g|| which can be large
        let local = larc.local_lr(100.0, 1.0);
        assert!(
            local <= global_lr + 1e-12,
            "LARC local_lr={local} must not exceed global_lr={global_lr}"
        );
    }

    #[test]
    fn test_lars_update_descends() {
        let mut lars = Lars::<f64>::new(0.01, 0.9, 0.0, 0.001, false).expect("should succeed");
        // param = [2.0, 2.0], grad = [1.0, 1.0] (points in same direction)
        let mut params = vec![make_param(&[2.0_f64, 2.0])];
        let grads = vec![make_param(&[1.0_f64, 1.0])];

        // dot(param, grad) > 0 before update
        let before_dot: f64 = params[0].iter().zip(grads[0].iter()).map(|(p, g)| p * g).sum();
        lars.update(&mut params, &grads).expect("update should succeed");
        // After stepping in -grad direction, params should be smaller
        let after_dot: f64 = params[0].iter().zip(grads[0].iter()).map(|(p, g)| p * g).sum();
        assert!(
            after_dot < before_dot,
            "Update should reduce param·grad: before={before_dot}, after={after_dot}"
        );
    }

    #[test]
    fn test_lars_zero_grad() {
        let mut lars = Lars::<f64>::new(0.01, 0.9, 0.0, 0.001, false).expect("should succeed");
        let initial = vec![3.0_f64, -1.0, 2.0];
        let mut params = vec![make_param(&initial)];
        let grads = vec![make_param(&[0.0_f64, 0.0, 0.0])];
        // Should not panic; params should remain unchanged (velocity accumulates 0)
        lars.update(&mut params, &grads).expect("zero grad update should succeed");
        for (p, &orig) in params[0].iter().zip(initial.iter()) {
            assert!(
                (*p - orig).abs() < 1e-12,
                "With zero grad and no weight_decay, params should not change: {p} vs {orig}"
            );
        }
    }

    #[test]
    fn test_lars_mismatched_lengths() {
        let mut lars = Lars::<f64>::default_with_lr(0.01).expect("should succeed");
        let mut params = vec![make_param(&[1.0_f64, 2.0])];
        let grads = vec![
            make_param(&[0.1_f64, 0.2]),
            make_param(&[0.3_f64, 0.4]),
        ];
        assert!(
            lars.update(&mut params, &grads).is_err(),
            "Mismatched param/grad counts should error"
        );
    }

    #[test]
    fn test_larc_default_creation() {
        let larc = Lars::<f64>::larc_with_lr(0.05).expect("should succeed");
        assert!(larc.is_larc());
        assert!((larc.get_learning_rate() - 0.05).abs() < 1e-12);
    }
}
