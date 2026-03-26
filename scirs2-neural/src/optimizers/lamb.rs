//! LAMB (Layer-wise Adaptive Moments for Batch training) optimizer.
//!
//! Extends Adam with per-layer trust ratio for large-batch training.
//! Designed for large-batch BERT training (You et al., 2019).
//!
//! The key difference from Adam is the LAMB trust ratio:
//! `trust = ||params|| / ||adam_update||`
//! which scales the effective learning rate per layer.
//!
//! # Examples
//! ```
//! use scirs2_core::ndarray::{Array, IxDyn};
//! use scirs2_neural::optimizers::{Lamb, Optimizer};
//!
//! let mut lamb = Lamb::<f64>::default_with_lr(0.001)
//!     .expect("optimizer creation should succeed");
//! ```

use crate::error::{NeuralError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;

/// LAMB optimizer for neural networks.
///
/// Implements LAMB from:
/// "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
/// by You et al. (2019).
///
/// The update rule is:
/// 1. Compute Adam update: `u = m_hat / (sqrt(v_hat) + eps) + wd * params`
/// 2. Compute trust ratio: `r = ||params|| / ||u||`
/// 3. Apply: `params -= lr * r * u`
#[derive(Debug, Clone)]
pub struct Lamb<F: Float + NumAssign + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: F,
    /// First moment decay (default 0.9)
    beta1: F,
    /// Second moment decay (default 0.999)
    beta2: F,
    /// Numerical stability term
    epsilon: F,
    /// Weight decay (L2 regularization, default 0.01)
    weight_decay: F,
    /// Optional trust ratio clamp (if Some(c), trust = min(trust, c))
    clamp_value: Option<F>,
    /// Whether to apply bias correction
    bias_correction: bool,
    /// First moment estimates per parameter tensor
    m: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
    /// Second moment estimates per parameter tensor
    v: Vec<Array<F, scirs2_core::ndarray::IxDyn>>,
    /// Current timestep
    t: usize,
}

impl<F: Float + NumAssign + ScalarOperand + Debug> Lamb<F> {
    /// Creates a new LAMB optimizer with given hyperparameters.
    ///
    /// # Arguments
    /// * `learning_rate` - Global learning rate
    /// * `beta1` - First moment decay (default 0.9)
    /// * `beta2` - Second moment decay (default 0.999)
    /// * `epsilon` - Numerical stability (default 1e-6)
    /// * `weight_decay` - Weight decay (default 0.01)
    /// * `clamp_value` - Optional trust ratio clamp
    /// * `bias_correction` - Whether to apply Adam bias correction
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        learning_rate: F,
        beta1: F,
        beta2: F,
        epsilon: F,
        weight_decay: F,
        clamp_value: Option<F>,
        bias_correction: bool,
    ) -> Result<Self> {
        if learning_rate <= F::zero() {
            return Err(NeuralError::InvalidArgument(
                "learning_rate must be positive".to_string(),
            ));
        }
        if beta1 < F::zero() || beta1 >= F::one() {
            return Err(NeuralError::InvalidArgument(
                "beta1 must be in [0, 1)".to_string(),
            ));
        }
        if beta2 < F::zero() || beta2 >= F::one() {
            return Err(NeuralError::InvalidArgument(
                "beta2 must be in [0, 1)".to_string(),
            ));
        }
        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            clamp_value,
            bias_correction,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        })
    }

    /// Creates a LAMB optimizer with default hyperparameters.
    pub fn default_with_lr(learning_rate: F) -> Result<Self> {
        let beta1 = F::from(0.9).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 0.9 to the floating point type".to_string(),
            )
        })?;
        let beta2 = F::from(0.999).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 0.999 to the floating point type".to_string(),
            )
        })?;
        let epsilon = F::from(1e-6).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 1e-6 to the floating point type".to_string(),
            )
        })?;
        let weight_decay = F::from(0.01).ok_or_else(|| {
            NeuralError::InvalidArgument(
                "Failed to convert 0.01 to the floating point type".to_string(),
            )
        })?;
        Self::new(learning_rate, beta1, beta2, epsilon, weight_decay, None, true)
    }

    /// Returns the trust ratio clamp value.
    pub fn get_clamp_value(&self) -> Option<F> {
        self.clamp_value
    }

    /// Sets the trust ratio clamp value.
    pub fn set_clamp_value(&mut self, clamp: Option<F>) {
        self.clamp_value = clamp;
    }

    /// Returns whether bias correction is enabled.
    pub fn is_bias_correction(&self) -> bool {
        self.bias_correction
    }

    /// Returns the current step count.
    pub fn get_step(&self) -> usize {
        self.t
    }

    /// Resets internal optimizer state.
    pub fn reset_state(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

impl<F: Float + NumAssign + ScalarOperand + Debug> Optimizer<F> for Lamb<F> {
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

        // Initialize moment buffers if needed
        if self.m.len() != params.len() {
            self.m = params
                .iter()
                .map(|p| Array::zeros(p.raw_dim()))
                .collect();
            self.v = params
                .iter()
                .map(|p| Array::zeros(p.raw_dim()))
                .collect();
        }

        let one_minus_beta1 = F::one() - self.beta1;
        let one_minus_beta2 = F::one() - self.beta2;
        let beta1_pow_t = self.beta1.powi(self.t as i32);
        let beta2_pow_t = self.beta2.powi(self.t as i32);
        let bc1 = F::one() - beta1_pow_t;
        let bc2 = F::one() - beta2_pow_t;

        for i in 0..params.len() {
            // Update first moment: m = beta1*m + (1-beta1)*g
            self.m[i] = &self.m[i] * self.beta1 + &(&grads[i] * one_minus_beta1);

            // Update second moment: v = beta2*v + (1-beta2)*g^2
            self.v[i] = &self.v[i] * self.beta2
                + &(grads[i].mapv(|x| x * x) * one_minus_beta2);

            // Bias-corrected moments
            let m_hat = if self.bias_correction {
                &self.m[i] / bc1
            } else {
                self.m[i].clone()
            };
            let v_hat = if self.bias_correction {
                &self.v[i] / bc2
            } else {
                self.v[i].clone()
            };

            // Adam update direction: m_hat / (sqrt(v_hat) + eps) + wd * params
            let adam_update = {
                let denom = v_hat.mapv(|x| x.sqrt()) + self.epsilon;
                let adam_part = &m_hat / &denom;
                if self.weight_decay > F::zero() {
                    &adam_part + &(&params[i] * self.weight_decay)
                } else {
                    adam_part
                }
            };

            // LAMB trust ratio: ||params|| / ||adam_update||
            let param_norm_sq: F = params[i].iter().fold(F::zero(), |acc, &x| acc + x * x);
            let param_norm = param_norm_sq.sqrt();

            let update_norm_sq: F = adam_update
                .iter()
                .fold(F::zero(), |acc, &x| acc + x * x);
            let update_norm = update_norm_sq.sqrt();

            let trust = if update_norm > F::zero() {
                let raw_trust = param_norm / update_norm;
                if let Some(clamp) = self.clamp_value {
                    if raw_trust < clamp {
                        raw_trust
                    } else {
                        clamp
                    }
                } else {
                    raw_trust
                }
            } else {
                F::one()
            };

            // params -= lr * trust * adam_update
            params[i] = &params[i] - &(&adam_update * (self.learning_rate * trust));
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
        "LAMB"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, IxDyn};

    fn make_param(vals: &[f64]) -> Array<f64, IxDyn> {
        Array1::from_vec(vals.to_vec()).into_dyn()
    }

    #[test]
    fn test_lamb_default_config() {
        let lamb = Lamb::<f64>::default_with_lr(0.001).expect("should succeed");
        assert!((lamb.get_learning_rate() - 0.001).abs() < 1e-15);
        assert!(lamb.is_bias_correction());
        assert!(lamb.get_clamp_value().is_none());
        assert_eq!(lamb.get_step(), 0);
    }

    #[test]
    fn test_lamb_adam_step_no_wd() {
        // With weight_decay=0 and trust=1 (param_norm == update_norm),
        // LAMB should behave similarly to Adam in terms of direction.
        let beta1 = 0.9_f64;
        let beta2 = 0.999_f64;
        let eps = 1e-6_f64;
        let lr = 0.01_f64;

        let mut lamb = Lamb::<f64>::new(lr, beta1, beta2, eps, 0.0, None, true)
            .expect("should succeed");

        let params_init = vec![1.0_f64, 2.0, 3.0];
        let mut params = vec![make_param(&params_init)];
        let grads = vec![make_param(&[0.1_f64, 0.2, 0.3])];

        lamb.update(&mut params, &grads).expect("update should succeed");

        // Params should have moved in the negative gradient direction
        for (p, &p0) in params[0].iter().zip(params_init.iter()) {
            assert!(
                *p < p0,
                "param should decrease: {p} < {p0}"
            );
        }
    }

    #[test]
    fn test_lamb_trust_ratio_clamp() {
        let clamp = 0.5_f64;
        let mut lamb = Lamb::<f64>::new(0.01, 0.9, 0.999, 1e-6, 0.0, Some(clamp), true)
            .expect("should succeed");

        // Large params, small grad → trust would be large without clamp
        let mut params = vec![make_param(&[100.0_f64; 10])];
        let grads = vec![make_param(&[0.001_f64; 10])];

        let params_before: Vec<f64> = params[0].iter().cloned().collect();
        lamb.update(&mut params, &grads).expect("update should succeed");

        // With clamped trust, the update should be limited
        // Check that the update is finite and params changed
        for (p, pb) in params[0].iter().zip(params_before.iter()) {
            assert!(p.is_finite(), "params should be finite after clamped LAMB");
            assert!(*p != *pb, "params should change after update");
        }
        assert_eq!(lamb.get_clamp_value(), Some(clamp));
    }

    #[test]
    fn test_lamb_converges_on_quadratic() {
        // Minimize f(x) = 0.5 * ||x||^2, grad = x
        // Optimal: x* = 0
        // LAMB computes trust ratio = ||params||/||adam_update||, which can vary;
        // use a higher lr and more steps to ensure convergence.
        let mut lamb = Lamb::<f64>::new(0.05, 0.9, 0.999, 1e-6, 0.0, None, true)
            .expect("should succeed");

        let mut params = vec![make_param(&[1.0_f64, -2.0, 3.0])];

        for _ in 0..500 {
            let grads: Vec<Array<f64, IxDyn>> = vec![params[0].clone()];
            lamb.update(&mut params, &grads).expect("update should succeed");
        }

        let norm_sq: f64 = params[0].iter().map(|x| x * x).sum();
        assert!(
            norm_sq < 0.1,
            "LAMB should converge on quadratic: ||x||^2 = {norm_sq}"
        );
    }

    #[test]
    fn test_lamb_mismatched_lengths() {
        let mut lamb = Lamb::<f64>::default_with_lr(0.001).expect("should succeed");
        let mut params = vec![make_param(&[1.0_f64])];
        let grads = vec![make_param(&[0.1_f64]), make_param(&[0.2_f64])];
        assert!(
            lamb.update(&mut params, &grads).is_err(),
            "Mismatched lengths should return error"
        );
    }

    #[test]
    fn test_lamb_zero_update_norm() {
        // When gradient is zero and wd=0, adam_update is zero → trust = 1.0
        let mut lamb = Lamb::<f64>::new(0.01, 0.9, 0.999, 1e-6, 0.0, None, true)
            .expect("should succeed");
        let initial = vec![1.0_f64, 2.0, 3.0];
        let mut params = vec![make_param(&initial)];
        let grads = vec![make_param(&[0.0_f64, 0.0, 0.0])];
        // Should not panic or produce NaN
        lamb.update(&mut params, &grads).expect("zero grad should succeed");
        for p in params[0].iter() {
            assert!(p.is_finite(), "params must remain finite with zero grad");
        }
    }
}
