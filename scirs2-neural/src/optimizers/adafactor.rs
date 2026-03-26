//! Adafactor: Adaptive Learning Rate with Factored Second Moments.
//!
//! Memory-efficient Adam variant using factored second moments (Shazeer & Stern, 2018).
//! For a matrix W ∈ R^{n×m}, stores O(n+m) instead of O(n×m) for the second moment.
//!
//! The factored approximation splits the second moment into row and column factors:
//! `v[i,j] ≈ row_factor[i] * col_factor[j] / mean(row_factor)`
//!
//! # Examples
//! ```
//! use scirs2_core::ndarray::{Array, IxDyn};
//! use scirs2_neural::optimizers::{Adafactor, Optimizer};
//!
//! let mut adafactor = Adafactor::<f64>::default_relative_step()
//!     .expect("optimizer creation should succeed");
//! ```

use crate::error::{NeuralError, Result};
use crate::optimizers::Optimizer;
use scirs2_core::ndarray::{Array, Array1, ScalarOperand};
use scirs2_core::numeric::{Float, NumAssign};
use std::fmt::Debug;

/// Second moment state for Adafactor — either full vector or factored (row + col factors).
///
/// For parameter vectors or small matrices (rows or cols < `min_dim_size_to_factor`),
/// the full second moment is stored. For large matrices, it is factored.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum SecondMomentState<F: Float> {
    /// Full second moment estimate (one value per parameter)
    Full(Array1<F>),
    /// Factored second moment: row and column factors
    Factored {
        /// Row factor (shape: [rows])
        row_factor: Array1<F>,
        /// Column factor (shape: [cols])
        col_factor: Array1<F>,
    },
}

/// Configuration for the Adafactor optimizer.
#[derive(Debug, Clone)]
pub struct AdafactorConfig {
    /// Fixed learning rate. If None, uses relative step size schedule.
    pub learning_rate: Option<f64>,
    /// Decay exponent for the second moment (default: 0.8).
    /// The decay factor at step t is: `1 - t^{-decay_factor}`.
    pub decay_factor: f64,
    /// Minimum parameter dimension to use factored second moment (default: 128).
    pub min_dim_size_to_factor: usize,
    /// Gradient clipping threshold via RMS (default: Some(1.0), None to disable).
    pub clipping_threshold: Option<f64>,
    /// Minimum learning rate when using relative step schedule (default: 1e-6).
    pub min_learning_rate: f64,
    /// Whether to use warmup init for relative step size (default: false).
    pub warmup_init: bool,
}

impl Default for AdafactorConfig {
    fn default() -> Self {
        Self {
            learning_rate: None,
            decay_factor: 0.8,
            min_dim_size_to_factor: 128,
            clipping_threshold: Some(1.0),
            min_learning_rate: 1e-6,
            warmup_init: false,
        }
    }
}

/// Adafactor optimizer for neural networks.
///
/// Implements "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
/// by Shazeer and Stern (2018).
///
/// Key properties:
/// - Sublinear memory: factored second moments for large weight matrices
/// - No fixed learning rate required (relative step size schedule)
/// - Gradient clipping via RMS normalization
#[derive(Debug)]
pub struct Adafactor<F: Float + NumAssign + ScalarOperand + Debug> {
    /// Optimizer configuration
    config: AdafactorConfig,
    /// Second moment state per parameter tensor
    second_moment: Vec<SecondMomentState<F>>,
    /// First moment (only used when learning_rate is fixed)
    first_moment: Vec<Option<Array1<F>>>,
    /// Current step count
    t: usize,
}

impl<F: Float + NumAssign + ScalarOperand + Debug> Adafactor<F> {
    /// Creates a new Adafactor optimizer with given configuration.
    ///
    /// The `param_shapes` slice provides the 2D shape `(rows, cols)` for each
    /// parameter tensor. Use `(n, 1)` for a vector of length `n`.
    pub fn new(config: AdafactorConfig, param_shapes: &[(usize, usize)]) -> Self {
        let second_moment = param_shapes
            .iter()
            .map(|&(rows, cols)| {
                let use_factor = rows >= config.min_dim_size_to_factor
                    && cols >= config.min_dim_size_to_factor;
                if use_factor {
                    SecondMomentState::Factored {
                        row_factor: Array1::ones(rows),
                        col_factor: Array1::ones(cols),
                    }
                } else {
                    SecondMomentState::Full(Array1::ones(rows * cols))
                }
            })
            .collect();

        let use_first_moment = config.learning_rate.is_some();
        let first_moment = param_shapes
            .iter()
            .map(|&(rows, cols)| {
                if use_first_moment {
                    Some(Array1::zeros(rows * cols))
                } else {
                    None
                }
            })
            .collect();

        Self {
            config,
            second_moment,
            first_moment,
            t: 0,
        }
    }

    /// Creates an Adafactor optimizer with the default relative step size schedule.
    pub fn default_relative_step() -> Result<Self> {
        Ok(Self::new(AdafactorConfig::default(), &[]))
    }

    /// Creates an Adafactor optimizer with a fixed learning rate.
    pub fn with_fixed_lr(learning_rate: f64) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(NeuralError::InvalidArgument(
                "learning_rate must be positive".to_string(),
            ));
        }
        let config = AdafactorConfig {
            learning_rate: Some(learning_rate),
            ..AdafactorConfig::default()
        };
        Ok(Self::new(config, &[]))
    }

    /// Relative step size: `ρ_t = max(min_lr, 1/sqrt(t))`
    pub fn relative_step_size(&self) -> f64 {
        let t = (self.t as f64).max(1.0);
        let step = if self.config.warmup_init {
            1e-6_f64.min(1.0 / t.sqrt())
        } else {
            1.0 / t.sqrt()
        };
        step.max(self.config.min_learning_rate)
    }

    /// Returns the current step count.
    pub fn get_step(&self) -> usize {
        self.t
    }

    /// Resets internal optimizer state (clears moment estimates).
    pub fn reset_state(&mut self) {
        for sm in &mut self.second_moment {
            match sm {
                SecondMomentState::Full(v) => v.fill(F::one()),
                SecondMomentState::Factored {
                    row_factor,
                    col_factor,
                } => {
                    row_factor.fill(F::one());
                    col_factor.fill(F::one());
                }
            }
        }
        for fm in &mut self.first_moment {
            if let Some(m) = fm {
                m.fill(F::zero());
            }
        }
        self.t = 0;
    }

    /// RMS gradient clipping: clips gradient so its RMS equals `threshold`.
    fn clip_gradient(grads: &[F], threshold: F) -> Vec<F> {
        let n = grads.len();
        if n == 0 {
            return Vec::new();
        }
        let sum_sq: F = grads.iter().fold(F::zero(), |acc, &g| acc + g * g);
        let rms = (sum_sq / F::from(n).unwrap_or(F::one())).sqrt();
        if rms > threshold {
            let scale = threshold / rms;
            grads.iter().map(|&g| g * scale).collect()
        } else {
            grads.to_vec()
        }
    }

    /// Reconstructs the approximate second moment estimate from factored state.
    ///
    /// For factored state: `v[i,j] = row_factor[i] * col_factor[j] / mean(row_factor)`
    fn reconstruct_factored(row_factor: &Array1<F>, col_factor: &Array1<F>) -> Vec<F> {
        let rows = row_factor.len();
        let cols = col_factor.len();
        let row_mean: F = row_factor.iter().fold(F::zero(), |acc, &x| acc + x)
            / F::from(rows).unwrap_or(F::one());
        let denom = if row_mean > F::zero() {
            row_mean
        } else {
            F::one()
        };
        let mut result = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                result.push(row_factor[r] * col_factor[c] / denom);
            }
        }
        result
    }

    /// Update second moment state given flat gradient and 2D shape.
    fn update_second_moment(
        &mut self,
        idx: usize,
        grads_flat: &[F],
        shape: (usize, usize),
        beta2: F,
    ) {
        let (rows, cols) = shape;
        let one_minus_beta2 = F::one() - beta2;

        match &mut self.second_moment[idx] {
            SecondMomentState::Full(v) => {
                for (vi, &gi) in v.iter_mut().zip(grads_flat.iter()) {
                    *vi = beta2 * (*vi) + one_minus_beta2 * gi * gi;
                }
            }
            SecondMomentState::Factored {
                row_factor,
                col_factor,
            } => {
                // Update row factor: mean over columns of g^2
                for r in 0..rows {
                    let row_mean_g2: F = if cols > 0 {
                        let sum: F = (0..cols)
                            .fold(F::zero(), |acc, c| {
                                let g = grads_flat[r * cols + c];
                                acc + g * g
                            });
                        sum / F::from(cols).unwrap_or(F::one())
                    } else {
                        F::zero()
                    };
                    row_factor[r] = beta2 * row_factor[r] + one_minus_beta2 * row_mean_g2;
                }
                // Update col factor: mean over rows of g^2
                for c in 0..cols {
                    let col_mean_g2: F = if rows > 0 {
                        let sum: F = (0..rows)
                            .fold(F::zero(), |acc, r| {
                                let g = grads_flat[r * cols + c];
                                acc + g * g
                            });
                        sum / F::from(rows).unwrap_or(F::one())
                    } else {
                        F::zero()
                    };
                    col_factor[c] = beta2 * col_factor[c] + one_minus_beta2 * col_mean_g2;
                }
            }
        }
    }

    /// Retrieve the second moment estimate as a flat vector.
    fn get_second_moment_flat(&self, idx: usize) -> Vec<F> {
        match &self.second_moment[idx] {
            SecondMomentState::Full(v) => v.to_vec(),
            SecondMomentState::Factored {
                row_factor,
                col_factor,
            } => Self::reconstruct_factored(row_factor, col_factor),
        }
    }
}

impl<F: Float + NumAssign + ScalarOperand + Debug> Optimizer<F> for Adafactor<F> {
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

        // Grow internal state if new parameters are added
        while self.second_moment.len() < params.len() {
            let n = params[self.second_moment.len()].len();
            // Default to full state for dynamically discovered params
            self.second_moment
                .push(SecondMomentState::Full(Array1::ones(n)));
            let use_first_moment = self.config.learning_rate.is_some();
            self.first_moment.push(if use_first_moment {
                Some(Array1::zeros(n))
            } else {
                None
            });
        }

        // Determine effective learning rate
        let lr_f64 = self
            .config
            .learning_rate
            .unwrap_or_else(|| self.relative_step_size());
        let lr = F::from(lr_f64).unwrap_or(F::from(1e-3).unwrap_or(F::one()));

        // Compute beta2 for this step: 1 - t^(-decay_factor)
        let t_f64 = self.t as f64;
        let beta2_f64 = 1.0 - t_f64.powf(-self.config.decay_factor);
        let beta2 = F::from(beta2_f64).unwrap_or(F::from(0.999).unwrap_or(F::one()));

        for i in 0..params.len() {
            let n = params[i].len();
            let grads_flat: Vec<F> = grads[i].iter().cloned().collect();

            // Apply gradient clipping if configured
            let grads_clipped: Vec<F> = if let Some(threshold) = self.config.clipping_threshold {
                let thresh = F::from(threshold).unwrap_or(F::one());
                Self::clip_gradient(&grads_flat, thresh)
            } else {
                grads_flat
            };

            // Infer 2D shape: if param has ndim >= 2, use first two dims; else treat as vector
            let shape = {
                let sh = params[i].shape();
                if sh.len() >= 2 {
                    (sh[0], n / sh[0])
                } else {
                    (n, 1)
                }
            };

            // Update second moment
            self.update_second_moment(i, &grads_clipped, shape, beta2);

            // Retrieve second moment as flat vector
            let v_flat = self.get_second_moment_flat(i);

            // Compute update: g / sqrt(v)
            let mut update: Vec<F> = grads_clipped
                .iter()
                .zip(v_flat.iter())
                .map(|(&g, &vi)| {
                    let denom = vi.sqrt().max(F::from(1e-30).unwrap_or(F::zero()));
                    g / denom
                })
                .collect();

            // Apply first moment (EMA) if using fixed learning rate
            if self.config.learning_rate.is_some() {
                // beta1 fixed at 0.9 for first-moment smoothing in fixed-lr mode
                let beta1_val =
                    F::from(0.9).unwrap_or(F::from(0.9_f32).unwrap_or(F::one() - F::one()));
                let one_minus_beta1 = F::one() - beta1_val;
                if let Some(ref mut m) = self.first_moment[i] {
                    for (mi, &ui) in m.iter_mut().zip(update.iter()) {
                        *mi = beta1_val * (*mi) + one_minus_beta1 * ui;
                    }
                    update = m.to_vec();
                }
            }

            // Apply update: params -= lr * update
            let params_flat: Vec<F> = params[i].iter().cloned().collect();
            let updated: Vec<F> = params_flat
                .iter()
                .zip(update.iter())
                .map(|(&p, &u)| p - lr * u)
                .collect();

            // Write back
            for (p, &u) in params[i].iter_mut().zip(updated.iter()) {
                *p = u;
            }
        }

        Ok(())
    }

    fn get_learning_rate(&self) -> F {
        if let Some(lr) = self.config.learning_rate {
            F::from(lr).unwrap_or(F::zero())
        } else {
            F::from(self.relative_step_size()).unwrap_or(F::zero())
        }
    }

    fn set_learning_rate(&mut self, lr: F) {
        let lr_f64 = lr.to_f64().unwrap_or(0.001);
        self.config.learning_rate = Some(lr_f64);
    }

    fn reset(&mut self) {
        self.reset_state();
    }

    fn name(&self) -> &'static str {
        "Adafactor"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2, IxDyn};

    fn make_param(vals: &[f64]) -> Array<f64, IxDyn> {
        Array1::from_vec(vals.to_vec()).into_dyn()
    }

    fn make_param_2d(vals: Vec<Vec<f64>>) -> Array<f64, IxDyn> {
        let rows = vals.len();
        let cols = vals[0].len();
        let flat: Vec<f64> = vals.into_iter().flatten().collect();
        Array2::from_shape_vec((rows, cols), flat)
            .expect("shape must match")
            .into_dyn()
    }

    #[test]
    fn test_adafactor_relative_step_schedule() {
        let adafactor = Adafactor::<f64>::new(AdafactorConfig::default(), &[]);
        // At step 0, t=0 → t.max(1) = 1 → 1/sqrt(1) = 1.0, but clamped to min_lr
        // Actually relative_step_size() does not use self.t directly in the base formula
        // We test by simulating multiple steps manually
        let mut opt = Adafactor::<f64>::new(AdafactorConfig::default(), &[(4, 1)]);
        let mut params = vec![make_param(&[1.0_f64, 2.0, 3.0, 4.0])];
        let grads = vec![make_param(&[0.1_f64, 0.1, 0.1, 0.1])];

        // After step 1, lr = 1/sqrt(1) = 1.0
        let _ = opt.update(&mut params, &grads);
        let step1_lr = opt.relative_step_size();

        // After step 10, lr = 1/sqrt(10) < 1/sqrt(1)
        for _ in 0..9 {
            let _ = opt.update(&mut params, &grads);
        }
        let step10_lr = opt.relative_step_size();

        assert!(
            step10_lr < step1_lr,
            "LR should decrease: step1={step1_lr}, step10={step10_lr}"
        );
        let _ = adafactor; // suppress unused warning
    }

    #[test]
    fn test_adafactor_gradient_clipping() {
        let threshold = 1.0_f64;
        let thresh_f = threshold;
        // Large gradient: RMS = sqrt(sum(g^2)/n) = sqrt(100) = 10 >> 1.0
        let grads = vec![10.0_f64; 4];
        let clipped = Adafactor::<f64>::clip_gradient(&grads, thresh_f);
        let rms = (clipped.iter().map(|g| g * g).sum::<f64>() / clipped.len() as f64).sqrt();
        assert!(
            (rms - threshold).abs() < 1e-6,
            "Clipped gradient RMS should equal threshold: {rms}"
        );
    }

    #[test]
    fn test_adafactor_full_small_matrix() {
        // 4x4 = 16 elements, min_dim_size_to_factor = 128 → should use Full state
        let config = AdafactorConfig {
            learning_rate: Some(0.01),
            min_dim_size_to_factor: 128,
            clipping_threshold: None,
            ..AdafactorConfig::default()
        };
        let opt = Adafactor::<f64>::new(config, &[(4, 4)]);
        match &opt.second_moment[0] {
            SecondMomentState::Full(_) => {} // expected
            SecondMomentState::Factored { .. } => {
                panic!("4x4 matrix should use Full state when min_dim=128")
            }
        }
    }

    #[test]
    fn test_adafactor_factored_state() {
        // 200x200 matrix: both dims >= 128 → should use Factored state
        let config = AdafactorConfig {
            learning_rate: Some(0.01),
            min_dim_size_to_factor: 128,
            clipping_threshold: None,
            ..AdafactorConfig::default()
        };
        let opt = Adafactor::<f64>::new(config, &[(200, 200)]);
        match &opt.second_moment[0] {
            SecondMomentState::Factored { row_factor, col_factor } => {
                assert_eq!(row_factor.len(), 200);
                assert_eq!(col_factor.len(), 200);
            }
            SecondMomentState::Full(_) => {
                panic!("200x200 matrix should use Factored state when min_dim=128")
            }
        }
    }

    #[test]
    fn test_adafactor_update_step() {
        // Single step should move params in negative gradient direction
        let config = AdafactorConfig {
            learning_rate: Some(0.1),
            clipping_threshold: None,
            min_dim_size_to_factor: 128,
            ..AdafactorConfig::default()
        };
        let mut opt = Adafactor::<f64>::new(config, &[(4, 1)]);
        let initial = vec![2.0_f64, -1.0, 3.0, 0.5];
        let mut params = vec![make_param(&initial)];
        // Positive gradients → params should decrease
        let grads = vec![make_param(&[1.0_f64, 1.0, 1.0, 1.0])];
        opt.update(&mut params, &grads).expect("update should succeed");
        for (p, &p0) in params[0].iter().zip(initial.iter()) {
            assert!(
                *p < p0,
                "Params should decrease with positive gradient: {p} < {p0}"
            );
        }
    }

    #[test]
    fn test_adafactor_mismatched_lengths() {
        let mut opt = Adafactor::<f64>::default_relative_step().expect("should succeed");
        let mut params = vec![make_param(&[1.0_f64])];
        let grads = vec![make_param(&[0.1_f64]), make_param(&[0.2_f64])];
        assert!(
            opt.update(&mut params, &grads).is_err(),
            "Mismatched lengths should return error"
        );
    }

    #[test]
    fn test_adafactor_convergence() {
        // Minimize f(x) = 0.5 * ||x||^2, grad = x
        let config = AdafactorConfig {
            learning_rate: Some(0.05),
            clipping_threshold: None,
            min_dim_size_to_factor: 128,
            ..AdafactorConfig::default()
        };
        let mut opt = Adafactor::<f64>::new(config, &[(3, 1)]);
        let mut params = vec![make_param(&[2.0_f64, -3.0, 1.5])];
        for _ in 0..300 {
            let grads: Vec<Array<f64, IxDyn>> = vec![params[0].clone()];
            opt.update(&mut params, &grads).expect("update should succeed");
        }
        let norm_sq: f64 = params[0].iter().map(|x| x * x).sum();
        assert!(
            norm_sq < 0.1,
            "Adafactor should converge: ||x||^2 = {norm_sq}"
        );
    }
}
