//! Plain-Rust gradient-descent optimizers and learning-rate schedules
//!
//! This module provides a self-contained, `Vec<f64>`-based optimizer API that
//! is independent of the autograd computation graph. It is the analogue of
//! PyTorch's `torch.optim` package but operating on plain numeric vectors.
//!
//! # Optimizers
//!
//! | Type | Description |
//! |------|-------------|
//! | [`Sgd`] | Stochastic gradient descent with optional momentum |
//! | [`AdamOptimizer`] | Adam with bias correction, optional weight decay |
//! | [`RmsProp`] | RMSprop with optional momentum |
//! | [`Adagrad`] | Adaptive gradient accumulation |
//!
//! # Learning-rate schedules
//!
//! | Type | Description |
//! |------|-------------|
//! | [`CosineAnnealingSchedule`] | Cosine decay from `lr_max` to `lr_min` over `t_max` steps |
//! | [`OneCycleSchedule`] | Triangular (1-cycle) schedule: warm-up then anneal |
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::plain_optimizers::{AdamOptimizer, Optimizer};
//!
//! let mut params = vec![0.0f64, 0.0];
//! let mut adam = AdamOptimizer::new(0.01);
//!
//! // Minimise f(x,y) = x^2 + y^2 starting from (0.5, -0.3)
//! let mut p = vec![0.5, -0.3];
//! for _ in 0..200 {
//!     // Gradient of x^2+y^2 is [2x, 2y]
//!     let grads = vec![2.0 * p[0], 2.0 * p[1]];
//!     adam.step(&mut p, &grads).expect("step");
//! }
//! assert!(p[0].abs() < 0.1);
//! assert!(p[1].abs() < 0.1);
//! ```

use crate::error::AutogradError;

// ---------------------------------------------------------------------------
// Optimizer trait
// ---------------------------------------------------------------------------

/// Stateful gradient-descent optimizer operating on plain `Vec<f64>`.
pub trait Optimizer {
    /// Apply one gradient-descent step, modifying `params` in-place.
    ///
    /// # Errors
    /// Returns `AutogradError` if `params` and `grads` lengths differ.
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) -> Result<(), AutogradError>;

    /// Reset internal optimizer state (moments, accumulators, …).
    fn zero_grad(&mut self);

    /// Current learning rate.
    fn learning_rate(&self) -> f64;
}

// ---------------------------------------------------------------------------
// SGD with momentum
// ---------------------------------------------------------------------------

/// Stochastic Gradient Descent with optional momentum.
///
/// Update rule:
/// ```text
/// v_t  = momentum * v_{t-1} + grads
/// params -= lr * v_t
/// ```
///
/// When `momentum == 0.0`, this reduces to vanilla SGD: `params -= lr * grads`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::plain_optimizers::{Sgd, Optimizer};
///
/// let mut sgd = Sgd::new(0.1, 0.9);
/// let mut p = vec![1.0, -1.0];
/// let grads = vec![2.0, -2.0];
/// sgd.step(&mut p, &grads).expect("step");
/// // p[0] decreases, p[1] increases
/// assert!(p[0] < 1.0);
/// assert!(p[1] > -1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Sgd {
    /// Learning rate.
    pub lr: f64,
    /// Momentum coefficient (`0.0` = no momentum).
    pub momentum: f64,
    /// Velocity buffers (one per parameter).
    velocity: Vec<f64>,
}

impl Sgd {
    /// Create a new SGD optimizer.
    ///
    /// # Arguments
    /// * `lr`       – Learning rate (must be > 0)
    /// * `momentum` – Momentum coefficient in `[0, 1)`
    pub fn new(lr: f64, momentum: f64) -> Self {
        Self {
            lr,
            momentum,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) -> Result<(), AutogradError> {
        if params.len() != grads.len() {
            return Err(AutogradError::ShapeMismatch(format!(
                "SGD: params length {} != grads length {}",
                params.len(),
                grads.len()
            )));
        }
        let n = params.len();
        // Lazy initialisation of velocity buffer
        if self.velocity.len() != n {
            self.velocity = vec![0.0f64; n];
        }
        for i in 0..n {
            self.velocity[i] = self.momentum * self.velocity[i] + grads[i];
            params[i] -= self.lr * self.velocity[i];
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for v in self.velocity.iter_mut() {
            *v = 0.0;
        }
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------

/// Adam optimizer (Adaptive Moment Estimation).
///
/// Update rule:
/// ```text
/// m_t  = β₁·m_{t-1} + (1-β₁)·g
/// v_t  = β₂·v_{t-1} + (1-β₂)·g²
/// m̂   = m_t / (1 - β₁^t)
/// v̂   = v_t / (1 - β₂^t)
/// params -= lr · m̂ / (√v̂ + ε)
/// ```
///
/// An optional `weight_decay` applies L2 regularisation before the moment update.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::plain_optimizers::{AdamOptimizer, Optimizer};
///
/// let mut adam = AdamOptimizer::new(0.01);
/// let mut p = vec![0.5, -0.3];
/// for _ in 0..300 {
///     let grads = vec![2.0 * p[0], 2.0 * p[1]];
///     adam.step(&mut p, &grads).expect("step");
/// }
/// assert!(p[0].abs() < 0.05);
/// ```
#[derive(Debug, Clone)]
pub struct AdamOptimizer {
    /// Learning rate.
    pub lr: f64,
    /// First-moment decay rate (default 0.9).
    pub beta1: f64,
    /// Second-moment decay rate (default 0.999).
    pub beta2: f64,
    /// Numerical stability constant (default 1e-8).
    pub eps: f64,
    /// L2 weight decay (default 0.0).
    pub weight_decay: f64,
    /// First moment estimate (`m`).
    m: Vec<f64>,
    /// Second moment estimate (`v`).
    v: Vec<f64>,
    /// Step counter (used for bias correction).
    t: usize,
}

impl AdamOptimizer {
    /// Create an Adam optimizer with default hyper-parameters.
    ///
    /// Defaults: β₁=0.9, β₂=0.999, ε=1e-8, weight_decay=0.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Create an Adam optimizer with explicit hyper-parameters.
    pub fn with_params(lr: f64, beta1: f64, beta2: f64, eps: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay: 0.0,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl Optimizer for AdamOptimizer {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) -> Result<(), AutogradError> {
        if params.len() != grads.len() {
            return Err(AutogradError::ShapeMismatch(format!(
                "Adam: params length {} != grads length {}",
                params.len(),
                grads.len()
            )));
        }
        let n = params.len();
        if self.m.len() != n {
            self.m = vec![0.0f64; n];
            self.v = vec![0.0f64; n];
        }
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        for i in 0..n {
            let mut g = grads[i];
            if self.weight_decay != 0.0 {
                g += self.weight_decay * params[i];
            }
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for m in self.m.iter_mut() {
            *m = 0.0;
        }
        for v in self.v.iter_mut() {
            *v = 0.0;
        }
        self.t = 0;
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }
}

// ---------------------------------------------------------------------------
// RMSprop
// ---------------------------------------------------------------------------

/// RMSprop optimizer with optional momentum.
///
/// Update rule:
/// ```text
/// E[g²]_t = α·E[g²]_{t-1} + (1-α)·g²
/// v_t      = momentum·v_{t-1} + lr·g / (√E[g²]_t + ε)
/// params  -= v_t
/// ```
///
/// When `momentum == 0.0`, the velocity buffer is bypassed.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::plain_optimizers::{RmsProp, Optimizer};
///
/// let mut rms = RmsProp::new(0.01);
/// let mut p = vec![1.0];
/// rms.step(&mut p, &[2.0]).expect("step");
/// assert!(p[0] < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct RmsProp {
    /// Learning rate.
    pub lr: f64,
    /// Smoothing constant α (default 0.99).
    pub alpha: f64,
    /// Numerical stability constant (default 1e-8).
    pub eps: f64,
    /// Momentum coefficient (default 0.0).
    pub momentum: f64,
    /// Running mean-square cache.
    cache: Vec<f64>,
    /// Momentum velocity buffer.
    velocity: Vec<f64>,
}

impl RmsProp {
    /// Create an RMSprop optimizer with default hyper-parameters.
    ///
    /// Defaults: α=0.99, ε=1e-8, momentum=0.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            cache: Vec::new(),
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for RmsProp {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) -> Result<(), AutogradError> {
        if params.len() != grads.len() {
            return Err(AutogradError::ShapeMismatch(format!(
                "RMSprop: params length {} != grads length {}",
                params.len(),
                grads.len()
            )));
        }
        let n = params.len();
        if self.cache.len() != n {
            self.cache = vec![0.0f64; n];
            self.velocity = vec![0.0f64; n];
        }
        for i in 0..n {
            let g = grads[i];
            self.cache[i] = self.alpha * self.cache[i] + (1.0 - self.alpha) * g * g;
            let rms = (self.cache[i] + self.eps).sqrt();
            let update = self.lr * g / rms;
            if self.momentum != 0.0 {
                self.velocity[i] = self.momentum * self.velocity[i] + update;
                params[i] -= self.velocity[i];
            } else {
                params[i] -= update;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for c in self.cache.iter_mut() {
            *c = 0.0;
        }
        for v in self.velocity.iter_mut() {
            *v = 0.0;
        }
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }
}

// ---------------------------------------------------------------------------
// Adagrad
// ---------------------------------------------------------------------------

/// Adagrad optimizer (Adaptive Gradient Algorithm).
///
/// Accumulates squared gradients and scales the learning rate individually:
/// ```text
/// G_t  = G_{t-1} + g²
/// params -= lr / (√G_t + ε) · g
/// ```
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::plain_optimizers::{Adagrad, Optimizer};
///
/// let mut ada = Adagrad::new(0.1);
/// let mut p = vec![1.0];
/// ada.step(&mut p, &[2.0]).expect("step");
/// assert!(p[0] < 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Adagrad {
    /// Learning rate.
    pub lr: f64,
    /// Numerical stability constant (default 1e-8).
    pub eps: f64,
    /// Accumulated squared-gradient sum.
    sum_sq_grad: Vec<f64>,
}

impl Adagrad {
    /// Create an Adagrad optimizer with default hyper-parameters.
    ///
    /// Defaults: ε=1e-8.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            eps: 1e-8,
            sum_sq_grad: Vec::new(),
        }
    }
}

impl Optimizer for Adagrad {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) -> Result<(), AutogradError> {
        if params.len() != grads.len() {
            return Err(AutogradError::ShapeMismatch(format!(
                "Adagrad: params length {} != grads length {}",
                params.len(),
                grads.len()
            )));
        }
        let n = params.len();
        if self.sum_sq_grad.len() != n {
            self.sum_sq_grad = vec![0.0f64; n];
        }
        for i in 0..n {
            let g = grads[i];
            self.sum_sq_grad[i] += g * g;
            let lr_scaled = self.lr / (self.sum_sq_grad[i].sqrt() + self.eps);
            params[i] -= lr_scaled * g;
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for s in self.sum_sq_grad.iter_mut() {
            *s = 0.0;
        }
    }

    fn learning_rate(&self) -> f64 {
        self.lr
    }
}

// ---------------------------------------------------------------------------
// CosineAnnealingSchedule
// ---------------------------------------------------------------------------

/// Cosine annealing learning-rate schedule.
///
/// `lr(t) = lr_min + (lr_max - lr_min) · (1 + cos(π·t / t_max)) / 2`
///
/// The schedule decays from `lr_max` at `t=0` to `lr_min` at `t=t_max`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::plain_optimizers::CosineAnnealingSchedule;
///
/// let mut sched = CosineAnnealingSchedule::new(0.1, 0.001, 100);
/// // Starts at lr_max
/// assert!((sched.get_lr() - 0.1).abs() < 1e-10);
/// // Advance to end
/// for _ in 0..100 { sched.step(); }
/// assert!((sched.get_lr() - 0.001).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingSchedule {
    /// Maximum (initial) learning rate.
    pub lr_max: f64,
    /// Minimum learning rate at the end of the cycle.
    pub lr_min: f64,
    /// Number of steps per cycle.
    pub t_max: usize,
    /// Internal step counter.
    pub step: usize,
}

impl CosineAnnealingSchedule {
    /// Create a new cosine annealing schedule.
    ///
    /// # Arguments
    /// * `lr_max` – Peak learning rate at step 0
    /// * `lr_min` – Minimum learning rate at step `t_max`
    /// * `t_max`  – Cycle length in steps
    pub fn new(lr_max: f64, lr_min: f64, t_max: usize) -> Self {
        Self {
            lr_max,
            lr_min,
            t_max,
            step: 0,
        }
    }

    /// Return the learning rate at the current internal step.
    pub fn get_lr(&self) -> f64 {
        if self.t_max == 0 {
            return self.lr_max;
        }
        let t = self.step as f64;
        let t_max = self.t_max as f64;
        let cos_val = (std::f64::consts::PI * t / t_max).cos();
        self.lr_min + (self.lr_max - self.lr_min) * (1.0 + cos_val) / 2.0
    }

    /// Advance the schedule by one step.
    pub fn step(&mut self) {
        self.step += 1;
    }
}

// ---------------------------------------------------------------------------
// OneCycleSchedule
// ---------------------------------------------------------------------------

/// One-cycle (triangular) learning-rate schedule.
///
/// The schedule consists of two phases:
/// 1. **Warm-up** (steps 0 .. `pct_start * total_steps`): linearly increase
///    from `max_lr / div_factor` to `max_lr`.
/// 2. **Anneal** (remaining steps): cosine decay from `max_lr` down to
///    `max_lr / final_div_factor`.
///
/// Default `div_factor = 25.0` and `final_div_factor = 1e4`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::plain_optimizers::OneCycleSchedule;
///
/// let mut sched = OneCycleSchedule::new(0.1, 100);
/// let initial_lr = sched.get_lr();
/// // Advance to peak
/// for _ in 0..30 { sched.step(); }
/// let peak_lr = sched.get_lr();
/// // Peak should be at or near max_lr
/// assert!(peak_lr >= initial_lr);
/// ```
#[derive(Debug, Clone)]
pub struct OneCycleSchedule {
    /// Peak learning rate.
    pub max_lr: f64,
    /// Total number of steps in the schedule.
    pub total_steps: usize,
    /// Fraction of steps spent in the warm-up phase (default 0.3).
    pub pct_start: f64,
    /// Internal step counter.
    pub step: usize,
    /// Initial LR divisor (start LR = max_lr / div_factor).
    div_factor: f64,
    /// Final LR divisor (min LR = max_lr / final_div_factor).
    final_div_factor: f64,
}

impl OneCycleSchedule {
    /// Create a one-cycle schedule with default hyper-parameters.
    ///
    /// Defaults: `pct_start=0.3`, `div_factor=25`, `final_div_factor=1e4`.
    ///
    /// # Arguments
    /// * `max_lr`      – Peak learning rate
    /// * `total_steps` – Total number of optimiser steps
    pub fn new(max_lr: f64, total_steps: usize) -> Self {
        Self {
            max_lr,
            total_steps,
            pct_start: 0.3,
            step: 0,
            div_factor: 25.0,
            final_div_factor: 1e4,
        }
    }

    /// Return the learning rate at the current internal step.
    pub fn get_lr(&self) -> f64 {
        if self.total_steps == 0 {
            return self.max_lr;
        }
        let warmup_steps = (self.pct_start * self.total_steps as f64) as usize;
        let start_lr = self.max_lr / self.div_factor;
        let min_lr = self.max_lr / self.final_div_factor;

        let t = self.step;
        if t <= warmup_steps {
            // Linear warm-up: start_lr → max_lr
            if warmup_steps == 0 {
                return self.max_lr;
            }
            let progress = t as f64 / warmup_steps as f64;
            start_lr + (self.max_lr - start_lr) * progress
        } else {
            // Cosine anneal: max_lr → min_lr
            let anneal_steps = self.total_steps.saturating_sub(warmup_steps);
            if anneal_steps == 0 {
                return min_lr;
            }
            let progress = (t - warmup_steps) as f64 / anneal_steps as f64;
            let cos_val = (std::f64::consts::PI * progress.min(1.0)).cos();
            min_lr + (self.max_lr - min_lr) * (1.0 + cos_val) / 2.0
        }
    }

    /// Advance the schedule by one step.
    pub fn step(&mut self) {
        self.step += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    // ----- SGD tests --------------------------------------------------------

    #[test]
    fn test_sgd_single_step_vanilla() {
        // Vanilla SGD (no momentum): params -= lr * grads
        let mut sgd = Sgd::new(0.1, 0.0);
        let mut p = vec![1.0, -1.0];
        sgd.step(&mut p, &[2.0, -4.0]).expect("sgd step");
        assert!((p[0] - (1.0 - 0.1 * 2.0)).abs() < TOL);
        assert!((p[1] - (-1.0 - 0.1 * (-4.0))).abs() < TOL);
    }

    #[test]
    fn test_sgd_reduces_loss_quadratic() {
        // f(x) = x^2, grad = 2x, minimum at 0
        let mut sgd = Sgd::new(0.01, 0.9);
        let mut p = vec![5.0];
        for _ in 0..500 {
            let g = vec![2.0 * p[0]];
            sgd.step(&mut p, &g).expect("sgd step");
        }
        assert!(p[0].abs() < 0.5, "SGD did not converge, p[0] = {}", p[0]);
    }

    #[test]
    fn test_sgd_dimension_mismatch_error() {
        let mut sgd = Sgd::new(0.1, 0.0);
        let mut p = vec![1.0, 2.0];
        let result = sgd.step(&mut p, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sgd_zero_grad_resets_velocity() {
        let mut sgd = Sgd::new(0.1, 0.9);
        let mut p = vec![1.0];
        sgd.step(&mut p, &[1.0]).expect("step");
        assert_ne!(sgd.velocity, vec![0.0]);
        sgd.zero_grad();
        assert_eq!(sgd.velocity, vec![0.0]);
    }

    // ----- Adam tests -------------------------------------------------------

    #[test]
    fn test_adam_single_step() {
        let mut adam = AdamOptimizer::new(0.01);
        let mut p = vec![1.0];
        let p_before = p[0];
        adam.step(&mut p, &[1.0]).expect("adam step");
        // Should move in negative gradient direction
        assert!(p[0] < p_before, "Adam should decrease p, got {}", p[0]);
    }

    #[test]
    fn test_adam_converges_on_quadratic() {
        let mut adam = AdamOptimizer::new(0.05);
        let mut p = vec![3.0, -3.0];
        for _ in 0..500 {
            let g = vec![2.0 * p[0], 2.0 * p[1]];
            adam.step(&mut p, &g).expect("adam step");
        }
        assert!(p[0].abs() < 0.1, "Adam p[0] did not converge: {}", p[0]);
        assert!(p[1].abs() < 0.1, "Adam p[1] did not converge: {}", p[1]);
    }

    #[test]
    fn test_adam_converges_faster_than_sgd_on_rosenbrock() {
        // Rosenbrock gradient
        let rosenbrock_grad = |x: &[f64]| -> Vec<f64> {
            let dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            let dy = 200.0 * (x[1] - x[0] * x[0]);
            vec![dx, dy]
        };
        let start = vec![-1.0, 1.0];

        // Adam 200 steps
        let mut adam = AdamOptimizer::new(0.001);
        let mut pa = start.clone();
        for _ in 0..200 {
            let g = rosenbrock_grad(&pa);
            adam.step(&mut pa, &g).expect("adam step");
        }
        let adam_dist = (pa[0] - 1.0).abs() + (pa[1] - 1.0).abs();

        // SGD 200 steps
        let mut sgd = Sgd::new(0.001, 0.0);
        let mut ps = start.clone();
        for _ in 0..200 {
            let g = rosenbrock_grad(&ps);
            sgd.step(&mut ps, &g).expect("sgd step");
        }
        let sgd_dist = (ps[0] - 1.0).abs() + (ps[1] - 1.0).abs();

        assert!(
            adam_dist < sgd_dist,
            "Adam dist {adam_dist:.4} should be < SGD dist {sgd_dist:.4}"
        );
    }

    #[test]
    fn test_adam_dimension_mismatch_error() {
        let mut adam = AdamOptimizer::new(0.01);
        let mut p = vec![1.0, 2.0];
        let result = adam.step(&mut p, &[1.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_adam_with_params() {
        let adam = AdamOptimizer::with_params(0.001, 0.95, 0.9999, 1e-7);
        assert_eq!(adam.beta1, 0.95);
        assert_eq!(adam.beta2, 0.9999);
        assert_eq!(adam.eps, 1e-7);
    }

    #[test]
    fn test_adam_zero_grad_resets_state() {
        let mut adam = AdamOptimizer::new(0.01);
        let mut p = vec![1.0];
        adam.step(&mut p, &[1.0]).expect("step");
        assert_eq!(adam.t, 1);
        adam.zero_grad();
        assert_eq!(adam.t, 0);
        assert_eq!(adam.m, vec![0.0]);
        assert_eq!(adam.v, vec![0.0]);
    }

    // ----- RMSprop tests ----------------------------------------------------

    #[test]
    fn test_rmsprop_single_step() {
        let mut rms = RmsProp::new(0.01);
        let mut p = vec![2.0];
        let p_before = p[0];
        rms.step(&mut p, &[4.0]).expect("rmsprop step");
        assert!(p[0] < p_before);
    }

    #[test]
    fn test_rmsprop_converges() {
        let mut rms = RmsProp::new(0.01);
        let mut p = vec![3.0];
        for _ in 0..500 {
            let grad = 2.0 * p[0];
            rms.step(&mut p, &[grad]).expect("step");
        }
        assert!(p[0].abs() < 1.0, "RMSprop did not converge: {}", p[0]);
    }

    #[test]
    fn test_rmsprop_dimension_mismatch_error() {
        let mut rms = RmsProp::new(0.01);
        let mut p = vec![1.0];
        let result = rms.step(&mut p, &[1.0, 2.0]);
        assert!(result.is_err());
    }

    // ----- Adagrad tests ----------------------------------------------------

    #[test]
    fn test_adagrad_single_step() {
        let mut ada = Adagrad::new(0.1);
        let mut p = vec![1.0];
        ada.step(&mut p, &[2.0]).expect("adagrad step");
        // lr_scaled = 0.1 / sqrt(4 + 1e-8) ≈ 0.05
        let expected = 1.0 - 0.1 / (4.0_f64.sqrt() + 1e-8) * 2.0;
        assert!((p[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_adagrad_accumulates_squared_grads() {
        let mut ada = Adagrad::new(0.1);
        let mut p = vec![5.0];
        // Constant gradient: effective lr decays over time
        let lr0_effective;
        {
            let mut p0 = p.clone();
            ada.step(&mut p0, &[1.0]).expect("step");
            lr0_effective = 5.0 - p0[0]; // should be ~0.1 / sqrt(1)
        }
        ada.zero_grad(); // reset for fresh comparison
        ada.step(&mut p, &[1.0]).expect("step 1");
        ada.step(&mut p, &[1.0]).expect("step 2");
        // After 2 steps sum_sq = 2, effective lr = 0.1/sqrt(2) < lr0_effective
        let lr1 = 5.0 - p[0] - lr0_effective;
        // Second step should use smaller effective lr than first
        assert!(lr1 < lr0_effective, "lr0={lr0_effective}, lr1={lr1}");
    }

    #[test]
    fn test_adagrad_dimension_mismatch_error() {
        let mut ada = Adagrad::new(0.1);
        let mut p = vec![1.0, 2.0];
        let result = ada.step(&mut p, &[1.0]);
        assert!(result.is_err());
    }

    // ----- CosineAnnealingSchedule tests ------------------------------------

    #[test]
    fn test_cosine_annealing_starts_at_max() {
        let sched = CosineAnnealingSchedule::new(0.1, 0.001, 100);
        assert!((sched.get_lr() - 0.1).abs() < TOL);
    }

    #[test]
    fn test_cosine_annealing_ends_at_min() {
        let mut sched = CosineAnnealingSchedule::new(0.1, 0.001, 100);
        for _ in 0..100 {
            sched.step();
        }
        assert!((sched.get_lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_is_monotone_decreasing() {
        let mut sched = CosineAnnealingSchedule::new(0.1, 0.001, 100);
        let mut prev = sched.get_lr();
        for _ in 0..100 {
            sched.step();
            let cur = sched.get_lr();
            assert!(
                cur <= prev + 1e-12,
                "LR increased from {prev} to {cur} at step {}",
                sched.step
            );
            prev = cur;
        }
    }

    #[test]
    fn test_cosine_annealing_midpoint() {
        let sched = CosineAnnealingSchedule::new(1.0, 0.0, 100);
        // At t=50: cos(π/2) = 0, lr = 0 + (1-0)*(1+0)/2 = 0.5
        let mut s = CosineAnnealingSchedule::new(1.0, 0.0, 100);
        for _ in 0..50 {
            s.step();
        }
        assert!((s.get_lr() - 0.5).abs() < 1e-6, "got {}", s.get_lr());
        let _ = sched; // suppress unused warning
    }

    // ----- OneCycleSchedule tests -------------------------------------------

    #[test]
    fn test_one_cycle_starts_below_max() {
        let sched = OneCycleSchedule::new(0.1, 100);
        // Initial LR = max_lr / div_factor = 0.1 / 25 = 0.004
        assert!(sched.get_lr() < sched.max_lr);
    }

    #[test]
    fn test_one_cycle_peaks_near_max_at_warmup_end() {
        let mut sched = OneCycleSchedule::new(0.1, 100);
        // Warm-up ends at step 30 (pct_start = 0.3)
        for _ in 0..30 {
            sched.step();
        }
        let peak = sched.get_lr();
        // Should be at max_lr (or very close)
        assert!(
            (peak - 0.1).abs() < 1e-9,
            "peak lr should be max_lr, got {}",
            peak
        );
    }

    #[test]
    fn test_one_cycle_decreases_after_warmup() {
        let mut sched = OneCycleSchedule::new(0.1, 100);
        // Advance past warm-up
        for _ in 0..30 {
            sched.step();
        }
        let peak = sched.get_lr();
        for _ in 0..40 {
            sched.step();
        }
        let later = sched.get_lr();
        assert!(later < peak, "lr should decrease after peak: {} vs {}", later, peak);
    }

    #[test]
    fn test_one_cycle_increases_during_warmup() {
        let mut sched = OneCycleSchedule::new(0.1, 100);
        let lr0 = sched.get_lr();
        sched.step();
        let lr1 = sched.get_lr();
        sched.step();
        let lr2 = sched.get_lr();
        assert!(lr1 > lr0, "LR should increase during warmup");
        assert!(lr2 > lr1, "LR should increase during warmup");
    }

    #[test]
    fn test_optimizer_learning_rate_accessor() {
        let sgd = Sgd::new(0.05, 0.0);
        assert_eq!(sgd.learning_rate(), 0.05);
        let adam = AdamOptimizer::new(0.003);
        assert_eq!(adam.learning_rate(), 0.003);
        let rms = RmsProp::new(0.01);
        assert_eq!(rms.learning_rate(), 0.01);
        let ada = Adagrad::new(0.1);
        assert_eq!(ada.learning_rate(), 0.1);
    }
}
