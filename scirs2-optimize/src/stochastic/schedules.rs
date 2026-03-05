//! Learning Rate Schedules
//!
//! Provides a trait and concrete implementations for learning rate scheduling
//! in stochastic optimization. These schedules control how the learning rate
//! changes over training to improve convergence and final performance.

use std::f64::consts::PI;

/// Trait for learning rate schedules.
///
/// Implementors compute the learning rate for a given epoch or step from the
/// base learning rate and any schedule-specific hyperparameters.
pub trait LrSchedule: Send + Sync {
    /// Compute the learning rate for the given epoch/step.
    ///
    /// # Arguments
    /// * `epoch` - Current training epoch or step (0-indexed)
    /// * `base_lr` - The initial/base learning rate
    ///
    /// # Returns
    /// The learning rate to use at `epoch`
    fn get_lr(&self, epoch: usize, base_lr: f64) -> f64;
}

// ─── Step Decay ──────────────────────────────────────────────────────────────

/// Step decay schedule: multiplies learning rate by `gamma` every `step_size` epochs.
///
/// `lr = base_lr * gamma^(floor(epoch / step_size))`
#[derive(Debug, Clone)]
pub struct StepDecay {
    /// Number of epochs between each decay step
    pub step_size: usize,
    /// Multiplicative decay factor (typically 0 < gamma < 1)
    pub gamma: f64,
}

impl StepDecay {
    /// Create a new step decay schedule.
    ///
    /// # Arguments
    /// * `step_size` - Epochs between decay applications
    /// * `gamma` - Multiplicative decay factor
    pub fn new(step_size: usize, gamma: f64) -> Self {
        Self { step_size, gamma }
    }
}

impl LrSchedule for StepDecay {
    fn get_lr(&self, epoch: usize, base_lr: f64) -> f64 {
        let steps = epoch / self.step_size.max(1);
        base_lr * self.gamma.powi(steps as i32)
    }
}

// ─── Cosine Annealing ────────────────────────────────────────────────────────

/// Cosine annealing schedule: smoothly decays the learning rate following a
/// cosine curve from `base_lr` down to `eta_min`.
///
/// `lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(π * epoch / t_max))`
///
/// Reference: Loshchilov & Hutter (2016), "SGDR: Stochastic Gradient Descent
/// with Warm Restarts".
#[derive(Debug, Clone)]
pub struct CosineAnnealing {
    /// Period of the cosine cycle (number of epochs for one full descent)
    pub t_max: usize,
    /// Minimum learning rate at the end of a cycle
    pub eta_min: f64,
}

impl CosineAnnealing {
    /// Create a new cosine annealing schedule.
    ///
    /// # Arguments
    /// * `t_max` - Period (epochs) for one cosine cycle
    /// * `eta_min` - Minimum learning rate
    pub fn new(t_max: usize, eta_min: f64) -> Self {
        Self { t_max, eta_min }
    }
}

impl LrSchedule for CosineAnnealing {
    fn get_lr(&self, epoch: usize, base_lr: f64) -> f64 {
        let t_max = self.t_max.max(1) as f64;
        let cos_val = (PI * epoch as f64 / t_max).cos();
        self.eta_min + 0.5 * (base_lr - self.eta_min) * (1.0 + cos_val)
    }
}

// ─── One Cycle ───────────────────────────────────────────────────────────────

/// Annealing strategy for the one-cycle policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnnealStrategy {
    /// Cosine annealing (smooth, recommended)
    Cos,
    /// Linear annealing (simple)
    Linear,
}

/// One-cycle learning rate policy.
///
/// Implements Smith & Touvron's 1cycle policy: the learning rate rises from
/// `base_lr` to `max_lr` over the first `pct_start` fraction of training,
/// then anneals back down to a minimum learning rate over the remainder.
///
/// Reference: Smith (2018), "A disciplined approach to neural network
/// hyper-parameters".
#[derive(Debug, Clone)]
pub struct OneCycle {
    /// Maximum learning rate (peak of the cycle)
    pub max_lr: f64,
    /// Fraction of total epochs for the increasing phase (0 < pct_start < 1)
    pub pct_start: f64,
    /// Annealing strategy for the decreasing phase
    pub anneal_strategy: AnnealStrategy,
    /// Total number of training epochs
    pub total_epochs: usize,
    /// Minimum (final) learning rate as a fraction of `base_lr`
    pub div_factor: f64,
    /// Final learning rate divisor (final_lr = base_lr / final_div_factor)
    pub final_div_factor: f64,
}

impl OneCycle {
    /// Create a new one-cycle schedule.
    ///
    /// # Arguments
    /// * `max_lr` - Peak learning rate
    /// * `pct_start` - Fraction of epochs for the warmup/increase phase
    /// * `anneal_strategy` - How to anneal during the decrease phase
    /// * `total_epochs` - Total training epochs
    pub fn new(
        max_lr: f64,
        pct_start: f64,
        anneal_strategy: AnnealStrategy,
        total_epochs: usize,
    ) -> Self {
        Self {
            max_lr,
            pct_start: pct_start.clamp(0.0, 1.0),
            anneal_strategy,
            total_epochs,
            div_factor: 25.0,
            final_div_factor: 1e4,
        }
    }

    /// Apply the chosen annealing strategy over the progress fraction [0,1].
    fn anneal(&self, start: f64, end: f64, pct: f64) -> f64 {
        let p = pct.clamp(0.0, 1.0);
        match self.anneal_strategy {
            AnnealStrategy::Cos => end + (start - end) / 2.0 * (1.0 + (PI * p).cos()),
            AnnealStrategy::Linear => start + (end - start) * p,
        }
    }
}

impl LrSchedule for OneCycle {
    fn get_lr(&self, epoch: usize, base_lr: f64) -> f64 {
        let total = self.total_epochs.max(1) as f64;
        let pct = epoch as f64 / total;
        let init_lr = base_lr / self.div_factor;
        let final_lr = init_lr / self.final_div_factor;

        if pct <= self.pct_start {
            // Warmup / increasing phase
            let phase_pct = if self.pct_start > 0.0 {
                pct / self.pct_start
            } else {
                1.0
            };
            self.anneal(init_lr, self.max_lr, phase_pct)
        } else {
            // Annealing phase
            let phase_pct = (pct - self.pct_start) / (1.0 - self.pct_start).max(1e-9);
            self.anneal(self.max_lr, final_lr, phase_pct)
        }
    }
}

// ─── Warmup + Cosine ─────────────────────────────────────────────────────────

/// Warmup followed by cosine decay schedule.
///
/// The learning rate increases linearly from 0 to `base_lr` over
/// `warmup_steps` steps, then decays following a cosine curve down to
/// `min_lr` over the remaining steps.
///
/// This is commonly used in Transformer training (Vaswani et al., 2017).
#[derive(Debug, Clone)]
pub struct WarmupCosine {
    /// Number of warmup epochs/steps (linear ramp from 0 → base_lr)
    pub warmup_steps: usize,
    /// Total training epochs/steps
    pub total_steps: usize,
    /// Minimum learning rate at the end of cosine decay
    pub min_lr: f64,
}

impl WarmupCosine {
    /// Create a new warmup + cosine decay schedule.
    ///
    /// # Arguments
    /// * `warmup_steps` - Epochs for linear warmup
    /// * `total_steps` - Total training epochs
    /// * `min_lr` - Minimum learning rate after full decay
    pub fn new(warmup_steps: usize, total_steps: usize, min_lr: f64) -> Self {
        Self {
            warmup_steps,
            total_steps,
            min_lr,
        }
    }
}

impl LrSchedule for WarmupCosine {
    fn get_lr(&self, epoch: usize, base_lr: f64) -> f64 {
        if epoch < self.warmup_steps {
            // Linear warmup
            let warmup = self.warmup_steps.max(1) as f64;
            base_lr * epoch as f64 / warmup
        } else {
            // Cosine decay from base_lr to min_lr
            let decay_steps = (self.total_steps.saturating_sub(self.warmup_steps)).max(1) as f64;
            let step = (epoch - self.warmup_steps) as f64;
            let cos_val = (PI * step / decay_steps).cos();
            self.min_lr + 0.5 * (base_lr - self.min_lr) * (1.0 + cos_val)
        }
    }
}

// ─── Exponential Decay ───────────────────────────────────────────────────────

/// Exponential decay schedule.
///
/// `lr = base_lr * gamma^epoch`
#[derive(Debug, Clone)]
pub struct ExponentialDecay {
    /// Decay factor per epoch (typically close to 1, e.g. 0.99)
    pub gamma: f64,
}

impl ExponentialDecay {
    /// Create a new exponential decay schedule.
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }
}

impl LrSchedule for ExponentialDecay {
    fn get_lr(&self, epoch: usize, base_lr: f64) -> f64 {
        base_lr * self.gamma.powi(epoch as i32)
    }
}

// ─── Constant Schedule ───────────────────────────────────────────────────────

/// Constant (no-op) schedule: always returns `base_lr` unchanged.
#[derive(Debug, Clone, Default)]
pub struct ConstantLr;

impl LrSchedule for ConstantLr {
    fn get_lr(&self, _epoch: usize, base_lr: f64) -> f64 {
        base_lr
    }
}

// ─── Polynomial Decay ────────────────────────────────────────────────────────

/// Polynomial decay schedule.
///
/// `lr = base_lr * (1 - epoch / total_epochs)^power`
#[derive(Debug, Clone)]
pub struct PolynomialDecay {
    /// Total epochs for decay
    pub total_epochs: usize,
    /// Power of the polynomial (1.0 = linear, 2.0 = quadratic)
    pub power: f64,
    /// Minimum learning rate floor
    pub end_lr: f64,
}

impl PolynomialDecay {
    /// Create a new polynomial decay schedule.
    pub fn new(total_epochs: usize, power: f64, end_lr: f64) -> Self {
        Self {
            total_epochs,
            power,
            end_lr,
        }
    }
}

impl LrSchedule for PolynomialDecay {
    fn get_lr(&self, epoch: usize, base_lr: f64) -> f64 {
        let total = self.total_epochs.max(1);
        if epoch >= total {
            return self.end_lr;
        }
        let decay = (1.0 - epoch as f64 / total as f64).powf(self.power);
        let lr = (base_lr - self.end_lr) * decay + self.end_lr;
        lr.max(self.end_lr)
    }
}

// ─── Cyclic LR ───────────────────────────────────────────────────────────────

/// Cyclic learning rate schedule.
///
/// Alternates between `min_lr` and `max_lr` in a triangular wave pattern.
/// Reference: Smith (2017), "Cyclical Learning Rates for Training Neural Networks".
#[derive(Debug, Clone)]
pub struct CyclicLr {
    /// Base (minimum) learning rate
    pub base_lr: f64,
    /// Maximum learning rate
    pub max_lr: f64,
    /// Half-cycle length in epochs
    pub step_size: usize,
}

impl CyclicLr {
    /// Create a new cyclic learning rate schedule.
    pub fn new(base_lr: f64, max_lr: f64, step_size: usize) -> Self {
        Self {
            base_lr,
            max_lr,
            step_size: step_size.max(1),
        }
    }
}

impl LrSchedule for CyclicLr {
    fn get_lr(&self, epoch: usize, _base_lr: f64) -> f64 {
        let cycle = epoch / (2 * self.step_size);
        let x = (epoch as f64 / self.step_size as f64) - 2.0 * cycle as f64 - 1.0;
        let scale = (1.0 - x.abs()).max(0.0);
        self.base_lr + (self.max_lr - self.base_lr) * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_step_decay() {
        let sched = StepDecay::new(10, 0.5);
        assert_abs_diff_eq!(sched.get_lr(0, 0.1), 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(sched.get_lr(9, 0.1), 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(sched.get_lr(10, 0.1), 0.05, epsilon = 1e-12);
        assert_abs_diff_eq!(sched.get_lr(20, 0.1), 0.025, epsilon = 1e-12);
    }

    #[test]
    fn test_cosine_annealing() {
        let sched = CosineAnnealing::new(100, 0.0);
        let lr_start = sched.get_lr(0, 1.0);
        let lr_mid = sched.get_lr(50, 1.0);
        let lr_end = sched.get_lr(100, 1.0);
        assert_abs_diff_eq!(lr_start, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(lr_mid, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(lr_end, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_one_cycle_warmup_peak() {
        let sched = OneCycle::new(0.1, 0.3, AnnealStrategy::Cos, 100);
        // At pct=0: init_lr = base_lr/div_factor
        let lr_start = sched.get_lr(0, 0.01);
        // At pct_start=30%: should be near max_lr
        let lr_peak = sched.get_lr(30, 0.01);
        assert!(lr_peak >= lr_start, "peak must exceed start");
        assert_abs_diff_eq!(lr_peak, sched.max_lr, epsilon = 1e-10);
    }

    #[test]
    fn test_warmup_cosine() {
        let sched = WarmupCosine::new(10, 100, 0.0);
        // During warmup: should be linear
        assert_abs_diff_eq!(sched.get_lr(0, 1.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sched.get_lr(5, 1.0), 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(sched.get_lr(10, 1.0), 1.0, epsilon = 1e-12);
        // After warmup: cosine decay
        let lr_after = sched.get_lr(55, 1.0);
        assert!(lr_after < 1.0, "should decay after warmup");
        assert!(lr_after >= 0.0, "should not go below min_lr");
    }

    #[test]
    fn test_exponential_decay() {
        let sched = ExponentialDecay::new(0.9);
        assert_abs_diff_eq!(sched.get_lr(0, 1.0), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sched.get_lr(1, 1.0), 0.9, epsilon = 1e-12);
        assert_abs_diff_eq!(sched.get_lr(2, 1.0), 0.81, epsilon = 1e-12);
    }

    #[test]
    fn test_constant_lr() {
        let sched = ConstantLr;
        for epoch in 0..100 {
            assert_abs_diff_eq!(sched.get_lr(epoch, 0.01), 0.01, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_cyclic_lr() {
        let sched = CyclicLr::new(0.001, 0.01, 5);
        // At epoch 0: base_lr
        let lr0 = sched.get_lr(0, 0.0);
        // At epoch 5: max_lr
        let lr5 = sched.get_lr(5, 0.0);
        assert_abs_diff_eq!(lr5, sched.max_lr, epsilon = 1e-10);
        // At epoch 10: back to base_lr
        let lr10 = sched.get_lr(10, 0.0);
        assert_abs_diff_eq!(lr10, sched.base_lr, epsilon = 1e-10);
        assert!(lr5 > lr0, "peak should exceed start");
    }
}
