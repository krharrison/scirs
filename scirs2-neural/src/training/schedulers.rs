//! Comprehensive learning rate schedulers for neural network training
//!
//! This module provides a suite of learning rate scheduling strategies commonly
//! used in deep learning. Each scheduler implements the [`LRScheduler`] trait,
//! allowing them to be used interchangeably in training loops.
//!
//! ## Available Schedulers
//!
//! | Scheduler | Description |
//! |-----------|-------------|
//! | [`StepLR`] | Decay LR by gamma every `step_size` epochs |
//! | [`MultiStepLR`] | Decay LR at specified milestone epochs |
//! | [`ExponentialLR`] | Exponential decay every epoch |
//! | [`CosineAnnealingLR`] | Cosine annealing to `eta_min` |
//! | [`CosineAnnealingWarmRestarts`] | Cosine annealing with warm restarts (SGDR) |
//! | [`LinearWarmup`] | Linear warmup from 0 to `base_lr` |
//! | [`WarmupCosine`] | Linear warmup followed by cosine decay |
//! | [`OneCycleLR`] | Super-convergence one-cycle policy |
//! | [`PolynomialLR`] | Polynomial decay schedule |
//! | [`ReduceOnPlateau`] | Reduce LR when a metric plateaus |
//! | [`CyclicLR`] | Cyclic LR between base and max |
//!
//! ## Example
//!
//! ```rust
//! use scirs2_neural::training::schedulers::{LRScheduler, CosineAnnealingLR};
//!
//! let scheduler = CosineAnnealingLR::new(0.01, 100, 1e-6);
//! for epoch in 0..100 {
//!     let lr = scheduler.get_lr(epoch);
//!     // use lr for training...
//! }
//! assert!(scheduler.get_lr(0) > scheduler.get_lr(50));
//! ```

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// LRScheduler trait
// ---------------------------------------------------------------------------

/// Trait for learning rate schedulers.
///
/// Implementors provide a deterministic mapping from epoch/step to learning rate.
/// Stateless schedulers only need to implement [`get_lr`](LRScheduler::get_lr)
/// and [`name`](LRScheduler::name); the default [`get_lr_at_step`](LRScheduler::get_lr_at_step)
/// delegates to `get_lr`.
pub trait LRScheduler {
    /// Return the learning rate for the given epoch (0-indexed).
    fn get_lr(&self, epoch: usize) -> f64;

    /// Return the learning rate for a specific step within an epoch.
    ///
    /// The default implementation ignores `step` and returns the epoch-level LR.
    fn get_lr_at_step(&self, _step: usize, epoch: usize) -> f64 {
        self.get_lr(epoch)
    }

    /// Human-readable name of this scheduler.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// StepLR
// ---------------------------------------------------------------------------

/// Decays the learning rate by `gamma` every `step_size` epochs.
///
/// `lr = base_lr * gamma^(epoch / step_size)`
#[derive(Debug, Clone)]
pub struct StepLR {
    base_lr: f64,
    step_size: usize,
    gamma: f64,
}

impl StepLR {
    /// Create a new `StepLR` scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `step_size` - Period of learning rate decay (in epochs)
    /// * `gamma` - Multiplicative factor of learning rate decay
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            base_lr,
            step_size: step_size.max(1),
            gamma,
        }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        let num_decays = epoch / self.step_size;
        self.base_lr * self.gamma.powi(num_decays as i32)
    }

    fn name(&self) -> &str {
        "StepLR"
    }
}

// ---------------------------------------------------------------------------
// MultiStepLR
// ---------------------------------------------------------------------------

/// Decays the learning rate by `gamma` at each milestone epoch.
///
/// At each milestone boundary the current LR is multiplied by `gamma`.
#[derive(Debug, Clone)]
pub struct MultiStepLR {
    base_lr: f64,
    milestones: Vec<usize>,
    gamma: f64,
}

impl MultiStepLR {
    /// Create a new `MultiStepLR` scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `milestones` - List of epoch indices at which to decay; will be sorted internally
    /// * `gamma` - Multiplicative factor applied at each milestone
    pub fn new(base_lr: f64, milestones: Vec<usize>, gamma: f64) -> Self {
        let mut sorted = milestones;
        sorted.sort_unstable();
        Self {
            base_lr,
            milestones: sorted,
            gamma,
        }
    }
}

impl LRScheduler for MultiStepLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        let num_passed = self.milestones.iter().filter(|&&m| epoch >= m).count();
        self.base_lr * self.gamma.powi(num_passed as i32)
    }

    fn name(&self) -> &str {
        "MultiStepLR"
    }
}

// ---------------------------------------------------------------------------
// ExponentialLR
// ---------------------------------------------------------------------------

/// Decays the learning rate exponentially every epoch.
///
/// `lr = base_lr * gamma^epoch`
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    base_lr: f64,
    gamma: f64,
}

impl ExponentialLR {
    /// Create a new `ExponentialLR` scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `gamma` - Multiplicative factor per epoch (typically close to but less than 1.0)
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        Self { base_lr, gamma }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        self.base_lr * self.gamma.powi(epoch as i32)
    }

    fn name(&self) -> &str {
        "ExponentialLR"
    }
}

// ---------------------------------------------------------------------------
// CosineAnnealingLR
// ---------------------------------------------------------------------------

/// Cosine annealing schedule (Loshchilov & Hutter, 2016).
///
/// ```text
/// lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * epoch / t_max))
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    base_lr: f64,
    t_max: usize,
    eta_min: f64,
}

impl CosineAnnealingLR {
    /// Create a new cosine annealing scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Maximum (initial) learning rate
    /// * `t_max` - Maximum number of epochs (half-period of the cosine)
    /// * `eta_min` - Minimum learning rate
    pub fn new(base_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self {
            base_lr,
            t_max: t_max.max(1),
            eta_min,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        let t = epoch.min(self.t_max) as f64;
        let t_max = self.t_max as f64;
        self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (PI * t / t_max).cos())
    }

    fn name(&self) -> &str {
        "CosineAnnealingLR"
    }
}

// ---------------------------------------------------------------------------
// CosineAnnealingWarmRestarts
// ---------------------------------------------------------------------------

/// SGDR: Cosine annealing with warm restarts (Loshchilov & Hutter, 2016).
///
/// After each restart the period is multiplied by `t_mult`, enabling
/// increasingly longer cosine cycles.
#[derive(Debug, Clone)]
pub struct CosineAnnealingWarmRestarts {
    base_lr: f64,
    t_0: usize,
    t_mult: usize,
    eta_min: f64,
}

impl CosineAnnealingWarmRestarts {
    /// Create a new SGDR scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Maximum (initial) learning rate
    /// * `t_0` - Number of epochs for the first restart
    /// * `t_mult` - Factor to increase `t_0` after each restart (>=1)
    /// * `eta_min` - Minimum learning rate
    pub fn new(base_lr: f64, t_0: usize, t_mult: usize, eta_min: f64) -> Self {
        Self {
            base_lr,
            t_0: t_0.max(1),
            t_mult: t_mult.max(1),
            eta_min,
        }
    }
}

impl LRScheduler for CosineAnnealingWarmRestarts {
    fn get_lr(&self, epoch: usize) -> f64 {
        let (t_cur, t_i) = if self.t_mult == 1 {
            // Constant period: simple modulo
            let t_cur = epoch % self.t_0;
            (t_cur as f64, self.t_0 as f64)
        } else {
            // Geometric series: T_0 + T_0*m + T_0*m^2 + ...
            // Find which restart cycle we are in
            let mut t_i = self.t_0;
            let mut cumulative = 0usize;
            loop {
                if cumulative + t_i > epoch {
                    break;
                }
                cumulative += t_i;
                // Saturating multiply to avoid overflow
                t_i = t_i.saturating_mul(self.t_mult);
                // Safety: if t_i somehow becomes 0, break
                if t_i == 0 {
                    t_i = 1;
                    break;
                }
            }
            let t_cur = epoch - cumulative;
            (t_cur as f64, t_i as f64)
        };
        self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (PI * t_cur / t_i).cos())
    }

    fn name(&self) -> &str {
        "CosineAnnealingWarmRestarts"
    }
}

// ---------------------------------------------------------------------------
// LinearWarmup
// ---------------------------------------------------------------------------

/// Linear warmup from 0 to `base_lr` over `warmup_steps` steps.
///
/// After warmup completes, the LR stays at `base_lr`.
#[derive(Debug, Clone)]
pub struct LinearWarmup {
    base_lr: f64,
    warmup_steps: usize,
}

impl LinearWarmup {
    /// Create a new linear warmup scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Target learning rate after warmup
    /// * `warmup_steps` - Number of steps/epochs for the warmup phase
    pub fn new(base_lr: f64, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps: warmup_steps.max(1),
        }
    }
}

impl LRScheduler for LinearWarmup {
    fn get_lr(&self, epoch: usize) -> f64 {
        if epoch >= self.warmup_steps {
            self.base_lr
        } else {
            self.base_lr * (epoch as f64 / self.warmup_steps as f64)
        }
    }

    fn get_lr_at_step(&self, step: usize, _epoch: usize) -> f64 {
        if step >= self.warmup_steps {
            self.base_lr
        } else {
            self.base_lr * (step as f64 / self.warmup_steps as f64)
        }
    }

    fn name(&self) -> &str {
        "LinearWarmup"
    }
}

// ---------------------------------------------------------------------------
// WarmupCosine
// ---------------------------------------------------------------------------

/// Linear warmup followed by cosine decay.
///
/// Phase 1 (step < warmup_steps): linear warmup from 0 to `base_lr`
/// Phase 2 (step >= warmup_steps): cosine annealing from `base_lr` to `eta_min`
#[derive(Debug, Clone)]
pub struct WarmupCosine {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    eta_min: f64,
}

impl WarmupCosine {
    /// Create a new warmup + cosine scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Peak learning rate (reached at end of warmup)
    /// * `warmup_steps` - Number of warmup steps
    /// * `total_steps` - Total number of training steps (including warmup)
    /// * `eta_min` - Minimum learning rate after cosine decay
    pub fn new(base_lr: f64, warmup_steps: usize, total_steps: usize, eta_min: f64) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps: total_steps.max(warmup_steps + 1),
            eta_min,
        }
    }

    /// Compute LR for a given global step.
    fn lr_at(&self, step: usize) -> f64 {
        if step < self.warmup_steps {
            // Linear warmup
            if self.warmup_steps == 0 {
                return self.base_lr;
            }
            self.base_lr * (step as f64 / self.warmup_steps as f64)
        } else {
            // Cosine decay
            let decay_steps = self.total_steps - self.warmup_steps;
            if decay_steps == 0 {
                return self.eta_min;
            }
            let progress = (step - self.warmup_steps) as f64 / decay_steps as f64;
            let progress = progress.min(1.0);
            self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (PI * progress).cos())
        }
    }
}

impl LRScheduler for WarmupCosine {
    fn get_lr(&self, epoch: usize) -> f64 {
        self.lr_at(epoch)
    }

    fn get_lr_at_step(&self, step: usize, _epoch: usize) -> f64 {
        self.lr_at(step)
    }

    fn name(&self) -> &str {
        "WarmupCosine"
    }
}

// ---------------------------------------------------------------------------
// OneCycleLR
// ---------------------------------------------------------------------------

/// Super-convergence one-cycle learning rate policy (Smith & Topin, 2019).
///
/// The schedule has three phases:
/// 1. Warmup: LR ramps from `max_lr / div_factor` to `max_lr` (cosine annealing)
/// 2. Annealing: LR decays from `max_lr` back to `max_lr / div_factor` (cosine)
/// 3. Final decay: LR decays from `max_lr / div_factor` to `max_lr / (div_factor * final_div_factor)` (cosine)
#[derive(Debug, Clone)]
pub struct OneCycleLR {
    max_lr: f64,
    total_steps: usize,
    pct_start: f64,
    div_factor: f64,
    final_div_factor: f64,
}

impl OneCycleLR {
    /// Create a new one-cycle scheduler.
    ///
    /// # Arguments
    /// * `max_lr` - Peak learning rate
    /// * `total_steps` - Total number of training steps
    /// * `pct_start` - Fraction of total steps spent in the warmup phase (0.0 to 1.0)
    /// * `div_factor` - Determines initial LR: `initial_lr = max_lr / div_factor`
    /// * `final_div_factor` - Determines final LR: `final_lr = initial_lr / final_div_factor`
    pub fn new(
        max_lr: f64,
        total_steps: usize,
        pct_start: f64,
        div_factor: f64,
        final_div_factor: f64,
    ) -> Self {
        Self {
            max_lr,
            total_steps: total_steps.max(1),
            pct_start: pct_start.clamp(0.0, 1.0),
            div_factor: if div_factor <= 0.0 { 25.0 } else { div_factor },
            final_div_factor: if final_div_factor <= 0.0 {
                1e4
            } else {
                final_div_factor
            },
        }
    }

    fn lr_at(&self, step: usize) -> f64 {
        let step = step.min(self.total_steps);
        let pct = step as f64 / self.total_steps as f64;
        let initial_lr = self.max_lr / self.div_factor;
        let final_lr = initial_lr / self.final_div_factor;

        if pct <= self.pct_start {
            // Phase 1: warmup from initial_lr to max_lr (cosine interpolation)
            let phase_pct = if self.pct_start > 0.0 {
                pct / self.pct_start
            } else {
                1.0
            };
            // Cosine annealing upward
            initial_lr + (self.max_lr - initial_lr) * 0.5 * (1.0 - (PI * phase_pct).cos())
        } else {
            // Phase 2+3: decay from max_lr to final_lr (cosine)
            let remaining_pct = 1.0 - self.pct_start;
            let phase_pct = if remaining_pct > 0.0 {
                (pct - self.pct_start) / remaining_pct
            } else {
                1.0
            };
            // Cosine decay from max_lr to final_lr
            final_lr + (self.max_lr - final_lr) * 0.5 * (1.0 + (PI * phase_pct).cos())
        }
    }
}

impl LRScheduler for OneCycleLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        self.lr_at(epoch)
    }

    fn get_lr_at_step(&self, step: usize, _epoch: usize) -> f64 {
        self.lr_at(step)
    }

    fn name(&self) -> &str {
        "OneCycleLR"
    }
}

// ---------------------------------------------------------------------------
// PolynomialLR
// ---------------------------------------------------------------------------

/// Polynomial decay learning rate scheduler.
///
/// ```text
/// lr = (base_lr - end_lr) * (1 - epoch / total_epochs)^power + end_lr
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialLR {
    base_lr: f64,
    total_epochs: usize,
    power: f64,
    end_lr: f64,
}

impl PolynomialLR {
    /// Create a new polynomial decay scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `total_epochs` - Total number of epochs
    /// * `power` - Power of the polynomial (1.0 = linear decay)
    /// * `end_lr` - Final learning rate
    pub fn new(base_lr: f64, total_epochs: usize, power: f64, end_lr: f64) -> Self {
        Self {
            base_lr,
            total_epochs: total_epochs.max(1),
            power,
            end_lr,
        }
    }
}

impl LRScheduler for PolynomialLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        if epoch >= self.total_epochs {
            return self.end_lr;
        }
        let fraction = 1.0 - (epoch as f64 / self.total_epochs as f64);
        (self.base_lr - self.end_lr) * fraction.powf(self.power) + self.end_lr
    }

    fn name(&self) -> &str {
        "PolynomialLR"
    }
}

// ---------------------------------------------------------------------------
// ReduceOnPlateau
// ---------------------------------------------------------------------------

/// Reduce learning rate when a monitored metric has stopped improving.
///
/// Unlike the other schedulers, this one is **stateful** and requires
/// calling [`step`](ReduceOnPlateau::step) with a metric value after each
/// evaluation epoch.
#[derive(Debug, Clone)]
pub struct ReduceOnPlateau {
    base_lr: f64,
    factor: f64,
    patience: usize,
    min_lr: f64,
    threshold: f64,
    // mutable state
    current_lr: f64,
    best_metric: f64,
    num_bad_epochs: usize,
    initialized: bool,
}

impl ReduceOnPlateau {
    /// Create a new ReduceOnPlateau scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `factor` - Factor by which the LR is reduced (new_lr = lr * factor)
    /// * `patience` - Number of epochs with no improvement after which LR is reduced
    /// * `min_lr` - Lower bound on the learning rate
    /// * `threshold` - Minimum improvement to qualify as "better"
    pub fn new(base_lr: f64, factor: f64, patience: usize, min_lr: f64, threshold: f64) -> Self {
        Self {
            base_lr,
            factor: factor.clamp(0.0, 1.0),
            patience,
            min_lr,
            threshold: threshold.abs(),
            current_lr: base_lr,
            best_metric: f64::INFINITY,
            num_bad_epochs: 0,
            initialized: false,
        }
    }

    /// Report a metric value and potentially reduce the learning rate.
    ///
    /// Returns the current learning rate after the update.
    pub fn step(&mut self, metric: f64) -> f64 {
        if !self.initialized {
            self.best_metric = metric;
            self.initialized = true;
            return self.current_lr;
        }

        if metric < self.best_metric - self.threshold {
            // Improvement
            self.best_metric = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        if self.num_bad_epochs > self.patience {
            self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
            self.num_bad_epochs = 0;
        }

        self.current_lr
    }

    /// Get the current learning rate without stepping.
    pub fn get_current_lr(&self) -> f64 {
        self.current_lr
    }

    /// Reset the scheduler state back to the initial configuration.
    pub fn reset(&mut self) {
        self.current_lr = self.base_lr;
        self.best_metric = f64::INFINITY;
        self.num_bad_epochs = 0;
        self.initialized = false;
    }
}

impl LRScheduler for ReduceOnPlateau {
    fn get_lr(&self, _epoch: usize) -> f64 {
        // ReduceOnPlateau is metric-driven; get_lr returns current state
        self.current_lr
    }

    fn name(&self) -> &str {
        "ReduceOnPlateau"
    }
}

// ---------------------------------------------------------------------------
// CyclicLR
// ---------------------------------------------------------------------------

/// Mode for cyclic learning rate policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CyclicMode {
    /// Basic triangular cycle: LR oscillates linearly between base and max.
    Triangular,
    /// Same as triangular but the amplitude is halved each cycle.
    Triangular2,
    /// Amplitude is scaled by `gamma^iteration` each cycle.
    ExpRange(f64),
}

/// Cyclic learning rate policy (Smith, 2017).
///
/// Cycles the learning rate between `base_lr` and `max_lr` according
/// to the chosen [`CyclicMode`].
#[derive(Debug, Clone)]
pub struct CyclicLR {
    base_lr: f64,
    max_lr: f64,
    step_size_up: usize,
    step_size_down: usize,
    mode: CyclicMode,
}

impl CyclicLR {
    /// Create a new cyclic LR scheduler.
    ///
    /// # Arguments
    /// * `base_lr` - Lower LR boundary in the cycle
    /// * `max_lr` - Upper LR boundary in the cycle
    /// * `step_size_up` - Number of steps in the increasing half of a cycle
    /// * `mode` - One of [`CyclicMode::Triangular`], [`CyclicMode::Triangular2`],
    ///   or [`CyclicMode::ExpRange`]
    pub fn new(base_lr: f64, max_lr: f64, step_size_up: usize, mode: CyclicMode) -> Self {
        Self {
            base_lr,
            max_lr,
            step_size_up: step_size_up.max(1),
            step_size_down: step_size_up.max(1), // symmetric by default
            mode,
        }
    }

    /// Create a cyclic LR scheduler with asymmetric up/down steps.
    ///
    /// # Arguments
    /// * `base_lr` - Lower LR boundary
    /// * `max_lr` - Upper LR boundary
    /// * `step_size_up` - Number of steps in the increasing half
    /// * `step_size_down` - Number of steps in the decreasing half
    /// * `mode` - Cycling mode
    pub fn with_step_sizes(
        base_lr: f64,
        max_lr: f64,
        step_size_up: usize,
        step_size_down: usize,
        mode: CyclicMode,
    ) -> Self {
        Self {
            base_lr,
            max_lr,
            step_size_up: step_size_up.max(1),
            step_size_down: step_size_down.max(1),
            mode,
        }
    }

    fn lr_at(&self, step: usize) -> f64 {
        let cycle_len = self.step_size_up + self.step_size_down;
        if cycle_len == 0 {
            return self.base_lr;
        }
        let cycle = step / cycle_len;
        let x = step % cycle_len;

        // Position within the cycle as a 0..1 fraction for the triangular wave
        let triangle = if x < self.step_size_up {
            // Ascending
            x as f64 / self.step_size_up as f64
        } else {
            // Descending
            1.0 - ((x - self.step_size_up) as f64 / self.step_size_down as f64)
        };

        let scale = match self.mode {
            CyclicMode::Triangular => 1.0,
            CyclicMode::Triangular2 => 1.0 / (1u64 << cycle.min(63)) as f64,
            CyclicMode::ExpRange(gamma) => gamma.powi(step as i32),
        };

        self.base_lr + (self.max_lr - self.base_lr) * triangle * scale
    }
}

impl LRScheduler for CyclicLR {
    fn get_lr(&self, epoch: usize) -> f64 {
        self.lr_at(epoch)
    }

    fn get_lr_at_step(&self, step: usize, _epoch: usize) -> f64 {
        self.lr_at(step)
    }

    fn name(&self) -> &str {
        "CyclicLR"
    }
}

// ---------------------------------------------------------------------------
// Composable scheduler combinator
// ---------------------------------------------------------------------------

/// A scheduler that chains two schedulers, switching at a given epoch.
///
/// Before `switch_epoch`: uses `first`. From `switch_epoch` onward: uses `second`.
#[derive(Debug)]
pub struct ChainedScheduler<A: LRScheduler, B: LRScheduler> {
    first: A,
    second: B,
    switch_epoch: usize,
}

impl<A: LRScheduler, B: LRScheduler> ChainedScheduler<A, B> {
    /// Create a chained scheduler that switches from `first` to `second` at `switch_epoch`.
    pub fn new(first: A, second: B, switch_epoch: usize) -> Self {
        Self {
            first,
            second,
            switch_epoch,
        }
    }
}

impl<A: LRScheduler, B: LRScheduler> LRScheduler for ChainedScheduler<A, B> {
    fn get_lr(&self, epoch: usize) -> f64 {
        if epoch < self.switch_epoch {
            self.first.get_lr(epoch)
        } else {
            self.second.get_lr(epoch - self.switch_epoch)
        }
    }

    fn get_lr_at_step(&self, step: usize, epoch: usize) -> f64 {
        if epoch < self.switch_epoch {
            self.first.get_lr_at_step(step, epoch)
        } else {
            self.second.get_lr_at_step(step, epoch - self.switch_epoch)
        }
    }

    fn name(&self) -> &str {
        "ChainedScheduler"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---- StepLR ----

    #[test]
    fn test_step_lr_initial() {
        let s = StepLR::new(0.1, 10, 0.5);
        assert!(approx_eq(s.get_lr(0), 0.1, EPS));
    }

    #[test]
    fn test_step_lr_decay() {
        let s = StepLR::new(0.1, 10, 0.5);
        assert!(approx_eq(s.get_lr(10), 0.05, EPS));
        assert!(approx_eq(s.get_lr(20), 0.025, EPS));
        assert!(approx_eq(s.get_lr(15), 0.05, EPS)); // between steps
    }

    #[test]
    fn test_step_lr_name() {
        let s = StepLR::new(0.1, 10, 0.5);
        assert_eq!(s.name(), "StepLR");
    }

    // ---- MultiStepLR ----

    #[test]
    fn test_multi_step_lr() {
        let s = MultiStepLR::new(0.1, vec![30, 80], 0.1);
        assert!(approx_eq(s.get_lr(0), 0.1, EPS));
        assert!(approx_eq(s.get_lr(29), 0.1, EPS));
        assert!(approx_eq(s.get_lr(30), 0.01, EPS));
        assert!(approx_eq(s.get_lr(79), 0.01, EPS));
        assert!(approx_eq(s.get_lr(80), 0.001, EPS));
    }

    #[test]
    fn test_multi_step_lr_unsorted() {
        // Milestones given out of order should still work
        let s = MultiStepLR::new(0.1, vec![80, 30], 0.1);
        assert!(approx_eq(s.get_lr(30), 0.01, EPS));
        assert!(approx_eq(s.get_lr(80), 0.001, EPS));
    }

    // ---- ExponentialLR ----

    #[test]
    fn test_exponential_lr() {
        let s = ExponentialLR::new(0.1, 0.9);
        assert!(approx_eq(s.get_lr(0), 0.1, EPS));
        assert!(approx_eq(s.get_lr(1), 0.09, EPS));
        assert!(approx_eq(s.get_lr(10), 0.1 * 0.9f64.powi(10), EPS));
    }

    #[test]
    fn test_exponential_lr_name() {
        let s = ExponentialLR::new(0.1, 0.9);
        assert_eq!(s.name(), "ExponentialLR");
    }

    // ---- CosineAnnealingLR ----

    #[test]
    fn test_cosine_annealing_boundaries() {
        let s = CosineAnnealingLR::new(0.1, 100, 0.0);
        // At epoch 0, LR should be base_lr
        assert!(approx_eq(s.get_lr(0), 0.1, EPS));
        // At epoch t_max, LR should be eta_min
        assert!(approx_eq(s.get_lr(100), 0.0, EPS));
    }

    #[test]
    fn test_cosine_annealing_midpoint() {
        let s = CosineAnnealingLR::new(0.1, 100, 0.0);
        // At the midpoint, cos(pi * 0.5) = 0, so lr = 0.5 * base_lr
        assert!(approx_eq(s.get_lr(50), 0.05, EPS));
    }

    #[test]
    fn test_cosine_annealing_with_eta_min() {
        let s = CosineAnnealingLR::new(0.1, 100, 0.001);
        assert!(approx_eq(s.get_lr(0), 0.1, EPS));
        assert!(approx_eq(s.get_lr(100), 0.001, EPS));
    }

    // ---- CosineAnnealingWarmRestarts ----

    #[test]
    fn test_warm_restarts_first_cycle() {
        let s = CosineAnnealingWarmRestarts::new(0.1, 10, 1, 0.0);
        assert!(approx_eq(s.get_lr(0), 0.1, EPS));
        assert!(approx_eq(s.get_lr(10), 0.1, EPS)); // restart
    }

    #[test]
    fn test_warm_restarts_multiplier() {
        let s = CosineAnnealingWarmRestarts::new(0.1, 10, 2, 0.0);
        // First cycle: epochs 0..10 (period 10)
        assert!(approx_eq(s.get_lr(0), 0.1, EPS));
        // Second cycle starts at epoch 10, has period 20
        assert!(approx_eq(s.get_lr(10), 0.1, EPS)); // restart
                                                    // Third cycle starts at epoch 30, has period 40
        assert!(approx_eq(s.get_lr(30), 0.1, EPS)); // restart
    }

    #[test]
    fn test_warm_restarts_decay_within_cycle() {
        let s = CosineAnnealingWarmRestarts::new(0.1, 10, 1, 0.0);
        // At the midpoint of first cycle (epoch 5), should be ~0.05
        assert!(approx_eq(s.get_lr(5), 0.05, 1e-6));
    }

    // ---- LinearWarmup ----

    #[test]
    fn test_linear_warmup() {
        let s = LinearWarmup::new(0.1, 10);
        assert!(approx_eq(s.get_lr(0), 0.0, EPS));
        assert!(approx_eq(s.get_lr(5), 0.05, EPS));
        assert!(approx_eq(s.get_lr(10), 0.1, EPS));
        assert!(approx_eq(s.get_lr(100), 0.1, EPS)); // stays flat
    }

    #[test]
    fn test_linear_warmup_step_level() {
        let s = LinearWarmup::new(0.1, 100);
        assert!(approx_eq(s.get_lr_at_step(0, 0), 0.0, EPS));
        assert!(approx_eq(s.get_lr_at_step(50, 0), 0.05, EPS));
        assert!(approx_eq(s.get_lr_at_step(100, 0), 0.1, EPS));
    }

    // ---- WarmupCosine ----

    #[test]
    fn test_warmup_cosine_phases() {
        let s = WarmupCosine::new(0.1, 10, 110, 0.0);
        // Warmup phase
        assert!(approx_eq(s.get_lr(0), 0.0, EPS));
        assert!(approx_eq(s.get_lr(5), 0.05, EPS));
        assert!(approx_eq(s.get_lr(10), 0.1, EPS));
        // Cosine decay phase
        assert!(s.get_lr(60) < 0.1);
        // End
        assert!(approx_eq(s.get_lr(110), 0.0, EPS));
    }

    #[test]
    fn test_warmup_cosine_with_eta_min() {
        let s = WarmupCosine::new(0.1, 10, 110, 0.001);
        assert!(approx_eq(s.get_lr(110), 0.001, EPS));
    }

    // ---- OneCycleLR ----

    #[test]
    fn test_one_cycle_boundaries() {
        let s = OneCycleLR::new(0.1, 100, 0.3, 25.0, 1e4);
        let initial_lr = 0.1 / 25.0; // 0.004
        let final_lr = initial_lr / 1e4; // 0.0000004

        // Step 0: should be close to initial_lr
        assert!(approx_eq(s.get_lr(0), initial_lr, 1e-8));
        // Step total: should be close to final_lr
        assert!(approx_eq(s.get_lr(100), final_lr, 1e-8));
    }

    #[test]
    fn test_one_cycle_peak() {
        let s = OneCycleLR::new(0.1, 1000, 0.3, 25.0, 1e4);
        // At 30% of total steps (step 300), should reach near max_lr
        let lr_at_peak = s.get_lr(300);
        assert!(approx_eq(lr_at_peak, 0.1, 1e-8));
    }

    #[test]
    fn test_one_cycle_increases_then_decreases() {
        let s = OneCycleLR::new(0.1, 100, 0.3, 25.0, 1e4);
        // LR should increase during warmup
        assert!(s.get_lr(15) > s.get_lr(0));
        // LR should decrease after peak
        assert!(s.get_lr(50) > s.get_lr(90));
    }

    // ---- PolynomialLR ----

    #[test]
    fn test_polynomial_lr_linear() {
        let s = PolynomialLR::new(0.1, 100, 1.0, 0.0);
        assert!(approx_eq(s.get_lr(0), 0.1, EPS));
        assert!(approx_eq(s.get_lr(50), 0.05, EPS));
        assert!(approx_eq(s.get_lr(100), 0.0, EPS));
    }

    #[test]
    fn test_polynomial_lr_quadratic() {
        let s = PolynomialLR::new(0.1, 100, 2.0, 0.0);
        // At epoch 50: (1 - 0.5)^2 = 0.25, lr = 0.025
        assert!(approx_eq(s.get_lr(50), 0.025, EPS));
    }

    #[test]
    fn test_polynomial_lr_with_end_lr() {
        let s = PolynomialLR::new(0.1, 100, 1.0, 0.01);
        assert!(approx_eq(s.get_lr(0), 0.1, EPS));
        assert!(approx_eq(s.get_lr(100), 0.01, EPS));
    }

    #[test]
    fn test_polynomial_lr_beyond_total() {
        let s = PolynomialLR::new(0.1, 100, 1.0, 0.01);
        assert!(approx_eq(s.get_lr(200), 0.01, EPS));
    }

    // ---- ReduceOnPlateau ----

    #[test]
    fn test_reduce_on_plateau_no_reduction() {
        let mut s = ReduceOnPlateau::new(0.1, 0.5, 3, 1e-6, 1e-4);
        // Continuously improving metric
        assert!(approx_eq(s.step(1.0), 0.1, EPS));
        assert!(approx_eq(s.step(0.9), 0.1, EPS));
        assert!(approx_eq(s.step(0.8), 0.1, EPS));
        assert!(approx_eq(s.step(0.7), 0.1, EPS));
    }

    #[test]
    fn test_reduce_on_plateau_reduction() {
        let mut s = ReduceOnPlateau::new(0.1, 0.5, 2, 1e-6, 1e-4);
        s.step(1.0); // init
        s.step(1.0); // no improvement, bad=1
        s.step(1.0); // no improvement, bad=2
        let lr = s.step(1.0); // bad=3 > patience=2 => reduce
        assert!(approx_eq(lr, 0.05, EPS));
    }

    #[test]
    fn test_reduce_on_plateau_min_lr() {
        let mut s = ReduceOnPlateau::new(0.001, 0.1, 0, 0.0001, 0.0);
        s.step(1.0); // init
                     // After patience=0, any bad epoch triggers reduction
        s.step(1.0); // bad=1 > 0 => reduce
                     // Should clamp at min_lr eventually
        for _ in 0..100 {
            s.step(1.0);
        }
        assert!(s.get_current_lr() >= 0.0001);
    }

    #[test]
    fn test_reduce_on_plateau_reset() {
        let mut s = ReduceOnPlateau::new(0.1, 0.5, 2, 1e-6, 1e-4);
        s.step(1.0);
        s.step(1.0);
        s.step(1.0);
        s.step(1.0); // reduced
        s.reset();
        assert!(approx_eq(s.get_current_lr(), 0.1, EPS));
    }

    #[test]
    fn test_reduce_on_plateau_get_lr_trait() {
        let s = ReduceOnPlateau::new(0.1, 0.5, 2, 1e-6, 1e-4);
        // Trait method should return current lr
        assert!(approx_eq(s.get_lr(999), 0.1, EPS));
    }

    // ---- CyclicLR ----

    #[test]
    fn test_cyclic_lr_triangular() {
        let s = CyclicLR::new(0.001, 0.01, 10, CyclicMode::Triangular);
        // At step 0: base_lr
        assert!(approx_eq(s.get_lr(0), 0.001, EPS));
        // At step 10 (peak of first cycle): max_lr
        assert!(approx_eq(s.get_lr(10), 0.01, EPS));
        // At step 20 (back to base): base_lr
        assert!(approx_eq(s.get_lr(20), 0.001, EPS));
    }

    #[test]
    fn test_cyclic_lr_triangular_midpoint() {
        let s = CyclicLR::new(0.001, 0.01, 10, CyclicMode::Triangular);
        // At step 5: halfway up
        let expected = 0.001 + (0.01 - 0.001) * 0.5;
        assert!(approx_eq(s.get_lr(5), expected, EPS));
    }

    #[test]
    fn test_cyclic_lr_triangular2() {
        let s = CyclicLR::new(0.001, 0.01, 10, CyclicMode::Triangular2);
        // First cycle: full amplitude
        assert!(approx_eq(s.get_lr(10), 0.01, EPS));
        // Second cycle (cycle=1): amplitude halved
        let expected = 0.001 + (0.01 - 0.001) * 1.0 * 0.5; // 0.0055
        assert!(approx_eq(s.get_lr(30), expected, EPS));
    }

    #[test]
    fn test_cyclic_lr_exp_range() {
        let s = CyclicLR::new(0.001, 0.01, 10, CyclicMode::ExpRange(0.99));
        // Should decay the amplitude exponentially
        let lr0 = s.get_lr(10); // peak at step 10
        let lr1 = s.get_lr(30); // peak at step 30
        assert!(lr1 < lr0); // amplitude should decrease
    }

    #[test]
    fn test_cyclic_lr_asymmetric() {
        let s = CyclicLR::with_step_sizes(0.001, 0.01, 10, 20, CyclicMode::Triangular);
        // Full cycle is 30 steps
        assert!(approx_eq(s.get_lr(0), 0.001, EPS));
        assert!(approx_eq(s.get_lr(10), 0.01, EPS)); // peak
        assert!(approx_eq(s.get_lr(30), 0.001, EPS)); // back to base
    }

    // ---- ChainedScheduler ----

    #[test]
    fn test_chained_scheduler() {
        let warmup = LinearWarmup::new(0.1, 10);
        let cosine = CosineAnnealingLR::new(0.1, 90, 0.0);
        let chained = ChainedScheduler::new(warmup, cosine, 10);

        // Warmup phase
        assert!(approx_eq(chained.get_lr(0), 0.0, EPS));
        assert!(approx_eq(chained.get_lr(5), 0.05, EPS));
        // Transition
        assert!(approx_eq(chained.get_lr(10), 0.1, EPS));
        // Cosine phase (epoch 10 maps to cosine epoch 0)
        assert!(chained.get_lr(55) < 0.1); // decaying
    }

    #[test]
    fn test_chained_scheduler_name() {
        let a = StepLR::new(0.1, 10, 0.5);
        let b = ExponentialLR::new(0.1, 0.9);
        let c = ChainedScheduler::new(a, b, 50);
        assert_eq!(c.name(), "ChainedScheduler");
    }

    // ---- Edge cases ----

    #[test]
    fn test_step_lr_zero_step_size_clamped() {
        // step_size is clamped to 1 to avoid division by zero
        let s = StepLR::new(0.1, 0, 0.5);
        // epoch 0 / 1 = 0 decays, so lr = 0.1
        assert!(approx_eq(s.get_lr(0), 0.1, EPS));
        // epoch 1 / 1 = 1 decay
        assert!(approx_eq(s.get_lr(1), 0.05, EPS));
    }

    #[test]
    fn test_cosine_annealing_beyond_t_max() {
        let s = CosineAnnealingLR::new(0.1, 100, 0.0);
        // Beyond t_max, epoch is clamped
        assert!(approx_eq(s.get_lr(200), 0.0, EPS));
    }

    #[test]
    fn test_all_schedulers_non_negative() {
        let schedulers: Vec<Box<dyn LRScheduler>> = vec![
            Box::new(StepLR::new(0.1, 10, 0.5)),
            Box::new(MultiStepLR::new(0.1, vec![30, 60, 90], 0.1)),
            Box::new(ExponentialLR::new(0.1, 0.95)),
            Box::new(CosineAnnealingLR::new(0.1, 100, 0.0)),
            Box::new(CosineAnnealingWarmRestarts::new(0.1, 10, 2, 0.0)),
            Box::new(LinearWarmup::new(0.1, 10)),
            Box::new(WarmupCosine::new(0.1, 10, 100, 0.0)),
            Box::new(OneCycleLR::new(0.1, 100, 0.3, 25.0, 1e4)),
            Box::new(PolynomialLR::new(0.1, 100, 2.0, 0.0)),
            Box::new(CyclicLR::new(0.001, 0.01, 10, CyclicMode::Triangular)),
        ];

        for s in &schedulers {
            for epoch in 0..200 {
                let lr = s.get_lr(epoch);
                assert!(
                    lr >= 0.0,
                    "{} returned negative LR {} at epoch {}",
                    s.name(),
                    lr,
                    epoch
                );
            }
        }
    }

    #[test]
    fn test_default_get_lr_at_step_delegates() {
        let s = StepLR::new(0.1, 10, 0.5);
        // Default implementation should delegate to get_lr(epoch)
        assert!(approx_eq(s.get_lr_at_step(999, 5), s.get_lr(5), EPS));
    }

    #[test]
    fn test_one_cycle_monotonic_warmup() {
        let s = OneCycleLR::new(0.1, 100, 0.3, 25.0, 1e4);
        // During warmup phase, LR should be monotonically increasing
        let mut prev = s.get_lr(0);
        for step in 1..30 {
            let curr = s.get_lr(step);
            assert!(
                curr >= prev - 1e-12,
                "OneCycleLR not monotonic at step {}: {} < {}",
                step,
                curr,
                prev
            );
            prev = curr;
        }
    }
}
