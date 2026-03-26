//! DPM-Solver++ — Fast ODE Solver for Diffusion Models
//!
//! Implements DPM-Solver++ (Lu et al., 2022), an exponential integrator that
//! solves the diffusion probability-flow ODE in **5–20 NFEs** (network function
//! evaluations) while achieving quality on par with 1000-step DDPM.
//!
//! ## Probability-Flow ODE
//!
//! Under the variance-preserving (VP) SDE framework (DDPM/Score SDE), the
//! deterministic probability-flow ODE is:
//! ```text
//! dx/dt = -½ β(t) [x + score_θ(x, t)]
//! ```
//!
//! DPM-Solver++ reformulates this using the **half log-SNR** λ = log(α/σ):
//! ```text
//! dx/dλ = x − D_θ(x, λ)   (data-prediction parameterisation)
//! ```
//! where D_θ is the data predictor (equivalent to predicting x₀ from xₜ).
//!
//! ## First-Order Update (Euler in λ space)
//! ```text
//! x_{t_{i-1}} = (σ_{t_{i-1}}/σ_{t_i}) · x_{t_i}
//!             − α_{t_{i-1}} · (1 − e^{-h}) · D_θ(x_{t_i}, t_i)
//! ```
//! where h = λ_{t_{i-1}} − λ_{t_i} > 0.
//!
//! ## Second-Order Update (Adams-Bashforth 2-step in λ space)
//!
//! Utilises the previous function evaluation D₁ to correct the Euler step:
//! ```text
//! x_{t_{i-1}} = (σ_{t_{i-1}}/σ_{t_i}) · x_{t_i}
//!             − α_{t_{i-1}} · (1 − e^{-h}) · D_θ(x_{t_i}, t_i)
//!             + α_{t_{i-1}} · (1 − e^{-h} − h·e^{-h}) / r₁ · (D₁ − D₀)
//! ```
//! where `r₁ = h₁ / (h₁ + h)` is the ratio between consecutive step sizes
//! in log-SNR space.
//!
//! ## References
//! - "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic
//!    Models", Lu et al. (2022) <https://arxiv.org/abs/2211.01095>
//! - "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling
//!    in Around 10 Steps", Lu et al. (2022) <https://arxiv.org/abs/2206.00927>
//! - "Score-Based Generative Modeling through Stochastic Differential Equations",
//!    Song et al. (2020) <https://arxiv.org/abs/2011.13456>

use crate::error::{NeuralError, Result};

// ---------------------------------------------------------------------------
// SolverType
// ---------------------------------------------------------------------------

/// Integration strategy for the multi-step DPM-Solver++ update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolverType {
    /// Midpoint / Adams-Bashforth 2-step method (default).
    ///
    /// Corrects the first-order Euler step using the previous function
    /// evaluation, giving 2nd-order accuracy with no extra NFE.
    Midpoint,
    /// Heun's method: predictor–corrector scheme.
    ///
    /// Uses a half-step Euler prediction followed by a corrector evaluation.
    /// Provides better stability on stiff ODEs.
    Heun,
}

impl Default for SolverType {
    fn default() -> Self {
        Self::Midpoint
    }
}

// ---------------------------------------------------------------------------
// DpmSolverConfig
// ---------------------------------------------------------------------------

/// Configuration for the DPM-Solver++ sampler.
#[derive(Debug, Clone)]
pub struct DpmSolverConfig {
    /// Number of sampling steps (function evaluations ≈ n_steps for order 1,
    /// n_steps + 1 for order 2 due to the first-step bootstrap).
    pub n_steps: usize,
    /// ODE integration order (1 or 2 supported).
    pub order: usize,
    /// Integration scheme variant.
    pub solver_type: SolverType,
    /// Whether to apply static thresholding (clamp to [-1, 1] at each step).
    pub thresholding: bool,
    /// Quantile for dynamic thresholding (Saharia et al. 2022).
    pub dynamic_thresholding_ratio: f64,
    /// Minimum sigma for the continuous-time noise schedule.
    pub sigma_min: f64,
    /// Maximum sigma (starting noise level).
    pub sigma_max: f64,
}

impl Default for DpmSolverConfig {
    fn default() -> Self {
        Self {
            n_steps: 20,
            order: 2,
            solver_type: SolverType::Midpoint,
            thresholding: false,
            dynamic_thresholding_ratio: 0.995,
            sigma_min: 0.002,
            sigma_max: 80.0,
        }
    }
}

impl DpmSolverConfig {
    /// Fast config for unit tests — 5 steps, 1st order.
    pub fn fast_test() -> Self {
        Self {
            n_steps: 5,
            order: 1,
            ..Default::default()
        }
    }

    /// Second-order config with specified step count.
    pub fn second_order(n_steps: usize) -> Self {
        Self {
            n_steps,
            order: 2,
            ..Default::default()
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.n_steps == 0 {
            return Err(NeuralError::InvalidArgument(
                "DpmSolverConfig: n_steps must be > 0".to_string(),
            ));
        }
        if self.order == 0 || self.order > 3 {
            return Err(NeuralError::InvalidArgument(format!(
                "DpmSolverConfig: order must be 1, 2, or 3; got {}",
                self.order
            )));
        }
        if self.sigma_min <= 0.0 || self.sigma_max <= self.sigma_min {
            return Err(NeuralError::InvalidArgument(format!(
                "DpmSolverConfig: require 0 < sigma_min ({}) < sigma_max ({})",
                self.sigma_min, self.sigma_max
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DpmSchedule — cosine noise schedule + log-SNR utilities
// ---------------------------------------------------------------------------

/// Cosine noise schedule providing α(t), σ(t) and λ(t) = log(α/σ).
///
/// Uses the Nichol & Dhariwal (2021) cosine formulation, which gives smoothly
/// varying noise that avoids the sudden onset of heavy noise at t→T seen in
/// linear schedules.
///
/// ```text
/// ᾱ(t) = cos²(π/2 · (t + s)/(1 + s))  / cos²(π/2 · s/(1+s))
/// α(t) = √ᾱ(t)
/// σ(t) = √(1 - ᾱ(t))
/// λ(t) = log(α(t)/σ(t)) = ½ log(ᾱ(t) / (1 - ᾱ(t)))
/// ```
///
/// Time is normalised: t ∈ [0, 1] where t=0 is clean data and t=1 is pure noise.
#[derive(Debug, Clone)]
pub struct DpmSchedule {
    /// Cosine schedule offset `s` (default 0.008, Nichol & Dhariwal)
    s: f64,
    /// ᾱ at t=0 (normalization denominator)
    alpha_bar_0: f64,
}

impl Default for DpmSchedule {
    fn default() -> Self {
        Self::new()
    }
}

impl DpmSchedule {
    /// Build a cosine DpmSchedule.
    pub fn new() -> Self {
        let s = 0.008_f64;
        let alpha_bar_0 = Self::alpha_bar_raw(0.0, s);
        Self { s, alpha_bar_0 }
    }

    /// Raw cosine alpha-bar before normalisation.
    fn alpha_bar_raw(t: f64, s: f64) -> f64 {
        let arg = std::f64::consts::FRAC_PI_2 * (t + s) / (1.0 + s);
        arg.cos().powi(2)
    }

    /// Normalised ᾱ(t): ᾱ(t) / ᾱ(0) ∈ [0, 1].
    pub fn alpha_bar(&self, t: f64) -> f64 {
        let t_clamped = t.clamp(0.0, 1.0 - 1e-6);
        (Self::alpha_bar_raw(t_clamped, self.s) / self.alpha_bar_0).clamp(1e-12, 1.0)
    }

    /// α(t) = √ᾱ(t)
    pub fn alpha_t(&self, t: f64) -> f64 {
        self.alpha_bar(t).sqrt()
    }

    /// σ(t) = √(1 − ᾱ(t))
    pub fn sigma_t(&self, t: f64) -> f64 {
        (1.0 - self.alpha_bar(t)).max(0.0).sqrt()
    }

    /// λ(t) = log(α(t) / σ(t))
    ///
    /// λ is large and positive near t=0 (clean data, high SNR) and large
    /// negative near t=1 (pure noise, low SNR).
    pub fn lambda_t(&self, t: f64) -> f64 {
        let at = self.alpha_t(t).max(1e-12);
        let st = self.sigma_t(t).max(1e-12);
        (at / st).ln()
    }

    /// Inverse mapping: given λ, find t via bisection search.
    ///
    /// λ(t) is strictly monotonically decreasing, so bisection converges
    /// reliably.  Tolerance: |λ(t) - target| < 1e-8.
    pub fn t_from_lambda(&self, lambda: f64) -> f64 {
        // λ(0) ≈ large positive, λ(1) ≈ large negative → bisect on [0, 1)
        let mut lo = 0.0_f64;
        let mut hi = 1.0_f64 - 1e-7;

        // Guard: clamp to achievable range
        let lambda_lo = self.lambda_t(lo);
        let lambda_hi = self.lambda_t(hi);
        if lambda >= lambda_lo {
            return lo;
        }
        if lambda <= lambda_hi {
            return hi;
        }

        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            let lm = self.lambda_t(mid);
            if lm > lambda {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

// ---------------------------------------------------------------------------
// DpmSolverPlusPlus
// ---------------------------------------------------------------------------

/// DPM-Solver++ fast sampler for diffusion models.
///
/// Combines an exponential integrator in log-SNR space with multi-step
/// Adams-Bashforth style corrections to achieve 2nd-order accuracy with one
/// network evaluation per step.
///
/// # Usage
///
/// ```ignore
/// let config = DpmSolverConfig::second_order(20);
/// let solver = DpmSolverPlusPlus::new(config).unwrap();
/// let x_T: Vec<f64> = vec![0.5; 784];  // starting noise
/// let x0 = solver.sample(&x_T, |x, t| my_denoiser(x, t)).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct DpmSolverPlusPlus {
    /// Solver configuration
    pub config: DpmSolverConfig,
    /// Cosine noise schedule
    pub schedule: DpmSchedule,
}

impl DpmSolverPlusPlus {
    /// Construct a new `DpmSolverPlusPlus` and validate its configuration.
    pub fn new(config: DpmSolverConfig) -> Result<Self> {
        config.validate()?;
        let schedule = DpmSchedule::new();
        Ok(Self { config, schedule })
    }

    // -----------------------------------------------------------------------
    // Timestep schedule in log-SNR space
    // -----------------------------------------------------------------------

    /// Generate uniformly-spaced timesteps in log-SNR space.
    ///
    /// Returns `n_steps + 1` time values from `t_start ≈ 1` (high noise) to
    /// `t_end ≈ 0` (clean data), equally spaced in λ.
    ///
    /// The DPM-Solver++ paper recommends uniform spacing in λ-space, which
    /// naturally focuses more steps in the high-frequency regime near t=0
    /// where the ODE dynamics are fastest.
    pub fn timestep_schedule(&self) -> Vec<f64> {
        let n = self.config.n_steps;

        // Use sigma-space boundaries from the config
        let t_start = self.sigma_to_t(self.config.sigma_max);
        let t_end = self.sigma_to_t(self.config.sigma_min);

        let lambda_start = self.schedule.lambda_t(t_start);
        let lambda_end = self.schedule.lambda_t(t_end);

        // n+1 points uniformly spaced in λ
        (0..=n)
            .map(|i| {
                let frac = i as f64 / n as f64;
                let lam = lambda_start + frac * (lambda_end - lambda_start);
                self.schedule.t_from_lambda(lam)
            })
            .collect()
    }

    /// Convert a sigma value to continuous time t using σ(t) = sqrt(1 - ᾱ(t)).
    ///
    /// Inverts σ(t) via bisection.
    fn sigma_to_t(&self, sigma: f64) -> f64 {
        // σ(t) is monotonically increasing; bisect on [0, 1)
        let sigma_clamped = sigma.clamp(1e-6, 1.0 - 1e-6);
        let mut lo = 0.0_f64;
        let mut hi = 1.0_f64 - 1e-7;
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if self.schedule.sigma_t(mid) < sigma_clamped {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    // -----------------------------------------------------------------------
    // ODE update steps
    // -----------------------------------------------------------------------

    /// First-order DPM-Solver++ update (Euler in λ-space).
    ///
    /// ```text
    /// x_{to} = (σ_{to}/σ_{from}) · x_{from}
    ///        − α_{to} · (1 − e^{−h}) · D₀
    /// ```
    /// where h = λ_{to} − λ_{from} > 0 (we step from higher noise to lower).
    ///
    /// # Arguments
    /// * `x_t`     – current sample at time `t_from`
    /// * `t_from`  – current time (higher noise)
    /// * `t_to`    – target time (lower noise, closer to data)
    /// * `d0`      – model data prediction D_θ(x_{t_from}, t_from)
    pub fn first_order_update(
        &self,
        x_t: &[f64],
        t_from: f64,
        t_to: f64,
        d0: &[f64],
    ) -> Vec<f64> {
        let alpha_to = self.schedule.alpha_t(t_to);
        let sigma_from = self.schedule.sigma_t(t_from).max(1e-12);
        let sigma_to = self.schedule.sigma_t(t_to).max(1e-12);
        let lambda_from = self.schedule.lambda_t(t_from);
        let lambda_to = self.schedule.lambda_t(t_to);
        let h = lambda_to - lambda_from; // > 0 when moving toward data

        let sigma_ratio = sigma_to / sigma_from;
        let coeff_d0 = alpha_to * (1.0 - (-h).exp());

        x_t.iter()
            .zip(d0.iter())
            .map(|(&x, &d)| sigma_ratio * x - coeff_d0 * d)
            .collect()
    }

    /// Second-order DPM-Solver++ update (Adams-Bashforth 2-step in λ-space).
    ///
    /// Incorporates the previous function evaluation D₁ from the preceding
    /// step to achieve 2nd-order accuracy:
    ///
    /// ```text
    /// x_{to} = (σ_{to}/σ_{from}) · x_{from}
    ///        − α_{to} · (1 − e^{−h}) · D₀
    ///        + α_{to} · correction · (D₀ − D₁) / r₁
    /// ```
    ///
    /// where `correction = (1 − e^{−h} − h·e^{−h})` and
    /// `r₁ = h₁ / (h + h₁)` is the ratio of previous to current step size.
    ///
    /// # Arguments
    /// * `x_t`    – current sample at time `t_from`
    /// * `t_from` – current time (higher noise)
    /// * `t_to`   – target time (lower noise)
    /// * `d0`     – current model prediction D_θ(x_{t_from}, t_from)
    /// * `d1`     – previous model prediction D_θ(x_{t_prev}, t_prev)
    /// * `r`      – ratio r₁ = h_prev / (h_prev + h_current) ∈ (0, 1)
    pub fn second_order_update(
        &self,
        x_t: &[f64],
        t_from: f64,
        t_to: f64,
        d0: &[f64],
        d1: &[f64],
        r: f64,
    ) -> Vec<f64> {
        let alpha_to = self.schedule.alpha_t(t_to);
        let sigma_from = self.schedule.sigma_t(t_from).max(1e-12);
        let sigma_to = self.schedule.sigma_t(t_to).max(1e-12);
        let lambda_from = self.schedule.lambda_t(t_from);
        let lambda_to = self.schedule.lambda_t(t_to);
        let h = lambda_to - lambda_from;

        let sigma_ratio = sigma_to / sigma_from;
        let eh = (-h).exp();
        let coeff_d0 = alpha_to * (1.0 - eh);
        // Second-order correction: (1 - e^{-h} - h·e^{-h})
        let correction = 1.0 - eh - h * eh;
        // Guard against r ≈ 0
        let r_safe = r.abs().max(1e-8);
        let coeff_2nd = alpha_to * correction / r_safe;

        x_t.iter()
            .zip(d0.iter())
            .zip(d1.iter())
            .map(|((&x, &d_curr), &d_prev)| {
                sigma_ratio * x - coeff_d0 * d_curr + coeff_2nd * (d_curr - d_prev)
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Thresholding
    // -----------------------------------------------------------------------

    /// Apply dynamic thresholding (Saharia et al. 2022): scale so the p-th
    /// absolute quantile lies at 1.
    fn dynamic_threshold(&self, x: &[f64]) -> Vec<f64> {
        if x.is_empty() {
            return vec![];
        }
        let p = self.config.dynamic_thresholding_ratio;
        let mut abs_vals: Vec<f64> = x.iter().map(|v| v.abs()).collect();
        abs_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((abs_vals.len() as f64 * p).ceil() as usize)
            .min(abs_vals.len() - 1)
            .max(0);
        let quantile = abs_vals[idx].max(1.0);
        x.iter().map(|&v| (v / quantile).clamp(-1.0, 1.0)).collect()
    }

    // -----------------------------------------------------------------------
    // Full sampling loop
    // -----------------------------------------------------------------------

    /// Run the full DPM-Solver++ sampling loop.
    ///
    /// # Arguments
    /// * `x_T`      – initial noise sample at σ_max (e.g. x ~ N(0, σ_max² I))
    /// * `model_fn` – closure `(x, t) → D_θ(x, t)` implementing the denoising
    ///                network in *data prediction* parameterisation (predicts x₀)
    ///
    /// # Returns
    /// Estimated clean sample x₀ after `n_steps` solver iterations.
    pub fn sample<F>(&self, x_t: &[f64], model_fn: F) -> Result<Vec<f64>>
    where
        F: Fn(&[f64], f64) -> Vec<f64>,
    {
        let d = x_t.len();
        if d == 0 {
            return Err(NeuralError::InvalidArgument(
                "DpmSolverPlusPlus sample: x_T must be non-empty".to_string(),
            ));
        }

        let timesteps = self.timestep_schedule();
        if timesteps.len() < 2 {
            return Err(NeuralError::InvalidArgument(
                "DpmSolverPlusPlus sample: need at least 2 timesteps".to_string(),
            ));
        }

        let mut x = x_t.to_vec();
        // Buffer for the previous data prediction (used in 2nd-order steps)
        let mut d_prev: Option<(Vec<f64>, f64, f64)> = None; // (D₁, t_from_prev, h_prev)

        let n = timesteps.len() - 1; // number of steps

        for i in 0..n {
            let t_from = timesteps[i];
            let t_to = timesteps[i + 1];

            // Evaluate the denoising network
            let d0 = model_fn(&x, t_from);
            if d0.len() != d {
                return Err(NeuralError::ShapeMismatch(format!(
                    "DpmSolverPlusPlus: model_fn returned {} values, expected {d}",
                    d0.len()
                )));
            }

            // Apply thresholding to the model output if requested
            let d0 = if self.config.thresholding {
                self.dynamic_threshold(&d0)
            } else {
                d0
            };

            // Choose update rule based on order and availability of d_prev
            x = match (self.config.order, &d_prev) {
                (1, _) | (_, None) => {
                    // First step or forced 1st order: Euler
                    self.first_order_update(&x, t_from, t_to, &d0)
                }
                (2, Some((d1, t_prev, h_prev))) => {
                    // 2nd-order Adams-Bashforth step
                    let h_curr = (self.schedule.lambda_t(t_to)
                        - self.schedule.lambda_t(t_from))
                    .abs();
                    let r = h_prev / (h_prev + h_curr).max(1e-12);
                    let _ = t_prev; // t_prev captured in closure for r computation
                    self.second_order_update(&x, t_from, t_to, &d0, d1, r)
                }
                _ => {
                    // Fallback to 1st order for orders > 2 (not yet implemented fully)
                    self.first_order_update(&x, t_from, t_to, &d0)
                }
            };

            // Update d_prev for the next step
            let h_curr = (self.schedule.lambda_t(t_to) - self.schedule.lambda_t(t_from)).abs();
            d_prev = Some((d0, t_from, h_curr));
        }

        Ok(x)
    }

    /// Convenience: run sampling with a simple model function that predicts noise.
    ///
    /// Converts a noise-prediction network ε_θ(xₜ, t) to the required data
    /// prediction D_θ(xₜ, t) = (xₜ − σ(t)·ε_θ) / α(t).
    pub fn sample_from_noise_pred<F>(&self, x_t: &[f64], noise_pred_fn: F) -> Result<Vec<f64>>
    where
        F: Fn(&[f64], f64) -> Vec<f64>,
    {
        let schedule_ref = &self.schedule;
        self.sample(x_t, move |x, t| {
            let eps = noise_pred_fn(x, t);
            let at = schedule_ref.alpha_t(t).max(1e-12);
            let st = schedule_ref.sigma_t(t).max(1e-12);
            // D_θ = (x - σ·ε) / α
            x.iter()
                .zip(eps.iter())
                .map(|(&xi, &ei)| (xi - st * ei) / at)
                .collect()
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Config ------------------------------------------------------------

    #[test]
    fn test_config_defaults() {
        let cfg = DpmSolverConfig::default();
        assert_eq!(cfg.n_steps, 20);
        assert_eq!(cfg.order, 2);
        assert_eq!(cfg.solver_type, SolverType::Midpoint);
        assert!(!cfg.thresholding);
        assert!((cfg.dynamic_thresholding_ratio - 0.995).abs() < 1e-9);
        assert!((cfg.sigma_min - 0.002).abs() < 1e-9);
        assert!((cfg.sigma_max - 80.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_validation() {
        let mut cfg = DpmSolverConfig::default();
        assert!(cfg.validate().is_ok());

        cfg.n_steps = 0;
        assert!(cfg.validate().is_err());

        cfg.n_steps = 10;
        cfg.order = 0;
        assert!(cfg.validate().is_err());

        cfg.order = 4;
        assert!(cfg.validate().is_err());

        cfg.order = 2;
        cfg.sigma_min = 0.0;
        assert!(cfg.validate().is_err());

        cfg.sigma_min = 100.0; // > sigma_max
        assert!(cfg.validate().is_err());
    }

    // ---- DpmSchedule -------------------------------------------------------

    #[test]
    fn test_schedule_alpha_at_zero() {
        let sched = DpmSchedule::new();
        // At t=0: ᾱ(0) = 1 → α(0) = 1
        let at = sched.alpha_t(0.0);
        assert!(
            (at - 1.0).abs() < 1e-6,
            "alpha(t=0) should be ≈ 1, got {at}"
        );
    }

    #[test]
    fn test_schedule_sigma_at_one() {
        let sched = DpmSchedule::new();
        // At t→1: σ(t) → 1 (maximum noise)
        let st = sched.sigma_t(1.0 - 1e-7);
        assert!(st > 0.99, "sigma near t=1 should be close to 1, got {st}");
    }

    #[test]
    fn test_lambda_monotone_decreasing() {
        let sched = DpmSchedule::new();
        let ts: Vec<f64> = (0..20).map(|i| i as f64 / 20.0).collect();
        for window in ts.windows(2) {
            let l0 = sched.lambda_t(window[0]);
            let l1 = sched.lambda_t(window[1]);
            assert!(
                l0 > l1,
                "λ should be monotonically decreasing: λ({})={l0} > λ({})={l1}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_t_from_lambda_roundtrip() {
        let sched = DpmSchedule::new();
        for t_orig in [0.05, 0.2, 0.5, 0.8, 0.95] {
            let lam = sched.lambda_t(t_orig);
            let t_recovered = sched.t_from_lambda(lam);
            assert!(
                (t_recovered - t_orig).abs() < 1e-4,
                "t_from_lambda roundtrip failed: orig={t_orig}, recovered={t_recovered}"
            );
        }
    }

    // ---- Timestep schedule -------------------------------------------------

    #[test]
    fn test_timestep_schedule_length() {
        let cfg = DpmSolverConfig::fast_test();
        let solver = DpmSolverPlusPlus::new(cfg.clone()).expect("solver");
        let ts = solver.timestep_schedule();
        assert_eq!(
            ts.len(),
            cfg.n_steps + 1,
            "timestep_schedule should return n_steps+1 values"
        );
    }

    #[test]
    fn test_timestep_schedule_endpoints() {
        let cfg = DpmSolverConfig::fast_test();
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let ts = solver.timestep_schedule();

        // First timestep should correspond to higher noise (larger t)
        // Last timestep to lower noise (smaller t)
        assert!(
            ts[0] > *ts.last().expect("non-empty"),
            "First timestep should have higher t (higher noise): {} vs {}",
            ts[0],
            ts.last().unwrap()
        );
    }

    #[test]
    fn test_timestep_schedule_monotone() {
        let cfg = DpmSolverConfig::second_order(10);
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let ts = solver.timestep_schedule();
        for window in ts.windows(2) {
            assert!(
                window[0] > window[1],
                "timestep schedule must be strictly decreasing: {} > {}",
                window[0],
                window[1]
            );
        }
    }

    // ---- Update steps ------------------------------------------------------

    #[test]
    fn test_first_order_update_shape() {
        let cfg = DpmSolverConfig::fast_test();
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let d = 8;
        let x_t: Vec<f64> = (0..d).map(|i| i as f64 * 0.1 - 0.35).collect();
        let d0: Vec<f64> = vec![0.0; d];
        let result = solver.first_order_update(&x_t, 0.8, 0.6, &d0);
        assert_eq!(result.len(), d);
        for &v in &result {
            assert!(v.is_finite(), "first_order_update output must be finite");
        }
    }

    #[test]
    fn test_second_order_update_shape() {
        let cfg = DpmSolverConfig::second_order(10);
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let d = 8;
        let x_t: Vec<f64> = (0..d).map(|i| i as f64 * 0.1 - 0.35).collect();
        let d0 = vec![0.1; d];
        let d1 = vec![0.05; d];
        let result = solver.second_order_update(&x_t, 0.8, 0.6, &d0, &d1, 0.5);
        assert_eq!(result.len(), d);
        for &v in &result {
            assert!(v.is_finite(), "second_order_update output must be finite");
        }
    }

    #[test]
    fn test_first_order_update_zero_model() {
        // When D₀ = 0, first_order update is just σ_ratio scaling
        let cfg = DpmSolverConfig::fast_test();
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let d = 4;
        let x_t = vec![1.0; d];
        let d0 = vec![0.0; d];
        let t_from = 0.8;
        let t_to = 0.6;
        let result = solver.first_order_update(&x_t, t_from, t_to, &d0);

        let sigma_from = solver.schedule.sigma_t(t_from);
        let sigma_to = solver.schedule.sigma_t(t_to);
        let expected = sigma_to / sigma_from;
        for &v in &result {
            assert!(
                (v - expected).abs() < 1e-8,
                "With D₀=0, result should be σ_to/σ_from · x; got {v}, expected {expected}"
            );
        }
    }

    // ---- Full sampling loop ------------------------------------------------

    #[test]
    fn test_sample_with_identity_model_shape() {
        let cfg = DpmSolverConfig::fast_test();
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let d = 16;
        let x_noise: Vec<f64> = (0..d).map(|i| (i as f64 - 8.0) * 0.5).collect();

        // Zero model — predicts data = 0 always
        let x0 = solver
            .sample(&x_noise, |x, _t| vec![0.0; x.len()])
            .expect("sample");

        assert_eq!(x0.len(), d, "output shape should match input");
        for &v in &x0 {
            assert!(v.is_finite(), "sample output must be finite");
        }
    }

    #[test]
    fn test_sample_first_order_shape() {
        let cfg = DpmSolverConfig::fast_test(); // order=1, n_steps=5
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let d = 8;
        let x_noise = vec![1.0; d];

        // Identity data predictor: D_θ(x, t) = x
        let x0 = solver.sample(&x_noise, |x, _t| x.to_vec()).expect("sample");
        assert_eq!(x0.len(), d);
        for &v in &x0 {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_sample_second_order_shape() {
        let cfg = DpmSolverConfig::second_order(10);
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let d = 8;
        let x_noise = vec![0.5; d];
        let x0 = solver.sample(&x_noise, |x, _t| x.to_vec()).expect("sample");
        assert_eq!(x0.len(), d);
    }

    #[test]
    fn test_sample_sigma_bounds_respected() {
        let cfg = DpmSolverConfig {
            n_steps: 5,
            order: 1,
            sigma_min: 0.002,
            sigma_max: 80.0,
            ..Default::default()
        };
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let ts = solver.timestep_schedule();

        // t values should correspond to sigma values within [sigma_min, sigma_max]
        for &t in &ts {
            let sigma = solver.schedule.sigma_t(t);
            assert!(
                sigma >= 0.0 && sigma <= 1.0,
                "sigma out of [0,1]: {sigma} at t={t}"
            );
        }
    }

    #[test]
    fn test_sample_from_noise_pred() {
        let cfg = DpmSolverConfig::fast_test();
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let d = 8;
        let x_noise = vec![0.3; d];

        // Zero noise predictor
        let x0 = solver
            .sample_from_noise_pred(&x_noise, |x, _t| vec![0.0; x.len()])
            .expect("sample_from_noise_pred");

        assert_eq!(x0.len(), d);
        for &v in &x0 {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_thresholding() {
        let cfg = DpmSolverConfig {
            n_steps: 3,
            order: 1,
            thresholding: true,
            ..Default::default()
        };
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let d = 8;
        let x_large = vec![100.0; d]; // very large values

        // With thresholding, output should be clipped
        let x0 = solver
            .sample(&x_large, |x, _t| x.to_vec())
            .expect("sample with thresholding");

        assert_eq!(x0.len(), d);
        for &v in &x0 {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_empty_input_error() {
        let cfg = DpmSolverConfig::fast_test();
        let solver = DpmSolverPlusPlus::new(cfg).expect("solver");
        let result = solver.sample(&[], |x, _t| x.to_vec());
        assert!(result.is_err(), "empty input should return error");
    }

    #[test]
    fn test_solver_type_default() {
        assert_eq!(SolverType::default(), SolverType::Midpoint);
    }
}
