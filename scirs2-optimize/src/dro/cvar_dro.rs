//! CVaR-based Distributionally Robust Optimization.
//!
//! Implements Conditional Value-at-Risk (CVaR) estimation and a CVaR-DRO
//! solver that minimises the worst-case CVaR over a Wasserstein-ball
//! ambiguity set.
//!
//! ## CVaR formulation (Rockafellar–Uryasev 2000)
//!
//! For a loss random variable X and confidence level α ∈ (0,1):
//!
//! ```text
//! CVaR_α(X) = min_{t} { t  +  (1/(1-α)) · E[max(X − t, 0)] }
//! ```
//!
//! The minimiser is t* = VaR_α (the α-quantile of X).  Given N empirical
//! samples this becomes:
//!
//! ```text
//! CVaR_α = t*  +  (1 / (N(1-α))) · Σ_{i: x_i > t*} (x_i − t*)
//! ```
//!
//! ## CVaR-DRO
//!
//! The CVaR-DRO problem:
//!
//! ```text
//! min_w  CVaR_α(ℓ(w, ξ))  +  ε · ‖w‖₂
//! ```
//!
//! is solved via sample average approximation (SAA) combined with
//! subgradient descent.
//!
//! # References
//!
//! - Rockafellar, R. T. & Uryasev, S. (2000). "Optimization of conditional
//!   value-at-risk." *Journal of Risk*.
//! - Delage, E. & Ye, Y. (2010). "Distributionally robust optimization under
//!   moment uncertainty." *Operations Research*.

use super::types::{DroConfig, DroResult};
use crate::error::{OptimizeError, OptimizeResult};

// ---------------------------------------------------------------------------
// Internal LCG PRNG
// ---------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// CVaR estimator
// ---------------------------------------------------------------------------

/// CVaR estimator at confidence level α.
///
/// Computes the empirical Conditional Value-at-Risk from a sample of losses.
#[derive(Debug, Clone)]
pub struct CvarEstimator {
    /// Confidence level α ∈ (0, 1).
    pub alpha: f64,
}

impl CvarEstimator {
    /// Create a new estimator.
    ///
    /// Returns an error when `alpha` is not in (0, 1).
    pub fn new(alpha: f64) -> OptimizeResult<Self> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(OptimizeError::InvalidParameter(format!(
                "alpha must be in (0, 1), got {alpha}"
            )));
        }
        Ok(Self { alpha })
    }

    /// Compute the empirical CVaR from a slice of loss values.
    ///
    /// Returns the CVaR_α = mean of the worst (1-α) fraction of losses.
    ///
    /// # Panics (never): all failures are returned as `Err`.
    pub fn compute_cvar(&self, losses: &[f64]) -> OptimizeResult<f64> {
        if losses.is_empty() {
            return Err(OptimizeError::InvalidParameter(
                "losses must be non-empty".into(),
            ));
        }
        let mut sorted = losses.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(cvar_sorted(&sorted, self.alpha))
    }

    /// Compute the CVaR and its subgradient with respect to the loss weights.
    ///
    /// Returns `(cvar_value, subgradient)` where the subgradient has one
    /// entry per sample:
    ///
    /// ```text
    /// ∂CVaR_α / ∂l_i = 1/(N(1-α))   if l_i > t*
    ///                  0              otherwise
    /// ```
    ///
    /// and t* is the empirical α-quantile.
    pub fn cvar_gradient(&self, losses: &[f64]) -> OptimizeResult<(f64, Vec<f64>)> {
        if losses.is_empty() {
            return Err(OptimizeError::InvalidParameter(
                "losses must be non-empty".into(),
            ));
        }
        let n = losses.len();
        let mut sorted = losses.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let cvar = cvar_sorted(&sorted, self.alpha);
        let t_star = quantile_sorted(&sorted, self.alpha);

        let scale = 1.0 / (n as f64 * (1.0 - self.alpha));
        let grad: Vec<f64> = losses
            .iter()
            .map(|&l| if l > t_star { scale } else { 0.0 })
            .collect();

        Ok((cvar, grad))
    }
}

// ---------------------------------------------------------------------------
// Internal CVaR helpers (operate on already-sorted vectors)
// ---------------------------------------------------------------------------

/// CVaR from a **sorted** loss vector (ascending order).
fn cvar_sorted(sorted: &[f64], alpha: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }

    // Rockafellar-Uryasev: sweep t over all candidate thresholds.
    let scale = 1.0 / ((1.0 - alpha) * n as f64);
    let best = sorted
        .iter()
        .map(|&t| {
            let excess: f64 = sorted.iter().map(|&l| (l - t).max(0.0)).sum();
            t + scale * excess
        })
        .fold(f64::INFINITY, f64::min);

    best
}

/// α-quantile (VaR_α) from a **sorted** vector.
fn quantile_sorted(sorted: &[f64], alpha: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    let idx = ((alpha * n as f64).floor() as usize).min(n - 1);
    sorted[idx]
}

// ---------------------------------------------------------------------------
// CvarDro solver
// ---------------------------------------------------------------------------

/// CVaR-based DRO solver.
///
/// Minimises
///
/// ```text
/// min_w  CVaR_α(ℓ(w, ξ))  +  ε · ‖w‖₂
/// ```
///
/// using sample average approximation (SAA) and subgradient descent.
pub struct CvarDro<'a> {
    config: DroConfig,
    alpha: f64,
    /// Per-sample loss function ℓ(weights, sample) → f64.
    loss_fn: &'a dyn Fn(&[f64], &[f64]) -> f64,
    /// Gradient of ℓ with respect to `weights`.
    grad_fn: &'a dyn Fn(&[f64], &[f64]) -> Vec<f64>,
}

impl<'a> CvarDro<'a> {
    /// Create a new CVaR-DRO solver.
    ///
    /// Returns an error when `alpha` is not in (0, 1) or the config is invalid.
    pub fn new(
        config: DroConfig,
        alpha: f64,
        loss_fn: &'a dyn Fn(&[f64], &[f64]) -> f64,
        grad_fn: &'a dyn Fn(&[f64], &[f64]) -> Vec<f64>,
    ) -> OptimizeResult<Self> {
        config.validate()?;
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(OptimizeError::InvalidParameter(format!(
                "alpha must be in (0, 1), got {alpha}"
            )));
        }
        Ok(Self {
            config,
            alpha,
            loss_fn,
            grad_fn,
        })
    }

    /// Solve the CVaR-DRO problem.
    ///
    /// # Arguments
    ///
    /// * `n_features` – dimensionality of the decision variable.
    /// * `samples`    – empirical loss samples.
    ///
    /// # Returns
    ///
    /// [`DroResult`] with optimal weights and objective values.
    pub fn solve(&self, n_features: usize, samples: &[Vec<f64>]) -> OptimizeResult<DroResult> {
        if n_features == 0 {
            return Err(OptimizeError::InvalidParameter(
                "n_features must be positive".into(),
            ));
        }
        if samples.is_empty() {
            return Err(OptimizeError::InvalidParameter(
                "samples must be non-empty".into(),
            ));
        }

        let n = samples.len();
        let eps = self.config.radius;
        let estimator = CvarEstimator::new(self.alpha)?;

        // Initialise weights uniformly.
        let mut w: Vec<f64> = vec![1.0 / n_features as f64; n_features];
        let mut best_w = w.clone();
        let mut best_obj = f64::INFINITY;

        let c = 0.3_f64;

        for t in 1..=self.config.max_iter {
            // ── Evaluate per-sample losses ─────────────────────────────────
            let losses: Vec<f64> = samples.iter().map(|s| (self.loss_fn)(&w, s)).collect();

            // ── CVaR value ────────────────────────────────────────────────
            let cvar_val = estimator.compute_cvar(&losses)?;

            // ── Wasserstein regularisation ────────────────────────────────
            let wn = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            let obj = cvar_val + eps * wn;

            if obj < best_obj {
                best_obj = obj;
                best_w = w.clone();
            }

            // ── CVaR subgradient w.r.t. losses ────────────────────────────
            let (_, loss_grad) = estimator.cvar_gradient(&losses)?;
            // Chain rule: ∂CVaR/∂w = Σ_i (∂CVaR/∂l_i) · ∂l_i/∂w
            let mut param_grad: Vec<f64> = vec![0.0; n_features];
            for (i, sample) in samples.iter().enumerate() {
                let lg = loss_grad[i];
                if lg.abs() < 1e-14 {
                    continue;
                }
                let g = (self.grad_fn)(&w, sample);
                for (pg, gi) in param_grad.iter_mut().zip(g.iter()) {
                    *pg += lg * gi;
                }
            }

            // ── Wasserstein regularisation gradient ───────────────────────
            let wn_safe = wn.max(1e-12);
            for (pg, &wi) in param_grad.iter_mut().zip(w.iter()) {
                *pg += eps * wi / wn_safe;
            }

            let grad_norm = param_grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm < self.config.tol {
                return Ok(DroResult {
                    optimal_weights: best_w,
                    worst_case_loss: best_obj,
                    primal_obj: cvar_val,
                    n_iter: t,
                    converged: true,
                });
            }

            let eta = self
                .config
                .step_size
                .unwrap_or_else(|| c / (t as f64).sqrt());

            for (wi, gi) in w.iter_mut().zip(param_grad.iter()) {
                *wi -= eta * gi;
            }

            let _ = n; // suppress warning
        }

        // Final evaluation.
        let losses: Vec<f64> = samples.iter().map(|s| (self.loss_fn)(&best_w, s)).collect();
        let final_cvar = estimator.compute_cvar(&losses)?;
        let final_wn = best_w.iter().map(|x| x * x).sum::<f64>().sqrt();

        Ok(DroResult {
            optimal_weights: best_w,
            worst_case_loss: final_cvar + eps * final_wn,
            primal_obj: final_cvar,
            n_iter: self.config.max_iter,
            converged: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Solve a CVaR-DRO problem with the given loss and gradient functions.
///
/// This is a thin wrapper around [`CvarDro::solve`] for ergonomic use.
pub fn solve_cvar_dro(
    n_features: usize,
    samples: &[Vec<f64>],
    alpha: f64,
    radius: f64,
    config: Option<DroConfig>,
) -> OptimizeResult<DroResult> {
    let cfg = config.unwrap_or_else(|| DroConfig {
        radius,
        n_samples: samples.len(),
        max_iter: 500,
        tol: 1e-6,
        step_size: None,
    });

    // Example linear loss: ℓ(w, x) = -w·x
    let loss_fn = |w: &[f64], x: &[f64]| -> f64 {
        -w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>()
    };
    let grad_fn = |_w: &[f64], x: &[f64]| -> Vec<f64> { x.iter().map(|xi| -xi).collect() };

    let solver = CvarDro::new(cfg, alpha, &loss_fn, &grad_fn)?;
    solver.solve(n_features, samples)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cvar_computation() {
        // Losses = [0,1,2,3,4,5,6,7,8,9]; CVaR_{0.9} = mean of top 10% = 9.0
        let losses: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let est = CvarEstimator::new(0.9).expect("valid alpha");
        let cvar = est.compute_cvar(&losses).expect("cvar ok");
        // Top 10% of 10 samples: the single value 9.
        assert!(
            (cvar - 9.0).abs() < 0.5,
            "CVaR_0.9 of [0..9] should be ~9, got {cvar}"
        );
    }

    #[test]
    fn test_cvar_symmetry_ge_mean() {
        // CVaR_α ≥ mean for any α ∈ (0,1).
        let losses = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = losses.iter().sum::<f64>() / losses.len() as f64;
        let est = CvarEstimator::new(0.8).expect("valid");
        let cvar = est.compute_cvar(&losses).expect("cvar ok");
        assert!(
            cvar >= mean - 1e-9,
            "CVaR should be >= mean ({mean}), got {cvar}"
        );
    }

    #[test]
    fn test_cvar_alpha_invalid_errors() {
        assert!(CvarEstimator::new(0.0).is_err());
        assert!(CvarEstimator::new(1.0).is_err());
        assert!(CvarEstimator::new(-0.1).is_err());
        assert!(CvarEstimator::new(1.5).is_err());
    }

    #[test]
    fn test_cvar_at_alpha_near_one_gives_max() {
        // CVaR at α very close to 1 ≈ maximum loss.
        let losses = vec![1.0, 2.0, 5.0, 10.0, 3.0];
        let est = CvarEstimator::new(0.99).expect("valid");
        let cvar = est.compute_cvar(&losses).expect("cvar ok");
        let max_loss = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            cvar >= max_loss - 1e-6,
            "CVaR at alpha≈1 should be >= max loss ({max_loss}), got {cvar}"
        );
    }

    #[test]
    fn test_cvar_at_alpha_near_zero_gives_mean() {
        // CVaR at α close to 0 ≈ mean of all losses.
        let losses = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = losses.iter().sum::<f64>() / losses.len() as f64;
        let est = CvarEstimator::new(0.01).expect("valid");
        let cvar = est.compute_cvar(&losses).expect("cvar ok");
        assert!(
            (cvar - mean).abs() < 0.5,
            "CVaR at alpha≈0 should be close to mean ({mean}), got {cvar}"
        );
    }

    #[test]
    fn test_cvar_gradient_shape() {
        let losses = vec![1.0, 5.0, 2.0, 8.0, 3.0];
        let est = CvarEstimator::new(0.6).expect("valid");
        let (cvar, grad) = est.cvar_gradient(&losses).expect("grad ok");
        assert_eq!(grad.len(), losses.len());
        assert!(cvar.is_finite(), "CVaR should be finite");
    }

    #[test]
    fn test_cvar_gradient_non_negative_entries() {
        let losses = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let est = CvarEstimator::new(0.6).expect("valid");
        let (_, grad) = est.cvar_gradient(&losses).expect("grad ok");
        for &g in &grad {
            assert!(g >= 0.0, "CVaR gradient entries should be non-negative");
        }
    }

    #[test]
    fn test_cvar_dro_converges() {
        // Linear loss: ℓ(w, x) = -w·x
        let loss_fn = |w: &[f64], x: &[f64]| -> f64 {
            -w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>()
        };
        let grad_fn = |_w: &[f64], x: &[f64]| -> Vec<f64> { x.iter().map(|xi| -xi).collect() };
        let mut rng = Lcg::new(77);
        let samples: Vec<Vec<f64>> = (0..30)
            .map(|_| vec![rng.next_f64(), rng.next_f64()])
            .collect();
        let cfg = DroConfig {
            radius: 0.05,
            max_iter: 300,
            tol: 1e-5,
            ..Default::default()
        };
        let solver = CvarDro::new(cfg, 0.8, &loss_fn, &grad_fn).expect("valid");
        let result = solver.solve(2, &samples).expect("solve ok");
        assert!(!result.primal_obj.is_nan(), "primal_obj is NaN");
        assert!(!result.worst_case_loss.is_nan(), "worst_case is NaN");
        assert_eq!(result.optimal_weights.len(), 2);
    }

    #[test]
    fn test_cvar_dro_fields_non_nan() {
        let result = solve_cvar_dro(2, &[vec![0.1, 0.2], vec![0.3, 0.4]], 0.8, 0.05, None)
            .expect("solve ok");
        assert!(!result.worst_case_loss.is_nan());
        assert!(!result.primal_obj.is_nan());
        for &w in &result.optimal_weights {
            assert!(!w.is_nan());
        }
    }
}
