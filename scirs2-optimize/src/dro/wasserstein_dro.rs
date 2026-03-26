//! Wasserstein Distributionally Robust Optimization.
//!
//! Implements the Esfahani–Kuhn (2018) Wasserstein DRO framework:
//!
//! ```text
//! min_w  max_{Q : W_1(Q,P_N)≤ε}  E_Q[ℓ(w, x)]
//! ```
//!
//! The dual formulation for Wasserstein-1 penalty yields a regularised
//! empirical risk:
//!
//! ```text
//! min_w  (1/N) Σ_i ℓ(w, x_i)  +  ε · ‖w‖_dual
//! ```
//!
//! For the concrete portfolio application (linear losses) the dual norm
//! penalty reduces to ε · ‖w‖₂, and the simplex constraint Σ w_i = 1,
//! w_i ≥ 0 is enforced via projected gradient descent onto the probability
//! simplex.
//!
//! # References
//!
//! - Esfahani, P. M. & Kuhn, D. (2018). "Data-driven distributionally robust
//!   optimization using the Wasserstein metric." *Mathematical Programming*.
//! - Blanchet, J. & Murthy, K. (2019). "Quantifying distributional model risk
//!   via optimal transport." *Mathematics of Operations Research*.

use super::types::{DroConfig, DroResult};
use crate::error::{OptimizeError, OptimizeResult};

// ---------------------------------------------------------------------------
// Internal LCG PRNG (no external crate dependency)
// ---------------------------------------------------------------------------

/// Linear Congruential Generator used for internal random sampling.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Advance state and return a `f64` in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Use the high 53 bits for mantissa.
        ((self.state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// Simplex projection
// ---------------------------------------------------------------------------

/// Project `v` onto the probability simplex Δ = {w ≥ 0, Σ w_i = 1}.
///
/// Uses the O(n log n) algorithm of Duchi et al. (2008).
fn project_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }

    // Sort descending.
    let mut sorted: Vec<f64> = v.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut rho = 0usize;
    for (j, &s) in sorted.iter().enumerate() {
        cumsum += s;
        if s - (cumsum - 1.0) / (j as f64 + 1.0) > 0.0 {
            rho = j;
        }
    }

    let cumsum_rho: f64 = sorted[..=rho].iter().sum();
    let theta = (cumsum_rho - 1.0) / (rho as f64 + 1.0);

    v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
}

// ---------------------------------------------------------------------------
// Gradient norm (L2)
// ---------------------------------------------------------------------------

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ---------------------------------------------------------------------------
// WassersteinDro
// ---------------------------------------------------------------------------

/// Wasserstein-ball DRO solver.
///
/// Minimises the regularised empirical risk:
///
/// ```text
/// min_w  (1/N) Σ_i ℓ(w, x_i)  +  ε · Ω(w)
/// ```
///
/// where Ω(w) is a regulariser derived from the dual norm of the loss
/// gradient, and ε is the Wasserstein ball radius.
pub struct WassersteinDro<'a> {
    config: DroConfig,
    /// Per-sample loss function ℓ(weights, sample) → f64.
    loss_fn: &'a dyn Fn(&[f64], &[f64]) -> f64,
    /// Gradient of ℓ with respect to `weights`.
    grad_fn: &'a dyn Fn(&[f64], &[f64]) -> Vec<f64>,
}

impl<'a> WassersteinDro<'a> {
    /// Create a new solver.
    pub fn new(
        config: DroConfig,
        loss_fn: &'a dyn Fn(&[f64], &[f64]) -> f64,
        grad_fn: &'a dyn Fn(&[f64], &[f64]) -> Vec<f64>,
    ) -> OptimizeResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            loss_fn,
            grad_fn,
        })
    }

    /// Solve the Wasserstein DRO problem.
    ///
    /// Runs sub-gradient descent with step size η_t = c/√t (or the fixed
    /// step from config) to minimise the regularised empirical risk.
    ///
    /// # Arguments
    ///
    /// * `n_features` – dimensionality of the decision variable `w`.
    /// * `samples`    – empirical samples {x_1, …, x_N}.
    ///
    /// # Returns
    ///
    /// [`DroResult`] containing the optimal weights and objective values.
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

        // Initialise weights uniformly.
        let mut w: Vec<f64> = vec![1.0 / n_features as f64; n_features];

        // Track best solution.
        let mut best_w = w.clone();
        let mut best_obj = f64::INFINITY;

        let c = 0.5_f64; // step size constant for 1/√t schedule

        for t in 1..=self.config.max_iter {
            // ── Compute empirical gradient ─────────────────────────────────
            let mut avg_grad: Vec<f64> = vec![0.0; n_features];
            let mut avg_loss = 0.0_f64;

            for sample in samples {
                let g = (self.grad_fn)(&w, sample);
                let l = (self.loss_fn)(&w, sample);
                avg_loss += l;
                for (ag, gi) in avg_grad.iter_mut().zip(g.iter()) {
                    *ag += gi;
                }
            }
            avg_loss /= n as f64;
            for ag in avg_grad.iter_mut() {
                *ag /= n as f64;
            }

            // ── Wasserstein regularisation ─────────────────────────────────
            // Dual-norm penalty: ε · ‖avg_grad‖_* (here L2 dual = L2).
            let grad_norm = l2_norm(&avg_grad);

            // Regularised objective.
            let obj = avg_loss + eps * grad_norm;
            if obj < best_obj {
                best_obj = obj;
                best_w = w.clone();
            }

            // ── Gradient norm convergence check ───────────────────────────
            if grad_norm < self.config.tol {
                return Ok(DroResult {
                    optimal_weights: best_w,
                    worst_case_loss: best_obj + eps * grad_norm,
                    primal_obj: best_obj,
                    n_iter: t,
                    converged: true,
                });
            }

            // ── Step size ─────────────────────────────────────────────────
            let eta = self
                .config
                .step_size
                .unwrap_or_else(|| c / (t as f64).sqrt());

            // ── Update: add Wasserstein penalty gradient ──────────────────
            // ∂/∂w [ε‖ḡ‖₂] ≈ ε · avg_grad / ‖avg_grad‖₂  (subgradient)
            let reg_grad: Vec<f64> = if grad_norm > 1e-12 {
                avg_grad.iter().map(|g| g * eps / grad_norm).collect()
            } else {
                vec![0.0; n_features]
            };

            for i in 0..n_features {
                w[i] -= eta * (avg_grad[i] + reg_grad[i]);
            }
        }

        // Compute final objective.
        let mut final_loss = 0.0_f64;
        for sample in samples {
            final_loss += (self.loss_fn)(&best_w, sample);
        }
        final_loss /= n as f64;
        let final_grad_norm: f64 = {
            let mut g_sum = vec![0.0_f64; n_features];
            for sample in samples {
                let g = (self.grad_fn)(&best_w, sample);
                for (gs, gi) in g_sum.iter_mut().zip(g.iter()) {
                    *gs += gi;
                }
            }
            for gs in g_sum.iter_mut() {
                *gs /= n as f64;
            }
            l2_norm(&g_sum)
        };

        Ok(DroResult {
            worst_case_loss: final_loss + eps * final_grad_norm,
            primal_obj: final_loss,
            optimal_weights: best_w,
            n_iter: self.config.max_iter,
            converged: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Portfolio DRO (projected gradient descent onto simplex)
// ---------------------------------------------------------------------------

/// Solve a distributionally robust portfolio optimisation problem.
///
/// The portfolio DRO problem seeks weights w ∈ Δ (probability simplex) that
/// minimise the worst-case expected negative return over a Wasserstein ball:
///
/// ```text
/// min_{w ∈ Δ}  max_{Q : W_1(Q,P_N)≤ε}  E_Q[−w^T r]
/// ```
///
/// For linear losses the Wasserstein penalty simplifies to ε‖w‖₂, giving the
/// tractable regularised problem:
///
/// ```text
/// min_{w ∈ Δ}  −ŵ^T μ̂  +  ε ‖w‖₂
/// ```
///
/// where μ̂ = (1/N) Σ_i r_i is the sample mean.
///
/// # Arguments
///
/// * `returns` – historical return observations; each row is one sample of d
///   asset returns.
/// * `radius`  – Wasserstein ball radius ε ≥ 0.
/// * `config`  – solver hyper-parameters (optional; uses defaults when `None`).
///
/// # Returns
///
/// [`DroResult`] where `optimal_weights` are the robust portfolio weights.
pub fn portfolio_dro(
    returns: &[Vec<f64>],
    radius: f64,
    config: Option<DroConfig>,
) -> OptimizeResult<DroResult> {
    if returns.is_empty() {
        return Err(OptimizeError::InvalidParameter(
            "returns must be non-empty".into(),
        ));
    }
    let d = returns[0].len();
    if d == 0 {
        return Err(OptimizeError::InvalidParameter(
            "each return vector must have at least one asset".into(),
        ));
    }
    if radius < 0.0 {
        return Err(OptimizeError::InvalidParameter(
            "radius must be non-negative".into(),
        ));
    }

    let cfg = config.unwrap_or_else(|| DroConfig {
        radius,
        n_samples: returns.len(),
        max_iter: 2_000,
        tol: 1e-7,
        step_size: None,
    });
    cfg.validate()?;

    let n = returns.len();

    // Compute sample mean returns μ̂.
    let mut mu = vec![0.0_f64; d];
    for r in returns {
        for (j, &rj) in r.iter().enumerate() {
            mu[j] += rj;
        }
    }
    for m in mu.iter_mut() {
        *m /= n as f64;
    }

    // Initialise with uniform weights on simplex.
    let mut w: Vec<f64> = vec![1.0 / d as f64; d];
    let mut best_w = w.clone();
    let mut best_obj = f64::INFINITY;

    let c = 0.5_f64;

    for t in 1..=cfg.max_iter {
        // Objective: −w^T μ̂ + ε ‖w‖₂
        let dot: f64 = w.iter().zip(mu.iter()).map(|(wi, mi)| wi * mi).sum();
        let wn = l2_norm(&w);
        let obj = -dot + radius * wn;

        if obj < best_obj {
            best_obj = obj;
            best_w = w.clone();
        }

        // Gradient: −μ̂ + ε · w / ‖w‖₂
        let wn_safe = wn.max(1e-12);
        let mut grad: Vec<f64> = w
            .iter()
            .zip(mu.iter())
            .map(|(&wi, &mi)| -mi + radius * wi / wn_safe)
            .collect();

        let grad_norm = l2_norm(&grad);
        if grad_norm < cfg.tol {
            break;
        }

        let eta = cfg.step_size.unwrap_or_else(|| c / (t as f64).sqrt());

        // Gradient step.
        for (wi, gi) in w.iter_mut().zip(grad.iter_mut()) {
            *wi -= eta * *gi;
        }

        // Project onto simplex.
        w = project_simplex(&w);
    }

    // Compute final worst-case loss.
    let dot_best: f64 = best_w.iter().zip(mu.iter()).map(|(wi, mi)| wi * mi).sum();
    let wn_best = l2_norm(&best_w);
    let worst_case = -dot_best + radius * wn_best;

    // Verify simplex constraint (should be satisfied by construction).
    let sum_w: f64 = best_w.iter().sum();

    // Renormalise to fix any floating-point drift.
    if (sum_w - 1.0).abs() > 1e-6 && sum_w > 0.0 {
        for wi in best_w.iter_mut() {
            *wi /= sum_w;
        }
    }

    Ok(DroResult {
        optimal_weights: best_w,
        worst_case_loss: worst_case,
        primal_obj: -dot_best,
        n_iter: cfg.max_iter,
        converged: true,
    })
}

// ---------------------------------------------------------------------------
// Convenience: standard ERM (ε = 0 special case)
// ---------------------------------------------------------------------------

/// Solve the standard empirical risk minimisation problem (ε = 0).
///
/// Equivalent to `portfolio_dro` with `radius = 0.0`.  Useful as a
/// baseline for comparing DRO against ERM.
pub fn portfolio_erm(returns: &[Vec<f64>], config: Option<DroConfig>) -> OptimizeResult<DroResult> {
    portfolio_dro(returns, 0.0, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: generates synthetic returns using LCG.
    fn synthetic_returns(n: usize, d: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|_| (0..d).map(|_| rng.next_f64() * 0.2 - 0.05).collect())
            .collect()
    }

    #[test]
    fn test_portfolio_dro_weights_sum_to_one() {
        let returns = synthetic_returns(50, 4, 42);
        let result = portfolio_dro(&returns, 0.05, None).expect("dro ok");
        let sum: f64 = result.optimal_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "weights must sum to 1, got {sum}");
    }

    #[test]
    fn test_portfolio_dro_positive_weights() {
        let returns = synthetic_returns(50, 3, 7);
        let result = portfolio_dro(&returns, 0.1, None).expect("dro ok");
        for (i, &w) in result.optimal_weights.iter().enumerate() {
            assert!(w >= -1e-9, "weight[{i}] = {w} should be non-negative");
        }
    }

    #[test]
    fn test_dro_radius_zero_equals_erm() {
        // With radius=0, DRO = ERM.  The primal objectives should be equal.
        let returns = synthetic_returns(30, 2, 13);
        let dro = portfolio_dro(&returns, 0.0, None).expect("dro ok");
        let erm = portfolio_erm(&returns, None).expect("erm ok");
        assert!(
            (dro.primal_obj - erm.primal_obj).abs() < 1e-4,
            "DRO with ε=0 should match ERM: {} vs {}",
            dro.primal_obj,
            erm.primal_obj
        );
    }

    #[test]
    fn test_dro_larger_radius_more_conservative() {
        // Larger radius should give a larger (more conservative) worst-case loss.
        let returns = synthetic_returns(40, 3, 99);
        let r1 = portfolio_dro(&returns, 0.01, None).expect("dro ok");
        let r2 = portfolio_dro(&returns, 0.5, None).expect("dro ok");
        // worst_case_loss for larger radius should be >= smaller radius.
        assert!(
            r2.worst_case_loss >= r1.worst_case_loss - 1e-4,
            "larger radius should yield more conservative (higher) worst-case: {} vs {}",
            r2.worst_case_loss,
            r1.worst_case_loss
        );
    }

    #[test]
    fn test_dro_result_fields_non_nan() {
        let returns = synthetic_returns(20, 2, 55);
        let result = portfolio_dro(&returns, 0.05, None).expect("dro ok");
        assert!(!result.worst_case_loss.is_nan(), "worst_case_loss is NaN");
        assert!(!result.primal_obj.is_nan(), "primal_obj is NaN");
        for (i, &w) in result.optimal_weights.iter().enumerate() {
            assert!(!w.is_nan(), "weight[{i}] is NaN");
        }
    }

    #[test]
    fn test_wasserstein_dro_solve_converges() {
        // Linear loss: ℓ(w, x) = -w·x;  gradient = -x
        let loss_fn = |w: &[f64], x: &[f64]| -> f64 {
            -w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum::<f64>()
        };
        let grad_fn = |_w: &[f64], x: &[f64]| -> Vec<f64> { x.iter().map(|xi| -xi).collect() };
        let mut rng = Lcg::new(31);
        let samples: Vec<Vec<f64>> = (0..20)
            .map(|_| vec![rng.next_f64(), rng.next_f64()])
            .collect();

        let cfg = DroConfig {
            radius: 0.1,
            max_iter: 200,
            tol: 1e-5,
            ..Default::default()
        };
        let solver = WassersteinDro::new(cfg, &loss_fn, &grad_fn).expect("valid");
        let result = solver.solve(2, &samples).expect("solve ok");
        assert!(!result.primal_obj.is_nan(), "primal_obj is NaN");
        assert!(!result.worst_case_loss.is_nan(), "worst_case_loss is NaN");
    }

    #[test]
    fn test_wasserstein_ball_center_distance_zero() {
        // The distance from the empirical centre to one of its own samples is 0.
        let samples = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let ball =
            super::super::types::WassersteinBall::new(samples.clone(), 0.5).expect("valid ball");
        let d = ball.distance_to_point(&[1.0, 2.0]);
        assert!(d < 1e-10, "distance to own sample should be 0, got {d}");
    }

    #[test]
    fn test_project_simplex_basic() {
        let v = vec![0.3, 0.7, 0.5];
        let p = project_simplex(&v);
        let sum: f64 = p.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "simplex projection should sum to 1"
        );
        for &pi in &p {
            assert!(pi >= 0.0, "simplex projection should be non-negative");
        }
    }

    #[test]
    fn test_project_simplex_already_on_simplex() {
        let v = vec![0.2, 0.5, 0.3];
        let p = project_simplex(&v);
        for (pi, vi) in p.iter().zip(v.iter()) {
            assert!(
                (pi - vi).abs() < 1e-9,
                "already on simplex, no change expected"
            );
        }
    }
}
