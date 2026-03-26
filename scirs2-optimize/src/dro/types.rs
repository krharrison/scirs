//! Types for Distributionally Robust Optimization (DRO).
//!
//! Provides configuration structs, result types, and core abstractions for
//! Wasserstein-ball and CVaR-based DRO.

use crate::error::{OptimizeError, OptimizeResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for distributionally robust optimization.
///
/// Controls the Wasserstein ball radius, sample count, and solver parameters.
#[derive(Debug, Clone)]
pub struct DroConfig {
    /// Wasserstein ball radius ε ≥ 0.  Larger values yield more conservative solutions.
    pub radius: f64,
    /// Number of empirical samples drawn from the reference distribution P_N.
    pub n_samples: usize,
    /// Maximum number of outer solver iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the gradient norm.
    pub tol: f64,
    /// Step size for (sub)gradient descent.  When `None` the solver uses
    /// the adaptive schedule 1/√t.
    pub step_size: Option<f64>,
}

impl Default for DroConfig {
    fn default() -> Self {
        Self {
            radius: 0.1,
            n_samples: 100,
            max_iter: 500,
            tol: 1e-6,
            step_size: None,
        }
    }
}

impl DroConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> OptimizeResult<()> {
        if self.radius < 0.0 {
            return Err(OptimizeError::InvalidParameter(
                "radius must be non-negative".into(),
            ));
        }
        if self.n_samples == 0 {
            return Err(OptimizeError::InvalidParameter(
                "n_samples must be positive".into(),
            ));
        }
        if self.max_iter == 0 {
            return Err(OptimizeError::InvalidParameter(
                "max_iter must be positive".into(),
            ));
        }
        if self.tol <= 0.0 {
            return Err(OptimizeError::InvalidParameter(
                "tol must be positive".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a distributionally robust optimization run.
#[derive(Debug, Clone)]
pub struct DroResult {
    /// Optimal decision variable weights (e.g. portfolio weights).
    pub optimal_weights: Vec<f64>,
    /// Worst-case expected loss under the Wasserstein-ball ambiguity set.
    pub worst_case_loss: f64,
    /// Primal objective value at the optimal weights.
    pub primal_obj: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the solver converged to the requested tolerance.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Wasserstein ball description
// ---------------------------------------------------------------------------

/// Describes a Wasserstein-1 ball around a set of centre samples.
///
/// The ball B_ε(P_N) = {Q : W_1(Q, P_N) ≤ ε} contains all probability
/// measures within Wasserstein distance ε of the empirical distribution P_N.
#[derive(Debug, Clone)]
pub struct WassersteinBall {
    /// Centre samples {x_1, …, x_N} defining the empirical distribution P_N.
    pub center_samples: Vec<Vec<f64>>,
    /// Ball radius ε ≥ 0.
    pub radius: f64,
}

impl WassersteinBall {
    /// Create a new Wasserstein ball.
    ///
    /// Returns an error when `radius < 0` or `center_samples` is empty.
    pub fn new(center_samples: Vec<Vec<f64>>, radius: f64) -> OptimizeResult<Self> {
        if radius < 0.0 {
            return Err(OptimizeError::InvalidParameter(
                "Wasserstein ball radius must be non-negative".into(),
            ));
        }
        if center_samples.is_empty() {
            return Err(OptimizeError::InvalidParameter(
                "center_samples must be non-empty".into(),
            ));
        }
        Ok(Self {
            center_samples,
            radius,
        })
    }

    /// Wasserstein-1 distance from the empirical centre to a single point `q`.
    ///
    /// For a discrete empirical distribution P_N the W_1 distance to the
    /// Dirac mass δ_q is min_i ‖x_i − q‖_2 (the nearest-centre distance).
    pub fn distance_to_point(&self, q: &[f64]) -> f64 {
        self.center_samples
            .iter()
            .map(|c| {
                c.iter()
                    .zip(q.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Check whether `q` is within the Wasserstein ball of the empirical centre.
    ///
    /// Returns `true` iff `distance_to_point(q) ≤ self.radius`.
    pub fn contains_point(&self, q: &[f64]) -> bool {
        self.distance_to_point(q) <= self.radius + f64::EPSILON
    }
}

// ---------------------------------------------------------------------------
// Robust objective variants
// ---------------------------------------------------------------------------

/// Selection of distributionally robust objective criterion.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum RobustObjective {
    /// Mean-variance trade-off: `E[loss] + lambda * Var[loss]`.
    MeanVariance {
        /// Trade-off parameter λ ≥ 0.  Larger values penalise variance more.
        lambda: f64,
    },
    /// Conditional Value-at-Risk at level α ∈ (0,1).
    CVaR {
        /// Confidence level α.  Must be in (0, 1).
        alpha: f64,
    },
    /// Pure worst-case (minimax) objective: `max_{Q in B_eps} E_Q[loss]`.
    WorstCase,
}

impl Default for RobustObjective {
    fn default() -> Self {
        Self::CVaR { alpha: 0.95 }
    }
}

// ---------------------------------------------------------------------------
// DroSolver
// ---------------------------------------------------------------------------

/// High-level handle to a DRO solver.
///
/// Stores configuration and exposes a uniform interface for different DRO
/// objectives.  Actual computation is delegated to the functions in
/// [`super::wasserstein_dro`] and [`super::cvar_dro`].
#[derive(Debug, Clone)]
pub struct DroSolver {
    /// Solver configuration.
    pub config: DroConfig,
    /// Objective criterion.
    pub objective: RobustObjective,
}

impl Default for DroSolver {
    fn default() -> Self {
        Self {
            config: DroConfig::default(),
            objective: RobustObjective::default(),
        }
    }
}

impl DroSolver {
    /// Create a new DRO solver with the given config and objective.
    pub fn new(config: DroConfig, objective: RobustObjective) -> OptimizeResult<Self> {
        config.validate()?;
        Ok(Self { config, objective })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dro_config_default_valid() {
        let cfg = DroConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_dro_config_negative_radius_error() {
        let cfg = DroConfig {
            radius: -0.1,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_wasserstein_ball_contains_center() {
        // Distance from a single-sample centre to itself is 0; should be in ball.
        let sample = vec![1.0, 2.0];
        let ball = WassersteinBall::new(vec![sample.clone()], 0.5).expect("valid ball");
        assert!(ball.contains_point(&sample));
    }

    #[test]
    fn test_wasserstein_ball_outside_radius() {
        let sample = vec![0.0, 0.0];
        let ball = WassersteinBall::new(vec![sample], 0.5).expect("valid ball");
        // Point at distance sqrt(2) ≈ 1.41 > 0.5
        assert!(!ball.contains_point(&[1.0, 1.0]));
    }

    #[test]
    fn test_wasserstein_ball_negative_radius_error() {
        assert!(WassersteinBall::new(vec![vec![0.0]], -0.1).is_err());
    }

    #[test]
    fn test_robust_objective_default() {
        let obj = RobustObjective::default();
        matches!(obj, RobustObjective::CVaR { .. });
    }

    #[test]
    fn test_dro_solver_default() {
        let solver = DroSolver::default();
        assert!(solver.config.radius >= 0.0);
    }
}
