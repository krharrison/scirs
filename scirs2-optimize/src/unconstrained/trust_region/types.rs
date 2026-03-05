//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::OptimizeError;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Result of a trust-region dogleg optimization.
///
/// Contains the solution, final function value, convergence information,
/// and the final trust-region radius.
#[derive(Debug, Clone)]
pub struct TrustRegionResult {
    /// Solution vector (the point that minimizes the objective)
    pub x: Array1<f64>,
    /// Objective function value at the solution
    pub f_val: f64,
    /// Number of outer iterations performed
    pub n_iter: usize,
    /// Whether the algorithm converged (gradient norm < tolerance)
    pub converged: bool,
    /// Final trust-region radius
    pub trust_radius_final: f64,
    /// Number of function evaluations
    pub n_fev: usize,
    /// Number of gradient evaluations
    pub n_gev: usize,
    /// Number of Hessian evaluations
    pub n_hev: usize,
    /// Final gradient norm
    pub grad_norm: f64,
    /// Status message describing termination reason
    pub message: String,
}
/// Configuration for the trust-region dogleg optimizer.
///
/// Controls the behavior of the trust-region update strategy, including
/// the initial radius, expansion/contraction factors, and convergence criteria.
#[derive(Debug, Clone)]
pub struct TrustRegionConfig {
    /// Initial trust-region radius (must be positive)
    pub initial_radius: f64,
    /// Maximum trust-region radius (upper bound on how large the region can grow)
    pub max_radius: f64,
    /// Lower threshold for the actual-to-predicted reduction ratio.
    /// If ratio < eta1, the step is rejected and the trust region is shrunk.
    pub eta1: f64,
    /// Upper threshold for the actual-to-predicted reduction ratio.
    /// If ratio > eta2 and we hit the boundary, the trust region is expanded.
    pub eta2: f64,
    /// Shrink factor for the trust-region radius when the step is poor (ratio < eta1).
    /// The radius is multiplied by gamma1.
    pub gamma1: f64,
    /// Expansion factor for the trust-region radius when the step is very good
    /// (ratio > eta2 and we hit the boundary). The radius is multiplied by gamma2.
    pub gamma2: f64,
    /// Maximum number of outer iterations
    pub max_iter: usize,
    /// Gradient norm tolerance for convergence. The algorithm terminates when
    /// ||grad(f)|| < tolerance.
    pub tolerance: f64,
    /// Function value tolerance for convergence. The algorithm terminates when
    /// |f_old - f_new| < ftol * (1 + |f|).
    pub ftol: f64,
    /// Finite difference step size for numerical gradient/Hessian computation.
    /// Only used when analytic gradient/Hessian are not provided.
    pub eps: f64,
    /// Minimum trust-region radius. If the radius falls below this, the
    /// algorithm terminates (likely converged or stuck).
    pub min_radius: f64,
}
impl TrustRegionConfig {
    /// Validate the configuration parameters.
    ///
    /// Returns an error if any parameter is invalid:
    /// - `initial_radius` must be positive
    /// - `max_radius` must be >= `initial_radius`
    /// - `eta1` must be in (0, 1) and `eta2` must be in (eta1, 1)
    /// - `gamma1` must be in (0, 1) and `gamma2` must be > 1
    /// - `tolerance` must be positive
    pub fn validate(&self) -> Result<(), OptimizeError> {
        if self.initial_radius <= 0.0 {
            return Err(OptimizeError::ValueError(
                "initial_radius must be positive".to_string(),
            ));
        }
        if self.max_radius < self.initial_radius {
            return Err(OptimizeError::ValueError(
                "max_radius must be >= initial_radius".to_string(),
            ));
        }
        if self.eta1 <= 0.0 || self.eta1 >= 1.0 {
            return Err(OptimizeError::ValueError(
                "eta1 must be in (0, 1)".to_string(),
            ));
        }
        if self.eta2 <= self.eta1 || self.eta2 >= 1.0 {
            return Err(OptimizeError::ValueError(
                "eta2 must be in (eta1, 1)".to_string(),
            ));
        }
        if self.gamma1 <= 0.0 || self.gamma1 >= 1.0 {
            return Err(OptimizeError::ValueError(
                "gamma1 must be in (0, 1)".to_string(),
            ));
        }
        if self.gamma2 <= 1.0 {
            return Err(OptimizeError::ValueError("gamma2 must be > 1".to_string()));
        }
        if self.tolerance <= 0.0 {
            return Err(OptimizeError::ValueError(
                "tolerance must be positive".to_string(),
            ));
        }
        if self.max_iter == 0 {
            return Err(OptimizeError::ValueError(
                "max_iter must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}
