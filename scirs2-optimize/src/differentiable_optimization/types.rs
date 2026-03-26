//! Types for differentiable optimization (OptNet-style LP/QP layers).
//!
//! Provides configuration, result, and gradient structures for differentiable
//! quadratic and linear programming.

/// Configuration for differentiable QP solving.
#[derive(Debug, Clone)]
pub struct DiffQPConfig {
    /// Convergence tolerance for the interior-point solver.
    pub tolerance: f64,
    /// Maximum number of interior-point iterations.
    pub max_iterations: usize,
    /// Tikhonov regularization added to Q diagonal for numerical stability.
    pub regularization: f64,
    /// Backward differentiation mode.
    pub backward_mode: BackwardMode,
}

impl Default for DiffQPConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 100,
            regularization: 1e-7,
            backward_mode: BackwardMode::FullDifferentiation,
        }
    }
}

/// Configuration for differentiable LP solving.
#[derive(Debug, Clone)]
pub struct DiffLPConfig {
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Maximum number of interior-point iterations.
    pub max_iterations: usize,
    /// Tolerance for identifying active inequality constraints.
    pub active_constraint_tol: f64,
    /// Tikhonov regularization for numerical stability.
    pub regularization: f64,
}

impl Default for DiffLPConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 100,
            active_constraint_tol: 1e-6,
            regularization: 1e-7,
        }
    }
}

/// Result of a differentiable QP forward solve.
#[derive(Debug, Clone)]
pub struct DiffQPResult {
    /// Optimal primal variable x*.
    pub optimal_x: Vec<f64>,
    /// Dual variables for inequality constraints (lambda).
    pub optimal_lambda: Vec<f64>,
    /// Dual variables for equality constraints (nu).
    pub optimal_nu: Vec<f64>,
    /// Optimal objective value: 0.5 x' Q x + c' x.
    pub objective: f64,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Number of iterations taken.
    pub iterations: usize,
}

/// Result of a differentiable LP forward solve.
#[derive(Debug, Clone)]
pub struct DiffLPResult {
    /// Optimal primal variable x*.
    pub optimal_x: Vec<f64>,
    /// Dual variables for inequality constraints (lambda).
    pub optimal_lambda: Vec<f64>,
    /// Dual variables for equality constraints (nu).
    pub optimal_nu: Vec<f64>,
    /// Optimal objective value: c' x.
    pub objective: f64,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Number of iterations taken.
    pub iterations: usize,
}

/// Implicit gradients of the loss w.r.t. problem parameters.
///
/// Given a downstream loss L(x*(θ)), these are the gradients dL/dθ
/// computed via implicit differentiation of the KKT conditions.
#[derive(Debug, Clone)]
pub struct ImplicitGradient {
    /// Gradient w.r.t. the quadratic cost matrix Q (n x n), if applicable.
    pub dl_dq: Option<Vec<Vec<f64>>>,
    /// Gradient w.r.t. the linear cost vector c (n).
    pub dl_dc: Vec<f64>,
    /// Gradient w.r.t. the inequality constraint matrix G (m x n), if applicable.
    pub dl_dg: Option<Vec<Vec<f64>>>,
    /// Gradient w.r.t. the inequality constraint rhs h (m).
    pub dl_dh: Vec<f64>,
    /// Gradient w.r.t. the equality constraint matrix A (p x n), if applicable.
    pub dl_da: Option<Vec<Vec<f64>>>,
    /// Gradient w.r.t. the equality constraint rhs b (p).
    pub dl_db: Vec<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// New unified layer types (DiffOptLayer trait, DiffOptParams, etc.)
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for a generic QP/LP optimization layer.
///
/// Holds the cost and constraint data for `min ½xᵀQx + cᵀx s.t. Ax=b, Gx≤h`.
#[derive(Debug, Clone, Default)]
pub struct DiffOptParams {
    /// Quadratic cost matrix Q (n×n). Empty means LP (Q=0).
    pub q: Vec<Vec<f64>>,
    /// Linear cost vector c (n).
    pub c: Vec<f64>,
    /// Equality constraint matrix A (p×n).
    pub a: Vec<Vec<f64>>,
    /// Equality rhs b (p).
    pub b: Vec<f64>,
    /// Inequality constraint matrix G (m×n): Gx ≤ h.
    pub g: Vec<Vec<f64>>,
    /// Inequality rhs h (m).
    pub h: Vec<f64>,
}

/// Result of a generic optimization layer forward pass.
#[derive(Debug, Clone)]
pub struct DiffOptResult {
    /// Optimal primal solution x*.
    pub x: Vec<f64>,
    /// Dual variables for inequality constraints λ.
    pub lambda: Vec<f64>,
    /// Dual variables for equality constraints ν.
    pub nu: Vec<f64>,
    /// Optimal objective value.
    pub objective: f64,
    /// Solver status.
    pub status: DiffOptStatus,
    /// Number of iterations taken.
    pub iterations: usize,
}

/// Solver status for an optimization layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DiffOptStatus {
    /// Optimal solution found within tolerance.
    Optimal,
    /// Maximum iterations reached without convergence.
    MaxIterations,
    /// Solver detected infeasibility.
    Infeasible,
    /// Solver detected unboundedness.
    Unbounded,
}

impl Default for DiffOptStatus {
    fn default() -> Self {
        DiffOptStatus::Optimal
    }
}

/// Gradient of a loss w.r.t. all optimization layer parameters.
#[derive(Debug, Clone)]
pub struct DiffOptGrad {
    /// Gradient dL/dQ (n×n).
    pub dl_dq: Option<Vec<Vec<f64>>>,
    /// Gradient dL/dc (n).
    pub dl_dc: Vec<f64>,
    /// Gradient dL/dA (p×n).
    pub dl_da: Option<Vec<Vec<f64>>>,
    /// Gradient dL/db (p).
    pub dl_db: Vec<f64>,
    /// Gradient dL/dG (m×n).
    pub dl_dg: Option<Vec<Vec<f64>>>,
    /// Gradient dL/dh (m).
    pub dl_dh: Vec<f64>,
}

/// Mode of backward differentiation through the optimization layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BackwardMode {
    /// Differentiate through all KKT conditions (full implicit differentiation).
    FullDifferentiation,
    /// Differentiate only through active inequality constraints, treating
    /// inactive constraints as absent. Faster but approximate when constraints
    /// are near the boundary.
    ActiveSetOnly,
}

/// KKT system residuals for monitoring convergence.
#[derive(Debug, Clone)]
pub struct KKTSystem {
    /// Stationarity residual: Qx + c + G'λ + A'ν.
    pub stationarity: Vec<f64>,
    /// Primal feasibility (equality): Ax - b.
    pub primal_eq: Vec<f64>,
    /// Primal feasibility (inequality): Gx - h (should be ≤ 0).
    pub primal_ineq: Vec<f64>,
    /// Complementary slackness: λ_i * (Gx - h)_i.
    pub complementarity: Vec<f64>,
    /// Maximum absolute residual across all conditions.
    pub max_residual: f64,
}

impl KKTSystem {
    /// Check whether all KKT residuals are below the given tolerance.
    pub fn is_satisfied(&self, tol: f64) -> bool {
        self.max_residual < tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diff_qp_config_default() {
        let cfg = DiffQPConfig::default();
        assert!((cfg.tolerance - 1e-8).abs() < 1e-15);
        assert_eq!(cfg.max_iterations, 100);
        assert!((cfg.regularization - 1e-7).abs() < 1e-15);
        assert_eq!(cfg.backward_mode, BackwardMode::FullDifferentiation);
    }

    #[test]
    fn test_diff_lp_config_default() {
        let cfg = DiffLPConfig::default();
        assert!((cfg.tolerance - 1e-8).abs() < 1e-15);
        assert_eq!(cfg.max_iterations, 100);
        assert!((cfg.active_constraint_tol - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn test_kkt_system_satisfied() {
        let kkt = KKTSystem {
            stationarity: vec![1e-10],
            primal_eq: vec![1e-10],
            primal_ineq: vec![-0.5],
            complementarity: vec![1e-12],
            max_residual: 1e-10,
        };
        assert!(kkt.is_satisfied(1e-8));
        assert!(!kkt.is_satisfied(1e-12));
    }

    #[test]
    fn test_backward_mode_non_exhaustive() {
        // Verify both modes exist and can be matched
        let mode = BackwardMode::ActiveSetOnly;
        match mode {
            BackwardMode::FullDifferentiation => panic!("wrong variant"),
            BackwardMode::ActiveSetOnly => {}
            _ => {}
        }
    }
}
