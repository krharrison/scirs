//! Differentiable Linear Programming.
//!
//! Solves the LP:
//!
//!   min  c'x
//!   s.t. Ax = b   (p equalities)
//!        Gx ≤ h   (m inequalities)
//!
//! and computes gradients of x* w.r.t. (c, A, b, G, h) via implicit
//! differentiation through the KKT conditions at the active constraints.
//!
//! At an LP optimum, the active inequality constraints define a polyhedron
//! face. We differentiate as if solving the equality-constrained system
//! formed by the active set, which is a degenerate QP with Q = 0 (plus
//! regularization for numerical stability).

use super::implicit_diff;
use super::types::{DiffLPConfig, DiffLPResult, ImplicitGradient};
use crate::error::{OptimizeError, OptimizeResult};

/// A differentiable LP layer.
#[derive(Debug, Clone)]
pub struct DifferentiableLP {
    /// Objective coefficient vector c (n).
    pub c: Vec<f64>,
    /// Equality constraint matrix A (p×n).
    pub a_eq: Vec<Vec<f64>>,
    /// Equality constraint rhs b (p).
    pub b_eq: Vec<f64>,
    /// Inequality constraint matrix G (m×n): Gx ≤ h.
    pub g: Vec<Vec<f64>>,
    /// Inequality constraint rhs h (m).
    pub h: Vec<f64>,
}

impl DifferentiableLP {
    /// Create a new differentiable LP.
    pub fn new(
        c: Vec<f64>,
        a_eq: Vec<Vec<f64>>,
        b_eq: Vec<f64>,
        g: Vec<Vec<f64>>,
        h: Vec<f64>,
    ) -> OptimizeResult<Self> {
        let n = c.len();
        for (i, row) in a_eq.iter().enumerate() {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput(format!(
                    "A_eq row {} has length {} but expected {}",
                    i,
                    row.len(),
                    n
                )));
            }
        }
        if a_eq.len() != b_eq.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "A_eq has {} rows but b_eq has length {}",
                a_eq.len(),
                b_eq.len()
            )));
        }
        for (i, row) in g.iter().enumerate() {
            if row.len() != n {
                return Err(OptimizeError::InvalidInput(format!(
                    "G row {} has length {} but expected {}",
                    i,
                    row.len(),
                    n
                )));
            }
        }
        if g.len() != h.len() {
            return Err(OptimizeError::InvalidInput(format!(
                "G has {} rows but h has length {}",
                g.len(),
                h.len()
            )));
        }

        Ok(Self {
            c,
            a_eq,
            b_eq,
            g,
            h,
        })
    }

    /// Number of primal variables.
    pub fn n(&self) -> usize {
        self.c.len()
    }

    /// Solve the LP (forward pass).
    ///
    /// Internally converts to a QP with Q = εI (small regularization) and
    /// solves via interior-point method.
    pub fn forward(&self, config: &DiffLPConfig) -> OptimizeResult<DiffLPResult> {
        let n = self.n();
        let m = self.h.len();
        let p = self.b_eq.len();

        // Convert LP to a lightly regularised QP: Q = reg * I
        let mut q = vec![vec![0.0; n]; n];
        for i in 0..n {
            q[i][i] = config.regularization;
        }

        // Use the QP solver from diff_qp
        let qp = super::diff_qp::DifferentiableQP {
            q,
            c: self.c.clone(),
            g: self.g.clone(),
            h: self.h.clone(),
            a: self.a_eq.clone(),
            b: self.b_eq.clone(),
        };

        let qp_config = super::types::DiffQPConfig {
            tolerance: config.tolerance,
            max_iterations: config.max_iterations,
            regularization: config.regularization,
            backward_mode: super::types::BackwardMode::FullDifferentiation,
        };

        let qp_result = qp.forward(&qp_config)?;

        // Compute LP objective (without the regularization term)
        let mut obj = 0.0;
        for i in 0..n {
            obj += self.c[i] * qp_result.optimal_x[i];
        }

        Ok(DiffLPResult {
            optimal_x: qp_result.optimal_x,
            optimal_lambda: qp_result.optimal_lambda,
            optimal_nu: qp_result.optimal_nu,
            objective: obj,
            converged: qp_result.converged,
            iterations: qp_result.iterations,
        })
    }

    /// Backward pass: compute gradients of loss w.r.t. LP parameters.
    ///
    /// Uses active-set implicit differentiation: at the LP optimum, only
    /// active inequality constraints matter, and we differentiate through
    /// the resulting equality system.
    pub fn backward(
        &self,
        result: &DiffLPResult,
        dl_dx: &[f64],
        config: &DiffLPConfig,
    ) -> OptimizeResult<ImplicitGradient> {
        let n = self.n();
        if dl_dx.len() != n {
            return Err(OptimizeError::InvalidInput(format!(
                "dl_dx length {} != n {}",
                dl_dx.len(),
                n
            )));
        }

        // Build Q = reg * I for the implicit differentiation
        let mut q = vec![vec![0.0; n]; n];
        for i in 0..n {
            q[i][i] = config.regularization;
        }

        // Use active-set implicit differentiation
        implicit_diff::compute_active_set_implicit_gradient(
            &q,
            &self.g,
            &self.h,
            &self.a_eq,
            &result.optimal_x,
            &result.optimal_lambda,
            &result.optimal_nu,
            dl_dx,
            config.active_constraint_tol,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple LP: min -x - y s.t. x + y <= 1, x >= 0, y >= 0
    /// Optimal at (1, 0) or (0, 1) — both give obj = -1
    /// With regularization, solution tends toward (0.5, 0.5)
    #[test]
    fn test_lp_forward_simple() {
        let lp = DifferentiableLP::new(
            vec![-1.0, -1.0],
            vec![],
            vec![],
            vec![
                vec![1.0, 1.0],  // x + y <= 1
                vec![-1.0, 0.0], // -x <= 0
                vec![0.0, -1.0], // -y <= 0
            ],
            vec![1.0, 0.0, 0.0],
        )
        .expect("LP creation failed");

        let config = DiffLPConfig::default();
        let result = lp.forward(&config).expect("Forward failed");

        assert!(result.converged, "LP should converge");
        // With regularization, x+y should be close to 1
        let sum: f64 = result.optimal_x.iter().sum();
        assert!((sum - 1.0).abs() < 0.1, "x+y = {} (expected ~1.0)", sum);
        // Objective should be close to -1
        assert!(
            (result.objective - (-1.0)).abs() < 0.1,
            "obj = {} (expected ~-1.0)",
            result.objective
        );
    }

    /// LP with equality constraint: min -x s.t. x + y = 1, x >= 0, y >= 0
    /// Optimal at x=1, y=0
    #[test]
    fn test_lp_with_equality() {
        let lp = DifferentiableLP::new(
            vec![-1.0, 0.0],
            vec![vec![1.0, 1.0]],
            vec![1.0],
            vec![
                vec![-1.0, 0.0], // -x <= 0
                vec![0.0, -1.0], // -y <= 0
            ],
            vec![0.0, 0.0],
        )
        .expect("LP creation failed");

        let config = DiffLPConfig::default();
        let result = lp.forward(&config).expect("Forward failed");

        assert!(result.converged);
        assert!(
            (result.optimal_x[0] - 1.0).abs() < 0.1,
            "x = {} (expected ~1.0)",
            result.optimal_x[0]
        );
    }

    #[test]
    fn test_lp_backward() {
        let lp = DifferentiableLP::new(
            vec![-1.0, -1.0],
            vec![],
            vec![],
            vec![vec![1.0, 1.0], vec![-1.0, 0.0], vec![0.0, -1.0]],
            vec![1.0, 0.0, 0.0],
        )
        .expect("LP creation failed");

        let config = DiffLPConfig::default();
        let result = lp.forward(&config).expect("Forward failed");

        let dl_dx = vec![1.0, 1.0];
        let grad = lp
            .backward(&result, &dl_dx, &config)
            .expect("Backward failed");

        // Gradient should be finite and non-empty
        assert_eq!(grad.dl_dc.len(), 2);
        assert!(grad.dl_dc[0].is_finite());
        assert!(grad.dl_dc[1].is_finite());
    }

    #[test]
    fn test_lp_dimension_validation() {
        let result = DifferentiableLP::new(
            vec![1.0, 2.0],
            vec![vec![1.0]], // wrong dimension
            vec![1.0],
            vec![],
            vec![],
        );
        assert!(result.is_err());
    }
}
