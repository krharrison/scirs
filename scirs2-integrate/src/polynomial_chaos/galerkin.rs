//! Stochastic Galerkin Method for ODEs with random parameters.
//!
//! Converts a stochastic ODE dx/dt = f(x, xi, t) into a coupled deterministic
//! system for the PCE coefficients c_k(t), then advances them using RK4.
//!
//! The Galerkin projection yields:
//!
//! dc_k/dt = E\[f(sum_j c_j Psi_j, xi, t) * Psi_k\] / ||Psi_k||^2
//!
//! ## Example
//!
//! ```ignore
//! // dx/dt = -a * x, where a ~ U[0.5, 1.5] (modeled as 1.0 + 0.5*xi, xi ~ U[-1,1])
//! let config = StochasticGalerkinConfig { ... };
//! let solver = StochasticGalerkinSolver::new(config)?;
//! let result = solver.solve(
//!     |x, t, xi| Ok(-(1.0 + 0.5 * xi[0]) * x),
//!     &[1.0, 0.0, 0.0],  // initial x = 1 (deterministic IC)
//! )?;
//! ```

use crate::error::{IntegrateError, IntegrateResult};

use super::basis::{
    basis_norm_squared_nd, evaluate_basis_nd, gauss_quadrature, generate_multi_indices,
};
use super::statistics;
use super::types::{PCEConfig, TruncationScheme};

/// Configuration for the stochastic Galerkin solver.
#[derive(Debug, Clone)]
pub struct StochasticGalerkinConfig {
    /// PCE configuration (defines basis, degree, truncation).
    pub pce_config: PCEConfig,
    /// Start time.
    pub t_start: f64,
    /// End time.
    pub t_end: f64,
    /// Time step for RK4 integration.
    pub dt: f64,
    /// Tolerance for adaptive features (reserved for future use).
    pub tolerance: f64,
}

/// Result of the stochastic Galerkin solver.
#[derive(Debug, Clone)]
pub struct StochasticGalerkinResult {
    /// Time values at each output step.
    pub t_values: Vec<f64>,
    /// PCE coefficient history: coefficient_history\[step\]\[k\] = c_k(t).
    pub coefficient_history: Vec<Vec<f64>>,
    /// Mean trajectory: E\[x(t)\] = c_0(t).
    pub mean_trajectory: Vec<f64>,
    /// Variance trajectory: Var\[x(t)\] = sum_{k>=1} c_k(t)^2 * ||Psi_k||^2.
    pub variance_trajectory: Vec<f64>,
}

/// Stochastic Galerkin solver.
///
/// Transforms a stochastic ODE into a coupled deterministic ODE system
/// for PCE coefficients and integrates with RK4.
pub struct StochasticGalerkinSolver {
    /// Solver configuration.
    pub config: StochasticGalerkinConfig,
    /// Multi-indices for the PCE basis.
    pub multi_indices: Vec<Vec<usize>>,
    /// Squared norms ||Psi_k||^2.
    basis_norms_squared: Vec<f64>,
    /// Pre-computed quadrature points and weights (tensor product).
    quad_points: Vec<Vec<f64>>,
    /// Pre-computed quadrature weights.
    quad_weights: Vec<f64>,
    /// Pre-computed basis evaluations at quadrature points: basis_at_quad\[q\]\[k\].
    basis_at_quad: Vec<Vec<f64>>,
}

impl StochasticGalerkinSolver {
    /// Create a new stochastic Galerkin solver.
    pub fn new(config: StochasticGalerkinConfig) -> IntegrateResult<Self> {
        if config.dt <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Time step dt must be positive".to_string(),
            ));
        }
        if config.t_end <= config.t_start {
            return Err(IntegrateError::ValueError(
                "t_end must be greater than t_start".to_string(),
            ));
        }

        let dim = config.pce_config.bases.len();
        let multi_indices = generate_multi_indices(
            dim,
            config.pce_config.max_degree,
            &config.pce_config.truncation,
        );

        let basis_norms_squared: Vec<f64> = multi_indices
            .iter()
            .map(|alpha| basis_norm_squared_nd(&config.pce_config.bases, alpha))
            .collect();

        // Determine quadrature order from coefficient method or use default
        let quad_order = match &config.pce_config.coefficient_method {
            super::types::CoefficientMethod::Projection { quadrature_order } => *quadrature_order,
            _ => config.pce_config.max_degree + 1,
        };

        // Pre-compute tensor-product quadrature
        let mut quad_rules = Vec::with_capacity(dim);
        for basis in &config.pce_config.bases {
            quad_rules.push(gauss_quadrature(basis, quad_order)?);
        }

        let mut quad_points = Vec::new();
        let mut quad_weights = Vec::new();
        build_tensor_product_quadrature(&quad_rules, dim, &mut quad_points, &mut quad_weights);

        // Pre-compute basis evaluations at quadrature points
        let n_terms = multi_indices.len();
        let n_quad = quad_points.len();
        let mut basis_at_quad = Vec::with_capacity(n_quad);
        for q in 0..n_quad {
            let mut basis_vals = Vec::with_capacity(n_terms);
            for alpha in &multi_indices {
                basis_vals.push(evaluate_basis_nd(
                    &config.pce_config.bases,
                    alpha,
                    &quad_points[q],
                )?);
            }
            basis_at_quad.push(basis_vals);
        }

        Ok(Self {
            config,
            multi_indices,
            basis_norms_squared,
            quad_points,
            quad_weights,
            basis_at_quad,
        })
    }

    /// Number of PCE terms.
    pub fn n_terms(&self) -> usize {
        self.multi_indices.len()
    }

    /// Solve the stochastic ODE using the Galerkin method.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side function f(x, t, xi) -> dx/dt.
    ///   `x` is the scalar state, `t` is time, `xi` is the random parameter vector.
    /// * `x0_coeffs` - Initial PCE coefficients. Length must match `n_terms()`.
    ///   Typically `[x0, 0, 0, ...]` for a deterministic initial condition x0.
    pub fn solve<F>(&self, rhs: F, x0_coeffs: &[f64]) -> IntegrateResult<StochasticGalerkinResult>
    where
        F: Fn(f64, f64, &[f64]) -> IntegrateResult<f64>,
    {
        let n_terms = self.n_terms();
        if x0_coeffs.len() != n_terms {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Initial coefficients length {} doesn't match n_terms {}",
                x0_coeffs.len(),
                n_terms
            )));
        }

        let dt = self.config.dt;
        let t_start = self.config.t_start;
        let t_end = self.config.t_end;

        let n_steps = ((t_end - t_start) / dt).ceil() as usize;

        let mut t_values = Vec::with_capacity(n_steps + 1);
        let mut coefficient_history = Vec::with_capacity(n_steps + 1);
        let mut mean_trajectory = Vec::with_capacity(n_steps + 1);
        let mut variance_trajectory = Vec::with_capacity(n_steps + 1);

        let mut coeffs = x0_coeffs.to_vec();
        let mut t = t_start;

        // Store initial state
        t_values.push(t);
        coefficient_history.push(coeffs.clone());
        mean_trajectory.push(statistics::pce_mean(&coeffs));
        variance_trajectory.push(statistics::pce_variance(&coeffs, &self.basis_norms_squared));

        // RK4 integration
        for _ in 0..n_steps {
            let actual_dt = dt.min(t_end - t);
            if actual_dt <= 0.0 {
                break;
            }

            let k1 = self.compute_rhs_galerkin(&coeffs, t, &rhs)?;
            let c_mid1 = add_scaled(&coeffs, &k1, 0.5 * actual_dt);
            let k2 = self.compute_rhs_galerkin(&c_mid1, t + 0.5 * actual_dt, &rhs)?;
            let c_mid2 = add_scaled(&coeffs, &k2, 0.5 * actual_dt);
            let k3 = self.compute_rhs_galerkin(&c_mid2, t + 0.5 * actual_dt, &rhs)?;
            let c_end = add_scaled(&coeffs, &k3, actual_dt);
            let k4 = self.compute_rhs_galerkin(&c_end, t + actual_dt, &rhs)?;

            // c_{n+1} = c_n + (dt/6)(k1 + 2*k2 + 2*k3 + k4)
            for i in 0..n_terms {
                coeffs[i] += actual_dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
            t += actual_dt;

            t_values.push(t);
            coefficient_history.push(coeffs.clone());
            mean_trajectory.push(statistics::pce_mean(&coeffs));
            variance_trajectory.push(statistics::pce_variance(&coeffs, &self.basis_norms_squared));
        }

        Ok(StochasticGalerkinResult {
            t_values,
            coefficient_history,
            mean_trajectory,
            variance_trajectory,
        })
    }

    /// Compute the Galerkin-projected RHS for the coupled coefficient ODE system.
    ///
    /// dc_k/dt = E\[f(x_pce, t, xi) * Psi_k(xi)\] / ||Psi_k||^2
    ///
    /// where x_pce(xi) = sum_j c_j * Psi_j(xi).
    fn compute_rhs_galerkin<F>(&self, coeffs: &[f64], t: f64, rhs: &F) -> IntegrateResult<Vec<f64>>
    where
        F: Fn(f64, f64, &[f64]) -> IntegrateResult<f64>,
    {
        let n_terms = self.n_terms();
        let n_quad = self.quad_points.len();

        // Evaluate x_pce at each quadrature point
        let mut x_at_quad = Vec::with_capacity(n_quad);
        for q in 0..n_quad {
            let mut x_val = 0.0_f64;
            for (j, &c_j) in coeffs.iter().enumerate() {
                x_val += c_j * self.basis_at_quad[q][j];
            }
            x_at_quad.push(x_val);
        }

        // Evaluate f(x_pce, t, xi) at each quadrature point
        let mut f_at_quad = Vec::with_capacity(n_quad);
        for q in 0..n_quad {
            f_at_quad.push(rhs(x_at_quad[q], t, &self.quad_points[q])?);
        }

        // Compute dc_k/dt = sum_q w_q * f_q * Psi_k(xi_q) / ||Psi_k||^2
        let mut dc_dt = vec![0.0_f64; n_terms];
        for k in 0..n_terms {
            let mut numerator = 0.0_f64;
            for q in 0..n_quad {
                numerator += self.quad_weights[q] * f_at_quad[q] * self.basis_at_quad[q][k];
            }
            let norm_sq = self.basis_norms_squared[k];
            if norm_sq.abs() > 1e-30 {
                dc_dt[k] = numerator / norm_sq;
            }
        }

        Ok(dc_dt)
    }
}

/// c + scale * v
fn add_scaled(c: &[f64], v: &[f64], scale: f64) -> Vec<f64> {
    c.iter()
        .zip(v.iter())
        .map(|(&ci, &vi)| ci + scale * vi)
        .collect()
}

/// Build tensor-product quadrature points and weights from 1-D rules.
fn build_tensor_product_quadrature(
    rules: &[(Vec<f64>, Vec<f64>)],
    dim: usize,
    points: &mut Vec<Vec<f64>>,
    weights: &mut Vec<f64>,
) {
    if dim == 0 {
        points.push(vec![]);
        weights.push(1.0);
        return;
    }

    let (ref nodes0, ref weights0) = rules[0];
    let mut current_points: Vec<Vec<f64>> = nodes0.iter().map(|&x| vec![x]).collect();
    let mut current_weights: Vec<f64> = weights0.clone();

    for d in 1..dim {
        let (ref nodes_d, ref weights_d) = rules[d];
        let mut new_points = Vec::with_capacity(current_points.len() * nodes_d.len());
        let mut new_weights = Vec::with_capacity(current_weights.len() * weights_d.len());

        for (i, pt) in current_points.iter().enumerate() {
            for (j, &xd) in nodes_d.iter().enumerate() {
                let mut new_pt = pt.clone();
                new_pt.push(xd);
                new_points.push(new_pt);
                new_weights.push(current_weights[i] * weights_d[j]);
            }
        }
        current_points = new_points;
        current_weights = new_weights;
    }

    *points = current_points;
    *weights = current_weights;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial_chaos::types::*;

    #[test]
    fn test_galerkin_coefficient_count() {
        // d=1, p=3: should have 4 coefficients (0,1,2,3)
        let config = StochasticGalerkinConfig {
            pce_config: PCEConfig {
                bases: vec![PolynomialBasis::Legendre],
                max_degree: 3,
                truncation: TruncationScheme::TotalDegree,
                coefficient_method: CoefficientMethod::Projection {
                    quadrature_order: 5,
                },
            },
            t_start: 0.0,
            t_end: 1.0,
            dt: 0.1,
            tolerance: 1e-6,
        };
        let solver = StochasticGalerkinSolver::new(config).expect("solver creation failed");
        assert_eq!(solver.n_terms(), 4);
    }

    #[test]
    fn test_galerkin_2d_coefficient_count() {
        // d=2, p=2: C(4,2) = 6 coefficients
        let config = StochasticGalerkinConfig {
            pce_config: PCEConfig {
                bases: vec![PolynomialBasis::Legendre, PolynomialBasis::Legendre],
                max_degree: 2,
                truncation: TruncationScheme::TotalDegree,
                coefficient_method: CoefficientMethod::Projection {
                    quadrature_order: 4,
                },
            },
            t_start: 0.0,
            t_end: 1.0,
            dt: 0.1,
            tolerance: 1e-6,
        };
        let solver = StochasticGalerkinSolver::new(config).expect("solver creation failed");
        assert_eq!(solver.n_terms(), 6);
    }

    #[test]
    fn test_galerkin_linear_ode() {
        // dx/dt = -a*x with a ~ U[0.5, 1.5] => a = 1 + 0.5*xi, xi ~ U[-1,1]
        // Initial condition: x(0) = 1 (deterministic)
        // Mean should decay exponentially, variance should grow then stabilize
        let config = StochasticGalerkinConfig {
            pce_config: PCEConfig {
                bases: vec![PolynomialBasis::Legendre],
                max_degree: 4,
                truncation: TruncationScheme::TotalDegree,
                coefficient_method: CoefficientMethod::Projection {
                    quadrature_order: 6,
                },
            },
            t_start: 0.0,
            t_end: 2.0,
            dt: 0.01,
            tolerance: 1e-6,
        };
        let solver = StochasticGalerkinSolver::new(config).expect("solver creation failed");

        let n_terms = solver.n_terms();
        let mut x0 = vec![0.0_f64; n_terms];
        x0[0] = 1.0; // deterministic IC: x(0) = 1

        let result = solver
            .solve(
                |x, _t, xi| {
                    let a = 1.0 + 0.5 * xi[0];
                    Ok(-a * x)
                },
                &x0,
            )
            .expect("solve failed");

        // Mean at t=0 should be 1.0
        assert!(
            (result.mean_trajectory[0] - 1.0).abs() < 1e-10,
            "Mean at t=0: {}",
            result.mean_trajectory[0]
        );

        // Mean should decay (should be < 1 at t > 0)
        let last_idx = result.mean_trajectory.len() - 1;
        assert!(
            result.mean_trajectory[last_idx] < 0.5,
            "Mean at t=2: {} (expected < 0.5)",
            result.mean_trajectory[last_idx]
        );

        // Variance at t=0 should be 0 (deterministic IC)
        assert!(
            result.variance_trajectory[0].abs() < 1e-10,
            "Variance at t=0: {}",
            result.variance_trajectory[0]
        );

        // Variance should be > 0 at later times
        assert!(
            result.variance_trajectory[last_idx] > 0.0,
            "Variance at t=2: {} (expected > 0)",
            result.variance_trajectory[last_idx]
        );
    }

    #[test]
    fn test_galerkin_deterministic_ode() {
        // dx/dt = -x (no randomness): solution x(t) = e^{-t}
        // All coefficients except c_0 should remain zero
        let config = StochasticGalerkinConfig {
            pce_config: PCEConfig {
                bases: vec![PolynomialBasis::Legendre],
                max_degree: 2,
                truncation: TruncationScheme::TotalDegree,
                coefficient_method: CoefficientMethod::Projection {
                    quadrature_order: 4,
                },
            },
            t_start: 0.0,
            t_end: 1.0,
            dt: 0.01,
            tolerance: 1e-6,
        };
        let solver = StochasticGalerkinSolver::new(config).expect("solver creation failed");

        let n_terms = solver.n_terms();
        let mut x0 = vec![0.0_f64; n_terms];
        x0[0] = 1.0;

        let result = solver
            .solve(
                |x, _t, _xi| Ok(-x), // no dependence on xi
                &x0,
            )
            .expect("solve failed");

        // Mean at t=1 should be ~e^{-1}
        let last_idx = result.mean_trajectory.len() - 1;
        let expected = (-1.0_f64).exp();
        assert!(
            (result.mean_trajectory[last_idx] - expected).abs() < 1e-3,
            "Mean at t=1: got {}, expected {}",
            result.mean_trajectory[last_idx],
            expected
        );

        // Variance should remain ~0
        assert!(
            result.variance_trajectory[last_idx].abs() < 1e-10,
            "Variance at t=1: {} (expected ~0)",
            result.variance_trajectory[last_idx]
        );
    }
}
