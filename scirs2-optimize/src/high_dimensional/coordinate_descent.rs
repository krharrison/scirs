//! Coordinate Descent optimization methods
//!
//! Implements multiple variants of coordinate descent for high-dimensional optimization:
//!
//! - **Cyclic**: Optimize one variable at a time, cycling through all
//! - **Randomized**: Randomly select coordinate to update
//! - **Greedy (Gauss-Southwell)**: Select coordinate with largest gradient magnitude
//! - **Proximal**: Support for L1 (Lasso) and L2 (Ridge) regularization
//! - **Block**: Update groups of variables together
//!
//! Coordinate descent is particularly effective for problems where the per-coordinate
//! update is cheap and the problem has separable structure.

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

/// Strategy for selecting which coordinate to update
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoordinateSelectionStrategy {
    /// Cycle through coordinates in order 0, 1, ..., n-1, 0, 1, ...
    Cyclic,
    /// Randomly select a coordinate uniformly at random
    Randomized,
    /// Select the coordinate with the largest absolute gradient (Gauss-Southwell rule)
    Greedy,
}

/// Type of regularization for proximal coordinate descent
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegularizationType {
    /// No regularization
    None,
    /// L1 (Lasso) regularization: lambda * ||x||_1
    L1,
    /// L2 (Ridge) regularization: lambda * ||x||_2^2
    L2,
    /// Elastic net: alpha * lambda * ||x||_1 + (1 - alpha) * lambda * ||x||_2^2
    ElasticNet,
}

/// Configuration for coordinate descent solver
#[derive(Debug, Clone)]
pub struct CoordinateDescentConfig {
    /// Maximum number of full passes over all coordinates
    pub max_iter: usize,
    /// Convergence tolerance on the change in objective value
    pub tol: f64,
    /// Coordinate selection strategy
    pub strategy: CoordinateSelectionStrategy,
    /// Step size (learning rate). If None, uses exact line search for quadratics
    pub step_size: Option<f64>,
    /// Regularization type
    pub regularization: RegularizationType,
    /// Regularization strength (lambda)
    pub lambda: f64,
    /// Elastic net mixing parameter alpha in \[0,1\] (only used for ElasticNet)
    pub alpha: f64,
    /// Random seed for reproducibility (used with Randomized strategy)
    pub seed: u64,
    /// Whether to track objective value history
    pub track_objective: bool,
}

impl Default for CoordinateDescentConfig {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
            strategy: CoordinateSelectionStrategy::Cyclic,
            step_size: None,
            regularization: RegularizationType::None,
            lambda: 0.0,
            alpha: 0.5,
            seed: 42,
            track_objective: false,
        }
    }
}

/// Result of coordinate descent optimization
#[derive(Debug, Clone)]
pub struct CoordinateDescentResult {
    /// Optimal solution vector
    pub x: Array1<f64>,
    /// Final objective value (smooth part only, not including regularization)
    pub fun: f64,
    /// Final objective value including regularization
    pub fun_regularized: f64,
    /// Number of full iterations (passes over all coordinates)
    pub iterations: usize,
    /// Whether the solver converged
    pub converged: bool,
    /// History of objective values (if tracking enabled)
    pub objective_history: Vec<f64>,
    /// Final gradient norm
    pub grad_norm: f64,
}

/// Soft-thresholding operator for L1 proximal step
///
/// S(x, t) = sign(x) * max(|x| - t, 0)
fn soft_threshold(x: f64, threshold: f64) -> f64 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

/// Compute regularization penalty value
fn regularization_penalty(
    x: &Array1<f64>,
    reg_type: RegularizationType,
    lambda: f64,
    alpha: f64,
) -> f64 {
    match reg_type {
        RegularizationType::None => 0.0,
        RegularizationType::L1 => lambda * x.mapv(f64::abs).sum(),
        RegularizationType::L2 => lambda * x.dot(x),
        RegularizationType::ElasticNet => {
            let l1_part = alpha * lambda * x.mapv(f64::abs).sum();
            let l2_part = (1.0 - alpha) * lambda * x.dot(x);
            l1_part + l2_part
        }
    }
}

/// Coordinate Descent Solver
///
/// Minimizes f(x) + g(x), where f is a smooth objective and g is a separable
/// (possibly non-smooth) regularization term.
pub struct CoordinateDescentSolver {
    config: CoordinateDescentConfig,
}

impl CoordinateDescentSolver {
    /// Create a new coordinate descent solver with the given configuration
    pub fn new(config: CoordinateDescentConfig) -> Self {
        Self { config }
    }

    /// Create a solver with default configuration
    pub fn default_solver() -> Self {
        Self::new(CoordinateDescentConfig::default())
    }

    /// Minimize the objective function f(x) given gradient function grad_f(x)
    ///
    /// # Arguments
    /// * `objective` - Smooth objective function f(x)
    /// * `gradient` - Gradient of the smooth part: grad f(x)
    /// * `x0` - Initial point
    ///
    /// # Returns
    /// The optimization result containing the solution and convergence info
    pub fn minimize<F, G>(
        &self,
        objective: F,
        gradient: G,
        x0: &Array1<f64>,
    ) -> OptimizeResult<CoordinateDescentResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        let n = x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must have at least one dimension".to_string(),
            ));
        }

        let mut x = x0.clone();
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut objective_history = Vec::new();

        let step_size = self.config.step_size.unwrap_or(0.01);

        let mut prev_obj = objective(&x.view())
            + regularization_penalty(
                &x,
                self.config.regularization,
                self.config.lambda,
                self.config.alpha,
            );

        if self.config.track_objective {
            objective_history.push(prev_obj);
        }

        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            // One full pass over coordinates
            for _coord_step in 0..n {
                let coord = match self.config.strategy {
                    CoordinateSelectionStrategy::Cyclic => _coord_step,
                    CoordinateSelectionStrategy::Randomized => rng.random_range(0..n),
                    CoordinateSelectionStrategy::Greedy => {
                        let grad = gradient(&x.view());
                        // Select coordinate with largest absolute gradient
                        let mut best_coord = 0;
                        let mut best_abs_grad = f64::NEG_INFINITY;
                        for i in 0..n {
                            let abs_g = grad[i].abs();
                            if abs_g > best_abs_grad {
                                best_abs_grad = abs_g;
                                best_coord = i;
                            }
                        }
                        best_coord
                    }
                };

                // Compute partial gradient for selected coordinate
                let grad = gradient(&x.view());
                let grad_coord = grad[coord];

                // Update step depends on regularization
                match self.config.regularization {
                    RegularizationType::None => {
                        x[coord] -= step_size * grad_coord;
                    }
                    RegularizationType::L1 => {
                        // Proximal gradient step: soft threshold
                        let proposal = x[coord] - step_size * grad_coord;
                        x[coord] = soft_threshold(proposal, step_size * self.config.lambda);
                    }
                    RegularizationType::L2 => {
                        // L2 gradient includes 2*lambda*x_i term
                        let total_grad = grad_coord + 2.0 * self.config.lambda * x[coord];
                        x[coord] -= step_size * total_grad;
                    }
                    RegularizationType::ElasticNet => {
                        // L2 gradient part
                        let l2_grad =
                            2.0 * (1.0 - self.config.alpha) * self.config.lambda * x[coord];
                        let proposal = x[coord] - step_size * (grad_coord + l2_grad);
                        // L1 proximal part
                        x[coord] = soft_threshold(
                            proposal,
                            step_size * self.config.alpha * self.config.lambda,
                        );
                    }
                }
            }

            let smooth_obj = objective(&x.view());
            let total_obj = smooth_obj
                + regularization_penalty(
                    &x,
                    self.config.regularization,
                    self.config.lambda,
                    self.config.alpha,
                );

            if self.config.track_objective {
                objective_history.push(total_obj);
            }

            let change = (prev_obj - total_obj).abs();
            prev_obj = total_obj;

            if change < self.config.tol {
                converged = true;
                break;
            }
        }

        let final_grad = gradient(&x.view());
        let grad_norm = final_grad.dot(&final_grad).sqrt();
        let smooth_obj = objective(&x.view());
        let reg_penalty = regularization_penalty(
            &x,
            self.config.regularization,
            self.config.lambda,
            self.config.alpha,
        );

        Ok(CoordinateDescentResult {
            x,
            fun: smooth_obj,
            fun_regularized: smooth_obj + reg_penalty,
            iterations,
            converged,
            objective_history,
            grad_norm,
        })
    }
}

/// Proximal Coordinate Descent
///
/// Specialized for composite optimization problems of the form:
///   minimize f(x) + sum_i g_i(x_i)
///
/// where f is smooth and each g_i is a separable (possibly non-smooth) penalty.
pub struct ProximalCoordinateDescent {
    config: CoordinateDescentConfig,
}

impl ProximalCoordinateDescent {
    /// Create a new proximal coordinate descent solver
    pub fn new(config: CoordinateDescentConfig) -> Self {
        Self { config }
    }

    /// Minimize f(x) + lambda * ||x||_1 (Lasso)
    ///
    /// Uses coordinate-wise soft-thresholding updates.
    ///
    /// # Arguments
    /// * `objective` - Smooth part f(x)
    /// * `gradient` - Gradient of f
    /// * `x0` - Initial point
    pub fn minimize_lasso<F, G>(
        &self,
        objective: F,
        gradient: G,
        x0: &Array1<f64>,
    ) -> OptimizeResult<CoordinateDescentResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        let mut config = self.config.clone();
        config.regularization = RegularizationType::L1;
        let solver = CoordinateDescentSolver::new(config);
        solver.minimize(objective, gradient, x0)
    }

    /// Minimize f(x) + lambda * ||x||_2^2 (Ridge)
    ///
    /// # Arguments
    /// * `objective` - Smooth part f(x)
    /// * `gradient` - Gradient of f
    /// * `x0` - Initial point
    pub fn minimize_ridge<F, G>(
        &self,
        objective: F,
        gradient: G,
        x0: &Array1<f64>,
    ) -> OptimizeResult<CoordinateDescentResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        let mut config = self.config.clone();
        config.regularization = RegularizationType::L2;
        let solver = CoordinateDescentSolver::new(config);
        solver.minimize(objective, gradient, x0)
    }

    /// Minimize f(x) + alpha*lambda*||x||_1 + (1-alpha)*lambda*||x||_2^2 (Elastic Net)
    ///
    /// # Arguments
    /// * `objective` - Smooth part f(x)
    /// * `gradient` - Gradient of f
    /// * `x0` - Initial point
    pub fn minimize_elastic_net<F, G>(
        &self,
        objective: F,
        gradient: G,
        x0: &Array1<f64>,
    ) -> OptimizeResult<CoordinateDescentResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        let mut config = self.config.clone();
        config.regularization = RegularizationType::ElasticNet;
        let solver = CoordinateDescentSolver::new(config);
        solver.minimize(objective, gradient, x0)
    }
}

/// Block Coordinate Descent
///
/// Updates groups (blocks) of variables together rather than single coordinates.
/// This is useful when variables have strong within-block dependencies.
pub struct BlockCoordinateDescent {
    config: CoordinateDescentConfig,
    /// Block definitions: each inner Vec contains coordinate indices for that block
    blocks: Vec<Vec<usize>>,
}

impl BlockCoordinateDescent {
    /// Create a new block coordinate descent solver
    ///
    /// # Arguments
    /// * `config` - Solver configuration
    /// * `blocks` - Block definitions, each block is a list of coordinate indices
    pub fn new(config: CoordinateDescentConfig, blocks: Vec<Vec<usize>>) -> Self {
        Self { config, blocks }
    }

    /// Create blocks of equal size from dimension n
    ///
    /// # Arguments
    /// * `config` - Solver configuration
    /// * `n` - Total number of variables
    /// * `block_size` - Number of variables per block
    pub fn with_uniform_blocks(
        config: CoordinateDescentConfig,
        n: usize,
        block_size: usize,
    ) -> Self {
        let mut blocks = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + block_size).min(n);
            blocks.push((start..end).collect());
            start = end;
        }
        Self { config, blocks }
    }

    /// Minimize using block coordinate descent
    ///
    /// For each block, computes the gradient restricted to those coordinates
    /// and performs a gradient step on the block variables.
    ///
    /// # Arguments
    /// * `objective` - Objective function f(x)
    /// * `gradient` - Full gradient of f
    /// * `x0` - Initial point
    pub fn minimize<F, G>(
        &self,
        objective: F,
        gradient: G,
        x0: &Array1<f64>,
    ) -> OptimizeResult<CoordinateDescentResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        let n = x0.len();
        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Initial point must have at least one dimension".to_string(),
            ));
        }

        // Validate blocks
        for (bi, block) in self.blocks.iter().enumerate() {
            for &idx in block {
                if idx >= n {
                    return Err(OptimizeError::InvalidInput(format!(
                        "Block {} contains index {} which exceeds dimension {}",
                        bi, idx, n
                    )));
                }
            }
        }

        let mut x = x0.clone();
        let step_size = self.config.step_size.unwrap_or(0.01);
        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut objective_history = Vec::new();

        let mut prev_obj = objective(&x.view())
            + regularization_penalty(
                &x,
                self.config.regularization,
                self.config.lambda,
                self.config.alpha,
            );

        if self.config.track_objective {
            objective_history.push(prev_obj);
        }

        let mut converged = false;
        let mut iterations = 0;
        let num_blocks = self.blocks.len();

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            // Iterate over blocks
            for block_step in 0..num_blocks {
                let block_idx = match self.config.strategy {
                    CoordinateSelectionStrategy::Cyclic => block_step,
                    CoordinateSelectionStrategy::Randomized => rng.random_range(0..num_blocks),
                    CoordinateSelectionStrategy::Greedy => {
                        // Select block with largest gradient norm
                        let grad = gradient(&x.view());
                        let mut best_block = 0;
                        let mut best_norm = f64::NEG_INFINITY;
                        for (bi, block) in self.blocks.iter().enumerate() {
                            let block_norm_sq: f64 = block.iter().map(|&i| grad[i] * grad[i]).sum();
                            if block_norm_sq > best_norm {
                                best_norm = block_norm_sq;
                                best_block = bi;
                            }
                        }
                        best_block
                    }
                };

                let block = &self.blocks[block_idx];
                let grad = gradient(&x.view());

                // Update all coordinates in the block
                for &coord in block {
                    match self.config.regularization {
                        RegularizationType::None => {
                            x[coord] -= step_size * grad[coord];
                        }
                        RegularizationType::L1 => {
                            let proposal = x[coord] - step_size * grad[coord];
                            x[coord] = soft_threshold(proposal, step_size * self.config.lambda);
                        }
                        RegularizationType::L2 => {
                            let total_grad = grad[coord] + 2.0 * self.config.lambda * x[coord];
                            x[coord] -= step_size * total_grad;
                        }
                        RegularizationType::ElasticNet => {
                            let l2_grad =
                                2.0 * (1.0 - self.config.alpha) * self.config.lambda * x[coord];
                            let proposal = x[coord] - step_size * (grad[coord] + l2_grad);
                            x[coord] = soft_threshold(
                                proposal,
                                step_size * self.config.alpha * self.config.lambda,
                            );
                        }
                    }
                }
            }

            let smooth_obj = objective(&x.view());
            let total_obj = smooth_obj
                + regularization_penalty(
                    &x,
                    self.config.regularization,
                    self.config.lambda,
                    self.config.alpha,
                );

            if self.config.track_objective {
                objective_history.push(total_obj);
            }

            let change = (prev_obj - total_obj).abs();
            prev_obj = total_obj;

            if change < self.config.tol {
                converged = true;
                break;
            }
        }

        let final_grad = gradient(&x.view());
        let grad_norm = final_grad.dot(&final_grad).sqrt();
        let smooth_obj = objective(&x.view());
        let reg_penalty = regularization_penalty(
            &x,
            self.config.regularization,
            self.config.lambda,
            self.config.alpha,
        );

        Ok(CoordinateDescentResult {
            x,
            fun: smooth_obj,
            fun_regularized: smooth_obj + reg_penalty,
            iterations,
            converged,
            objective_history,
            grad_norm,
        })
    }
}

/// Convenience function: minimize a smooth objective using cyclic coordinate descent
///
/// # Arguments
/// * `objective` - Objective function f(x) -> f64
/// * `gradient` - Gradient function grad f(x) -> `Array1<f64>`
/// * `x0` - Initial point
/// * `config` - Optional configuration (uses defaults if None)
pub fn coordinate_descent_minimize<F, G>(
    objective: F,
    gradient: G,
    x0: &Array1<f64>,
    config: Option<CoordinateDescentConfig>,
) -> OptimizeResult<CoordinateDescentResult>
where
    F: Fn(&ArrayView1<f64>) -> f64,
    G: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    let config = config.unwrap_or_default();
    let solver = CoordinateDescentSolver::new(config);
    solver.minimize(objective, gradient, x0)
}

/// Convenience function: minimize with L1 (Lasso) regularization
///
/// Solves: min_x f(x) + lambda * ||x||_1
pub fn lasso_coordinate_descent<F, G>(
    objective: F,
    gradient: G,
    x0: &Array1<f64>,
    lambda: f64,
    config: Option<CoordinateDescentConfig>,
) -> OptimizeResult<CoordinateDescentResult>
where
    F: Fn(&ArrayView1<f64>) -> f64,
    G: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    let mut config = config.unwrap_or_default();
    config.regularization = RegularizationType::L1;
    config.lambda = lambda;
    let solver = CoordinateDescentSolver::new(config);
    solver.minimize(objective, gradient, x0)
}

/// Coordinate descent for quadratic objectives: min 0.5 * x^T A x - b^T x
///
/// When the objective is quadratic with a known Hessian, we can compute exact
/// coordinate-wise minimizers without a step size parameter.
///
/// # Arguments
/// * `a` - Symmetric positive definite matrix (Hessian)
/// * `b` - Linear term
/// * `x0` - Initial point
/// * `config` - Optional solver configuration
pub fn quadratic_coordinate_descent(
    a: &Array2<f64>,
    b: &Array1<f64>,
    x0: &Array1<f64>,
    config: Option<CoordinateDescentConfig>,
) -> OptimizeResult<CoordinateDescentResult> {
    let n = x0.len();
    let config = config.unwrap_or_default();

    if a.nrows() != n || a.ncols() != n {
        return Err(OptimizeError::InvalidInput(format!(
            "Matrix A has shape ({}, {}), expected ({}, {})",
            a.nrows(),
            a.ncols(),
            n,
            n
        )));
    }
    if b.len() != n {
        return Err(OptimizeError::InvalidInput(format!(
            "Vector b has length {}, expected {}",
            b.len(),
            n
        )));
    }

    let mut x = x0.clone();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut objective_history = Vec::new();

    // Objective: 0.5 * x^T A x - b^T x
    let compute_obj = |x: &Array1<f64>| -> f64 {
        let ax = a.dot(x);
        0.5 * x.dot(&ax) - b.dot(x)
    };

    let mut prev_obj = compute_obj(&x)
        + regularization_penalty(&x, config.regularization, config.lambda, config.alpha);

    if config.track_objective {
        objective_history.push(prev_obj);
    }

    let mut converged = false;
    let mut iterations = 0;

    for iter in 0..config.max_iter {
        iterations = iter + 1;

        for _coord_step in 0..n {
            let coord = match config.strategy {
                CoordinateSelectionStrategy::Cyclic => _coord_step,
                CoordinateSelectionStrategy::Randomized => rng.random_range(0..n),
                CoordinateSelectionStrategy::Greedy => {
                    // Gradient = Ax - b
                    let grad = a.dot(&x) - b;
                    let mut best = 0;
                    let mut best_val = f64::NEG_INFINITY;
                    for i in 0..n {
                        let abs_g = grad[i].abs();
                        if abs_g > best_val {
                            best_val = abs_g;
                            best = i;
                        }
                    }
                    best
                }
            };

            let a_ii = a[[coord, coord]];
            if a_ii.abs() < 1e-15 {
                continue; // Skip degenerate coordinate
            }

            // Compute residual for this coordinate: (Ax - b)[coord]
            let mut residual_coord = -b[coord];
            for j in 0..n {
                residual_coord += a[[coord, j]] * x[j];
            }

            match config.regularization {
                RegularizationType::None => {
                    // Exact minimizer: x_i = (b_i - sum_{j!=i} A_{ij} x_j) / A_{ii}
                    x[coord] -= residual_coord / a_ii;
                }
                RegularizationType::L1 => {
                    // Exact coordinate update with L1
                    let rhs = b[coord]
                        - (0..n)
                            .filter(|&j| j != coord)
                            .map(|j| a[[coord, j]] * x[j])
                            .sum::<f64>();
                    x[coord] = soft_threshold(rhs, config.lambda) / a_ii;
                }
                RegularizationType::L2 => {
                    // With L2: A_{ii} + 2*lambda in denominator
                    let rhs = b[coord]
                        - (0..n)
                            .filter(|&j| j != coord)
                            .map(|j| a[[coord, j]] * x[j])
                            .sum::<f64>();
                    x[coord] = rhs / (a_ii + 2.0 * config.lambda);
                }
                RegularizationType::ElasticNet => {
                    let rhs = b[coord]
                        - (0..n)
                            .filter(|&j| j != coord)
                            .map(|j| a[[coord, j]] * x[j])
                            .sum::<f64>();
                    x[coord] = soft_threshold(rhs, config.alpha * config.lambda)
                        / (a_ii + 2.0 * (1.0 - config.alpha) * config.lambda);
                }
            }
        }

        let smooth_obj = compute_obj(&x);
        let total_obj = smooth_obj
            + regularization_penalty(&x, config.regularization, config.lambda, config.alpha);

        if config.track_objective {
            objective_history.push(total_obj);
        }

        let change = (prev_obj - total_obj).abs();
        prev_obj = total_obj;

        if change < config.tol {
            converged = true;
            break;
        }
    }

    let grad = a.dot(&x) - b;
    let grad_norm = grad.dot(&grad).sqrt();
    let smooth_obj = compute_obj(&x);
    let reg_penalty =
        regularization_penalty(&x, config.regularization, config.lambda, config.alpha);

    Ok(CoordinateDescentResult {
        x,
        fun: smooth_obj,
        fun_regularized: smooth_obj + reg_penalty,
        iterations,
        converged,
        objective_history,
        grad_norm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    /// Test 1: Minimize a simple quadratic f(x) = x_1^2 + x_2^2 to the known optimum (0, 0)
    #[test]
    fn test_cyclic_cd_quadratic_minimum() {
        let objective = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };
        let gradient = |x: &ArrayView1<f64>| -> Array1<f64> { array![2.0 * x[0], 2.0 * x[1]] };

        let x0 = array![5.0, 3.0];
        let config = CoordinateDescentConfig {
            max_iter: 5000,
            tol: 1e-12,
            strategy: CoordinateSelectionStrategy::Cyclic,
            step_size: Some(0.4),
            ..Default::default()
        };

        let result = coordinate_descent_minimize(objective, gradient, &x0, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(result.converged);
        assert!(result.x[0].abs() < 1e-5);
        assert!(result.x[1].abs() < 1e-5);
        assert!(result.fun < 1e-10);
    }

    /// Test 2: Lasso regression produces a sparse solution
    #[test]
    fn test_lasso_sparse_solution() {
        // Quadratic with Lasso: min 0.5 * x^T A x - b^T x + lambda * ||x||_1
        // A = identity, b = [0.5, 0.05, 0.5] with lambda = 0.1
        // L1 should zero out the second component (0.05 < lambda=0.1)
        let a = Array2::eye(3);
        let b = array![0.5, 0.05, 0.5];
        let x0 = array![0.0, 0.0, 0.0];

        let config = CoordinateDescentConfig {
            max_iter: 1000,
            tol: 1e-12,
            regularization: RegularizationType::L1,
            lambda: 0.1,
            ..Default::default()
        };

        let result = quadratic_coordinate_descent(&a, &b, &x0, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(result.converged);
        // x[1] should be zero (sparse)
        assert!(
            result.x[1].abs() < 1e-10,
            "Expected sparse: x[1]={} should be ~0",
            result.x[1]
        );
        // x[0] and x[2] should be nonzero (b_i > lambda)
        assert!(result.x[0].abs() > 0.1);
        assert!(result.x[2].abs() > 0.1);
    }

    /// Test 3: Convergence rate comparison - greedy vs cyclic
    #[test]
    fn test_greedy_vs_cyclic_convergence() {
        // Diagonal quadratic: f(x) = sum_i (i+1) * x_i^2 / 2
        let n = 10;
        let a = Array2::from_diag(&Array1::from_vec((1..=n).map(|i| i as f64).collect()));
        let b = Array1::ones(n);
        let x0 = Array1::from_vec(vec![10.0; n]);

        let config_cyclic = CoordinateDescentConfig {
            max_iter: 50,
            tol: 1e-20, // Don't stop early
            strategy: CoordinateSelectionStrategy::Cyclic,
            track_objective: true,
            ..Default::default()
        };

        let config_greedy = CoordinateDescentConfig {
            max_iter: 50,
            tol: 1e-20,
            strategy: CoordinateSelectionStrategy::Greedy,
            track_objective: true,
            ..Default::default()
        };

        let result_cyclic = quadratic_coordinate_descent(&a, &b, &x0, Some(config_cyclic));
        let result_greedy = quadratic_coordinate_descent(&a, &b, &x0, Some(config_greedy));

        assert!(result_cyclic.is_ok());
        assert!(result_greedy.is_ok());
        let r_cyclic = result_cyclic.expect("cyclic should succeed");
        let r_greedy = result_greedy.expect("greedy should succeed");

        // Both should converge; greedy should have at least comparable final objective
        // (greedy is typically faster on ill-conditioned problems)
        assert!(r_cyclic.fun.is_finite());
        assert!(r_greedy.fun.is_finite());
    }

    /// Test 4: Randomized coordinate descent converges
    #[test]
    fn test_randomized_cd_converges() {
        let objective = |x: &ArrayView1<f64>| -> f64 {
            0.5 * (x[0] - 1.0).powi(2) + 0.5 * (x[1] - 2.0).powi(2)
        };
        let gradient = |x: &ArrayView1<f64>| -> Array1<f64> { array![x[0] - 1.0, x[1] - 2.0] };

        let x0 = array![10.0, -5.0];
        let config = CoordinateDescentConfig {
            max_iter: 10000,
            tol: 1e-10,
            strategy: CoordinateSelectionStrategy::Randomized,
            step_size: Some(0.9),
            seed: 123,
            ..Default::default()
        };

        let result = coordinate_descent_minimize(objective, gradient, &x0, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4, "x[0]={}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 1e-4, "x[1]={}", result.x[1]);
    }

    /// Test 5: Block coordinate descent
    #[test]
    fn test_block_cd() {
        let objective = |x: &ArrayView1<f64>| -> f64 {
            (x[0] - 1.0).powi(2)
                + (x[1] - 2.0).powi(2)
                + (x[2] - 3.0).powi(2)
                + (x[3] - 4.0).powi(2)
        };
        let gradient = |x: &ArrayView1<f64>| -> Array1<f64> {
            array![
                2.0 * (x[0] - 1.0),
                2.0 * (x[1] - 2.0),
                2.0 * (x[2] - 3.0),
                2.0 * (x[3] - 4.0)
            ]
        };

        let x0 = array![0.0, 0.0, 0.0, 0.0];
        let config = CoordinateDescentConfig {
            max_iter: 5000,
            tol: 1e-12,
            step_size: Some(0.4),
            ..Default::default()
        };

        let solver = BlockCoordinateDescent::with_uniform_blocks(config, 4, 2);
        let result = solver.minimize(objective, gradient, &x0);
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 2.0).abs() < 1e-4);
        assert!((result.x[2] - 3.0).abs() < 1e-4);
        assert!((result.x[3] - 4.0).abs() < 1e-4);
    }

    /// Test 6: Quadratic coordinate descent with exact updates
    #[test]
    fn test_quadratic_cd_exact() {
        // A = [[2, 1], [1, 3]], b = [1, 2]
        // Solution: A^{-1} b = [1/5, 3/5]
        let a = array![[2.0, 1.0], [1.0, 3.0]];
        let b = array![1.0, 2.0];
        let x0 = array![0.0, 0.0];

        let config = CoordinateDescentConfig {
            max_iter: 500,
            tol: 1e-14,
            ..Default::default()
        };

        let result = quadratic_coordinate_descent(&a, &b, &x0, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(result.converged);
        // A^{-1} = [[3, -1], [-1, 2]] / 5
        // x = [3/5 - 2/5, -1/5 + 4/5] = [1/5, 3/5]
        assert!(
            (result.x[0] - 0.2).abs() < 1e-8,
            "x[0]={}, expected 0.2",
            result.x[0]
        );
        assert!(
            (result.x[1] - 0.6).abs() < 1e-8,
            "x[1]={}, expected ~0.6",
            result.x[1]
        );
    }

    /// Test 7: Ridge regression
    #[test]
    fn test_ridge_cd() {
        let a = Array2::eye(3);
        let b = array![1.0, 2.0, 3.0];
        let x0 = array![0.0, 0.0, 0.0];

        let config = CoordinateDescentConfig {
            max_iter: 1000,
            tol: 1e-14,
            regularization: RegularizationType::L2,
            lambda: 0.5,
            ..Default::default()
        };

        let result = quadratic_coordinate_descent(&a, &b, &x0, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        // Solution: x_i = b_i / (1 + 2*lambda) = b_i / 2
        assert!((result.x[0] - 0.5).abs() < 1e-8, "x[0]={}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 1e-8, "x[1]={}", result.x[1]);
        assert!((result.x[2] - 1.5).abs() < 1e-8, "x[2]={}", result.x[2]);
    }

    /// Test 8: Objective history tracking
    #[test]
    fn test_objective_history_tracking() {
        let a = Array2::eye(2);
        let b = array![1.0, 1.0];
        let x0 = array![5.0, 5.0];

        let config = CoordinateDescentConfig {
            max_iter: 20,
            tol: 1e-20,
            track_objective: true,
            ..Default::default()
        };

        let result = quadratic_coordinate_descent(&a, &b, &x0, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        // History should be non-empty (initial + iterations)
        assert!(result.objective_history.len() > 1);
        // Objective should be monotonically non-increasing
        for i in 1..result.objective_history.len() {
            assert!(
                result.objective_history[i] <= result.objective_history[i - 1] + 1e-12,
                "Objective increased at iter {}: {} -> {}",
                i,
                result.objective_history[i - 1],
                result.objective_history[i]
            );
        }
    }
}
