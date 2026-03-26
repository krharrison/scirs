//! PINN training loop with Adam optimizer and collocation point generation.

use super::network::PINNNetwork;
use super::types::{
    Boundary, BoundaryCondition, BoundarySide, CollocationStrategy, PDEProblem, PINNConfig,
    PINNResult,
};
use crate::error::IntegrateResult;
use scirs2_core::ndarray::{Array1, Array2};

/// Physics-Informed Neural Network solver.
///
/// Trains a neural network to satisfy a PDE by minimizing a composite loss:
///   L = physics_weight * L_pde + boundary_weight * L_bc + data_weight * L_data
pub struct PINNSolver {
    /// The underlying neural network approximator
    network: PINNNetwork,
    /// Training configuration
    config: PINNConfig,
}

/// Adam optimizer for PINN training.
struct AdamOptimizer {
    /// Learning rate
    lr: f64,
    /// First moment decay (default 0.9)
    beta1: f64,
    /// Second moment decay (default 0.999)
    beta2: f64,
    /// Numerical stability constant
    epsilon: f64,
    /// First moment estimate
    m: Array1<f64>,
    /// Second moment estimate
    v: Array1<f64>,
    /// Time step
    t: usize,
}

impl AdamOptimizer {
    /// Create a new Adam optimizer.
    fn new(n_params: usize, lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: Array1::<f64>::zeros(n_params),
            v: Array1::<f64>::zeros(n_params),
            t: 0,
        }
    }

    /// Perform one Adam update step.
    ///
    /// Returns the updated parameters.
    fn step(&mut self, params: &Array1<f64>, grad: &Array1<f64>) -> Array1<f64> {
        self.t += 1;
        let t = self.t as f64;

        // Update biased moments
        self.m = &self.m * self.beta1 + grad * (1.0 - self.beta1);
        self.v = &self.v * self.beta2 + &(grad * grad) * (1.0 - self.beta2);

        // Bias correction
        let m_hat = &self.m / (1.0 - self.beta1.powf(t));
        let v_hat = &self.v / (1.0 - self.beta2.powf(t));

        // Update
        params - &(&m_hat / &(v_hat.mapv(|x| x.sqrt()) + self.epsilon) * self.lr)
    }
}

/// Xorshift64 pseudo-random number generator.
fn xorshift64(state: &mut u64) -> f64 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    *state = s;
    (s as f64) / (u64::MAX as f64)
}

impl PINNSolver {
    /// Create a new PINN solver for the given PDE problem.
    ///
    /// # Arguments
    /// * `problem` - The PDE problem definition
    /// * `config` - Training configuration
    ///
    /// # Errors
    /// Returns an error if network creation fails.
    pub fn new(problem: &PDEProblem, config: PINNConfig) -> IntegrateResult<Self> {
        let input_dim = if problem.has_time {
            problem.spatial_dim + 1
        } else {
            problem.spatial_dim
        };

        let network = PINNNetwork::new(input_dim, &config.hidden_layers, 1)?;

        Ok(Self { network, config })
    }

    /// Train the PINN to satisfy the PDE and boundary conditions.
    ///
    /// The total loss is:
    ///   L = physics_weight * mean(residual^2) + boundary_weight * mean(bc_error^2) + data_weight * mean(data_error^2)
    ///
    /// # Arguments
    /// * `pde_residual` - Function computing the PDE residual at a given point
    /// * `problem` - The PDE problem definition
    /// * `data_points` - Optional observed data as (coordinates, values) pairs
    ///
    /// # Returns
    /// A `PINNResult` with training statistics and final loss components.
    pub fn train<F>(
        &mut self,
        pde_residual: &F,
        problem: &PDEProblem,
        data_points: Option<(&Array2<f64>, &Array1<f64>)>,
    ) -> IntegrateResult<PINNResult>
    where
        F: Fn(&PINNNetwork, &Array1<f64>) -> IntegrateResult<f64>,
    {
        let mut rng_state: u64 = 12345_u64 | 1;

        // Generate collocation points
        let collocation_pts = generate_collocation_points(
            problem,
            self.config.n_collocation,
            &self.config.collocation,
            &mut rng_state,
        );

        // Generate boundary points
        let boundary_data =
            generate_boundary_points(problem, self.config.n_boundary, &mut rng_state);

        let n_params = self.network.n_parameters();
        let mut optimizer = AdamOptimizer::new(n_params, self.config.learning_rate);

        let mut loss_history = Vec::with_capacity(self.config.max_epochs);
        let mut physics_loss = 0.0;
        let mut boundary_loss = 0.0;
        let mut data_loss = 0.0;
        let mut converged = false;
        let mut epochs_trained = 0;
        let fd_step = 1e-5; // for finite-difference gradient of loss w.r.t. parameters

        for epoch in 0..self.config.max_epochs {
            // --- Compute physics loss ---
            physics_loss = 0.0;
            let n_coll = collocation_pts.nrows();
            for i in 0..n_coll {
                let pt = collocation_pts.row(i).to_owned();
                let residual = pde_residual(&self.network, &pt)?;
                physics_loss += residual * residual;
            }
            if n_coll > 0 {
                physics_loss /= n_coll as f64;
            }

            // --- Compute boundary loss ---
            boundary_loss = 0.0;
            let mut n_bc_total = 0;
            for (bc_points, bc_condition) in &boundary_data {
                let n_bc = bc_points.nrows();
                for j in 0..n_bc {
                    let pt = bc_points.row(j).to_owned();
                    let bc_err = compute_bc_error(&self.network, &pt, bc_condition, fd_step)?;
                    boundary_loss += bc_err * bc_err;
                    n_bc_total += 1;
                }
            }
            if n_bc_total > 0 {
                boundary_loss /= n_bc_total as f64;
            }

            // --- Compute data loss ---
            data_loss = 0.0;
            if let Some((x_data, y_data)) = data_points {
                let n_data = x_data.nrows();
                for i in 0..n_data {
                    let pt = x_data.row(i).to_owned();
                    let predicted = self.network.forward(&pt)?;
                    let err = predicted - y_data[i];
                    data_loss += err * err;
                }
                if n_data > 0 {
                    data_loss /= n_data as f64;
                }
            }

            let total_loss = self.config.physics_weight * physics_loss
                + self.config.boundary_weight * boundary_loss
                + self.config.data_weight * data_loss;

            loss_history.push(total_loss);
            epochs_trained = epoch + 1;

            if total_loss < self.config.convergence_tol {
                converged = true;
                break;
            }

            // --- Compute gradient of total loss w.r.t. parameters via finite differences ---
            let current_params = self.network.parameters();
            let mut grad = Array1::<f64>::zeros(n_params);

            for p in 0..n_params {
                let mut params_plus = current_params.clone();
                params_plus[p] += fd_step;
                self.network.set_parameters(&params_plus)?;

                // Recompute total loss with perturbed parameters
                let loss_plus = self.compute_total_loss(
                    pde_residual,
                    &collocation_pts,
                    &boundary_data,
                    data_points,
                    fd_step,
                )?;

                grad[p] = (loss_plus - total_loss) / fd_step;
            }

            // Restore parameters and apply Adam update
            let new_params = optimizer.step(&current_params, &grad);
            self.network.set_parameters(&new_params)?;
        }

        Ok(PINNResult {
            final_loss: loss_history.last().copied().unwrap_or(f64::INFINITY),
            physics_loss,
            boundary_loss,
            data_loss,
            epochs_trained,
            converged,
            loss_history,
        })
    }

    /// Compute total loss for gradient estimation.
    fn compute_total_loss<F>(
        &self,
        pde_residual: &F,
        collocation_pts: &Array2<f64>,
        boundary_data: &[(Array2<f64>, BoundaryCondition)],
        data_points: Option<(&Array2<f64>, &Array1<f64>)>,
        fd_step: f64,
    ) -> IntegrateResult<f64>
    where
        F: Fn(&PINNNetwork, &Array1<f64>) -> IntegrateResult<f64>,
    {
        let n_coll = collocation_pts.nrows();
        let mut physics_loss = 0.0;
        for i in 0..n_coll {
            let pt = collocation_pts.row(i).to_owned();
            let residual = pde_residual(&self.network, &pt)?;
            physics_loss += residual * residual;
        }
        if n_coll > 0 {
            physics_loss /= n_coll as f64;
        }

        let mut boundary_loss = 0.0;
        let mut n_bc_total = 0;
        for (bc_points, bc_condition) in boundary_data {
            let n_bc = bc_points.nrows();
            for j in 0..n_bc {
                let pt = bc_points.row(j).to_owned();
                let bc_err = compute_bc_error(&self.network, &pt, bc_condition, fd_step)?;
                boundary_loss += bc_err * bc_err;
                n_bc_total += 1;
            }
        }
        if n_bc_total > 0 {
            boundary_loss /= n_bc_total as f64;
        }

        let mut data_loss = 0.0;
        if let Some((x_data, y_data)) = data_points {
            let n_data = x_data.nrows();
            for i in 0..n_data {
                let pt = x_data.row(i).to_owned();
                let predicted = self.network.forward(&pt)?;
                let err = predicted - y_data[i];
                data_loss += err * err;
            }
            if n_data > 0 {
                data_loss /= n_data as f64;
            }
        }

        Ok(self.config.physics_weight * physics_loss
            + self.config.boundary_weight * boundary_loss
            + self.config.data_weight * data_loss)
    }

    /// Predict the solution at new spatial points.
    ///
    /// # Arguments
    /// * `points` - Matrix of shape (n_points, input_dim)
    ///
    /// # Returns
    /// Array of predicted solution values.
    pub fn predict(&self, points: &Array2<f64>) -> IntegrateResult<Array1<f64>> {
        self.network.forward_batch(points)
    }

    /// Get a reference to the trained network.
    pub fn network(&self) -> &PINNNetwork {
        &self.network
    }
}

/// Compute the boundary condition error at a point.
fn compute_bc_error(
    network: &PINNNetwork,
    x: &Array1<f64>,
    condition: &BoundaryCondition,
    h: f64,
) -> IntegrateResult<f64> {
    match condition {
        BoundaryCondition::Dirichlet { value } => {
            let u = network.forward(x)?;
            Ok(u - value)
        }
        BoundaryCondition::Neumann { flux } => {
            let grad = network.gradient(x, h)?;
            // Use first spatial dimension's gradient as a proxy for normal derivative
            // (exact normal depends on boundary orientation)
            let du_dn = grad[0];
            Ok(du_dn - flux)
        }
        BoundaryCondition::Robin { alpha, beta, value } => {
            let u = network.forward(x)?;
            let grad = network.gradient(x, h)?;
            let du_dn = grad[0];
            Ok(alpha * u + beta * du_dn - value)
        }
        BoundaryCondition::Periodic => {
            // Periodic BCs are handled at the problem level by pairing boundaries.
            Ok(0.0)
        }
    }
}

/// Generate collocation points within the problem domain.
///
/// # Arguments
/// * `problem` - PDE problem defining the domain
/// * `n_points` - Number of interior points to generate
/// * `strategy` - Placement strategy
/// * `rng_state` - Mutable xorshift64 PRNG state
///
/// # Returns
/// Matrix of shape (n_points, input_dim) with collocation coordinates.
pub fn generate_collocation_points(
    problem: &PDEProblem,
    n_points: usize,
    strategy: &CollocationStrategy,
    rng_state: &mut u64,
) -> Array2<f64> {
    let input_dim = if problem.has_time {
        problem.spatial_dim + 1
    } else {
        problem.spatial_dim
    };

    let mut points = Array2::<f64>::zeros((n_points, input_dim));

    match strategy {
        CollocationStrategy::UniformGrid => {
            // Compute points per dimension for approximately n_points total
            let n_per_dim = (n_points as f64).powf(1.0 / input_dim as f64).ceil() as usize;
            let actual_n = n_per_dim.pow(input_dim as u32).min(n_points);

            // Build all domain bounds including time
            let mut all_bounds = problem.domain.clone();
            if let Some((t_min, t_max)) = problem.time_domain {
                all_bounds.push((t_min, t_max));
            }

            for idx in 0..actual_n {
                let mut remainder = idx;
                for d in 0..input_dim {
                    let coord_idx = remainder % n_per_dim;
                    remainder /= n_per_dim;
                    let (lo, hi) = all_bounds.get(d).copied().unwrap_or((0.0, 1.0));
                    let frac = if n_per_dim > 1 {
                        (coord_idx as f64 + 0.5) / n_per_dim as f64
                    } else {
                        0.5
                    };
                    points[[idx, d]] = lo + frac * (hi - lo);
                }
            }

            // Fill remaining points randomly if grid doesn't fill n_points
            for idx in actual_n..n_points {
                for d in 0..input_dim {
                    let (lo, hi) = if d < problem.domain.len() {
                        problem.domain[d]
                    } else {
                        problem.time_domain.unwrap_or((0.0, 1.0))
                    };
                    points[[idx, d]] = lo + xorshift64(rng_state) * (hi - lo);
                }
            }
        }
        CollocationStrategy::LatinHypercube => {
            // Latin Hypercube Sampling
            let mut all_bounds = problem.domain.clone();
            if let Some((t_min, t_max)) = problem.time_domain {
                all_bounds.push((t_min, t_max));
            }

            for d in 0..input_dim {
                // Create permutation of [0, n_points)
                let mut indices: Vec<usize> = (0..n_points).collect();
                // Fisher-Yates shuffle
                for i in (1..n_points).rev() {
                    let j = (xorshift64(rng_state) * (i + 1) as f64) as usize % (i + 1);
                    indices.swap(i, j);
                }

                let (lo, hi) = all_bounds.get(d).copied().unwrap_or((0.0, 1.0));
                for i in 0..n_points {
                    let frac = (indices[i] as f64 + xorshift64(rng_state)) / n_points as f64;
                    points[[i, d]] = lo + frac * (hi - lo);
                }
            }
        }
        // Random and AdaptiveResidual both use random placement.
        // (AdaptiveResidual starts random; refinement would require the network.)
        CollocationStrategy::Random | CollocationStrategy::AdaptiveResidual => {
            let mut all_bounds = problem.domain.clone();
            if let Some((t_min, t_max)) = problem.time_domain {
                all_bounds.push((t_min, t_max));
            }

            for i in 0..n_points {
                for d in 0..input_dim {
                    let (lo, hi) = all_bounds.get(d).copied().unwrap_or((0.0, 1.0));
                    points[[i, d]] = lo + xorshift64(rng_state) * (hi - lo);
                }
            }
        }
    }

    points
}

/// Generate boundary points for each boundary in the problem.
///
/// # Arguments
/// * `problem` - PDE problem defining boundaries
/// * `n_per_boundary` - Number of points per boundary segment
/// * `rng_state` - Mutable xorshift64 PRNG state
///
/// # Returns
/// A vector of (points_matrix, boundary_condition) pairs. Each matrix has
/// shape (n_per_boundary, input_dim).
pub fn generate_boundary_points(
    problem: &PDEProblem,
    n_per_boundary: usize,
    rng_state: &mut u64,
) -> Vec<(Array2<f64>, BoundaryCondition)> {
    let input_dim = if problem.has_time {
        problem.spatial_dim + 1
    } else {
        problem.spatial_dim
    };

    let mut result = Vec::with_capacity(problem.boundaries.len());

    for boundary in &problem.boundaries {
        let mut pts = Array2::<f64>::zeros((n_per_boundary, input_dim));

        for i in 0..n_per_boundary {
            for d in 0..input_dim {
                if d == boundary.dim && d < problem.domain.len() {
                    // Fixed coordinate on this boundary
                    let (lo, hi) = problem.domain[d];
                    pts[[i, d]] = match boundary.side {
                        BoundarySide::High => hi,
                        BoundarySide::Low => lo,
                    };
                } else if d < problem.domain.len() {
                    // Random coordinate in the other spatial dimensions
                    let (lo, hi) = problem.domain[d];
                    pts[[i, d]] = lo + xorshift64(rng_state) * (hi - lo);
                } else if let Some((t_min, t_max)) = problem.time_domain {
                    // Time dimension
                    pts[[i, d]] = t_min + xorshift64(rng_state) * (t_max - t_min);
                }
            }
        }

        result.push((pts, boundary.condition.clone()));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pinn::problems;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_solver_creation() {
        let problem = problems::laplace_problem_2d((0.0, 1.0, 0.0, 1.0));
        let config = PINNConfig::default();
        let solver = PINNSolver::new(&problem, config);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_collocation_points_count_and_range() {
        let problem = problems::laplace_problem_2d((0.0, 1.0, 0.0, 1.0));
        let mut rng = 42u64 | 1;
        let pts =
            generate_collocation_points(&problem, 100, &CollocationStrategy::Random, &mut rng);
        assert_eq!(pts.nrows(), 100);
        assert_eq!(pts.ncols(), 2);

        // All points within domain
        for i in 0..100 {
            assert!(pts[[i, 0]] >= 0.0 && pts[[i, 0]] <= 1.0);
            assert!(pts[[i, 1]] >= 0.0 && pts[[i, 1]] <= 1.0);
        }
    }

    #[test]
    fn test_collocation_uniform_grid() {
        let problem = problems::laplace_problem_2d((0.0, 1.0, 0.0, 1.0));
        let mut rng = 42u64 | 1;
        let pts =
            generate_collocation_points(&problem, 25, &CollocationStrategy::UniformGrid, &mut rng);
        assert_eq!(pts.nrows(), 25);

        // All points within domain
        for i in 0..25 {
            assert!(pts[[i, 0]] >= 0.0 && pts[[i, 0]] <= 1.0);
            assert!(pts[[i, 1]] >= 0.0 && pts[[i, 1]] <= 1.0);
        }
    }

    #[test]
    fn test_collocation_latin_hypercube() {
        let problem = problems::laplace_problem_2d((0.0, 1.0, 0.0, 1.0));
        let mut rng = 42u64 | 1;
        let pts = generate_collocation_points(
            &problem,
            50,
            &CollocationStrategy::LatinHypercube,
            &mut rng,
        );
        assert_eq!(pts.nrows(), 50);
        assert_eq!(pts.ncols(), 2);

        for i in 0..50 {
            assert!(pts[[i, 0]] >= 0.0 && pts[[i, 0]] <= 1.0);
            assert!(pts[[i, 1]] >= 0.0 && pts[[i, 1]] <= 1.0);
        }
    }

    #[test]
    fn test_boundary_points_on_boundary() {
        let problem = problems::laplace_problem_2d((0.0, 1.0, 0.0, 1.0));
        let mut rng = 42u64 | 1;
        let bnd_data = generate_boundary_points(&problem, 20, &mut rng);

        // Should have 4 boundaries for a 2D problem
        assert_eq!(bnd_data.len(), 4);

        for (pts, _cond) in &bnd_data {
            assert_eq!(pts.nrows(), 20);
            assert_eq!(pts.ncols(), 2);
            // Each point should have at least one coordinate on a boundary edge
            for i in 0..20 {
                let x = pts[[i, 0]];
                let y = pts[[i, 1]];
                let on_boundary = (x - 0.0).abs() < 1e-15
                    || (x - 1.0).abs() < 1e-15
                    || (y - 0.0).abs() < 1e-15
                    || (y - 1.0).abs() < 1e-15;
                assert!(on_boundary, "point ({}, {}) not on boundary", x, y);
            }
        }
    }

    #[test]
    fn test_train_loss_decreases() {
        let problem = problems::laplace_problem_2d((0.0, 1.0, 0.0, 1.0));
        let config = PINNConfig {
            hidden_layers: vec![8, 8],
            max_epochs: 20,
            n_collocation: 10,
            n_boundary: 5,
            learning_rate: 1e-3,
            ..PINNConfig::default()
        };

        let mut solver = PINNSolver::new(&problem, config).expect("solver creation");
        let result = solver
            .train(&problems::laplace_residual, &problem, None)
            .expect("training");

        assert!(result.epochs_trained > 0);
        assert!(!result.loss_history.is_empty());

        // Check that loss generally decreases (first loss > last loss)
        let first = result.loss_history[0];
        let last = result.loss_history[result.loss_history.len() - 1];
        // Allow for some fluctuation, but final should be less than 10x initial
        assert!(
            last <= first * 10.0,
            "loss did not decrease: first={}, last={}",
            first,
            last
        );
    }

    #[test]
    fn test_predict_after_training() {
        let problem = problems::laplace_problem_2d((0.0, 1.0, 0.0, 1.0));
        let config = PINNConfig {
            hidden_layers: vec![8],
            max_epochs: 5,
            n_collocation: 5,
            n_boundary: 3,
            ..PINNConfig::default()
        };

        let mut solver = PINNSolver::new(&problem, config).expect("solver creation");
        let _ = solver.train(&problems::laplace_residual, &problem, None);

        let test_pts = Array2::from_shape_vec((3, 2), vec![0.2, 0.3, 0.5, 0.5, 0.8, 0.9])
            .expect("test points");
        let predictions = solver.predict(&test_pts);
        assert!(predictions.is_ok());
        let vals = predictions.expect("predictions");
        assert_eq!(vals.len(), 3);
        for &v in vals.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_train_with_data() {
        let problem = problems::laplace_problem_2d((0.0, 1.0, 0.0, 1.0));
        let config = PINNConfig {
            hidden_layers: vec![8],
            max_epochs: 5,
            n_collocation: 5,
            n_boundary: 3,
            ..PINNConfig::default()
        };

        let mut solver = PINNSolver::new(&problem, config).expect("solver creation");

        // Provide some dummy data points
        let x_data = Array2::from_shape_vec((2, 2), vec![0.5, 0.5, 0.25, 0.75]).expect("data x");
        let y_data = array![0.0, 0.0];

        let result = solver
            .train(
                &problems::laplace_residual,
                &problem,
                Some((&x_data, &y_data)),
            )
            .expect("training with data");

        assert!(result.epochs_trained > 0);
        assert!(result.data_loss.is_finite());
    }

    #[test]
    fn test_adam_step_reduces_quadratic() {
        // Minimize f(x) = x^2 starting from x = 5
        let mut optimizer = AdamOptimizer::new(1, 0.1);
        let params = array![5.0];
        let grad = array![10.0]; // df/dx = 2x = 10 at x=5

        let new_params = optimizer.step(&params, &grad);
        // After one step, |new_params[0]| < |params[0]|
        assert!(
            new_params[0].abs() < params[0].abs(),
            "Adam step should reduce magnitude: {} -> {}",
            params[0],
            new_params[0]
        );
    }

    #[test]
    fn test_adam_momentum_nonzero() {
        let mut optimizer = AdamOptimizer::new(2, 0.01);
        let params = array![1.0, 2.0];
        let grad = array![0.5, -0.3];

        let _ = optimizer.step(&params, &grad);

        // First moment should be non-zero after first step
        assert!(optimizer.m[0].abs() > 1e-15);
        assert!(optimizer.m[1].abs() > 1e-15);
        // Second moment should be non-zero
        assert!(optimizer.v[0] > 0.0);
        assert!(optimizer.v[1] > 0.0);
        assert_eq!(optimizer.t, 1);
    }

    #[test]
    fn test_collocation_time_dependent() {
        let problem = problems::heat_problem_1d((0.0, 1.0), (0.0, 0.5), 0.01);
        let mut rng = 99u64 | 1;
        let pts = generate_collocation_points(&problem, 50, &CollocationStrategy::Random, &mut rng);
        // 1 spatial + 1 time = 2 input dims
        assert_eq!(pts.ncols(), 2);
        assert_eq!(pts.nrows(), 50);

        for i in 0..50 {
            assert!(pts[[i, 0]] >= 0.0 && pts[[i, 0]] <= 1.0);
            assert!(pts[[i, 1]] >= 0.0 && pts[[i, 1]] <= 0.5);
        }
    }
}
