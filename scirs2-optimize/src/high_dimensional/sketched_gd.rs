//! Sketched Gradient Descent
//!
//! Dimensionality reduction via random sketching applied to gradient descent.
//! Instead of using the full gradient in R^n, the gradient is projected into
//! a lower-dimensional sketch space of dimension k << n, reducing computation
//! and communication costs.
//!
//! ## Sketch Types
//!
//! - **Gaussian sketch**: Multiply gradient by a random Gaussian matrix S in R^{k x n}
//! - **Count sketch**: Hash-based dimensionality reduction (sparse, fast)
//! - **Sketch-and-project**: Project iterate onto solution of sketched system
//!
//! ## Reference
//!
//! - Gower, R.M. and Richtarik, P. (2015). "Randomized Iterative Methods
//!   for Linear Systems"

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

/// Type of random sketch to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SketchType {
    /// Gaussian sketch: entries of S are iid N(0, 1/k)
    Gaussian,
    /// Count sketch: each column of S has exactly one nonzero entry (+1 or -1)
    CountSketch,
}

/// Configuration for sketched gradient descent
#[derive(Debug, Clone)]
pub struct SketchedGdConfig {
    /// Sketch dimension k (number of rows in sketch matrix)
    pub sketch_dim: usize,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Step size (learning rate)
    pub step_size: f64,
    /// Type of sketch
    pub sketch_type: SketchType,
    /// Random seed
    pub seed: u64,
    /// Whether to regenerate the sketch matrix each iteration
    pub resample_sketch: bool,
    /// Whether to track objective history
    pub track_objective: bool,
}

impl Default for SketchedGdConfig {
    fn default() -> Self {
        Self {
            sketch_dim: 10,
            max_iter: 1000,
            tol: 1e-6,
            step_size: 0.01,
            sketch_type: SketchType::Gaussian,
            seed: 42,
            resample_sketch: true,
            track_objective: false,
        }
    }
}

/// Result of sketched gradient descent
#[derive(Debug, Clone)]
pub struct SketchedGdResult {
    /// Solution vector
    pub x: Array1<f64>,
    /// Final objective value
    pub fun: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Whether converged
    pub converged: bool,
    /// Objective value history (if tracking enabled)
    pub objective_history: Vec<f64>,
    /// Final gradient norm
    pub grad_norm: f64,
}

/// Generate a Gaussian sketch matrix of shape (k x n) with entries ~ N(0, 1/k)
fn generate_gaussian_sketch(k: usize, n: usize, rng: &mut StdRng) -> Array2<f64> {
    let scale = 1.0 / (k as f64).sqrt();
    let mut s = Array2::zeros((k, n));
    for i in 0..k {
        for j in 0..n {
            // Box-Muller transform for standard normal
            let u1: f64 = rng.random::<f64>().max(1e-30);
            let u2: f64 = rng.random::<f64>();
            let z: f64 = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
            s[[i, j]] = z * scale;
        }
    }
    s
}

/// Generate a count sketch matrix of shape (k x n)
///
/// Each column j has exactly one nonzero entry at a random row h(j),
/// with value +1 or -1 with equal probability.
fn generate_count_sketch(k: usize, n: usize, rng: &mut StdRng) -> Array2<f64> {
    let mut s = Array2::zeros((k, n));
    for j in 0..n {
        let row = rng.random_range(0..k);
        let sign: f64 = if rng.random_range(0..2_u32) == 0 {
            1.0
        } else {
            -1.0
        };
        s[[row, j]] = sign;
    }
    s
}

/// Sketched Gradient Descent solver
///
/// Minimizes f(x) using gradient descent where the gradient is compressed
/// via a random sketch before being applied as an update direction.
///
/// The update rule is:
///   x_{t+1} = x_t - eta * S^T * S * grad f(x_t)
///
/// where S is a k x n sketch matrix and eta is the step size.
pub struct SketchedGradientDescent {
    config: SketchedGdConfig,
}

impl SketchedGradientDescent {
    /// Create a new sketched gradient descent solver
    pub fn new(config: SketchedGdConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default_solver() -> Self {
        Self::new(SketchedGdConfig::default())
    }

    /// Minimize the objective function
    ///
    /// # Arguments
    /// * `objective` - Objective function f(x) -> f64
    /// * `gradient` - Gradient function grad f(x) -> `Array1<f64>`
    /// * `x0` - Initial point
    pub fn minimize<F, G>(
        &self,
        objective: F,
        gradient: G,
        x0: &Array1<f64>,
    ) -> OptimizeResult<SketchedGdResult>
    where
        F: Fn(&ArrayView1<f64>) -> f64,
        G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    {
        let n = x0.len();
        let k = self.config.sketch_dim.min(n);

        if n == 0 {
            return Err(OptimizeError::InvalidInput(
                "Dimension must be at least 1".to_string(),
            ));
        }
        if k == 0 {
            return Err(OptimizeError::InvalidInput(
                "Sketch dimension must be at least 1".to_string(),
            ));
        }

        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut x = x0.clone();
        let mut objective_history = Vec::new();

        // Generate initial sketch
        let mut sketch = match self.config.sketch_type {
            SketchType::Gaussian => generate_gaussian_sketch(k, n, &mut rng),
            SketchType::CountSketch => generate_count_sketch(k, n, &mut rng),
        };

        let mut prev_obj = objective(&x.view());
        if self.config.track_objective {
            objective_history.push(prev_obj);
        }

        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            // Optionally resample sketch
            if self.config.resample_sketch && iter > 0 {
                sketch = match self.config.sketch_type {
                    SketchType::Gaussian => generate_gaussian_sketch(k, n, &mut rng),
                    SketchType::CountSketch => generate_count_sketch(k, n, &mut rng),
                };
            }

            // Compute full gradient
            let grad = gradient(&x.view());

            // Sketch the gradient: s_grad = S * grad (k-dimensional)
            let s_grad = sketch.dot(&grad);

            // Lift back: direction = S^T * s_grad (n-dimensional)
            let direction = sketch.t().dot(&s_grad);

            // Update: x <- x - step_size * direction
            for j in 0..n {
                x[j] -= self.config.step_size * direction[j];
            }

            let cur_obj = objective(&x.view());
            if self.config.track_objective {
                objective_history.push(cur_obj);
            }

            let change = (prev_obj - cur_obj).abs();
            prev_obj = cur_obj;

            if change < self.config.tol {
                converged = true;
                break;
            }
        }

        let final_grad = gradient(&x.view());
        let grad_norm = final_grad.dot(&final_grad).sqrt();

        Ok(SketchedGdResult {
            x,
            fun: prev_obj,
            iterations,
            converged,
            objective_history,
            grad_norm,
        })
    }

    /// Sketch-and-project for quadratic: min 0.5 x^T A x - b^T x
    ///
    /// Each iteration projects the current iterate onto the solution set
    /// of a sketched system: S*A*x = S*b, where S is a sketch matrix.
    ///
    /// # Arguments
    /// * `a` - Symmetric positive definite matrix
    /// * `b` - Linear term
    /// * `x0` - Initial point
    pub fn sketch_and_project(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
        x0: &Array1<f64>,
    ) -> OptimizeResult<SketchedGdResult> {
        let n = x0.len();
        let k = self.config.sketch_dim.min(n);

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

        let mut rng = StdRng::seed_from_u64(self.config.seed);
        let mut x = x0.clone();
        let mut objective_history = Vec::new();

        let compute_obj = |x: &Array1<f64>| -> f64 {
            let ax = a.dot(x);
            0.5 * x.dot(&ax) - b.dot(x)
        };

        let mut prev_obj = compute_obj(&x);
        if self.config.track_objective {
            objective_history.push(prev_obj);
        }

        let mut converged = false;
        let mut iterations = 0;

        for iter in 0..self.config.max_iter {
            iterations = iter + 1;

            // Generate sketch
            let sketch = match self.config.sketch_type {
                SketchType::Gaussian => generate_gaussian_sketch(k, n, &mut rng),
                SketchType::CountSketch => generate_count_sketch(k, n, &mut rng),
            };

            // Gradient: Ax - b
            let grad = a.dot(&x) - b;

            // Sketched gradient: S * grad
            let s_grad = sketch.dot(&grad);

            // Sketched Hessian: S * A * S^T (k x k)
            let sa = sketch.dot(a); // k x n
            let sa_st = sa.dot(&sketch.t()); // k x n * n x k = k x k

            // Solve (S A S^T) z = S * grad for z
            let z = match solve_sketched_system(&sa_st, &s_grad) {
                Some(z) => z,
                None => {
                    // Fallback to simple sketched gradient step
                    let direction = sketch.t().dot(&s_grad);
                    for j in 0..n {
                        x[j] -= self.config.step_size * direction[j];
                    }
                    let cur_obj = compute_obj(&x);
                    if self.config.track_objective {
                        objective_history.push(cur_obj);
                    }
                    prev_obj = cur_obj;
                    continue;
                }
            };

            // Update: x <- x - S^T * z
            let update = sketch.t().dot(&z);
            for j in 0..n {
                x[j] -= update[j];
            }

            let cur_obj = compute_obj(&x);
            if self.config.track_objective {
                objective_history.push(cur_obj);
            }

            let change = (prev_obj - cur_obj).abs();
            prev_obj = cur_obj;

            if change < self.config.tol {
                converged = true;
                break;
            }
        }

        let final_grad = a.dot(&x) - b;
        let grad_norm = final_grad.dot(&final_grad).sqrt();

        Ok(SketchedGdResult {
            x,
            fun: prev_obj,
            iterations,
            converged,
            objective_history,
            grad_norm,
        })
    }
}

/// Solve a small system via Gaussian elimination (same as in kaczmarz.rs)
fn solve_sketched_system(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    let n = a.nrows();
    if n == 0 || a.ncols() != n || b.len() != n {
        return None;
    }

    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    for col in 0..n {
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return None;
        }

        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let val = aug[[col, j]];
                aug[[row, j]] -= factor * val;
            }
        }
    }

    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        if aug[[i, i]].abs() < 1e-14 {
            return None;
        }
        x[i] = sum / aug[[i, i]];
    }

    Some(x)
}

/// Convenience function: minimize using sketched gradient descent
pub fn sketched_gradient_descent<F, G>(
    objective: F,
    gradient: G,
    x0: &Array1<f64>,
    config: Option<SketchedGdConfig>,
) -> OptimizeResult<SketchedGdResult>
where
    F: Fn(&ArrayView1<f64>) -> f64,
    G: Fn(&ArrayView1<f64>) -> Array1<f64>,
{
    let config = config.unwrap_or_default();
    let solver = SketchedGradientDescent::new(config);
    solver.minimize(objective, gradient, x0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1, Array2};

    /// Test 1: Simple quadratic convergence with Gaussian sketch
    #[test]
    fn test_gaussian_sketch_quadratic() {
        let objective = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };
        let gradient = |x: &ArrayView1<f64>| -> Array1<f64> { array![2.0 * x[0], 2.0 * x[1]] };

        let x0 = array![5.0, 3.0];
        let config = SketchedGdConfig {
            sketch_dim: 2,
            max_iter: 5000,
            tol: 1e-10,
            step_size: 0.1,
            sketch_type: SketchType::Gaussian,
            seed: 42,
            resample_sketch: false, // Fixed sketch for consistency
            ..Default::default()
        };

        let result = sketched_gradient_descent(objective, gradient, &x0, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(result.converged, "Did not converge, fun={}", result.fun);
        assert!(result.fun < 1e-4, "fun={}", result.fun);
    }

    /// Test 2: Count sketch convergence
    #[test]
    fn test_count_sketch_convergence() {
        let objective = |x: &ArrayView1<f64>| -> f64 {
            (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2) + (x[2] - 3.0).powi(2)
        };
        let gradient = |x: &ArrayView1<f64>| -> Array1<f64> {
            array![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0), 2.0 * (x[2] - 3.0)]
        };

        let x0 = array![0.0, 0.0, 0.0];
        let config = SketchedGdConfig {
            sketch_dim: 3,
            max_iter: 10000,
            tol: 1e-8,
            step_size: 0.1,
            sketch_type: SketchType::CountSketch,
            seed: 123,
            resample_sketch: true,
            ..Default::default()
        };

        let result = sketched_gradient_descent(objective, gradient, &x0, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        // Should get reasonably close
        assert!(result.fun < 1.0, "fun={}", result.fun);
    }

    /// Test 3: Sketch-and-project for quadratic
    #[test]
    fn test_sketch_and_project() {
        let a = array![[2.0, 0.5], [0.5, 3.0]];
        let b = array![1.0, 2.0];
        let x0 = array![0.0, 0.0];

        let config = SketchedGdConfig {
            sketch_dim: 2,
            max_iter: 200,
            tol: 1e-10,
            sketch_type: SketchType::Gaussian,
            seed: 42,
            ..Default::default()
        };

        let solver = SketchedGradientDescent::new(config);
        let result = solver.sketch_and_project(&a, &b, &x0);
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(result.converged, "Did not converge, fun={}", result.fun);

        // Verify solution is close to A^{-1} b
        let residual = a.dot(&result.x) - &b;
        let res_norm = residual.dot(&residual).sqrt();
        assert!(res_norm < 1e-4, "Residual norm={}", res_norm);
    }

    /// Test 4: Higher-dimensional problem
    #[test]
    fn test_high_dimensional() {
        let n = 20;
        // Diagonal quadratic: f(x) = sum x_i^2
        let objective = |x: &ArrayView1<f64>| -> f64 { x.dot(x) };
        let gradient = |x: &ArrayView1<f64>| -> Array1<f64> { x.mapv(|xi| 2.0 * xi) };

        let x0 = Array1::from_vec(vec![1.0; n]);
        let config = SketchedGdConfig {
            sketch_dim: 5, // k << n
            max_iter: 10000,
            tol: 1e-6,
            step_size: 0.05,
            sketch_type: SketchType::Gaussian,
            seed: 77,
            resample_sketch: true,
            ..Default::default()
        };

        let result = sketched_gradient_descent(objective, gradient, &x0, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(result.fun < 1.0, "fun={}, expected < 1.0", result.fun);
    }

    /// Test 5: Objective history is monotonically decreasing (fixed sketch)
    #[test]
    fn test_objective_history() {
        let objective = |x: &ArrayView1<f64>| -> f64 { x[0] * x[0] + x[1] * x[1] };
        let gradient = |x: &ArrayView1<f64>| -> Array1<f64> { array![2.0 * x[0], 2.0 * x[1]] };

        let x0 = array![3.0, 4.0];
        let config = SketchedGdConfig {
            sketch_dim: 2,
            max_iter: 50,
            tol: 1e-20,
            step_size: 0.05,
            sketch_type: SketchType::Gaussian,
            seed: 42,
            resample_sketch: false, // Fixed sketch => deterministic descent
            track_objective: true,
            ..Default::default()
        };

        let result = sketched_gradient_descent(objective, gradient, &x0, Some(config));
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert!(result.objective_history.len() > 1);

        // With fixed sketch and small step size, objective should decrease
        for i in 1..result.objective_history.len() {
            assert!(
                result.objective_history[i] <= result.objective_history[i - 1] + 1e-10,
                "Objective increased at iter {}: {} -> {}",
                i,
                result.objective_history[i - 1],
                result.objective_history[i]
            );
        }
    }

    /// Test 6: Zero-dimensional input error
    #[test]
    fn test_zero_dimension_error() {
        let objective = |_x: &ArrayView1<f64>| -> f64 { 0.0 };
        let gradient = |_x: &ArrayView1<f64>| -> Array1<f64> { Array1::zeros(0) };

        let x0 = Array1::zeros(0);
        let result = sketched_gradient_descent(objective, gradient, &x0, None);
        assert!(result.is_err());
    }

    /// Test 7: Sketch-and-project dimension mismatch
    #[test]
    fn test_sketch_project_mismatch() {
        let a = Array2::eye(3);
        let b = array![1.0, 2.0]; // Wrong dimension
        let x0 = array![0.0, 0.0, 0.0];

        let solver = SketchedGradientDescent::default_solver();
        let result = solver.sketch_and_project(&a, &b, &x0);
        assert!(result.is_err());
    }
}
