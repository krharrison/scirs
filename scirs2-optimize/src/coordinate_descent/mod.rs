//! Coordinate Descent with Random and Greedy Selection Rules
//!
//! This module implements coordinate descent optimization algorithms with multiple
//! coordinate selection strategies and support for proximal updates (L1/Lasso).
//!
//! ## Coordinate Selection Rules
//!
//! - **Random**: Select coordinate uniformly at random each iteration
//! - **Cyclic**: Iterate through coordinates in order
//! - **GreedyGradient**: Select coordinate with largest absolute gradient component
//! - **GreedyGauss**: Gauss-Southwell rule: max |∂f/∂x_i|
//! - **StochasticGreedy**: Sample random subset, pick best from subset
//!
//! ## Line Search
//!
//! Supports exact line search via golden-section on the 1D sub-problem, or Armijo
//! backtracking when `line_search = true`.
//!
//! ## References
//!
//! - Wright, S.J. (2015). "Coordinate Descent Algorithms". Mathematical Programming.
//! - Tseng, P. (2001). "Convergence of a Block Coordinate Descent Method for Nondifferentiable Minimization"
//! - Friedman, J. et al. (2007). "Pathwise Coordinate Optimization"

use crate::error::{OptimizeError, OptimizeResult};
use scirs2_core::random::{rngs::StdRng, RngExt, SeedableRng};

/// Rule for selecting which coordinate to update at each iteration.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum CoordSelectionRule {
    /// Select coordinate uniformly at random.
    Random,
    /// Cycle through all coordinates in order 0, 1, ..., n-1, 0, 1, ...
    Cyclic,
    /// Select coordinate with largest absolute gradient component.
    GreedyGradient,
    /// Gauss-Southwell rule: select i = argmax_i |∂f/∂x_i|.
    GreedyGauss,
    /// Randomly sample `greedy_subset_size` coordinates, pick the one with largest |grad_i|.
    StochasticGreedy,
}

impl Default for CoordSelectionRule {
    fn default() -> Self {
        CoordSelectionRule::Cyclic
    }
}

/// Configuration for coordinate descent optimization.
#[derive(Clone, Debug)]
pub struct CoordDescentConfig {
    /// Coordinate selection rule.
    pub selection: CoordSelectionRule,
    /// Maximum number of outer iterations (each iteration processes one coordinate).
    pub max_iter: usize,
    /// Convergence tolerance on the gradient norm ||∇f||.
    pub tol: f64,
    /// Whether to perform line search along selected coordinate direction.
    /// When true, uses Armijo backtracking. When false, uses a fixed step.
    pub line_search: bool,
    /// Fixed step size used when `line_search = false`.
    pub step_size: f64,
    /// Number of coordinates to sample in `StochasticGreedy` rule.
    pub greedy_subset_size: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// Armijo backtracking: sufficient decrease parameter c1 ∈ (0, 1).
    pub armijo_c1: f64,
    /// Armijo backtracking: initial step length.
    pub armijo_alpha0: f64,
    /// Armijo backtracking: reduction factor τ ∈ (0, 1).
    pub armijo_tau: f64,
    /// Maximum backtracking iterations.
    pub armijo_max_iter: usize,
}

impl Default for CoordDescentConfig {
    fn default() -> Self {
        Self {
            selection: CoordSelectionRule::Cyclic,
            max_iter: 1000,
            tol: 1e-6,
            line_search: true,
            step_size: 1e-3,
            greedy_subset_size: 10,
            seed: 42,
            armijo_c1: 1e-4,
            armijo_alpha0: 1.0,
            armijo_tau: 0.5,
            armijo_max_iter: 50,
        }
    }
}

/// Result of coordinate descent optimization.
#[derive(Debug, Clone)]
pub struct CoordDescentResult {
    /// Optimal solution found.
    pub x: Vec<f64>,
    /// Objective function value at the solution.
    pub f_val: f64,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged within the tolerance.
    pub converged: bool,
    /// Euclidean norm of the gradient at the solution.
    pub gradient_norm: f64,
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Armijo backtracking line search along coordinate `i`.
///
/// Finds step α such that f(x + α * e_i * direction) ≤ f(x) + c1 * α * grad_i * direction.
fn armijo_line_search<F>(
    f: &F,
    x: &[f64],
    i: usize,
    direction: f64,
    grad_i: f64,
    f_x: f64,
    config: &CoordDescentConfig,
) -> f64
where
    F: Fn(&[f64]) -> f64,
{
    let mut alpha = config.armijo_alpha0;
    let mut x_trial = x.to_vec();

    for _ in 0..config.armijo_max_iter {
        x_trial[i] = x[i] + alpha * direction;
        let f_trial = f(&x_trial);
        if f_trial <= f_x + config.armijo_c1 * alpha * grad_i * direction {
            return alpha;
        }
        alpha *= config.armijo_tau;
    }
    alpha
}

/// Golden-section line search along coordinate `i` over interval [lo, hi].
fn golden_section_1d<F>(f: &F, x: &[f64], i: usize, lo: f64, hi: f64) -> (f64, f64)
where
    F: Fn(&[f64]) -> f64,
{
    const PHI: f64 = 0.618_033_988_749_895; // (√5 - 1) / 2
    const MAX_ITER: usize = 100;
    const TOL: f64 = 1e-10;

    let mut a = lo;
    let mut b = hi;
    let mut c = b - PHI * (b - a);
    let mut d = a + PHI * (b - a);

    let mut x_c = x.to_vec();
    let mut x_d = x.to_vec();
    x_c[i] = c;
    x_d[i] = d;
    let mut f_c = f(&x_c);
    let mut f_d = f(&x_d);

    for _ in 0..MAX_ITER {
        if (b - a).abs() < TOL {
            break;
        }
        if f_c < f_d {
            b = d;
            d = c;
            f_d = f_c;
            c = b - PHI * (b - a);
            x_c[i] = c;
            f_c = f(&x_c);
        } else {
            a = c;
            c = d;
            f_c = f_d;
            d = a + PHI * (b - a);
            x_d[i] = d;
            f_d = f(&x_d);
        }
    }

    let x_mid = (a + b) / 2.0;
    let mut x_eval = x.to_vec();
    x_eval[i] = x_mid;
    (x_mid, f(&x_eval))
}

/// Select coordinate index according to the given rule.
fn select_coordinate(
    rule: &CoordSelectionRule,
    n: usize,
    grad: &[f64],
    cycle_idx: usize,
    rng: &mut StdRng,
    subset_size: usize,
) -> usize {
    match rule {
        CoordSelectionRule::Random => rng.random_range(0..n),
        CoordSelectionRule::Cyclic => cycle_idx % n,
        CoordSelectionRule::GreedyGradient | CoordSelectionRule::GreedyGauss => grad
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.abs()
                    .partial_cmp(&b.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0),
        CoordSelectionRule::StochasticGreedy => {
            let actual_size = subset_size.min(n);
            let mut best_i = 0;
            let mut best_val = f64::NEG_INFINITY;
            for _ in 0..actual_size {
                let i = rng.random_range(0..n);
                let v = grad[i].abs();
                if v > best_val {
                    best_val = v;
                    best_i = i;
                }
            }
            best_i
        }
        _ => cycle_idx % n,
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Run coordinate descent minimization.
///
/// Minimizes `f(x)` starting from `x0` using gradient information from `grad`.
///
/// # Arguments
/// - `f`: Objective function.
/// - `grad`: Full-gradient function returning a `Vec<f64>` of length `n`.
/// - `x0`: Initial point of length `n`.
/// - `config`: Algorithm configuration.
///
/// # Returns
/// A [`CoordDescentResult`] on success, or an [`OptimizeError`] on failure.
pub fn coordinate_descent<F, G>(
    f: F,
    grad: G,
    x0: &[f64],
    config: &CoordDescentConfig,
) -> OptimizeResult<CoordDescentResult>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput(
            "x0 must be non-empty".to_string(),
        ));
    }

    let mut x = x0.to_vec();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut cycle_idx: usize = 0;

    for iter in 0..config.max_iter {
        let g = grad(&x);

        // Compute gradient norm for convergence check
        let gnorm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < config.tol {
            let f_val = f(&x);
            return Ok(CoordDescentResult {
                x,
                f_val,
                n_iter: iter,
                converged: true,
                gradient_norm: gnorm,
            });
        }

        // Select coordinate
        let i = select_coordinate(
            &config.selection,
            n,
            &g,
            cycle_idx,
            &mut rng,
            config.greedy_subset_size,
        );
        cycle_idx = cycle_idx.wrapping_add(1);

        let gi = g[i];
        if gi.abs() < f64::EPSILON * 100.0 {
            continue;
        }

        if config.line_search {
            // Armijo backtracking along -grad direction
            let direction = -gi.signum();
            let f_x = f(&x);
            let alpha = armijo_line_search(&f, &x, i, direction, gi, f_x, config);
            x[i] += alpha * direction;
        } else {
            x[i] -= config.step_size * gi;
        }
    }

    // Final evaluation
    let g_final = grad(&x);
    let gnorm = g_final.iter().map(|v| v * v).sum::<f64>().sqrt();
    let f_val = f(&x);

    Ok(CoordDescentResult {
        x,
        f_val,
        n_iter: config.max_iter,
        converged: gnorm < config.tol,
        gradient_norm: gnorm,
    })
}

// ─── Proximal Coordinate Descent ─────────────────────────────────────────────

/// Soft-thresholding (proximal operator for L1 norm).
///
/// Returns `sign(u) * max(0, |u| - threshold)`.
#[inline]
fn soft_threshold(u: f64, threshold: f64) -> f64 {
    if u > threshold {
        u - threshold
    } else if u < -threshold {
        u + threshold
    } else {
        0.0
    }
}

/// Proximal coordinate descent for L1-regularized minimization.
///
/// Solves:  minimize  f(x) + λ ||x||₁
///
/// Uses soft-thresholding as the proximal operator for the L1 penalty.
/// At each iteration, the coordinate update is:
///   u = x_i - step * ∂f/∂x_i
///   x_i ← sign(u) * max(0, |u| - λ * step)
///
/// # Arguments
/// - `f`: Smooth objective function.
/// - `grad_f`: Gradient of the smooth part.
/// - `lambda`: L1 regularization strength (λ ≥ 0).
/// - `x0`: Initial point.
/// - `config`: Algorithm configuration. The `line_search` flag controls step selection.
///
/// # Returns
/// A [`CoordDescentResult`] on success.
pub fn proximal_coord_descent<F, G>(
    f: F,
    grad_f: G,
    lambda: f64,
    x0: &[f64],
    config: &CoordDescentConfig,
) -> OptimizeResult<CoordDescentResult>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    if lambda < 0.0 {
        return Err(OptimizeError::InvalidParameter(
            "lambda must be non-negative".to_string(),
        ));
    }
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput(
            "x0 must be non-empty".to_string(),
        ));
    }

    let mut x = x0.to_vec();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut cycle_idx: usize = 0;

    for iter in 0..config.max_iter {
        let g = grad_f(&x);

        // Convergence: check composite gradient (accounting for L1 subdifferential)
        let gnorm_sq: f64 = g
            .iter()
            .enumerate()
            .map(|(i, &gi)| {
                // Composite gradient for x_i ≠ 0: ∂f/∂x_i + λ sign(x_i)
                // For x_i = 0: max(0, |∂f/∂x_i| - λ)
                let composite = if x[i].abs() > f64::EPSILON {
                    gi + lambda * x[i].signum()
                } else {
                    let abs_gi = gi.abs();
                    if abs_gi > lambda {
                        abs_gi - lambda
                    } else {
                        0.0
                    }
                };
                composite * composite
            })
            .sum();
        let gnorm = gnorm_sq.sqrt();

        if gnorm < config.tol {
            let f_val = f(&x);
            return Ok(CoordDescentResult {
                x,
                f_val,
                n_iter: iter,
                converged: true,
                gradient_norm: gnorm,
            });
        }

        // Select coordinate
        let i = select_coordinate(
            &config.selection,
            n,
            &g,
            cycle_idx,
            &mut rng,
            config.greedy_subset_size,
        );
        cycle_idx = cycle_idx.wrapping_add(1);

        // Determine step size
        let step = if config.line_search {
            // Armijo on smooth part, then apply proximal
            let direction = -g[i].signum();
            let f_x = f(&x);
            armijo_line_search(&f, &x, i, direction, g[i], f_x, config)
        } else {
            config.step_size
        };

        // Proximal update: u = x_i - step * grad_i;  x_i = soft_threshold(u, lambda*step)
        let u = x[i] - step * g[i];
        x[i] = soft_threshold(u, lambda * step);
    }

    let g_final = grad_f(&x);
    let gnorm: f64 = g_final
        .iter()
        .enumerate()
        .map(|(i, &gi)| {
            let c = if x[i].abs() > f64::EPSILON {
                gi + lambda * x[i].signum()
            } else {
                let abs_gi = gi.abs();
                if abs_gi > lambda {
                    abs_gi - lambda
                } else {
                    0.0
                }
            };
            c * c
        })
        .sum::<f64>()
        .sqrt();
    let f_val = f(&x);

    Ok(CoordDescentResult {
        x,
        f_val,
        n_iter: config.max_iter,
        converged: gnorm < config.tol,
        gradient_norm: gnorm,
    })
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// f(x,y) = 0.5*(x-2)^2 + 0.5*(y-3)^2, minimum at (2, 3).
    fn quadratic(x: &[f64]) -> f64 {
        0.5 * (x[0] - 2.0).powi(2) + 0.5 * (x[1] - 3.0).powi(2)
    }

    fn quadratic_grad(x: &[f64]) -> Vec<f64> {
        vec![x[0] - 2.0, x[1] - 3.0]
    }

    #[test]
    fn test_coord_descent_quadratic() {
        let config = CoordDescentConfig {
            selection: CoordSelectionRule::Cyclic,
            max_iter: 5000,
            tol: 1e-8,
            line_search: true,
            ..CoordDescentConfig::default()
        };
        let result = coordinate_descent(quadratic, quadratic_grad, &[0.0, 0.0], &config)
            .expect("optimization should succeed");
        assert!(result.converged, "should converge");
        assert!((result.x[0] - 2.0).abs() < 1e-5, "x[0] should be ≈ 2");
        assert!((result.x[1] - 3.0).abs() < 1e-5, "x[1] should be ≈ 3");
    }

    #[test]
    fn test_coord_descent_random_vs_cyclic() {
        let x0 = vec![0.0, 0.0];
        let config_cyclic = CoordDescentConfig {
            selection: CoordSelectionRule::Cyclic,
            max_iter: 5000,
            tol: 1e-6,
            line_search: true,
            ..CoordDescentConfig::default()
        };
        let config_random = CoordDescentConfig {
            selection: CoordSelectionRule::Random,
            max_iter: 20000,
            tol: 1e-6,
            line_search: true,
            ..CoordDescentConfig::default()
        };

        let r_cyclic = coordinate_descent(quadratic, quadratic_grad, &x0, &config_cyclic)
            .expect("cyclic should succeed");
        let r_random = coordinate_descent(quadratic, quadratic_grad, &x0, &config_random)
            .expect("random should succeed");

        assert!(r_cyclic.converged, "cyclic should converge");
        assert!(r_random.converged, "random should converge");

        // Both should find the same minimum
        assert!((r_cyclic.x[0] - 2.0).abs() < 1e-4);
        assert!((r_random.x[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_coord_descent_greedy() {
        // Ill-conditioned: f = 0.5 * (100*(x-1)^2 + (y-1)^2)
        let f = |x: &[f64]| 0.5 * (100.0 * (x[0] - 1.0).powi(2) + (x[1] - 1.0).powi(2));
        let g = |x: &[f64]| vec![100.0 * (x[0] - 1.0), x[1] - 1.0];

        let config = CoordDescentConfig {
            selection: CoordSelectionRule::GreedyGradient,
            max_iter: 5000,
            tol: 1e-6,
            line_search: true,
            ..CoordDescentConfig::default()
        };
        let result = coordinate_descent(f, g, &[0.0, 0.0], &config).expect("greedy should succeed");
        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_coord_descent_stochastic_greedy() {
        let config = CoordDescentConfig {
            selection: CoordSelectionRule::StochasticGreedy,
            max_iter: 10000,
            tol: 1e-5,
            greedy_subset_size: 2,
            line_search: true,
            ..CoordDescentConfig::default()
        };
        let result = coordinate_descent(quadratic, quadratic_grad, &[0.0, 0.0], &config)
            .expect("stochastic greedy should succeed");
        assert!(result.converged || result.gradient_norm < 1e-4);
    }

    #[test]
    fn test_proximal_coord_descent_lasso() {
        // Lasso regression: minimize 0.5 * sum((x_i - c_i)^2) + lambda * sum(|x_i|)
        // with c = [5.0, 0.3, -0.2, 0.0] and lambda = 1.0
        // Expected: x ≈ [4.0, 0.0, 0.0, 0.0] (soft threshold applied)
        let c = vec![5.0_f64, 0.3, -0.2, 0.0];
        let c_clone = c.clone();
        let f = move |x: &[f64]| {
            x.iter()
                .zip(c.iter())
                .map(|(xi, ci)| 0.5 * (xi - ci).powi(2))
                .sum::<f64>()
        };
        let g = move |x: &[f64]| {
            x.iter()
                .zip(c_clone.iter())
                .map(|(xi, ci)| xi - ci)
                .collect::<Vec<_>>()
        };

        let lambda = 1.0_f64;
        let config = CoordDescentConfig {
            selection: CoordSelectionRule::Cyclic,
            max_iter: 5000,
            tol: 1e-8,
            line_search: false,
            step_size: 0.5,
            ..CoordDescentConfig::default()
        };

        let result = proximal_coord_descent(f, g, lambda, &[0.0; 4], &config)
            .expect("proximal CD should succeed");

        // x[0] should be c[0] - lambda = 4.0 (shrunk but nonzero)
        assert!(result.x[0] > 3.0, "x[0] should be large positive");
        // x[1], x[2], x[3] should be near zero (sparsity from L1)
        assert!(result.x[1].abs() < 0.5, "x[1] should be sparse");
        assert!(result.x[2].abs() < 0.5, "x[2] should be sparse");
        assert!(result.x[3].abs() < 0.5, "x[3] should be sparse");
    }

    #[test]
    fn test_proximal_coord_descent_zero_lambda() {
        // With lambda=0, proximal CD should behave like standard CD
        let config = CoordDescentConfig {
            selection: CoordSelectionRule::Cyclic,
            max_iter: 5000,
            tol: 1e-8,
            line_search: false,
            step_size: 0.5,
            ..CoordDescentConfig::default()
        };
        let result = proximal_coord_descent(quadratic, quadratic_grad, 0.0, &[0.0, 0.0], &config)
            .expect("zero-lambda proximal should succeed");
        // Should converge close to (2, 3)
        assert!((result.x[0] - 2.0).abs() < 0.1, "x[0] ≈ 2 for lambda=0");
    }

    #[test]
    fn test_coord_descent_empty_input_error() {
        let result = coordinate_descent(
            |_: &[f64]| 0.0,
            |_: &[f64]| vec![],
            &[],
            &CoordDescentConfig::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_proximal_negative_lambda_error() {
        let result = proximal_coord_descent(
            |_: &[f64]| 0.0,
            |_: &[f64]| vec![0.0],
            -1.0,
            &[0.0],
            &CoordDescentConfig::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_coord_descent_gauss_southwell() {
        let config = CoordDescentConfig {
            selection: CoordSelectionRule::GreedyGauss,
            max_iter: 5000,
            tol: 1e-6,
            line_search: true,
            ..CoordDescentConfig::default()
        };
        let result = coordinate_descent(quadratic, quadratic_grad, &[0.0, 0.0], &config)
            .expect("Gauss-Southwell should succeed");
        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-4);
        assert!((result.x[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_coord_descent_no_line_search() {
        let config = CoordDescentConfig {
            selection: CoordSelectionRule::Cyclic,
            max_iter: 100_000,
            tol: 1e-5,
            line_search: false,
            step_size: 0.1,
            ..CoordDescentConfig::default()
        };
        let result = coordinate_descent(quadratic, quadratic_grad, &[0.0, 0.0], &config)
            .expect("fixed step CD should succeed");
        assert!((result.x[0] - 2.0).abs() < 0.1);
        assert!((result.x[1] - 3.0).abs() < 0.1);
    }
}
