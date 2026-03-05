//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::OptimizeError;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::{finite_difference_gradient, finite_difference_hessian};
use crate::unconstrained::Options;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

use super::functions_2::{
    calculate_predicted_reduction, trust_region_lanczos_subproblem, trust_region_subproblem,
};
use super::types::{TrustRegionConfig, TrustRegionResult};

/// Computes the Cauchy point for the trust-region subproblem.
///
/// The Cauchy point is the minimizer of the quadratic model along the
/// steepest descent direction, constrained to lie within the trust region.
///
/// Given the quadratic model:
///   m(p) = f + g^T p + 0.5 p^T B p
///
/// The Cauchy point is:
///   p_c = -tau * (delta / ||g||) * g
///
/// where tau is chosen to minimize the model along the steepest descent direction:
/// - If g^T B g <= 0 (negative curvature), take full step to trust-region boundary: tau = 1
/// - Otherwise, tau = min(||g||^3 / (delta * g^T B g), 1)
///
/// # Arguments
/// * `gradient` - The gradient of the objective function at the current point
/// * `hessian` - The Hessian (or Hessian approximation) at the current point
/// * `trust_radius` - The current trust-region radius
///
/// # Returns
/// A tuple of (cauchy_point, hits_boundary) where hits_boundary indicates
/// if the Cauchy point lies on the trust-region boundary.
pub fn cauchy_point(
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
    trust_radius: f64,
) -> (Array1<f64>, bool) {
    let g_norm = gradient.dot(gradient).sqrt();
    if g_norm < 1e-15 {
        return (Array1::zeros(gradient.len()), false);
    }
    let g_bg = gradient.dot(&hessian.dot(gradient));
    let tau = if g_bg <= 0.0 {
        1.0
    } else {
        let tau_unconstrained = g_norm.powi(3) / (trust_radius * g_bg);
        tau_unconstrained.min(1.0)
    };
    let step_length = tau * trust_radius / g_norm;
    let p_cauchy = gradient * (-step_length);
    let hits_boundary = (tau - 1.0).abs() < 1e-12;
    (p_cauchy, hits_boundary)
}
/// Computes Powell's dogleg step for the trust-region subproblem.
///
/// The dogleg path is a piecewise-linear path from the origin through the
/// Cauchy point to the Newton point (full Newton step). The dogleg step
/// is the point on this path that intersects the trust-region boundary
/// (or the Newton point if it lies within the trust region).
///
/// The path is parameterized as:
/// - For tau in [0, 1]: p(tau) = tau * p_cauchy  (toward the Cauchy point)
/// - For tau in [1, 2]: p(tau) = p_cauchy + (tau - 1) * (p_newton - p_cauchy)
///   (from the Cauchy point toward the Newton point)
///
/// # Arguments
/// * `gradient` - The gradient of the objective function
/// * `hessian` - The Hessian (or Hessian approximation)
/// * `trust_radius` - The current trust-region radius
///
/// # Returns
/// A tuple of (dogleg_step, hits_boundary) where hits_boundary indicates
/// whether the step lies on the trust-region boundary.
pub fn dogleg_step(
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
    trust_radius: f64,
) -> (Array1<f64>, bool) {
    let n = gradient.len();
    let g_norm = gradient.dot(gradient).sqrt();
    if g_norm < 1e-15 {
        return (Array1::zeros(n), false);
    }
    let newton_step = solve_symmetric_system(hessian, &(-gradient.clone()));
    let newton_valid = newton_step.iter().all(|v| v.is_finite());
    if newton_valid {
        let newton_norm = newton_step.dot(&newton_step).sqrt();
        if newton_norm <= trust_radius {
            return (newton_step, false);
        }
    }
    let (p_cauchy, _) = cauchy_point(gradient, hessian, trust_radius);
    let cauchy_norm = p_cauchy.dot(&p_cauchy).sqrt();
    if cauchy_norm >= trust_radius {
        let scale = trust_radius / cauchy_norm;
        return (p_cauchy * scale, true);
    }
    if !newton_valid {
        if cauchy_norm > 1e-15 {
            let scale = trust_radius / cauchy_norm;
            return (p_cauchy * scale, true);
        }
        return (Array1::zeros(n), false);
    }
    let diff = &newton_step - &p_cauchy;
    let diff_norm_sq = diff.dot(&diff);
    let cauchy_dot_diff = p_cauchy.dot(&diff);
    let cauchy_norm_sq = p_cauchy.dot(&p_cauchy);
    let a_coeff = diff_norm_sq;
    let b_coeff = 2.0 * cauchy_dot_diff;
    let c_coeff = cauchy_norm_sq - trust_radius * trust_radius;
    let discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
    let discriminant = discriminant.max(0.0);
    let tau = if a_coeff.abs() < 1e-15 {
        0.0
    } else {
        (-b_coeff + discriminant.sqrt()) / (2.0 * a_coeff)
    };
    let tau = tau.clamp(0.0, 1.0);
    let step = &p_cauchy + &(&diff * tau);
    (step, true)
}
/// Solve a symmetric positive definite system H * x = b using Cholesky-like factorization.
///
/// Falls back to regularized solve if the system is ill-conditioned.
fn solve_symmetric_system(hess: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    let n = hess.nrows();
    let regularization_levels = [0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2];
    for &reg in &regularization_levels {
        let mut l = Array2::<f64>::zeros((n, n));
        let mut success = true;
        for i in 0..n {
            for j in 0..=i {
                let mut sum = hess[[i, j]];
                if i == j {
                    sum += reg;
                }
                for k in 0..j {
                    sum -= l[[i, k]] * l[[j, k]];
                }
                if i == j {
                    if sum <= 0.0 {
                        success = false;
                        break;
                    }
                    l[[i, j]] = sum.sqrt();
                } else {
                    if l[[j, j]].abs() < 1e-15 {
                        success = false;
                        break;
                    }
                    l[[i, j]] = sum / l[[j, j]];
                }
            }
            if !success {
                break;
            }
        }
        if !success {
            continue;
        }
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut sum = rhs[i];
            for j in 0..i {
                sum -= l[[i, j]] * y[j];
            }
            if l[[i, i]].abs() < 1e-15 {
                success = false;
                break;
            }
            y[i] = sum / l[[i, i]];
        }
        if !success {
            continue;
        }
        let mut x = Array1::<f64>::zeros(n);
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= l[[j, i]] * x[j];
            }
            if l[[i, i]].abs() < 1e-15 {
                success = false;
                break;
            }
            x[i] = sum / l[[i, i]];
        }
        if success && x.iter().all(|v| v.is_finite()) {
            return x;
        }
    }
    let max_diag = (0..n).map(|i| hess[[i, i]].abs()).fold(1.0, f64::max);
    rhs / max_diag
}
/// Solve the trust-region subproblem using the dogleg method.
///
/// The subproblem is:
///   minimize   m(p) = g^T p + 0.5 p^T B p
///   subject to ||p|| <= delta
///
/// where g is the gradient, B is the Hessian, and delta is the trust radius.
///
/// This function returns the approximate solution using Powell's dogleg approach.
///
/// # Arguments
/// * `gradient` - The gradient at the current point
/// * `hessian` - The Hessian (or approximation) at the current point
/// * `trust_radius` - The current trust-region radius
///
/// # Returns
/// A tuple of (step, predicted_reduction, hits_boundary).
pub fn solve_trust_subproblem(
    gradient: &Array1<f64>,
    hessian: &Array2<f64>,
    trust_radius: f64,
) -> (Array1<f64>, f64, bool) {
    let (step, hits_boundary) = dogleg_step(gradient, hessian, trust_radius);
    let g_dot_p = gradient.dot(&step);
    let p_bp = step.dot(&hessian.dot(&step));
    let predicted_reduction = -(g_dot_p + 0.5 * p_bp);
    (step, predicted_reduction, hits_boundary)
}
/// Trust-region minimization using the dogleg method with Cauchy point.
///
/// This implements a classical trust-region algorithm that:
/// 1. Computes the gradient and Hessian at the current point
/// 2. Solves the trust-region subproblem using Powell's dogleg method
/// 3. Evaluates the actual vs. predicted reduction ratio
/// 4. Updates the trust-region radius based on the ratio
/// 5. Accepts or rejects the step
///
/// The dogleg method combines the Cauchy (steepest descent) direction with
/// the Newton direction, producing a piecewise-linear path that approximates
/// the full trust-region solution efficiently.
///
/// # Arguments
/// * `objective` - Closure computing the objective function value
/// * `gradient_fn` - Closure computing the gradient (or None to use finite differences)
/// * `hessian_fn` - Closure computing the Hessian (or None to use finite differences)
/// * `x0` - Initial guess
/// * `config` - Optional configuration (uses defaults if None)
///
/// # Returns
/// A `TrustRegionResult` containing the solution and convergence information.
///
/// # Example
/// ```
/// use scirs2_optimize::unconstrained::trust_region::{trust_region_minimize, TrustRegionConfig};
/// use scirs2_core::ndarray::{Array1, ArrayView1};
///
/// // Rosenbrock function
/// let objective = |x: &ArrayView1<f64>| -> f64 {
///     (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
/// };
///
/// let gradient_fn = |x: &ArrayView1<f64>| -> Array1<f64> {
///     let g0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2));
///     let g1 = 200.0 * (x[1] - x[0].powi(2));
///     Array1::from_vec(vec![g0, g1])
/// };
///
/// let hessian_fn = |x: &ArrayView1<f64>| -> scirs2_core::ndarray::Array2<f64> {
///     let h00 = 2.0 - 400.0 * x[1] + 1200.0 * x[0].powi(2);
///     let h01 = -400.0 * x[0];
///     let h11 = 200.0;
///     scirs2_core::ndarray::Array2::from_shape_vec(
///         (2, 2), vec![h00, h01, h01, h11]
///     ).expect("valid shape")
/// };
///
/// let x0 = Array1::from_vec(vec![-1.0, 1.0]);
/// let config = TrustRegionConfig::default();
/// let result = trust_region_minimize(
///     objective,
///     Some(gradient_fn),
///     Some(hessian_fn),
///     x0,
///     Some(config),
/// ).expect("optimization should succeed");
///
/// assert!(result.converged);
/// ```
pub fn trust_region_minimize<F, G, H>(
    mut objective: F,
    gradient_fn: Option<G>,
    hessian_fn: Option<H>,
    x0: Array1<f64>,
    config: Option<TrustRegionConfig>,
) -> Result<TrustRegionResult, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> f64,
    G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    H: Fn(&ArrayView1<f64>) -> Array2<f64>,
{
    let config = config.unwrap_or_default();
    config.validate()?;
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::ValueError(
            "Initial guess must have at least one element".to_string(),
        ));
    }
    let mut x = x0;
    let mut trust_radius = config.initial_radius;
    let mut n_fev: usize = 0;
    let mut n_gev: usize = 0;
    let mut n_hev: usize = 0;
    let mut f_val = objective(&x.view());
    n_fev += 1;
    if !f_val.is_finite() {
        return Err(OptimizeError::ComputationError(
            "Initial function value is not finite".to_string(),
        ));
    }
    let mut grad = match &gradient_fn {
        Some(gf) => {
            n_gev += 1;
            gf(&x.view())
        }
        None => {
            let g = finite_difference_gradient(&mut objective, &x.view(), config.eps)?;
            n_fev += 2 * n;
            g
        }
    };
    let mut converged = false;
    let mut n_iter = 0;
    let mut message = String::new();
    for iter in 0..config.max_iter {
        n_iter = iter + 1;
        let grad_norm = grad.dot(&grad).sqrt();
        if grad_norm < config.tolerance {
            converged = true;
            message = format!(
                "Converged: gradient norm {:.2e} < tolerance {:.2e}",
                grad_norm, config.tolerance
            );
            break;
        }
        let hess = match &hessian_fn {
            Some(hf) => {
                n_hev += 1;
                hf(&x.view())
            }
            None => {
                let h = finite_difference_hessian(&mut objective, &x.view(), config.eps)?;
                n_fev += 1 + 4 * n * (n + 1) / 2;
                h
            }
        };
        let (step, predicted_reduction, hits_boundary) =
            solve_trust_subproblem(&grad, &hess, trust_radius);
        let x_trial = &x + &step;
        let f_trial = objective(&x_trial.view());
        n_fev += 1;
        let actual_reduction = f_val - f_trial;
        let ratio = if predicted_reduction.abs() < 1e-15 {
            if actual_reduction.abs() < 1e-15 {
                1.0
            } else if actual_reduction > 0.0 {
                1.0
            } else {
                0.0
            }
        } else {
            actual_reduction / predicted_reduction
        };
        if ratio < config.eta1 {
            trust_radius *= config.gamma1;
        } else if ratio > config.eta2 && hits_boundary {
            trust_radius = (trust_radius * config.gamma2).min(config.max_radius);
        }
        if ratio > config.eta1 {
            x = x_trial;
            f_val = f_trial;
            grad = match &gradient_fn {
                Some(gf) => {
                    n_gev += 1;
                    gf(&x.view())
                }
                None => {
                    let g = finite_difference_gradient(&mut objective, &x.view(), config.eps)?;
                    n_fev += 2 * n;
                    g
                }
            };
            if actual_reduction.abs() < config.ftol * (1.0 + f_val.abs()) {
                converged = true;
                message = format!(
                    "Converged: function change {:.2e} < ftol {:.2e}",
                    actual_reduction.abs(),
                    config.ftol * (1.0 + f_val.abs())
                );
                break;
            }
        }
        if trust_radius < config.min_radius {
            converged = true;
            message = format!(
                "Converged: trust radius {:.2e} < minimum {:.2e}",
                trust_radius, config.min_radius
            );
            break;
        }
    }
    if !converged {
        message = format!("Maximum iterations ({}) reached", config.max_iter);
    }
    let grad_norm = grad.dot(&grad).sqrt();
    Ok(TrustRegionResult {
        x,
        f_val,
        n_iter,
        converged,
        trust_radius_final: trust_radius,
        n_fev,
        n_gev,
        n_hev,
        grad_norm,
        message,
    })
}
/// Implements the Trust-Region Newton Conjugate Gradient method for optimization
#[allow(dead_code)]
pub fn minimize_trust_ncg<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64> + Clone,
{
    let ftol = options.ftol;
    let gtol = options.gtol;
    let max_iter = options.max_iter;
    let eps = options.eps;
    let initial_trust_radius = options.trust_radius.unwrap_or(1.0);
    let max_trust_radius = options.max_trust_radius.unwrap_or(1000.0);
    let min_trust_radius = options.min_trust_radius.unwrap_or(1e-10);
    let eta = options.trust_eta.unwrap_or(1e-4);
    let n = x0.len();
    let mut x = x0.to_owned();
    let mut nfev = 0;
    let mut f = fun(&x.view()).into();
    nfev += 1;
    let mut g = finite_difference_gradient(&mut fun, &x.view(), eps)?;
    nfev += n;
    let mut trust_radius = initial_trust_radius;
    let mut iter = 0;
    while iter < max_iter {
        let g_norm = g.dot(&g).sqrt();
        if g_norm < gtol {
            break;
        }
        let f_old = f;
        let hess = finite_difference_hessian(&mut fun, &x.view(), eps)?;
        nfev += n * n;
        let (step, hits_boundary) = trust_region_subproblem(&g, &hess, trust_radius);
        let pred_reduction = calculate_predicted_reduction(&g, &hess, &step);
        let x_new = &x + &step;
        let f_new = fun(&x_new.view()).into();
        nfev += 1;
        let actual_reduction = f - f_new;
        let ratio = if pred_reduction.abs() < 1e-8 {
            1.0
        } else {
            actual_reduction / pred_reduction
        };
        if ratio < 0.25 {
            trust_radius *= 0.25;
        } else if ratio > 0.75 && hits_boundary {
            trust_radius = f64::min(2.0 * trust_radius, max_trust_radius);
        }
        if ratio > eta {
            x = x_new;
            f = f_new;
            g = finite_difference_gradient(&mut fun, &x.view(), eps)?;
            nfev += n;
        }
        if trust_radius < min_trust_radius {
            break;
        }
        if ratio > eta && (f_old - f).abs() < ftol * (1.0 + f.abs()) {
            break;
        }
        iter += 1;
    }
    let final_fun = fun(&x.view());
    Ok(OptimizeResult {
        x,
        fun: final_fun,
        nit: iter,
        func_evals: nfev,
        nfev,
        success: iter < max_iter,
        message: if iter < max_iter {
            "Optimization terminated successfully.".to_string()
        } else {
            "Maximum iterations reached.".to_string()
        },
        jacobian: Some(g),
        hessian: None,
    })
}
/// Implements the Trust-Region truncated generalized Lanczos / conjugate gradient algorithm
#[allow(dead_code)]
pub fn minimize_trust_krylov<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S,
    S: Into<f64> + Clone,
{
    let ftol = options.ftol;
    let gtol = options.gtol;
    let max_iter = options.max_iter;
    let eps = options.eps;
    let initial_trust_radius = options.trust_radius.unwrap_or(1.0);
    let max_trust_radius = options.max_trust_radius.unwrap_or(1000.0);
    let min_trust_radius = options.min_trust_radius.unwrap_or(1e-10);
    let eta = options.trust_eta.unwrap_or(1e-4);
    let n = x0.len();
    let mut x = x0.to_owned();
    let mut nfev = 0;
    let mut f = fun(&x.view()).into();
    nfev += 1;
    let mut g = finite_difference_gradient(&mut fun, &x.view(), eps)?;
    nfev += n;
    let mut trust_radius = initial_trust_radius;
    let mut iter = 0;
    while iter < max_iter {
        let g_norm = g.dot(&g).sqrt();
        if g_norm < gtol {
            break;
        }
        let f_old = f;
        let hess = finite_difference_hessian(&mut fun, &x.view(), eps)?;
        nfev += n * n;
        let (step, hits_boundary) = trust_region_lanczos_subproblem(&g, &hess, trust_radius);
        let pred_reduction = calculate_predicted_reduction(&g, &hess, &step);
        let x_new = &x + &step;
        let f_new = fun(&x_new.view()).into();
        nfev += 1;
        let actual_reduction = f - f_new;
        let ratio = if pred_reduction.abs() < 1e-8 {
            1.0
        } else {
            actual_reduction / pred_reduction
        };
        if ratio < 0.25 {
            trust_radius *= 0.25;
        } else if ratio > 0.75 && hits_boundary {
            trust_radius = f64::min(2.0 * trust_radius, max_trust_radius);
        }
        if ratio > eta {
            x = x_new;
            f = f_new;
            g = finite_difference_gradient(&mut fun, &x.view(), eps)?;
            nfev += n;
        }
        if trust_radius < min_trust_radius {
            break;
        }
        if ratio > eta && (f_old - f).abs() < ftol * (1.0 + f.abs()) {
            break;
        }
        iter += 1;
    }
    let final_fun = fun(&x.view());
    Ok(OptimizeResult {
        x,
        fun: final_fun,
        nit: iter,
        func_evals: nfev,
        nfev,
        success: iter < max_iter,
        message: if iter < max_iter {
            "Optimization terminated successfully.".to_string()
        } else {
            "Maximum iterations reached.".to_string()
        },
        jacobian: Some(g),
        hessian: None,
    })
}
