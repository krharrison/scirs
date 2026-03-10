//! BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm for unconstrained optimization

use crate::error::OptimizeError;
use crate::unconstrained::line_search::backtracking_line_search;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::{
    array_diff_norm, check_convergence, compute_gradient_with_jacobian, finite_difference_gradient,
};
use crate::unconstrained::{Jacobian, Options};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};

/// Implements the BFGS algorithm with optional bounds support and a `Jacobian` enum
/// for gradient computation.
///
/// This is the core implementation. Both `minimize_bfgs` and `minimize_bfgs_no_grad`
/// delegate to this function.
#[allow(dead_code)]
pub fn minimize_bfgs_with_jacobian<F, S>(
    mut fun: F,
    x0: Array1<f64>,
    jacobian: Option<&Jacobian<'_>>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S + Clone,
    S: Into<f64> + Clone,
{
    // Get options or use defaults
    let ftol = options.ftol;
    let gtol = options.gtol;
    let max_iter = options.max_iter;
    let eps = options.eps;
    let bounds = options.bounds.as_ref();

    // Initialize variables
    let n = x0.len();
    let mut x = x0.to_owned();

    // Ensure initial point is within bounds
    if let Some(bounds) = bounds {
        let slice = x.as_slice_mut().ok_or_else(|| {
            OptimizeError::ComputationError(
                "Failed to get mutable slice for bounds projection".to_string(),
            )
        })?;
        bounds.project(slice);
    }

    let mut f = fun(&x.view()).into();

    // Calculate initial gradient
    let mut nfev: usize = 1; // count the initial f evaluation
    let mut g = match jacobian {
        Some(jac) => compute_gradient_with_jacobian(&mut fun, &x.view(), jac, eps, &mut nfev)?,
        None => {
            nfev += n;
            finite_difference_gradient(&mut fun, &x.view(), eps)?
        }
    };

    // Initialize approximation of inverse Hessian with identity matrix
    let mut h_inv = Array2::eye(n);

    // Initialize iteration counter
    let mut iter = 0;

    // Main loop
    while iter < max_iter {
        // Check convergence on gradient
        if g.mapv(|gi| gi.abs()).sum() < gtol {
            break;
        }

        // Compute search direction
        let mut p = -h_inv.dot(&g);

        // Project search direction for bounded optimization
        if let Some(bounds) = bounds {
            for i in 0..n {
                let mut can_decrease = true;
                let mut can_increase = true;

                // Check if at boundary
                if let Some(lb) = bounds.lower[i] {
                    if x[i] <= lb + eps {
                        can_decrease = false;
                    }
                }
                if let Some(ub) = bounds.upper[i] {
                    if x[i] >= ub - eps {
                        can_increase = false;
                    }
                }

                // Project gradient component
                if (g[i] > 0.0 && !can_decrease) || (g[i] < 0.0 && !can_increase) {
                    p[i] = 0.0;
                }
            }

            // If no movement is possible, we're at a constrained optimum
            if p.mapv(|pi| pi.abs()).sum() < 1e-10 {
                break;
            }
        }

        // Line search
        let alpha_init = 1.0;
        let (alpha, f_new) = backtracking_line_search(
            &mut fun,
            &x.view(),
            f,
            &p.view(),
            &g.view(),
            alpha_init,
            0.0001,
            0.5,
            bounds,
        );

        nfev += 1; // Count line search evaluations

        // Update position
        let s = alpha * &p;
        let x_new = &x + &s;

        // Check step size convergence
        if array_diff_norm(&x_new.view(), &x.view()) < options.xtol {
            x = x_new;
            break;
        }

        // Calculate new gradient
        let g_new = match jacobian {
            Some(jac) => {
                compute_gradient_with_jacobian(&mut fun, &x_new.view(), jac, eps, &mut nfev)?
            }
            None => {
                let g_fd = finite_difference_gradient(&mut fun, &x_new.view(), eps)?;
                nfev += n;
                g_fd
            }
        };

        // Gradient difference
        let y = &g_new - &g;

        // Check convergence on function value
        if check_convergence(
            f - f_new,
            0.0,
            g_new.mapv(|xi| xi.abs()).sum(),
            ftol,
            0.0,
            gtol,
        ) {
            x = x_new;
            g = g_new;
            break;
        }

        // Update inverse Hessian approximation using BFGS formula
        let s_dot_y = s.dot(&y);
        if s_dot_y > 1e-10 {
            let rho = 1.0 / s_dot_y;
            let i_mat = Array2::eye(n);

            // Compute (I - rho y s^T)
            let y_col = y.clone().insert_axis(Axis(1));
            let s_row = s.clone().insert_axis(Axis(0));
            let y_s_t = y_col.dot(&s_row);
            let term1 = &i_mat - &(&y_s_t * rho);

            // Compute (I - rho s y^T)
            let s_col = s.clone().insert_axis(Axis(1));
            let y_row = y.clone().insert_axis(Axis(0));
            let s_y_t = s_col.dot(&y_row);
            let term2 = &i_mat - &(&s_y_t * rho);

            // Update H_inv = (I - rho y s^T) H (I - rho s y^T) + rho s s^T
            let term3 = term1.dot(&h_inv);
            h_inv = term3.dot(&term2) + rho * s_col.dot(&s_row);
        }

        // Update variables for next iteration
        x = x_new;
        f = f_new;
        g = g_new;

        iter += 1;
    }

    // Final check for bounds
    if let Some(bounds) = bounds {
        let slice = x.as_slice_mut().ok_or_else(|| {
            OptimizeError::ComputationError(
                "Failed to get mutable slice for bounds projection".to_string(),
            )
        })?;
        bounds.project(slice);
    }

    // Use original function for final value
    let final_fun = fun(&x.view());

    // Create and return result
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

/// Implements the BFGS algorithm with optional bounds support and optional user-provided gradient.
///
/// This is the legacy API. It delegates to `minimize_bfgs_with_jacobian` internally,
/// converting the `Option<G>` gradient function to a `Jacobian` enum.
#[allow(dead_code)]
pub fn minimize_bfgs<F, G, S>(
    fun: F,
    grad: Option<G>,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S + Clone,
    G: Fn(&ArrayView1<f64>) -> Array1<f64>,
    S: Into<f64> + Clone,
{
    let jac = grad.map(|g| Jacobian::Function(Box::new(g)));
    minimize_bfgs_with_jacobian(fun, x0, jac.as_ref(), options)
}

/// Backward-compatible wrapper: calls `minimize_bfgs` with no user-provided gradient.
///
/// Gradients are computed using forward finite differences. This function is provided
/// for call sites that do not have an analytic gradient available.
#[allow(dead_code)]
pub fn minimize_bfgs_no_grad<F, S>(
    fun: F,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S + Clone,
    S: Into<f64> + Clone,
{
    minimize_bfgs_with_jacobian(fun, x0, None, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unconstrained::Bounds;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_bfgs_quadratic() {
        let quadratic = |x: &ArrayView1<f64>| -> f64 {
            let a =
                Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).expect("test: shape vec");
            let b = Array1::from_vec(vec![-4.0, -6.0]);
            0.5 * x.dot(&a.dot(x)) + b.dot(x)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let options = Options::default();

        let result = minimize_bfgs(
            quadratic,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            x0,
            &options,
        )
        .expect("test: bfgs quadratic");

        assert!(result.success);
        // Optimal solution: x = A^(-1) * (-b) = [2.0, 2.0]
        assert_abs_diff_eq!(result.x[0], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            let a = 1.0;
            let b = 100.0;
            (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
        };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();
        options.max_iter = 2000; // More iterations for Rosenbrock

        let result = minimize_bfgs(
            rosenbrock,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            x0,
            &options,
        )
        .expect("test: bfgs rosenbrock");

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 3e-3);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 5e-3);
    }

    #[test]
    fn test_bfgs_with_bounds() {
        let quadratic =
            |x: &ArrayView1<f64>| -> f64 { (x[0] - 2.0).powi(2) + (x[1] - 3.0).powi(2) };

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();

        // Constrain solution to [0, 1] x [0, 1]
        let bounds = Bounds::new(&[(Some(0.0), Some(1.0)), (Some(0.0), Some(1.0))]);
        options.bounds = Some(bounds);

        let result = minimize_bfgs(
            quadratic,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            x0,
            &options,
        )
        .expect("test: bfgs with bounds");

        assert!(result.success);
        // The optimal point (2, 3) is outside the bounds, so we should get (1, 1)
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bfgs_with_analytic_gradient() {
        // Quadratic function f(x) = x[0]^2 + x[1]^2, minimum at (0, 0)
        let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + x[1].powi(2) };

        // Analytic gradient: g(x) = [2*x[0], 2*x[1]]
        let grad_fn = |x: &ArrayView1<f64>| array![2.0 * x[0], 2.0 * x[1]];

        let x0 = Array1::from_vec(vec![3.0, -2.0]);
        let options = Options::default();

        let result =
            minimize_bfgs(fun, Some(grad_fn), x0, &options).expect("test: bfgs analytic grad");

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-6);
        assert!(result.fun < 1e-10);
    }

    #[test]
    fn test_bfgs_with_user_jacobian() {
        // Rosenbrock function
        let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
            (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
        };

        // Analytic gradient of Rosenbrock
        let jac = Jacobian::Function(Box::new(|x: &ArrayView1<f64>| {
            array![
                -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
                200.0 * (x[1] - x[0].powi(2))
            ]
        }));

        let x0 = Array1::from_vec(vec![0.0, 0.0]);
        let mut options = Options::default();
        options.max_iter = 2000;

        let result: OptimizeResult<f64> =
            minimize_bfgs_with_jacobian(rosenbrock, x0, Some(&jac), &options)
                .expect("test: bfgs with jacobian");

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 1.0, epsilon = 5e-3);
        assert_abs_diff_eq!(result.x[1], 1.0, epsilon = 5e-3);
    }

    #[test]
    fn test_gradient_verification() {
        // Verify that analytic gradient matches finite difference gradient
        let fun = |x: &ArrayView1<f64>| -> f64 { x[0].powi(2) + 2.0 * x[1].powi(2) + x[0] * x[1] };

        let x = array![1.0, 2.0];

        // Analytic gradient: [2*x0 + x1, 4*x1 + x0]
        let analytic_grad = array![2.0 * x[0] + x[1], 4.0 * x[1] + x[0]];

        // Finite difference gradient
        let mut fun_clone = fun;
        let fd_grad = finite_difference_gradient(&mut fun_clone, &x.view(), 1e-8)
            .expect("test: finite diff gradient");

        // They should be close
        assert_abs_diff_eq!(analytic_grad[0], fd_grad[0], epsilon = 1e-5);
        assert_abs_diff_eq!(analytic_grad[1], fd_grad[1], epsilon = 1e-5);

        // Now verify via Jacobian enum
        let jac = Jacobian::Function(Box::new(move |x: &ArrayView1<f64>| {
            array![2.0 * x[0] + x[1], 4.0 * x[1] + x[0]]
        }));

        let mut nfev = 0usize;
        let mut fun_mut = fun;
        let jac_grad =
            compute_gradient_with_jacobian(&mut fun_mut, &x.view(), &jac, 1e-8, &mut nfev)
                .expect("test: jacobian gradient");

        assert_abs_diff_eq!(analytic_grad[0], jac_grad[0], epsilon = 1e-12);
        assert_abs_diff_eq!(analytic_grad[1], jac_grad[1], epsilon = 1e-12);
        // User-provided jacobian should not increment nfev
        assert_eq!(nfev, 0);
    }
}
