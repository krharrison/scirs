//! BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm for unconstrained optimization

use crate::error::OptimizeError;
use crate::unconstrained::line_search::backtracking_line_search;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::{array_diff_norm, check_convergence, finite_difference_gradient};
use crate::unconstrained::Options;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};

/// Implements the BFGS algorithm with optional bounds support and optional user-provided gradient
#[allow(dead_code)]
pub fn minimize_bfgs<F, G, S>(
    mut fun: F,
    grad: Option<G>,
    x0: Array1<f64>,
    options: &Options,
) -> Result<OptimizeResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S + Clone,
    G: Fn(&ArrayView1<f64>) -> Array1<f64>,
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
        bounds.project(x.as_slice_mut().expect("Operation failed"));
    }

    let mut f = fun(&x.view()).into();

    // Calculate initial gradient: use user-provided gradient if available,
    // otherwise fall back to finite differences
    let mut g = if let Some(ref grad_fn) = grad {
        grad_fn(&x.view())
    } else {
        finite_difference_gradient(&mut fun, &x.view(), eps)?
    };

    // Initialize approximation of inverse Hessian with identity matrix
    let mut h_inv = Array2::eye(n);

    // Initialize counters
    // When using a user-provided gradient, only count the initial function evaluation.
    // When using finite differences, also count the n gradient evaluations.
    let mut nfev = if grad.is_some() { 1 } else { 1 + n };

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

        // Calculate new gradient: use user-provided gradient if available,
        // otherwise fall back to finite differences (and count those evaluations)
        let g_new = if let Some(ref grad_fn) = grad {
            grad_fn(&x_new.view())
        } else {
            let g_fd = finite_difference_gradient(&mut fun, &x_new.view(), eps)?;
            nfev += n;
            g_fd
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

            // Compute (I - ρ y s^T)
            let y_col = y.clone().insert_axis(Axis(1));
            let s_row = s.clone().insert_axis(Axis(0));
            let y_s_t = y_col.dot(&s_row);
            let term1 = &i_mat - &(&y_s_t * rho);

            // Compute (I - ρ s y^T)
            let s_col = s.clone().insert_axis(Axis(1));
            let y_row = y.clone().insert_axis(Axis(0));
            let s_y_t = s_col.dot(&y_row);
            let term2 = &i_mat - &(&s_y_t * rho);

            // Update H_inv = (I - ρ y s^T) H (I - ρ s y^T) + ρ s s^T
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
        bounds.project(x.as_slice_mut().expect("Operation failed"));
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
    minimize_bfgs(
        fun,
        None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
        x0,
        options,
    )
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
                Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).expect("Operation failed");
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
        .expect("Operation failed");

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
        .expect("Operation failed");

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
        .expect("Operation failed");

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

        let result = minimize_bfgs(fun, Some(grad_fn), x0, &options).expect("Operation failed");

        assert!(result.success);
        assert_abs_diff_eq!(result.x[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.x[1], 0.0, epsilon = 1e-6);
        assert!(result.fun < 1e-10);
    }
}
