//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::OptimizeError;
use crate::unconstrained::result::OptimizeResult;
use crate::unconstrained::utils::{finite_difference_gradient, finite_difference_hessian};
use crate::unconstrained::Options;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Implements the Trust-region nearly exact algorithm
#[allow(dead_code)]
pub fn minimize_trust_exact<F, S>(
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
        let (step, hits_boundary) = trust_region_exact_subproblem(&g, &hess, trust_radius);
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
/// Solve the trust-region subproblem using the conjugate gradient method
#[allow(dead_code)]
pub(super) fn trust_region_subproblem(
    g: &Array1<f64>,
    hess: &Array2<f64>,
    trust_radius: f64,
) -> (Array1<f64>, bool) {
    let n = g.len();
    let mut p = -g.clone();
    if g.dot(g) < 1e-10 {
        return (Array1::zeros(n), false);
    }
    let mut s = Array1::zeros(n);
    let mut r = g.clone();
    let g_norm = g.dot(g).sqrt();
    let cg_tol = f64::min(0.1, g_norm);
    let max_cg_iters = n * 2;
    let mut hits_boundary = false;
    for _ in 0..max_cg_iters {
        let hp = hess.dot(&p);
        let php = p.dot(&hp);
        if php <= 0.0 {
            let (_alpha, boundary_step) = find_boundary_step(&s, &p, trust_radius);
            hits_boundary = true;
            return (boundary_step, hits_boundary);
        }
        let alpha = r.dot(&r) / php;
        let s_next = &s + &(&p * alpha);
        if s_next.dot(&s_next).sqrt() >= trust_radius {
            let (_alpha, boundary_step) = find_boundary_step(&s, &p, trust_radius);
            hits_boundary = true;
            return (boundary_step, hits_boundary);
        }
        s = s_next;
        r = &r + &(&hp * alpha);
        if r.dot(&r).sqrt() < cg_tol {
            break;
        }
        let r_new_norm_squared = r.dot(&r);
        let r_old_norm_squared = p.dot(&p);
        let beta = r_new_norm_squared / r_old_norm_squared;
        p = -&r + &(&p * beta);
    }
    (s, hits_boundary)
}
/// Solve the trust region subproblem using the Lanczos method
#[allow(dead_code)]
pub(super) fn trust_region_lanczos_subproblem(
    g: &Array1<f64>,
    hess: &Array2<f64>,
    trust_radius: f64,
) -> (Array1<f64>, bool) {
    let n = g.len();
    let mut v1 = -g.clone();
    v1 = &v1 / v1.dot(&v1).sqrt();
    if g.dot(g) < 1e-10 {
        return (Array1::zeros(n), false);
    }
    let max_lanczos_iters = 10.min(n);
    let mut v = Vec::with_capacity(max_lanczos_iters);
    v.push(v1);
    let mut alpha = Vec::with_capacity(max_lanczos_iters);
    let mut beta = Vec::with_capacity(max_lanczos_iters);
    let mut w = hess.dot(&v[0]);
    alpha.push(w.dot(&v[0]));
    let mut hits_boundary = false;
    for j in 1..max_lanczos_iters {
        if j > 1 {
            w -= &(&v[j - 2] * beta[j - 2]);
        }
        w -= &(&v[j - 1] * alpha[j - 1]);
        for vi in v.iter().take(j) {
            let projection = w.dot(vi);
            w -= &(vi * projection);
        }
        let b = w.dot(&w).sqrt();
        beta.push(b);
        if b < 1e-10 {
            break;
        }
        let vj = &w / b;
        v.push(vj.clone());
        w = hess.dot(&vj);
        alpha.push(w.dot(&vj));
    }
    let k = alpha.len();
    let mut t = Array2::zeros((k, k));
    for i in 0..k {
        t[[i, i]] = alpha[i];
        if i < k - 1 {
            t[[i, i + 1]] = beta[i];
            t[[i + 1, i]] = beta[i];
        }
    }
    let mut lambda_min = alpha[0];
    for &a in alpha.iter().take(k).skip(1) {
        lambda_min = f64::min(lambda_min, a);
    }
    let mut lambda = if lambda_min < 0.0 {
        -lambda_min + 0.1
    } else {
        0.0
    };
    let mut b = Array1::zeros(k);
    b[0] = g.dot(g).sqrt();
    let mut s = Array1::zeros(k);
    let mut inside_trust_region = false;
    let max_tr_iters = 10;
    for _ in 0..max_tr_iters {
        s = solve_tridiagonal_system(&t, &b, lambda);
        let norm_s = s.dot(&s).sqrt();
        if (norm_s - trust_radius).abs() < 1e-6 * trust_radius {
            inside_trust_region = false;
            hits_boundary = true;
            break;
        } else if norm_s < trust_radius {
            inside_trust_region = true;
            if lambda < 1e-10 {
                break;
            }
            lambda /= 4.0;
        } else {
            lambda *= 2.0;
        }
    }
    let mut step: Array1<f64> = Array1::zeros(n);
    for (i, vi) in v.iter().take(k).enumerate() {
        step += &(vi * s[i]);
    }
    if inside_trust_region && lambda > 1e-10 {
        let norm_step = step.dot(&step).sqrt();
        step = &step * (trust_radius / norm_step);
        hits_boundary = true;
    }
    (step, hits_boundary)
}
/// Solve the trust region subproblem using the exact method with eigendecomposition
#[allow(dead_code)]
fn trust_region_exact_subproblem(
    g: &Array1<f64>,
    hess: &Array2<f64>,
    trust_radius: f64,
) -> (Array1<f64>, bool) {
    let n = g.len();
    if g.dot(g) < 1e-10 {
        return (Array1::zeros(n), false);
    }
    let (eigvals, eigvecs) = compute_eig_decomposition(hess);
    let min_eigval = eigvals.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut g_transformed = Array1::zeros(n);
    for i in 0..n {
        let eigvec_i = eigvecs.column(i);
        g_transformed[i] = -eigvec_i.dot(g);
    }
    if min_eigval > 0.0 {
        let mut newton_step = Array1::zeros(n);
        for i in 0..n {
            newton_step[i] = g_transformed[i] / eigvals[i];
        }
        let mut step: Array1<f64> = Array1::zeros(n);
        for i in 0..n {
            let eigvec_i = eigvecs.column(i);
            step += &(&eigvec_i * newton_step[i]);
        }
        let step_norm = step.dot(&step).sqrt();
        if step_norm <= trust_radius {
            return (step, false);
        }
    }
    let phi = |lambda: f64| -> f64 {
        let mut norm_squared = 0.0;
        for i in 0..n {
            let step_i = g_transformed[i] / (eigvals[i] + lambda);
            norm_squared += step_i * step_i;
        }
        norm_squared.sqrt() - trust_radius
    };
    let lambda_min = if min_eigval > 0.0 {
        0.0
    } else {
        -min_eigval + 1e-6
    };
    let lambda_max = lambda_min + 1000.0;
    let lambda = find_lambda_bisection(lambda_min, lambda_max, phi);
    let mut opt_step_transformed = Array1::zeros(n);
    for i in 0..n {
        opt_step_transformed[i] = g_transformed[i] / (eigvals[i] + lambda);
    }
    let mut step: Array1<f64> = Array1::zeros(n);
    for i in 0..n {
        let eigvec_i = eigvecs.column(i);
        step += &(&eigvec_i * opt_step_transformed[i]);
    }
    (step, true)
}
/// Find a step that lies on the trust region boundary
#[allow(dead_code)]
fn find_boundary_step(s: &Array1<f64>, p: &Array1<f64>, trust_radius: f64) -> (f64, Array1<f64>) {
    let s_norm_squared = s.dot(s);
    let p_norm_squared = p.dot(p);
    let s_dot_p = s.dot(p);
    let a = p_norm_squared;
    let b = 2.0 * s_dot_p;
    let c = s_norm_squared - trust_radius * trust_radius;
    let disc = b * b - 4.0 * a * c;
    let disc = f64::max(disc, 0.0);
    let alpha = (-b + disc.sqrt()) / (2.0 * a);
    let boundary_step = s + &(p * alpha);
    (alpha, boundary_step)
}
/// Calculate the predicted reduction in the quadratic model
#[allow(dead_code)]
pub(super) fn calculate_predicted_reduction(
    g: &Array1<f64>,
    hess: &Array2<f64>,
    step: &Array1<f64>,
) -> f64 {
    let g_dot_s = g.dot(step);
    let s_dot_bs = step.dot(&hess.dot(step));
    -g_dot_s - 0.5 * s_dot_bs
}
/// Solve a tridiagonal system (T + lambda*I)x = b
#[allow(dead_code)]
fn solve_tridiagonal_system(t: &Array2<f64>, b: &Array1<f64>, lambda: f64) -> Array1<f64> {
    let n = t.shape()[0];
    let mut d = Array1::zeros(n);
    let mut e = Array1::zeros(n - 1);
    for i in 0..n {
        d[i] = t[[i, i]] + lambda;
        if i < n - 1 {
            e[i] = t[[i, i + 1]];
        }
    }
    let mut u = Array1::zeros(n);
    let mut w = d[0];
    u[0] = b[0] / w;
    for i in 1..n {
        let temp = e[i - 1] / w;
        d[i] -= temp * e[i - 1];
        w = d[i];
        u[i] = (b[i] - temp * u[i - 1]) / w;
    }
    let mut x = Array1::zeros(n);
    x[n - 1] = u[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = u[i] - e[i] * x[i + 1] / d[i];
    }
    x
}
/// Compute eigendecomposition of a matrix (simplified version)
#[allow(dead_code)]
fn compute_eig_decomposition(mat: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = mat.nrows();
    let mut eigvals = Array1::zeros(n);
    let mut eigvecs = Array2::zeros((n, n));
    let mut mat_copy = mat.clone();
    for k in 0..n {
        let mut v = Array1::zeros(n);
        v[k % n] = 1.0;
        for i in 0..n {
            v[i] += 0.01 * (i as f64);
        }
        v = &v / v.dot(&v).sqrt();
        for _ in 0..50 {
            let w = mat_copy.dot(&v);
            let norm = w.dot(&w).sqrt();
            if norm < 1e-10 {
                break;
            }
            v = &w / norm;
        }
        let eigval = v.dot(&mat_copy.dot(&v));
        eigvals[k] = eigval;
        for i in 0..n {
            eigvecs[[i, k]] = v[i];
        }
        for i in 0..n {
            for j in 0..n {
                mat_copy[[i, j]] -= eigval * v[i] * v[j];
            }
        }
    }
    (eigvals, eigvecs)
}
/// Find the optimal lambda using the bisection method
#[allow(dead_code)]
fn find_lambda_bisection<F>(a: f64, b: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut a = a;
    let mut b = b;
    let tol = 1e-10;
    let max_iter = 100;
    let mut fa = f(a);
    if fa > 0.0 {
        let mut b_temp = a + 1.0;
        let mut fb_temp = f(b_temp);
        while fb_temp > 0.0 && b_temp < 1e6 {
            b_temp *= 2.0;
            fb_temp = f(b_temp);
        }
        if fb_temp > 0.0 {
            return a;
        }
        b = b_temp;
    } else if fa == 0.0 {
        return a;
    }
    let mut iter = 0;
    while (b - a) > tol && iter < max_iter {
        let c = (a + b) / 2.0;
        let fc = f(c);
        if fc.abs() < tol {
            return c;
        }
        if fc * fa < 0.0 {
            b = c;
        } else {
            a = c;
            fa = fc;
        }
        iter += 1;
    }
    (a + b) / 2.0
}
