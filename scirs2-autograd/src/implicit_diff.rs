//! Implicit differentiation and fixed-point differentiation
//!
//! This module implements:
//!
//! 1. **Implicit Function Theorem** ([`implicit_function_theorem`]): given a system
//!    `F(x, y) = 0`, computes `∂y/∂x = -(∂F/∂y)⁻¹ ∂F/∂x` using central finite
//!    differences for the Jacobians and Gaussian elimination for the linear solve.
//!
//! 2. **Fixed-point differentiation** ([`fixed_point_diff`]): given a contractive map
//!    `T(y, θ)` with fixed point `y* = T(y*, θ)`, computes `∂y*/∂θ` via the identity
//!    `∂y*/∂θ = (I - ∂T/∂y)⁻¹ ∂T/∂θ`.
//!
//! 3. **Argmin differentiation** ([`argmin_diff`]): given a loss `L(params, θ)`, the
//!    minimiser `params*(θ) = argmin_p L(p, θ)` satisfies `∇_p L = 0`.  By the
//!    implicit function theorem, `∂params*/∂θ = -(∂²L/∂p²)⁻¹ ∂²L/∂p∂θ`.
//!
//! All Jacobians are computed via central finite differences with step `h = 1e-5`.
//! Linear systems `A · x = b` are solved with Gaussian elimination with partial pivoting.
//!
//! # Applications
//!
//! These primitives are the core building blocks for:
//! - **Meta-learning** (MAML, iMAML): differentiate through an inner-loop optimisation.
//! - **Bilevel optimisation**: upper-level gradient through a lower-level argmin.
//! - **Neural ODEs / equilibrium networks**: differentiate through implicit layers.
//!
//! # Example — implicit function theorem
//!
//! ```rust
//! use scirs2_autograd::implicit_diff::implicit_function_theorem;
//!
//! // F(x, y) = y - 2*x = 0, so y*(x) = 2*x, dy/dx = 2
//! let dy_dx = implicit_function_theorem(
//!     |x, y| vec![y[0] - 2.0 * x[0]],
//!     &[1.0],
//!     &[2.0],
//! ).expect("IFT");
//! assert!((dy_dx[[0, 0]] - 2.0).abs() < 1e-3, "dy/dx={}", dy_dx[[0, 0]]);
//! ```

use crate::error::AutogradError;
use scirs2_core::ndarray::Array2;

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Central-FD step size.
const H: f64 = 1e-5;

/// Compute the Jacobian `∂F/∂z ∈ R^{m × dim_z}` of a vector function
/// `F: R^{dim_x} × R^{dim_y} → R^m` with respect to the `z` argument at
/// `(x, y)`.  The closure `eval` evaluates `F` given `(x, y)` and returns
/// a `Vec<f64>` of length `m`.
///
/// `z_ref` points to the slice that should be perturbed (either `x` or `y`),
/// and `eval` is called with the perturbed version.
fn jacobian_wrt(
    eval: &dyn Fn(&[f64]) -> Vec<f64>,
    z: &[f64],
    m: usize,
) -> Array2<f64> {
    let dz = z.len();
    let two_h = 2.0 * H;
    let mut jac = Array2::<f64>::zeros((m, dz));
    let mut zp = z.to_vec();
    let mut zm = z.to_vec();
    for j in 0..dz {
        zp[j] = z[j] + H;
        zm[j] = z[j] - H;
        let fp = eval(&zp);
        let fm = eval(&zm);
        for i in 0..m {
            jac[[i, j]] = (fp[i] - fm[i]) / two_h;
        }
        zp[j] = z[j];
        zm[j] = z[j];
    }
    jac
}

/// Solve the linear system `A · X = B` where `A ∈ R^{m×m}` and `B ∈ R^{m×k}`
/// using Gaussian elimination with partial pivoting.
///
/// Returns `X ∈ R^{m×k}`.
///
/// # Errors
/// Returns an error if `A` is singular (pivot below `1e-14`).
fn solve_linear(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, AutogradError> {
    let m = a.nrows();
    let k = b.ncols();
    debug_assert_eq!(a.ncols(), m, "A must be square");
    debug_assert_eq!(b.nrows(), m, "B rows must equal A rows");

    // Augment: [A | B]
    let mut aug = Array2::<f64>::zeros((m, m + k));
    for i in 0..m {
        for j in 0..m {
            aug[[i, j]] = a[[i, j]];
        }
        for j in 0..k {
            aug[[i, m + j]] = b[[i, j]];
        }
    }

    // Forward elimination with partial pivoting
    for col in 0..m {
        // Find pivot row
        let mut max_val = aug[[col, col]].abs();
        let mut pivot_row = col;
        for row in (col + 1)..m {
            let v = aug[[row, col]].abs();
            if v > max_val {
                max_val = v;
                pivot_row = row;
            }
        }
        if max_val < 1e-14 {
            return Err(AutogradError::OperationError(
                "solve_linear: singular matrix (pivot near zero)".to_string(),
            ));
        }
        // Swap rows
        if pivot_row != col {
            for j in 0..(m + k) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = tmp;
            }
        }
        // Eliminate below
        let pivot = aug[[col, col]];
        for row in (col + 1)..m {
            let factor = aug[[row, col]] / pivot;
            for j in col..(m + k) {
                let val = aug[[col, j]];
                aug[[row, j]] -= factor * val;
            }
        }
    }

    // Back substitution
    let mut x = Array2::<f64>::zeros((m, k));
    for rhs_col in 0..k {
        for i in (0..m).rev() {
            let mut s = aug[[i, m + rhs_col]];
            for j in (i + 1)..m {
                s -= aug[[i, j]] * x[[j, rhs_col]];
            }
            let piv = aug[[i, i]];
            if piv.abs() < 1e-14 {
                return Err(AutogradError::OperationError(
                    "solve_linear: zero diagonal during back-substitution".to_string(),
                ));
            }
            x[[i, rhs_col]] = s / piv;
        }
    }

    Ok(x)
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Implicit Function Theorem
// ─────────────────────────────────────────────────────────────────────────────

/// Differentiate implicitly through the constraint `F(x, y) = 0`.
///
/// Given:
/// - `F: R^{n_x} × R^{n_y} → R^m` (with `m = n_y` for the IFT to apply)
/// - A point `(x, y)` satisfying `F(x, y) ≈ 0`
///
/// Computes `∂y/∂x ∈ R^{n_y × n_x}` via the implicit function theorem:
///
/// ```text
/// ∂y/∂x = -(∂F/∂y)⁻¹ · (∂F/∂x)
/// ```
///
/// Both Jacobians are computed via central finite differences.
/// The linear system is solved by Gaussian elimination.
///
/// # Arguments
/// * `F`   – Constraint function `(x, y) -> Vec<f64>` of length `n_y`
/// * `x`   – Current value of the "independent" variables (length `n_x`)
/// * `y`   – Current value of the "dependent" variables (length `n_y`)
///
/// # Returns
/// `Array2<f64>` of shape `(n_y, n_x)`.
///
/// # Errors
/// Returns `AutogradError` if the system is ill-posed (singular `∂F/∂y`) or
/// dimensions are inconsistent.
///
/// # Example
/// ```rust
/// use scirs2_autograd::implicit_diff::implicit_function_theorem;
///
/// // F(x, y) = y - 3*x = 0, dy/dx = 3
/// let dy_dx = implicit_function_theorem(
///     |x, y| vec![y[0] - 3.0 * x[0]],
///     &[1.0],
///     &[3.0],
/// ).expect("IFT");
/// assert!((dy_dx[[0, 0]] - 3.0).abs() < 1e-3, "dy/dx={}", dy_dx[[0, 0]]);
/// ```
pub fn implicit_function_theorem(
    f_constraint: impl Fn(&[f64], &[f64]) -> Vec<f64>,
    x: &[f64],
    y: &[f64],
) -> Result<Array2<f64>, AutogradError> {
    let n_x = x.len();
    let n_y = y.len();
    if n_x == 0 {
        return Err(AutogradError::OperationError(
            "implicit_function_theorem: x must be non-empty".to_string(),
        ));
    }
    if n_y == 0 {
        return Err(AutogradError::OperationError(
            "implicit_function_theorem: y must be non-empty".to_string(),
        ));
    }

    // Probe output dimension
    let f0 = f_constraint(x, y);
    let m = f0.len();
    if m == 0 {
        return Err(AutogradError::OperationError(
            "implicit_function_theorem: F output must be non-empty".to_string(),
        ));
    }
    if m != n_y {
        return Err(AutogradError::ShapeMismatch(format!(
            "implicit_function_theorem: F output length {} must equal |y|={} for IFT to apply",
            m, n_y
        )));
    }

    let x_owned = x.to_vec();
    let y_owned = y.to_vec();

    // ∂F/∂y ∈ R^{m × n_y}
    let dfy = {
        let x_c = x_owned.clone();
        let eval_y = |yp: &[f64]| f_constraint(&x_c, yp);
        jacobian_wrt(&eval_y, y, m)
    };

    // ∂F/∂x ∈ R^{m × n_x}
    let dfx = {
        let y_c = y_owned.clone();
        let eval_x = |xp: &[f64]| f_constraint(xp, &y_c);
        jacobian_wrt(&eval_x, x, m)
    };

    // dy/dx = -(∂F/∂y)^{-1} · (∂F/∂x)
    // Solve: (∂F/∂y) · Z = -(∂F/∂x)
    let neg_dfx = dfx.mapv(|v| -v);
    let z = solve_linear(&dfy, &neg_dfx)?;

    Ok(z)
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Fixed-point differentiation
// ─────────────────────────────────────────────────────────────────────────────

/// Differentiate through a fixed point `y* = T(y*, θ)`.
///
/// Given a contractive map `T: R^{n_y} × R^{n_θ} → R^{n_y}` and a
/// current approximate fixed point `y_star` (so that `T(y_star, θ) ≈ y_star`),
/// computes `∂y*/∂θ ∈ R^{n_y × n_θ}` via:
///
/// ```text
/// ∂y*/∂θ = (I - ∂T/∂y)⁻¹ · (∂T/∂θ)
/// ```
///
/// Both Jacobians are computed via central finite differences.
///
/// # Arguments
/// * `T`      – Fixed-point operator `(y, theta) -> Vec<f64>` of length `n_y`
/// * `y_star` – Approximate fixed point `y* ≈ T(y*, θ)` (length `n_y`)
/// * `theta`  – Parameter vector (length `n_θ`)
///
/// # Returns
/// `Array2<f64>` of shape `(n_y, n_θ)`.
///
/// # Errors
/// Returns `AutogradError` if the operator `(I - ∂T/∂y)` is singular or
/// if dimension constraints are violated.
///
/// # Example
/// ```rust
/// use scirs2_autograd::implicit_diff::fixed_point_diff;
///
/// // T(y, θ) = θ[0] (scalar contraction to constant)
/// // y* = θ[0], ∂y*/∂θ = [1.0]
/// let dy_dtheta = fixed_point_diff(
///     |y, theta| vec![theta[0]],
///     &[2.0],
///     &[2.0],
/// ).expect("fixed point");
/// assert!((dy_dtheta[[0, 0]] - 1.0).abs() < 1e-3, "dy*/dθ={}", dy_dtheta[[0, 0]]);
/// ```
pub fn fixed_point_diff(
    t_map: impl Fn(&[f64], &[f64]) -> Vec<f64>,
    y_star: &[f64],
    theta: &[f64],
) -> Result<Array2<f64>, AutogradError> {
    let n_y = y_star.len();
    let n_theta = theta.len();
    if n_y == 0 {
        return Err(AutogradError::OperationError(
            "fixed_point_diff: y_star must be non-empty".to_string(),
        ));
    }
    if n_theta == 0 {
        return Err(AutogradError::OperationError(
            "fixed_point_diff: theta must be non-empty".to_string(),
        ));
    }

    // Probe output dimension
    let t0 = t_map(y_star, theta);
    let m = t0.len();
    if m != n_y {
        return Err(AutogradError::ShapeMismatch(format!(
            "fixed_point_diff: T output length {} must equal |y_star|={}",
            m, n_y
        )));
    }

    let theta_owned = theta.to_vec();
    let y_owned = y_star.to_vec();

    // ∂T/∂y ∈ R^{n_y × n_y}
    let dt_dy = {
        let th_c = theta_owned.clone();
        let eval_y = |yp: &[f64]| t_map(yp, &th_c);
        jacobian_wrt(&eval_y, y_star, m)
    };

    // ∂T/∂θ ∈ R^{n_y × n_θ}
    let dt_dtheta = {
        let y_c = y_owned.clone();
        let eval_th = |thp: &[f64]| t_map(&y_c, thp);
        jacobian_wrt(&eval_th, theta, m)
    };

    // ∂y*/∂θ = (I - ∂T/∂y)^{-1} · (∂T/∂θ)
    let eye = Array2::<f64>::eye(n_y);
    let lhs = eye - dt_dy; // (I - ∂T/∂y)
    let rhs = dt_dtheta;   // ∂T/∂θ

    let dy_dtheta = solve_linear(&lhs, &rhs)?;
    Ok(dy_dtheta)
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Argmin differentiation
// ─────────────────────────────────────────────────────────────────────────────

/// Differentiate through an argmin: `params*(θ) = argmin_p L(p, θ)`.
///
/// At the optimum `params*`, the first-order stationarity condition holds:
/// `∇_p L(params*, θ) = 0`.
///
/// By the implicit function theorem applied to `F(p, θ) = ∇_p L(p, θ)`:
///
/// ```text
/// ∂params*/∂θ = -(∂²L/∂p²)⁻¹ · (∂²L/∂p∂θ)
/// ```
///
/// All second derivatives are approximated via central finite differences of
/// gradients (which are themselves central-FD approximations).
///
/// # Arguments
/// * `loss`   – Scalar loss function `(params, theta) -> f64`
/// * `params` – Approximate argmin `params* ≈ argmin_p L(p, θ)` (length `n_p`)
/// * `theta`  – Parameter vector (length `n_θ`)
///
/// # Returns
/// `Array2<f64>` of shape `(n_p, n_θ)` — the sensitivity of the argmin to `θ`.
///
/// # Errors
/// Returns `AutogradError` if the Hessian w.r.t. `params` is singular or
/// if dimensions are empty.
///
/// # Application (meta-learning / bilevel optimisation)
/// In MAML-style meta-learning, `params` are the adapted inner-loop parameters
/// and `theta` are the meta-parameters.  `argmin_diff` gives the meta-gradient.
///
/// # Example
/// ```rust
/// use scirs2_autograd::implicit_diff::argmin_diff;
///
/// // L(p, θ) = (p[0] - θ[0])^2
/// // p* = θ[0], ∂p*/∂θ = I
/// let dp_dtheta = argmin_diff(
///     |p, th| (p[0] - th[0]).powi(2),
///     &[2.0],
///     &[2.0],
/// ).expect("argmin diff");
/// assert!((dp_dtheta[[0, 0]] - 1.0).abs() < 1e-2, "dp*/dθ={}", dp_dtheta[[0, 0]]);
/// ```
pub fn argmin_diff(
    loss: impl Fn(&[f64], &[f64]) -> f64,
    params: &[f64],
    theta: &[f64],
) -> Result<Array2<f64>, AutogradError> {
    let n_p = params.len();
    let n_theta = theta.len();
    if n_p == 0 {
        return Err(AutogradError::OperationError(
            "argmin_diff: params must be non-empty".to_string(),
        ));
    }
    if n_theta == 0 {
        return Err(AutogradError::OperationError(
            "argmin_diff: theta must be non-empty".to_string(),
        ));
    }

    let theta_owned = theta.to_vec();
    let params_owned = params.to_vec();

    // ∂²L/∂p² ∈ R^{n_p × n_p} (Hessian of L w.r.t. params)
    // We compute it as the Jacobian of ∇_p L w.r.t. params.
    let hess_pp = {
        let th_c = theta_owned.clone();
        // ∇_p L is a function of params
        let grad_p = |p: &[f64]| -> Vec<f64> {
            let mut gp = vec![0.0f64; n_p];
            let mut pp = p.to_vec();
            let mut pm = p.to_vec();
            let two_h = 2.0 * H;
            for i in 0..n_p {
                pp[i] = p[i] + H;
                pm[i] = p[i] - H;
                gp[i] = (loss(&pp, &th_c) - loss(&pm, &th_c)) / two_h;
                pp[i] = p[i];
                pm[i] = p[i];
            }
            gp
        };
        jacobian_wrt(&grad_p, params, n_p)
    };

    // ∂²L/∂p∂θ ∈ R^{n_p × n_θ} (cross Hessian)
    // = Jacobian of (∇_p L) w.r.t. θ
    let hess_ptheta = {
        let p_c = params_owned.clone();
        // ∇_p L as a function of theta
        let grad_p_wrt_theta = |th: &[f64]| -> Vec<f64> {
            let mut gp = vec![0.0f64; n_p];
            let mut pp = p_c.clone();
            let mut pm = p_c.clone();
            let two_h = 2.0 * H;
            for i in 0..n_p {
                pp[i] = p_c[i] + H;
                pm[i] = p_c[i] - H;
                gp[i] = (loss(&pp, th) - loss(&pm, th)) / two_h;
                pp[i] = p_c[i];
                pm[i] = p_c[i];
            }
            gp
        };
        jacobian_wrt(&grad_p_wrt_theta, theta, n_p)
    };

    // ∂params*/∂θ = -(∂²L/∂p²)^{-1} · (∂²L/∂p∂θ)
    let neg_hess_ptheta = hess_ptheta.mapv(|v| -v);
    let dp_dtheta = solve_linear(&hess_pp, &neg_hess_ptheta)?;
    Ok(dp_dtheta)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-2;

    // ── implicit_function_theorem ─────────────────────────────────────────

    #[test]
    fn test_ift_linear_scalar() {
        // F(x, y) = y - 2*x = 0, y* = 2x, dy/dx = 2
        let dy_dx = implicit_function_theorem(
            |x, y| vec![y[0] - 2.0 * x[0]],
            &[1.0],
            &[2.0],
        )
        .expect("IFT linear scalar");
        assert!((dy_dx[[0, 0]] - 2.0).abs() < TOL, "dy/dx={}", dy_dx[[0, 0]]);
    }

    #[test]
    fn test_ift_linear_2d() {
        // F(x, y) = y - A*x = 0 where A = [[2,0],[0,3]]
        // dy/dx = A
        let dy_dx = implicit_function_theorem(
            |x, y| vec![
                y[0] - 2.0 * x[0],
                y[1] - 3.0 * x[1],
            ],
            &[1.0, 1.0],
            &[2.0, 3.0],
        )
        .expect("IFT linear 2D");
        assert!((dy_dx[[0, 0]] - 2.0).abs() < TOL, "dy0/dx0={}", dy_dx[[0, 0]]);
        assert!(dy_dx[[0, 1]].abs() < TOL, "dy0/dx1={}", dy_dx[[0, 1]]);
        assert!(dy_dx[[1, 0]].abs() < TOL, "dy1/dx0={}", dy_dx[[1, 0]]);
        assert!((dy_dx[[1, 1]] - 3.0).abs() < TOL, "dy1/dx1={}", dy_dx[[1, 1]]);
    }

    #[test]
    fn test_ift_nonlinear() {
        // F(x, y) = y^2 - x = 0, y* = sqrt(x)
        // dy/dx = 1/(2y*) = 1/(2*sqrt(2)) at x=2, y=sqrt(2)
        let y_star = 2.0_f64.sqrt();
        let expected = 1.0 / (2.0 * y_star);
        let dy_dx = implicit_function_theorem(
            |x, y| vec![y[0] * y[0] - x[0]],
            &[2.0],
            &[y_star],
        )
        .expect("IFT nonlinear");
        assert!(
            (dy_dx[[0, 0]] - expected).abs() < TOL,
            "dy/dx={} expected={}",
            dy_dx[[0, 0]],
            expected
        );
    }

    #[test]
    fn test_ift_empty_x_error() {
        let r = implicit_function_theorem(|_, y| vec![y[0]], &[], &[1.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_ift_dimension_mismatch_error() {
        // F returns 2 values but |y|=1 — IFT requires m = n_y
        let r = implicit_function_theorem(
            |x, y| vec![y[0], x[0]],
            &[1.0],
            &[1.0],
        );
        assert!(r.is_err());
    }

    // ── fixed_point_diff ──────────────────────────────────────────────────

    #[test]
    fn test_fixed_point_constant() {
        // T(y, θ) = θ[0] (contraction to constant)
        // y* = θ[0], ∂y*/∂θ = 1
        let dy_dth = fixed_point_diff(
            |_y, theta| vec![theta[0]],
            &[2.0],
            &[2.0],
        )
        .expect("fixed point constant");
        assert!((dy_dth[[0, 0]] - 1.0).abs() < TOL, "dy*/dθ={}", dy_dth[[0, 0]]);
    }

    #[test]
    fn test_fixed_point_affine() {
        // T(y, θ) = 0.5*y + θ[0], y* = 2*θ[0], ∂y*/∂θ = 2
        let y_star = 4.0; // 2 * 2.0
        let dy_dth = fixed_point_diff(
            |y, theta| vec![0.5 * y[0] + theta[0]],
            &[y_star],
            &[2.0],
        )
        .expect("fixed point affine");
        assert!((dy_dth[[0, 0]] - 2.0).abs() < TOL, "dy*/dθ={}", dy_dth[[0, 0]]);
    }

    #[test]
    fn test_fixed_point_2d() {
        // T(y, θ) = [0.5*y0 + θ0, 0.3*y1 + θ1]
        // y*0 = 2*θ0, y*1 = (1/0.7)*θ1
        // ∂y*0/∂θ0 = 2, ∂y*1/∂θ1 = 1/(1-0.3) ≈ 1.4286
        let y0 = 4.0; // 2 * 2.0
        let y1 = 3.0 / 0.7; // (1/0.7) * 3.0 * 0.7 = 3.0
        let dy_dth = fixed_point_diff(
            |y, theta| vec![0.5 * y[0] + theta[0], 0.3 * y[1] + theta[1]],
            &[y0, y1],
            &[2.0, 3.0],
        )
        .expect("fixed point 2D");
        assert!((dy_dth[[0, 0]] - 2.0).abs() < TOL, "dy0*/dθ0={}", dy_dth[[0, 0]]);
        let expected_11 = 1.0 / (1.0 - 0.3);
        assert!(
            (dy_dth[[1, 1]] - expected_11).abs() < TOL,
            "dy1*/dθ1={} expected={}",
            dy_dth[[1, 1]],
            expected_11
        );
    }

    #[test]
    fn test_fixed_point_empty_error() {
        let r = fixed_point_diff(|_, th| vec![th[0]], &[], &[1.0]);
        assert!(r.is_err());
    }

    // ── argmin_diff ───────────────────────────────────────────────────────

    #[test]
    fn test_argmin_quadratic_scalar() {
        // L(p, θ) = (p - θ)^2, p* = θ, ∂p*/∂θ = 1
        let dp_dth = argmin_diff(
            |p, th| (p[0] - th[0]).powi(2),
            &[2.0],
            &[2.0],
        )
        .expect("argmin quadratic");
        assert!((dp_dth[[0, 0]] - 1.0).abs() < TOL, "dp*/dθ={}", dp_dth[[0, 0]]);
    }

    #[test]
    fn test_argmin_weighted_quadratic() {
        // L(p, θ) = (p - θ[0])^2 + θ[1] * p^2 = (1 + θ1)*p^2 - 2*θ0*p + θ0^2
        // p* = θ0 / (1 + θ1)
        // ∂p*/∂θ0 = 1/(1+θ1), ∂p*/∂θ1 = -θ0/(1+θ1)^2
        let theta = vec![2.0, 1.0]; // θ0=2, θ1=1
        let p_star = theta[0] / (1.0 + theta[1]); // = 1.0
        let dp_dth = argmin_diff(
            |p, th| (p[0] - th[0]).powi(2) + th[1] * p[0] * p[0],
            &[p_star],
            &theta,
        )
        .expect("argmin weighted quadratic");
        let expected_dpdth0 = 1.0 / (1.0 + theta[1]); // 0.5
        let expected_dpdth1 = -theta[0] / (1.0 + theta[1]).powi(2); // -0.5
        assert!(
            (dp_dth[[0, 0]] - expected_dpdth0).abs() < TOL,
            "dp*/dθ0={} expected={}",
            dp_dth[[0, 0]],
            expected_dpdth0
        );
        assert!(
            (dp_dth[[0, 1]] - expected_dpdth1).abs() < TOL,
            "dp*/dθ1={} expected={}",
            dp_dth[[0, 1]],
            expected_dpdth1
        );
    }

    #[test]
    fn test_argmin_empty_params_error() {
        let r = argmin_diff(|_p, _th| 0.0, &[], &[1.0]);
        assert!(r.is_err());
    }

    #[test]
    fn test_argmin_empty_theta_error() {
        let r = argmin_diff(|_p, _th| 0.0, &[1.0], &[]);
        assert!(r.is_err());
    }

    // ── solve_linear internal ─────────────────────────────────────────────

    #[test]
    fn test_solve_linear_identity() {
        use scirs2_core::ndarray::Array2;
        let a = Array2::<f64>::eye(3);
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("b shape");
        let x = solve_linear(&a, &b).expect("solve identity");
        for i in 0..3 {
            for j in 0..2 {
                assert!((x[[i, j]] - b[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_solve_linear_2x2() {
        use scirs2_core::ndarray::Array2;
        // [[2,1],[5,7]] * [x,y] = [11, 13]  => x=3.909, y=3.182
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 5.0, 7.0]).expect("a");
        let b = Array2::from_shape_vec((2, 1), vec![11.0, 13.0]).expect("b");
        let x = solve_linear(&a, &b).expect("solve 2x2");
        // Verify A*x ≈ b
        let check0 = 2.0 * x[[0, 0]] + 1.0 * x[[1, 0]];
        let check1 = 5.0 * x[[0, 0]] + 7.0 * x[[1, 0]];
        assert!((check0 - 11.0).abs() < 1e-8, "check0={}", check0);
        assert!((check1 - 13.0).abs() < 1e-8, "check1={}", check1);
    }
}
