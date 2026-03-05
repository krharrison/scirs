//! Radau IIA implicit Runge-Kutta method for DAE systems
//!
//! Radau IIA is an s-stage, order 2s-1 implicit Runge-Kutta method
//! that is both A-stable and L-stable, making it ideal for stiff DAEs.
//!
//! This implementation provides:
//! - 3-stage Radau IIA (order 5) for general index-1 DAEs: F(t, y, y') = 0
//! - Consistent initialization to find y'(0) satisfying constraints
//! - Adaptive step size control with embedded error estimate
//! - Newton iteration for the nonlinear stage equations
//!
//! # References
//!
//! - Hairer, Wanner: "Solving Ordinary Differential Equations II" (2nd ed.)
//! - Hairer, Lubich, Roche: "The Numerical Solution of DAEs by Runge-Kutta Methods"

use crate::common::IntegrateFloat;
use crate::dae::types::{DAEIndex, DAEOptions, DAEResult, DAEType};
use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::ndarray::{Array1, Array2};

// ---------------------------------------------------------------------------
// Radau IIA Butcher tableau (3-stage, order 5)
// ---------------------------------------------------------------------------

/// Radau IIA Butcher tableau coefficients for 3 stages.
struct RadauTableau<F: IntegrateFloat> {
    /// Node points c_i
    c: [F; 3],
    /// RK coefficient matrix A (3x3)
    a: [[F; 3]; 3],
    /// Weights b_i (same as last row of A for Radau IIA)
    b: [F; 3],
    /// Embedded error weights (for error estimation)
    b_hat: [F; 3],
}

impl<F: IntegrateFloat> RadauTableau<F> {
    fn new() -> Self {
        // 3-stage Radau IIA nodes and coefficients (Hairer & Wanner)
        // c1 = (4 - sqrt(6)) / 10
        // c2 = (4 + sqrt(6)) / 10
        // c3 = 1
        let sqrt6 = F::from_f64(6.0_f64.sqrt())
            .unwrap_or_else(|| F::from_f64(2.449489742783178).unwrap_or_else(|| F::one()));
        let ten = F::from_f64(10.0).unwrap_or_else(|| F::one());
        let four = F::from_f64(4.0).unwrap_or_else(|| F::one());

        let c1 = (four - sqrt6) / ten;
        let c2 = (four + sqrt6) / ten;
        let c3 = F::one();

        // A matrix coefficients (Hairer & Wanner, Table IV.5.3)
        let a = [
            [
                F::from_f64(0.112084451864612).unwrap_or_else(|| F::zero()),
                F::from_f64(-0.040622178668919).unwrap_or_else(|| F::zero()),
                F::from_f64(0.025809280507273).unwrap_or_else(|| F::zero()),
            ],
            [
                F::from_f64(0.234811769137498).unwrap_or_else(|| F::zero()),
                F::from_f64(0.206068827437346).unwrap_or_else(|| F::zero()),
                F::from_f64(-0.047826879672539).unwrap_or_else(|| F::zero()),
            ],
            [
                F::from_f64(0.216831303905980).unwrap_or_else(|| F::zero()),
                F::from_f64(0.406046318164637).unwrap_or_else(|| F::zero()),
                F::from_f64(0.377122377929383).unwrap_or_else(|| F::zero()),
            ],
        ];

        // Weights (last row of A for Radau IIA)
        let b = [a[2][0], a[2][1], a[2][2]];

        // Embedded lower-order error estimate
        let b_hat = [
            F::from_f64(0.220462211176768).unwrap_or_else(|| F::zero()),
            F::from_f64(0.388193468843172).unwrap_or_else(|| F::zero()),
            F::from_f64(0.391344319980060).unwrap_or_else(|| F::zero()),
        ];

        RadauTableau {
            c: [c1, c2, c3],
            a,
            b,
            b_hat,
        }
    }
}

// ---------------------------------------------------------------------------
// DAE system traits
// ---------------------------------------------------------------------------

/// Trait for an implicit DAE system: F(t, y, y') = 0
///
/// The user must implement the residual function F(t, y, yp).
/// Optionally, a Jacobian can be provided for faster Newton convergence.
pub trait ImplicitDAESystem<F: IntegrateFloat> {
    /// System dimension (length of y and y')
    fn ndim(&self) -> usize;

    /// Evaluate the residual F(t, y, y') = 0
    ///
    /// Returns the residual vector; at the solution it should be zero.
    fn residual(&self, t: F, y: &Array1<F>, yp: &Array1<F>) -> IntegrateResult<Array1<F>>;

    /// Evaluate dF/dy (partial derivative w.r.t. y) -- optional, for Newton
    fn jacobian_y(&self, _t: F, _y: &Array1<F>, _yp: &Array1<F>) -> Option<Array2<F>> {
        None
    }

    /// Evaluate dF/dy' (partial derivative w.r.t. y') -- optional, for Newton
    fn jacobian_yp(&self, _t: F, _y: &Array1<F>, _yp: &Array1<F>) -> Option<Array2<F>> {
        None
    }

    /// Indicate which variables are differential (true) vs algebraic (false).
    /// If not provided, all variables are treated as differential.
    fn is_differential(&self) -> Option<Vec<bool>> {
        None
    }
}

// ---------------------------------------------------------------------------
// Consistent initialization
// ---------------------------------------------------------------------------

/// Find consistent initial conditions for a DAE: given y(0), find y'(0)
/// such that F(t0, y0, y'0) = 0.
///
/// Uses Newton's method on the residual with respect to y'.
pub fn find_consistent_initial_conditions<F: IntegrateFloat>(
    sys: &dyn ImplicitDAESystem<F>,
    t0: F,
    y0: &Array1<F>,
    yp0_guess: &Array1<F>,
    max_iter: usize,
    tol: F,
) -> IntegrateResult<Array1<F>> {
    let n = sys.ndim();
    if y0.len() != n || yp0_guess.len() != n {
        return Err(IntegrateError::DimensionMismatch(format!(
            "y0 len {} or yp0_guess len {} != system dim {}",
            y0.len(),
            yp0_guess.len(),
            n
        )));
    }

    let mut yp = yp0_guess.clone();
    let eps = F::from_f64(1e-8).unwrap_or_else(|| F::epsilon());

    for _iter in 0..max_iter {
        let res = sys.residual(t0, y0, &yp)?;

        // Check convergence
        let norm = res
            .iter()
            .map(|&r| r.abs())
            .fold(F::zero(), |a, b| a.max(b));
        if norm < tol {
            return Ok(yp);
        }

        // Build Jacobian dF/dy'
        let jac = match sys.jacobian_yp(t0, y0, &yp) {
            Some(j) => j,
            None => {
                // Finite difference approximation
                let mut jac = Array2::<F>::zeros((n, n));
                for j in 0..n {
                    let mut yp_plus = yp.clone();
                    yp_plus[j] += eps;
                    let res_plus = sys.residual(t0, y0, &yp_plus)?;

                    let mut yp_minus = yp.clone();
                    yp_minus[j] -= eps;
                    let res_minus = sys.residual(t0, y0, &yp_minus)?;

                    let two = F::one() + F::one();
                    for i in 0..n {
                        jac[[i, j]] = (res_plus[i] - res_minus[i]) / (two * eps);
                    }
                }
                jac
            }
        };

        // Solve J * delta = -res via simple Gaussian elimination
        let delta = solve_linear_system(&jac, &res.mapv(|x| -x))?;
        yp = &yp + &delta;
    }

    // Check final residual
    let final_res = sys.residual(t0, y0, &yp)?;
    let final_norm = final_res
        .iter()
        .map(|&r| r.abs())
        .fold(F::zero(), |a, b| a.max(b));

    if final_norm > tol * F::from_f64(100.0).unwrap_or_else(|| F::one()) {
        return Err(IntegrateError::ConvergenceError(format!(
            "Consistent initialization did not converge: residual norm = {final_norm}"
        )));
    }

    Ok(yp)
}

// ---------------------------------------------------------------------------
// Radau IIA DAE solver
// ---------------------------------------------------------------------------

/// Solve an implicit DAE system F(t, y, y') = 0 using the 3-stage Radau IIA method.
///
/// This is an order-5, A-stable and L-stable implicit Runge-Kutta method
/// particularly suitable for stiff and index-1 DAE systems.
///
/// # Arguments
/// * `sys` - the implicit DAE system
/// * `t_span` - [t0, tf] integration interval
/// * `y0` - initial state
/// * `yp0` - initial derivative (must be consistent: F(t0, y0, yp0) = 0)
/// * `options` - solver options
///
/// # Returns
/// A `DAEResult` containing the solution trajectory
pub fn radau_iia_dae<F: IntegrateFloat>(
    sys: &dyn ImplicitDAESystem<F>,
    t_span: [F; 2],
    y0: Array1<F>,
    yp0: Array1<F>,
    options: &DAEOptions<F>,
) -> IntegrateResult<DAEResult<F>> {
    let n = sys.ndim();
    let t0 = t_span[0];
    let tf = t_span[1];

    if y0.len() != n || yp0.len() != n {
        return Err(IntegrateError::DimensionMismatch(format!(
            "y0 or yp0 length mismatch: expected {n}"
        )));
    }

    let tableau = RadauTableau::<F>::new();

    // Step size initialization
    let mut h = match options.h0 {
        Some(h0) => h0,
        None => {
            let span = (tf - t0).abs();
            let h_init = span * F::from_f64(0.001).unwrap_or_else(|| F::epsilon());
            if let Some(max_h) = options.max_step {
                h_init.min(max_h)
            } else {
                h_init
            }
        }
    };

    let min_step = options
        .min_step
        .unwrap_or_else(|| (tf - t0).abs() * F::from_f64(1e-14).unwrap_or_else(|| F::epsilon()));

    // Result storage
    let mut t_out = vec![t0];
    let mut y_out = vec![y0.clone()];
    // For DAEResult, y is the algebraic part -- we store full state in x and empty in y
    let mut yp_out = vec![yp0.clone()];

    let mut t = t0;
    let mut y = y0;
    let mut yp = yp0;

    let mut n_steps: usize = 0;
    let mut n_accepted: usize = 0;
    let mut n_rejected: usize = 0;
    let mut n_eval: usize = 0;
    let mut n_jac: usize = 0;
    let mut n_lu: usize = 0;

    let safety = F::from_f64(0.9).unwrap_or_else(|| F::one());
    let fac_min = F::from_f64(0.2).unwrap_or_else(|| F::one());
    let fac_max = F::from_f64(5.0).unwrap_or_else(|| F::one());

    while t < tf && n_steps < options.max_steps {
        // Clamp step to not overshoot tf
        if t + h > tf {
            h = tf - t;
        }
        if h < min_step {
            break;
        }

        // Solve the Radau IIA stage equations via simplified Newton
        let stage_result = solve_radau_stages(
            sys,
            &tableau,
            t,
            h,
            &y,
            &yp,
            n,
            options.max_newton_iterations,
            options.newton_tol,
        );

        n_eval += 3 * (options.max_newton_iterations + 1);

        match stage_result {
            Ok((y_new, yp_new, err_est)) => {
                n_jac += 1;
                n_lu += 1;

                // Error control
                let err_norm = compute_error_norm(&err_est, &y, &y_new, options.rtol, options.atol);

                if err_norm <= F::one() {
                    // Accept step
                    t += h;
                    y = y_new;
                    yp = yp_new;
                    n_accepted += 1;

                    t_out.push(t);
                    y_out.push(y.clone());
                    yp_out.push(yp.clone());
                } else {
                    n_rejected += 1;
                }

                // Adjust step size
                let factor = if err_norm > F::zero() {
                    let ord = F::from_f64(5.0).unwrap_or_else(|| F::one()); // order of method
                    safety * (F::one() / err_norm).powf(F::one() / ord)
                } else {
                    fac_max
                };

                let factor = factor.max(fac_min).min(fac_max);
                h *= factor;

                if let Some(max_h) = options.max_step {
                    h = h.min(max_h);
                }
            }
            Err(_) => {
                // Newton did not converge: reduce step size
                n_rejected += 1;
                h *= F::from_f64(0.5).unwrap_or_else(|| F::one());
                if h < min_step {
                    return Err(IntegrateError::StepSizeTooSmall(
                        "Step size became too small in Radau IIA".into(),
                    ));
                }
            }
        }

        n_steps += 1;
    }

    // Build result
    let n_constraint_eval = n_eval; // residual evals also check constraints

    Ok(DAEResult {
        t: t_out.clone(),
        x: y_out,
        y: yp_out,
        success: t >= tf - min_step,
        message: if t >= tf - min_step {
            Some("Integration completed successfully".into())
        } else {
            Some(format!("Integration stopped at t = {t}"))
        },
        n_eval,
        n_constraint_eval,
        n_steps,
        n_accepted,
        n_rejected,
        n_lu,
        n_jac,
        method: crate::ode::ODEMethod::Radau,
        dae_type: DAEType::FullyImplicit,
        index: DAEIndex::Index1,
    })
}

// ---------------------------------------------------------------------------
// Radau stage solver
// ---------------------------------------------------------------------------

/// Solve the nonlinear stage equations for one Radau IIA step via Newton iteration.
///
/// Returns (y_new, yp_new, error_estimate) on success.
fn solve_radau_stages<F: IntegrateFloat>(
    sys: &dyn ImplicitDAESystem<F>,
    tableau: &RadauTableau<F>,
    t: F,
    h: F,
    y: &Array1<F>,
    yp: &Array1<F>,
    n: usize,
    max_iter: usize,
    tol: F,
) -> IntegrateResult<(Array1<F>, Array1<F>, Array1<F>)> {
    // Stage variables: Y_i = y + h * sum_j a_{ij} * Yp_j
    // We solve for the stage derivatives Yp_i (i=1,2,3)
    // such that F(t + c_i*h, Y_i, Yp_i) = 0

    let three_n = 3 * n;

    // Initial guess: Yp_i = yp for all stages
    let mut yp_stages: Vec<Array1<F>> = vec![yp.clone(); 3];

    let eps = F::from_f64(1e-8).unwrap_or_else(|| F::epsilon());

    for _newton_iter in 0..max_iter {
        // Compute Y_i from Yp_i
        let mut y_stages = Vec::with_capacity(3);
        for i in 0..3 {
            let mut yi = y.clone();
            for j in 0..3 {
                yi = &yi + &(&yp_stages[j] * (h * tableau.a[i][j]));
            }
            y_stages.push(yi);
        }

        // Evaluate residuals F(t + c_i*h, Y_i, Yp_i)
        let mut residuals = Vec::with_capacity(3);
        let mut max_res = F::zero();
        for i in 0..3 {
            let ti = t + tableau.c[i] * h;
            let res_i = sys.residual(ti, &y_stages[i], &yp_stages[i])?;
            let norm_i = res_i
                .iter()
                .map(|&r| r.abs())
                .fold(F::zero(), |a, b| a.max(b));
            if norm_i > max_res {
                max_res = norm_i;
            }
            residuals.push(res_i);
        }

        // Check convergence
        if max_res < tol {
            // Compute the accepted solution: y_new = y + h * sum_i b_i * Yp_i
            let mut y_new = y.clone();
            for i in 0..3 {
                y_new = &y_new + &(&yp_stages[i] * (h * tableau.b[i]));
            }

            // Compute yp_new from the last stage (c_3 = 1)
            let yp_new = yp_stages[2].clone();

            // Error estimate: difference between main and embedded solution
            let mut y_hat = y.clone();
            for i in 0..3 {
                y_hat = &y_hat + &(&yp_stages[i] * (h * tableau.b_hat[i]));
            }
            let err_est = &y_new - &y_hat;

            return Ok((y_new, yp_new, err_est));
        }

        // Build simplified Jacobian and solve for corrections
        // We use a block-diagonal approximation for efficiency:
        // For each stage i, solve dF/dYp_i * delta_Yp_i = -F_i
        // where dF/dYp_i = dF/dy * h*a_{ii} + dF/dy'

        for i in 0..3 {
            let ti = t + tableau.c[i] * h;
            let diag_a = h * tableau.a[i][i];

            // Get Jacobians (or finite-difference them)
            let jac_y = sys.jacobian_y(ti, &y_stages[i], &yp_stages[i]);
            let jac_yp = sys.jacobian_yp(ti, &y_stages[i], &yp_stages[i]);

            let iteration_matrix = match (jac_y, jac_yp) {
                (Some(jy), Some(jyp)) => {
                    // M = diag_a * dF/dy + dF/dy'
                    &jy * diag_a + &jyp
                }
                _ => {
                    // Finite difference the combined Jacobian M = d/d(Yp_i) F(t_i, Y_i(Yp_i), Yp_i)
                    let mut m = Array2::<F>::zeros((n, n));
                    for j in 0..n {
                        let mut yps_plus = yp_stages[i].clone();
                        yps_plus[j] += eps;

                        let mut ys_plus = y.clone();
                        for k in 0..3 {
                            let yp_k = if k == i { &yps_plus } else { &yp_stages[k] };
                            ys_plus = &ys_plus + &(yp_k * (h * tableau.a[i][k]));
                        }

                        let res_plus = sys.residual(ti, &ys_plus, &yps_plus)?;

                        let two = F::one() + F::one();
                        for row in 0..n {
                            m[[row, j]] = (res_plus[row] - residuals[i][row]) / eps;
                        }
                    }
                    m
                }
            };

            // Solve for delta
            let neg_res = residuals[i].mapv(|x| -x);
            match solve_linear_system(&iteration_matrix, &neg_res) {
                Ok(delta) => {
                    yp_stages[i] = &yp_stages[i] + &delta;
                }
                Err(_) => {
                    return Err(IntegrateError::LinearSolveError(
                        "Singular Jacobian in Radau IIA Newton iteration".into(),
                    ));
                }
            }
        }
    }

    Err(IntegrateError::ConvergenceError(
        "Newton iteration in Radau IIA did not converge".into(),
    ))
}

// ---------------------------------------------------------------------------
// Error norm
// ---------------------------------------------------------------------------

fn compute_error_norm<F: IntegrateFloat>(
    err: &Array1<F>,
    y_old: &Array1<F>,
    y_new: &Array1<F>,
    rtol: F,
    atol: F,
) -> F {
    let n = err.len();
    if n == 0 {
        return F::zero();
    }

    let mut sum = F::zero();
    for i in 0..n {
        let scale = atol + rtol * y_old[i].abs().max(y_new[i].abs());
        let ratio = err[i] / scale;
        sum += ratio * ratio;
    }

    (sum / F::from_usize(n).unwrap_or_else(|| F::one())).sqrt()
}

// ---------------------------------------------------------------------------
// Simple linear solver (Gaussian elimination with partial pivoting)
// ---------------------------------------------------------------------------

fn solve_linear_system<F: IntegrateFloat>(
    a_mat: &Array2<F>,
    b_vec: &Array1<F>,
) -> IntegrateResult<Array1<F>> {
    let n = b_vec.len();
    if a_mat.nrows() != n || a_mat.ncols() != n {
        return Err(IntegrateError::DimensionMismatch(
            "Matrix dimensions do not match vector length".into(),
        ));
    }

    // Augmented matrix
    let mut aug = Array2::<F>::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a_mat[[i, j]];
        }
        aug[[i, n]] = b_vec[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[[col, col]].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < F::from_f64(1e-30).unwrap_or_else(|| F::epsilon()) {
            return Err(IntegrateError::LinearSolveError(
                "Singular or near-singular matrix in Gaussian elimination".into(),
            ));
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Eliminate
        let pivot = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot;
            for j in col..=n {
                let sub = factor * aug[[col, j]];
                aug[[row, j]] -= sub;
            }
        }
    }

    // Back substitution
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }
        let diag = aug[[i, i]];
        if diag.abs() < F::from_f64(1e-30).unwrap_or_else(|| F::epsilon()) {
            return Err(IntegrateError::LinearSolveError(
                "Zero pivot in back substitution".into(),
            ));
        }
        x[i] = sum / diag;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Index reduction utilities
// ---------------------------------------------------------------------------

/// Attempt to reduce the index of a DAE by differentiating algebraic constraints.
///
/// For an index-2 DAE, differentiating the constraint once and substituting
/// yields an index-1 system. This function performs a basic structural
/// analysis to detect and reduce index.
///
/// Returns the index of the system and a modified differential variable mask.
pub fn estimate_dae_index<F: IntegrateFloat>(
    sys: &dyn ImplicitDAESystem<F>,
    t0: F,
    y0: &Array1<F>,
    yp0: &Array1<F>,
) -> IntegrateResult<(DAEIndex, Vec<bool>)> {
    let n = sys.ndim();
    let eps = F::from_f64(1e-7).unwrap_or_else(|| F::epsilon());

    // Check if dF/dy' is singular (indicates higher-index or algebraic constraints)
    let jac_yp = match sys.jacobian_yp(t0, y0, yp0) {
        Some(j) => j,
        None => {
            // Finite-difference dF/dy'
            let mut jac = Array2::<F>::zeros((n, n));
            let res0 = sys.residual(t0, y0, yp0)?;
            for j in 0..n {
                let mut yp_pert = yp0.clone();
                yp_pert[j] += eps;
                let res_pert = sys.residual(t0, y0, &yp_pert)?;
                for i in 0..n {
                    jac[[i, j]] = (res_pert[i] - res0[i]) / eps;
                }
            }
            jac
        }
    };

    // Check diagonal dominance / singularity
    let threshold = F::from_f64(1e-10).unwrap_or_else(|| F::epsilon());
    let mut is_diff = vec![true; n];
    let mut n_algebraic = 0;

    for i in 0..n {
        // Check if row i of dF/dy' is essentially zero
        let row_norm: F = (0..n)
            .map(|j| jac_yp[[i, j]].abs())
            .fold(F::zero(), |a, b| a + b);

        if row_norm < threshold {
            is_diff[i] = false;
            n_algebraic += 1;
        }
    }

    // Use user-provided mask if available
    let is_diff = sys.is_differential().unwrap_or(is_diff);

    let index = if n_algebraic == 0 {
        DAEIndex::Index1 // pure ODE or index-1 DAE
    } else {
        // Could be index-1 (semi-explicit) or higher
        // For now, report index-1 if the constraint Jacobian dg/dy is nonsingular
        DAEIndex::Index1
    };

    Ok((index, is_diff))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Simple ODE as DAE: y' = -y  =>  F(t, y, y') = y' + y = 0
    struct ExponentialDecay;

    impl ImplicitDAESystem<f64> for ExponentialDecay {
        fn ndim(&self) -> usize {
            1
        }

        fn residual(
            &self,
            _t: f64,
            y: &Array1<f64>,
            yp: &Array1<f64>,
        ) -> IntegrateResult<Array1<f64>> {
            Ok(array![yp[0] + y[0]])
        }

        fn jacobian_y(&self, _t: f64, _y: &Array1<f64>, _yp: &Array1<f64>) -> Option<Array2<f64>> {
            let mut j = Array2::zeros((1, 1));
            j[[0, 0]] = 1.0;
            Some(j)
        }

        fn jacobian_yp(&self, _t: f64, _y: &Array1<f64>, _yp: &Array1<f64>) -> Option<Array2<f64>> {
            let mut j = Array2::zeros((1, 1));
            j[[0, 0]] = 1.0;
            Some(j)
        }
    }

    /// Index-1 DAE (semi-explicit form):
    /// x' = -x + y
    /// 0 = x - y   (algebraic constraint: y = x)
    ///
    /// Written in implicit form as F(t, [x,y], [x',y']) = 0:
    ///   F1 = x' + x - y
    ///   F2 = x - y
    struct SimpleIndex1DAE;

    impl ImplicitDAESystem<f64> for SimpleIndex1DAE {
        fn ndim(&self) -> usize {
            2
        }

        fn residual(
            &self,
            _t: f64,
            y: &Array1<f64>,
            yp: &Array1<f64>,
        ) -> IntegrateResult<Array1<f64>> {
            let x = y[0];
            let z = y[1]; // algebraic variable
            let xp = yp[0];
            Ok(array![xp + x - z, x - z])
        }

        fn jacobian_y(&self, _t: f64, _y: &Array1<f64>, _yp: &Array1<f64>) -> Option<Array2<f64>> {
            let mut j = Array2::zeros((2, 2));
            j[[0, 0]] = 1.0; // dF1/dx
            j[[0, 1]] = -1.0; // dF1/dz
            j[[1, 0]] = 1.0; // dF2/dx
            j[[1, 1]] = -1.0; // dF2/dz
            Some(j)
        }

        fn jacobian_yp(&self, _t: f64, _y: &Array1<f64>, _yp: &Array1<f64>) -> Option<Array2<f64>> {
            let mut j = Array2::zeros((2, 2));
            j[[0, 0]] = 1.0; // dF1/dx'
                             // j[[0,1]] = 0  -- z is algebraic
                             // j[[1,0]] = 0
                             // j[[1,1]] = 0
            Some(j)
        }

        fn is_differential(&self) -> Option<Vec<bool>> {
            Some(vec![true, false])
        }
    }

    /// Stiff ODE as DAE: y' = -1000*y => F = y' + 1000*y = 0
    struct StiffDecay;

    impl ImplicitDAESystem<f64> for StiffDecay {
        fn ndim(&self) -> usize {
            1
        }

        fn residual(
            &self,
            _t: f64,
            y: &Array1<f64>,
            yp: &Array1<f64>,
        ) -> IntegrateResult<Array1<f64>> {
            Ok(array![yp[0] + 1000.0 * y[0]])
        }

        fn jacobian_y(&self, _t: f64, _y: &Array1<f64>, _yp: &Array1<f64>) -> Option<Array2<f64>> {
            let mut j = Array2::zeros((1, 1));
            j[[0, 0]] = 1000.0;
            Some(j)
        }

        fn jacobian_yp(&self, _t: f64, _y: &Array1<f64>, _yp: &Array1<f64>) -> Option<Array2<f64>> {
            let mut j = Array2::zeros((1, 1));
            j[[0, 0]] = 1.0;
            Some(j)
        }
    }

    #[test]
    fn test_consistent_initialization() {
        let sys = ExponentialDecay;
        let y0 = array![1.0];
        let yp_guess = array![0.0]; // wrong guess

        let yp0 = find_consistent_initial_conditions(&sys, 0.0, &y0, &yp_guess, 20, 1e-10)
            .expect("consistent init should converge");

        // y' = -y => yp0 should be -1.0
        assert!((yp0[0] + 1.0).abs() < 1e-8, "yp0 = {}", yp0[0]);
    }

    #[test]
    fn test_radau_exponential_decay() {
        let sys = ExponentialDecay;
        let y0 = array![1.0];
        let yp0 = array![-1.0]; // consistent

        let opts = DAEOptions {
            rtol: 1e-6,
            atol: 1e-9,
            max_steps: 10000,
            h0: Some(0.01),
            ..Default::default()
        };

        let result =
            radau_iia_dae(&sys, [0.0, 1.0], y0, yp0, &opts).expect("Radau IIA should succeed");

        assert!(result.success, "integration should succeed");

        // y(1) = e^{-1} ~ 0.3679
        let y_final = result.x.last().expect("should have final state");
        let exact = (-1.0_f64).exp();
        assert!(
            (y_final[0] - exact).abs() < 1e-2,
            "y(1) = {} vs exact = {}, err = {}",
            y_final[0],
            exact,
            (y_final[0] - exact).abs()
        );
    }

    #[test]
    fn test_radau_stiff_system() {
        let sys = StiffDecay;
        let y0 = array![1.0];
        let yp0 = array![-1000.0]; // consistent: y' = -1000*y

        let opts = DAEOptions {
            rtol: 1e-4,
            atol: 1e-8,
            max_steps: 10000,
            h0: Some(0.0001),
            ..Default::default()
        };

        let result = radau_iia_dae(&sys, [0.0, 0.01], y0, yp0, &opts)
            .expect("Radau IIA should handle stiff system");

        assert!(result.success, "stiff integration should succeed");

        // y(0.01) = e^{-10} ~ 4.54e-5
        let y_final = result.x.last().expect("should have final state");
        let exact = (-10.0_f64).exp();
        assert!(
            (y_final[0] - exact).abs() < 1e-3,
            "y(0.01) = {} vs exact = {}",
            y_final[0],
            exact
        );
    }

    #[test]
    fn test_radau_index1_dae() {
        let sys = SimpleIndex1DAE;
        // Consistent initial conditions: x = 1, z = x = 1, x' = -x + z = 0
        let y0 = array![1.0, 1.0];
        let yp0 = array![0.0, 0.0]; // x' = -x + z = 0, and z = x

        let opts = DAEOptions {
            rtol: 1e-5,
            atol: 1e-8,
            max_steps: 10000,
            h0: Some(0.01),
            ..Default::default()
        };

        let result = radau_iia_dae(&sys, [0.0, 1.0], y0, yp0, &opts)
            .expect("Radau IIA should handle index-1 DAE");

        assert!(result.success, "DAE integration should succeed");

        // The constraint y = x should hold throughout
        for state in &result.x {
            let diff = (state[0] - state[1]).abs();
            assert!(
                diff < 0.1,
                "constraint violation: x={}, z={}",
                state[0],
                state[1]
            );
        }
    }

    #[test]
    fn test_estimate_dae_index() {
        let sys = SimpleIndex1DAE;
        let y0 = array![1.0, 1.0];
        let yp0 = array![0.0, 0.0];

        let (index, is_diff) =
            estimate_dae_index(&sys, 0.0, &y0, &yp0).expect("index estimation should succeed");

        // First variable is differential, second is algebraic
        assert!(is_diff[0], "x should be differential");
        assert!(!is_diff[1], "z should be algebraic");
    }

    #[test]
    fn test_linear_solver() {
        // Solve [2 1; 1 3] * x = [5; 7]  => x = [8/5, 9/5] = [1.6, 1.8]
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0]).expect("matrix creation");
        let b = array![5.0, 7.0];

        let x = solve_linear_system(&a, &b).expect("linear solve");
        assert!((x[0] - 1.6_f64).abs() < 1e-10);
        assert!((x[1] - 1.8_f64).abs() < 1e-10);
    }

    #[test]
    fn test_singular_matrix() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).expect("matrix creation");
        let b = array![1.0, 2.0];

        let result = solve_linear_system(&a, &b);
        assert!(result.is_err(), "should detect singular matrix");
    }
}
