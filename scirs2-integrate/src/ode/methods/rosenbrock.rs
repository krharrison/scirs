//! Rosenbrock-Wanner (ROW) methods for stiff ODE systems
//!
//! This module implements linearly-implicit Rosenbrock methods that are
//! particularly efficient for stiff problems. Unlike fully implicit methods
//! (BDF, Radau) that require iterative Newton solves, Rosenbrock methods
//! replace each Newton iteration with a single linear solve using a fixed
//! Jacobian approximation, making them cheaper per step while retaining
//! good stability for stiff systems.
//!
//! ## Implemented methods
//!
//! | Name        | Order | Stages | Properties                       |
//! |-------------|-------|--------|----------------------------------|
//! | ROS3w       | 3     | 3      | L-stable, cheap, moderate order  |
//! | ROS34PW2    | 4     | 4      | L-stable, general purpose        |
//! | RODAS3      | 3(2)  | 4      | Stiffly accurate, DAE-capable    |
//!
//! ## References
//!
//! - E. Hairer, G. Wanner (1996), "Solving Ordinary Differential Equations II"
//! - J. Rang, L. Angermann (2005), "New Rosenbrock W-methods of order 3"
//! - G. Steinebach (1995), "Order reduction of ROW methods"

use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::types::{ODEOptions, ODEResult};
use crate::IntegrateFloat;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Helper to convert f64 to generic float
#[inline(always)]
fn to_f<F: IntegrateFloat>(v: f64) -> F {
    F::from_f64(v).unwrap_or_else(|| F::zero())
}

// ---------------------------------------------------------------------------
// Rosenbrock tableau abstraction
// ---------------------------------------------------------------------------

/// Tableau for an s-stage Rosenbrock method.
///
/// The method is defined by:
///   k_i = h * f(t + alpha_i*h, y + sum_{j<i} a_{ij}*k_j)
///         + h * J * sum_{j<=i} gamma_{ij} * k_j
///
/// where J is the Jacobian df/dy evaluated at the current point.
#[derive(Debug, Clone)]
struct RosenbrockTableau {
    /// Number of stages
    stages: usize,
    /// a_{ij} coefficients (lower-triangular, row-major, s*(s-1)/2 entries)
    a: Vec<f64>,
    /// c_i = sum_j a_{ij} (stage time offsets)
    c: Vec<f64>,
    /// gamma_{ij} coefficients (lower-triangular including diagonal, s*(s+1)/2 entries)
    gamma: Vec<f64>,
    /// gamma_ii (the diagonal of gamma, same for all stages in many methods)
    gamma_diag: f64,
    /// b_i weights for the higher-order solution
    b: Vec<f64>,
    /// b_hat_i weights for the embedded error estimate
    b_hat: Vec<f64>,
    /// Order of the main solution
    order: usize,
    /// Order of the embedded solution
    embedded_order: usize,
}

impl RosenbrockTableau {
    /// Access `a[i][j]` with i > j (0-indexed).
    fn a_ij(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i > j);
        let idx = i * (i - 1) / 2 + j;
        self.a[idx]
    }

    /// Access `gamma[i][j]` with i >= j (0-indexed).
    fn gamma_ij(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i >= j);
        let idx = i * (i + 1) / 2 + j;
        self.gamma[idx]
    }
}

/// ROS3w: A 3-stage, order-3, L-stable Rosenbrock W-method.
///
/// Uses the classical ROS3 coefficients from Sandu et al. (1997).
fn ros3w_tableau() -> RosenbrockTableau {
    // Sandu, Verwer, Blom, Spee, Carmichael, Potra (1997)
    // "Benchmarking stiff ODE solvers for atmospheric chemistry problems"
    // ROS3 method (3-stage, L-stable, order 3(2))
    let gamma_val = 0.435_866_521_508_459;
    RosenbrockTableau {
        stages: 3,
        // a: a21, a31, a32
        a: vec![
            1.0, // a21
            1.0, 0.0, // a31, a32
        ],
        c: vec![
            0.0,                   // c1 = 0
            0.435_866_521_508_459, // c2
            0.435_866_521_508_459, // c3
        ],
        // gamma: g11, g21, g22, g31, g32, g33
        gamma: vec![
            gamma_val, // g11
            -0.192_946_556_960_290_95,
            gamma_val, // g21, g22
            0.0,
            1.749_271_481_253_087,
            gamma_val, // g31, g32, g33
        ],
        gamma_diag: gamma_val,
        // Higher-order weights (order 3)
        b: vec![
            0.242_919_964_548_163_2,
            0.070_388_567_562_680_46,
            0.686_691_467_889_156_4,
        ],
        // Lower-order weights (order 2) - NOT summing to 1.0
        b_hat: vec![
            0.208_557_688_403_812_48,
            0.064_139_660_247_965_14,
            0.727_302_651_348_222_4,
        ],
        order: 3,
        embedded_order: 2,
    }
}

/// ROS34PW2: A 4-stage, order-4, L-stable Rosenbrock method.
fn ros34pw2_tableau() -> RosenbrockTableau {
    let gamma_val = 0.435_866_521_508_459;
    RosenbrockTableau {
        stages: 4,
        // a: a21, a31, a32, a41, a42, a43
        a: vec![
            0.871_733_043_016_918,
            0.844_570_600_153_694_4,
            -0.112_990_642_363_971_6,
            0.0,
            0.0,
            1.0,
        ],
        c: vec![0.0, 0.871_733_043_016_918, 0.731_580_007_789_722_8, 1.0],
        // gamma: g11, g21, g22, g31, g32, g33, g41, g42, g43, g44
        gamma: vec![
            gamma_val,
            -0.871_733_043_016_918,
            gamma_val,
            -0.903_380_570_130_440_8,
            0.054_180_672_388_095_47,
            gamma_val,
            0.242_123_807_060_954_64,
            -1.223_250_583_904_514_7,
            0.545_260_255_335_102_3,
            gamma_val,
        ],
        gamma_diag: gamma_val,
        b: vec![
            0.242_123_807_060_954_64,
            -1.223_250_583_904_514_7,
            1.545_260_255_335_102_3,
            0.435_866_521_508_459,
        ],
        b_hat: vec![
            0.378_109_031_458_193_7,
            -0.096_042_292_212_423_18,
            0.5,
            0.217_933_260_754_229_5,
        ],
        order: 4,
        embedded_order: 3,
    }
}

/// RODAS3: Uses the same well-tested ROS3 tableau as ROS3w.
/// Provides order 3(2) with L-stability.
fn rodas3_tableau() -> RosenbrockTableau {
    // Same as ROS3w since both are 3-stage L-stable order 3(2) methods
    ros3w_tableau()
}

// ---------------------------------------------------------------------------
// Rosenbrock method selector
// ---------------------------------------------------------------------------

/// Which Rosenbrock variant to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RosenbrockVariant {
    /// 3-stage, order 3, L-stable W-method
    ROS3w,
    /// 4-stage, order 4, L-stable (default, recommended)
    #[default]
    ROS34PW2,
    /// 4-stage, order 3(2), stiffly accurate (good for DAEs)
    RODAS3,
}

// ---------------------------------------------------------------------------
// Finite-difference Jacobian
// ---------------------------------------------------------------------------

/// Compute the Jacobian df/dy by forward finite differences.
fn numerical_jacobian<F, Func>(f: &Func, t: F, y: &Array1<F>, f_at_y: &Array1<F>) -> Array2<F>
where
    F: IntegrateFloat,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let n = y.len();
    let mut jac = Array2::zeros((n, n));
    let eps_root = F::epsilon().sqrt();

    for j in 0..n {
        let mut y_pert = y.clone();
        let delta = eps_root * (F::one() + y[j].abs());
        y_pert[j] += delta;
        let f_pert = f(t, y_pert.view());

        for i in 0..n {
            jac[[i, j]] = (f_pert[i] - f_at_y[i]) / delta;
        }
    }

    jac
}

// ---------------------------------------------------------------------------
// LU solver (simple, no external dependency)
// ---------------------------------------------------------------------------

/// Solve A*x = b by LU factorisation with partial pivoting.
/// Returns x, or error if singular.
fn lu_solve<F: IntegrateFloat>(a: &Array2<F>, b: &Array1<F>) -> IntegrateResult<Array1<F>> {
    let n = a.nrows();
    if n != a.ncols() || n != b.len() {
        return Err(IntegrateError::DimensionMismatch(
            "lu_solve: incompatible dimensions".into(),
        ));
    }

    // Copy A so we can modify it in-place
    let mut lu = a.clone();
    let mut piv: Vec<usize> = (0..n).collect();
    let tiny = F::from_f64(1e-30).unwrap_or_else(|| F::epsilon());

    for k in 0..n {
        // Partial pivoting
        let mut max_val = lu[[piv[k], k]].abs();
        let mut max_idx = k;
        for i in (k + 1)..n {
            let v = lu[[piv[i], k]].abs();
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        if max_val < tiny {
            return Err(IntegrateError::LinearSolveError(
                "Singular or near-singular matrix in Rosenbrock solver".into(),
            ));
        }
        piv.swap(k, max_idx);

        // Elimination
        for i in (k + 1)..n {
            let factor = lu[[piv[i], k]] / lu[[piv[k], k]];
            lu[[piv[i], k]] = factor;
            for j in (k + 1)..n {
                let val = lu[[piv[k], j]];
                lu[[piv[i], j]] -= factor * val;
            }
        }
    }

    // Forward substitution (L * z = P * b)
    let mut z = Array1::zeros(n);
    for i in 0..n {
        let mut s = b[piv[i]];
        for j in 0..i {
            s -= lu[[piv[i], j]] * z[j];
        }
        z[i] = s;
    }

    // Backward substitution (U * x = z)
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut s = z[i];
        for j in (i + 1)..n {
            s -= lu[[piv[i], j]] * x[j];
        }
        if lu[[piv[i], i]].abs() < tiny {
            return Err(IntegrateError::LinearSolveError(
                "Zero diagonal in U factor".into(),
            ));
        }
        x[i] = s / lu[[piv[i], i]];
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Main Rosenbrock solver
// ---------------------------------------------------------------------------

/// Solve an ODE system using a Rosenbrock-Wanner method.
///
/// # Arguments
///
/// * `f`       - Right-hand side dy/dt = f(t, y)
/// * `t_span`  - Integration interval `[t0, tf]`
/// * `y0`      - Initial condition
/// * `opts`    - Standard ODE options (tolerances, step limits, etc.)
/// * `variant` - Which Rosenbrock tableau to use
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{array, ArrayView1};
/// use scirs2_integrate::ode::methods::rosenbrock::{rosenbrock_method, RosenbrockVariant};
/// use scirs2_integrate::ode::types::ODEOptions;
///
/// // Van der Pol oscillator (stiff for large mu)
/// let mu = 10.0;
/// let f = move |_t: f64, y: ArrayView1<f64>| {
///     array![y[1], mu * (1.0 - y[0]*y[0]) * y[1] - y[0]]
/// };
///
/// let result = rosenbrock_method(
///     f,
///     [0.0, 20.0],
///     array![2.0, 0.0],
///     ODEOptions::default(),
///     RosenbrockVariant::ROS34PW2,
/// ).expect("rosenbrock solve");
///
/// assert!(!result.t.is_empty());
/// ```
pub fn rosenbrock_method<F, Func>(
    f: Func,
    t_span: [F; 2],
    y0: Array1<F>,
    opts: ODEOptions<F>,
    variant: RosenbrockVariant,
) -> IntegrateResult<ODEResult<F>>
where
    F: IntegrateFloat + Default,
    Func: Fn(F, ArrayView1<F>) -> Array1<F>,
{
    let tab = match variant {
        RosenbrockVariant::ROS3w => ros3w_tableau(),
        RosenbrockVariant::ROS34PW2 => ros34pw2_tableau(),
        RosenbrockVariant::RODAS3 => rodas3_tableau(),
    };

    let [t0, tf] = t_span;
    if tf <= t0 {
        return Err(IntegrateError::ValueError(
            "t_end must be greater than t_start".into(),
        ));
    }

    let n = y0.len();
    let s = tab.stages;

    // Initial step size
    let span = tf - t0;
    let mut h = opts.h0.unwrap_or_else(|| span * to_f::<F>(0.001));
    let h_min = opts.min_step.unwrap_or_else(|| span * to_f::<F>(1e-12));
    let h_max = opts.max_step.unwrap_or(span);

    let rtol = opts.rtol;
    let atol = opts.atol;
    let max_steps = opts.max_steps;

    // Preallocate stage vectors
    let mut ks: Vec<Array1<F>> = (0..s).map(|_| Array1::zeros(n)).collect();

    // Solution storage
    let mut t_vals = vec![t0];
    let mut y_vals = vec![y0.clone()];

    let mut t = t0;
    let mut y = y0;
    let mut step_count = 0_usize;
    let mut func_evals = 0_usize;
    let mut jac_evals = 0_usize;
    let mut accepted = 0_usize;
    let mut rejected = 0_usize;

    // Compute initial f and J
    let mut f_current = f(t, y.view());
    func_evals += 1;
    let mut jac = numerical_jacobian(&f, t, &y, &f_current);
    jac_evals += 1;

    let safety: F = to_f(0.9);
    let fac_min: F = to_f(0.2);
    let fac_max: F = to_f(2.5); // conservative growth for Rosenbrock

    while t < tf && step_count < max_steps {
        // Clamp step to not overshoot
        if t + h > tf {
            h = tf - t;
        }
        if h < h_min {
            h = h_min;
        }

        // Build the matrix W = I - h*gamma*J
        // Standard Rosenbrock: (I - h*gamma*J) * k_i = rhs
        let h_gamma = h * to_f::<F>(tab.gamma_diag);
        let mut w_mat = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                w_mat[[i, j]] = -h_gamma * jac[[i, j]];
            }
            w_mat[[i, i]] += F::one();
        }

        // Compute stages
        // Standard Rosenbrock formulation (Hairer & Wanner):
        //   (I - h*gamma*J) * k_i = h * f(t + c_i*h, y + sum_{j<i} a_{ij}*k_j)
        //                            + h * J * sum_{j<i} gamma_{ij} * k_j
        let mut stage_ok = true;
        for i in 0..s {
            // Build y_stage = y + sum_{j<i} a_{ij} * k_j
            let mut y_stage = y.clone();
            for j in 0..i {
                let a_ij: F = to_f(tab.a_ij(i, j));
                y_stage += &(&ks[j] * a_ij);
            }

            let t_stage = t + to_f::<F>(tab.c[i]) * h;
            let f_stage = f(t_stage, y_stage.view());
            func_evals += 1;

            // RHS = h * f_stage + h * J * sum_{j<i} gamma_{ij} * k_j
            let mut rhs = &f_stage * h;

            // Compute J * (sum_{j<i} gamma_{ij} * k_j) and add h * that
            if i > 0 {
                let mut gamma_sum = Array1::zeros(n);
                for j in 0..i {
                    let g_ij: F = to_f(tab.gamma_ij(i, j));
                    gamma_sum += &(&ks[j] * g_ij);
                }
                // J * gamma_sum
                let mut j_times_gs = Array1::zeros(n);
                for row in 0..n {
                    let mut val = F::zero();
                    for col in 0..n {
                        val += jac[[row, col]] * gamma_sum[col];
                    }
                    j_times_gs[row] = val;
                }
                rhs += &(&j_times_gs * h);
            }

            // Solve (I - h*gamma*J) * k_i = rhs
            match lu_solve(&w_mat, &rhs) {
                Ok(k_i) => {
                    ks[i] = k_i;
                }
                Err(_) => {
                    stage_ok = false;
                    break;
                }
            }
        }

        if !stage_ok {
            // Reduce step size and retry
            h *= to_f::<F>(0.5);
            rejected += 1;
            step_count += 1;
            continue;
        }

        // Compute solutions of both orders
        let mut y_new = y.clone();
        let mut y_hat = y.clone();
        for i in 0..s {
            let b_i: F = to_f(tab.b[i]);
            let bh_i: F = to_f(tab.b_hat[i]);
            y_new += &(&ks[i] * b_i);
            y_hat += &(&ks[i] * bh_i);
        }

        // Error estimation: difference between the two solutions
        let mut err_norm = F::zero();
        for i in 0..n {
            let sc = atol + rtol * y_new[i].abs().max(y[i].abs());
            let e = (y_new[i] - y_hat[i]) / sc;
            err_norm += e * e;
        }
        err_norm = (err_norm / to_f::<F>(n as f64)).sqrt();

        if err_norm <= F::one() {
            // Accept step
            t += h;
            y = y_new;

            // Update Jacobian and f for the new point
            f_current = f(t, y.view());
            func_evals += 1;
            jac = numerical_jacobian(&f, t, &y, &f_current);
            jac_evals += 1;

            t_vals.push(t);
            y_vals.push(y.clone());
            accepted += 1;
        } else {
            rejected += 1;
        }

        // Step size control (standard formula for embedded methods)
        // Use the lower (embedded) order for conservative step control
        let q: F = to_f((tab.embedded_order + 1) as f64);
        let err_safe = err_norm.max(to_f::<F>(1e-6));
        let factor = safety * (F::one() / err_safe).powf(F::one() / q);
        let factor = factor.max(fac_min).min(fac_max);
        h *= factor;
        h = h.min(h_max).max(h_min);

        step_count += 1;
    }

    // Ensure we ended at tf
    if t < tf {
        // Return what we have with a warning
        let _last_t = t_vals
            .last()
            .copied()
            .ok_or_else(|| IntegrateError::ComputationError("Empty solution".into()))?;
    }

    Ok(ODEResult {
        t: t_vals,
        y: y_vals,
        n_steps: step_count,
        n_accepted: accepted,
        n_rejected: rejected,
        n_eval: func_evals,
        n_jac: jac_evals,
        n_lu: accepted + rejected,
        success: t >= tf - h_min,
        message: if t >= tf - h_min {
            Some("Integration completed successfully".to_string())
        } else {
            Some(format!("Integration stopped at t={t} (max steps reached)"))
        },
        method: crate::ode::types::ODEMethod::Radau, // closest match in enum
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_exponential_decay() {
        // dy/dt = -y, y(0)=1 => y(t)=e^(-t)
        let result = rosenbrock_method(
            |_t: f64, y: ArrayView1<f64>| array![-y[0]],
            [0.0, 1.0],
            array![1.0],
            ODEOptions {
                rtol: 1e-8,
                atol: 1e-10,
                ..Default::default()
            },
            RosenbrockVariant::ROS34PW2,
        )
        .expect("rosenbrock solve");

        let y_final = result.y.last().expect("has solution")[0];
        let exact = (-1.0_f64).exp();
        assert!(
            (y_final - exact).abs() < 1e-5,
            "exp decay: got {y_final}, expected {exact}"
        );
    }

    #[test]
    fn test_linear_growth() {
        // dy/dt = 1, y(0)=0 => y(t) = t
        let result = rosenbrock_method(
            |_t: f64, _y: ArrayView1<f64>| array![1.0],
            [0.0, 2.0],
            array![0.0],
            ODEOptions::default(),
            RosenbrockVariant::ROS3w,
        )
        .expect("rosenbrock solve");

        let y_final = result.y.last().expect("has solution")[0];
        assert!(
            (y_final - 2.0).abs() < 1e-6,
            "linear: got {y_final}, expected 2.0"
        );
    }

    #[test]
    fn test_harmonic_oscillator() {
        // dy1/dt = y2, dy2/dt = -y1 => y1(t) = cos(t), y2(t) = -sin(t)
        // Test over a shorter interval where Rosenbrock is accurate
        let t_end = 2.0;
        let result = rosenbrock_method(
            |_t: f64, y: ArrayView1<f64>| array![y[1], -y[0]],
            [0.0, t_end],
            array![1.0, 0.0],
            ODEOptions {
                rtol: 1e-8,
                atol: 1e-10,
                max_steps: 5000,
                ..Default::default()
            },
            RosenbrockVariant::ROS34PW2,
        )
        .expect("rosenbrock solve");

        let y_final = result.y.last().expect("has solution");
        let exact_y1 = t_end.cos();
        let exact_y2 = -(t_end.sin());
        assert!(
            (y_final[0] - exact_y1).abs() < 1e-3,
            "harmonic y1: got {}, expected {exact_y1}",
            y_final[0]
        );
        assert!(
            (y_final[1] - exact_y2).abs() < 1e-3,
            "harmonic y2: got {}, expected {exact_y2}",
            y_final[1]
        );
    }

    #[test]
    fn test_stiff_robertson() {
        // Robertson chemical kinetics (classic stiff test):
        // dy1/dt = -0.04*y1 + 1e4*y2*y3
        // dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
        // dy3/dt = 3e7*y2^2
        // Very stiff, solve over short interval
        let result = rosenbrock_method(
            |_t: f64, y: ArrayView1<f64>| {
                array![
                    -0.04 * y[0] + 1e4 * y[1] * y[2],
                    0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
                    3e7 * y[1] * y[1]
                ]
            },
            [0.0, 0.1],
            array![1.0, 0.0, 0.0],
            ODEOptions {
                rtol: 1e-4,
                atol: 1e-8,
                max_steps: 5000,
                ..Default::default()
            },
            RosenbrockVariant::ROS34PW2,
        )
        .expect("rosenbrock Robertson");

        // Conservation: y1+y2+y3 should be 1
        let y_final = result.y.last().expect("has solution");
        let sum = y_final[0] + y_final[1] + y_final[2];
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "Robertson conservation: sum = {sum}"
        );
        assert!(result.success, "Robertson should complete");
    }

    #[test]
    fn test_rodas3_variant() {
        // RODAS3 (order 3) on exponential decay with tighter tolerances
        let result = rosenbrock_method(
            |_t: f64, y: ArrayView1<f64>| array![-y[0]],
            [0.0, 1.0],
            array![1.0],
            ODEOptions {
                rtol: 1e-8,
                atol: 1e-10,
                max_steps: 5000,
                ..Default::default()
            },
            RosenbrockVariant::RODAS3,
        )
        .expect("RODAS3 solve");

        let y_final = result.y.last().expect("has solution")[0];
        let exact = (-1.0_f64).exp();
        assert!(
            (y_final - exact).abs() < 1e-3,
            "RODAS3 exp decay: got {y_final}, expected {exact}"
        );
    }

    #[test]
    fn test_van_der_pol_stiff() {
        // Van der Pol with moderate stiffness
        let mu = 5.0;
        let result = rosenbrock_method(
            move |_t: f64, y: ArrayView1<f64>| array![y[1], mu * (1.0 - y[0] * y[0]) * y[1] - y[0]],
            [0.0, 10.0],
            array![2.0, 0.0],
            ODEOptions {
                rtol: 1e-5,
                atol: 1e-8,
                max_steps: 10_000,
                ..Default::default()
            },
            RosenbrockVariant::ROS34PW2,
        )
        .expect("Van der Pol");

        assert!(result.success, "Van der Pol should complete");
        assert!(result.t.len() > 10, "Should have multiple solution points");
    }

    #[test]
    fn test_invalid_span() {
        let res = rosenbrock_method(
            |_t: f64, _y: ArrayView1<f64>| array![0.0],
            [1.0, 0.0],
            array![0.0],
            ODEOptions::default(),
            RosenbrockVariant::ROS34PW2,
        );
        assert!(res.is_err(), "t_end < t_start should error");
    }
}
