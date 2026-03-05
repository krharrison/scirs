//! Operator Splitting Methods for PDEs
//!
//! This module implements operator-splitting time integrators that decompose
//! a PDE `du/dt = A(u) + B(u)` into sub-problems and solve each separately.
//!
//! # Available methods
//!
//! | Method | Order | Reference |
//! |---|---|---|
//! | Lie-Trotter (sequential) | 1 | Trotter 1959 |
//! | Strang splitting | 2 | Strang 1968 |
//! | Yoshida 4th-order | 4 | Yoshida 1990 |
//!
//! # Usage pattern
//! Each sub-solver receives the current state `u` and the sub-step size `dt`
//! and must advance the sub-problem by exactly `dt`.  The functions compose
//! these solvers to produce the full trajectory at equally-spaced time points.
//!
//! # Examples
//! ```rust,no_run
//! use scirs2_integrate::specialized::pde::splitting::{strang_split, lie_trotter_split};
//! use scirs2_core::ndarray::Array1;
//!
//! // Simple advection-diffusion: du/dt = -v*du/dx + D*d²u/dx²
//! // A: pure advection, B: pure diffusion
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let u0 = Array1::from_vec(vec![1.0, 0.5, 0.0, 0.0]);
//! let solve_a = |u: &Array1<f64>, dt: f64| -> Result<Array1<f64>, scirs2_integrate::error::IntegrateError> {
//!     // advection step (placeholder)
//!     Ok(u.clone())
//! };
//! let solve_b = |u: &Array1<f64>, dt: f64| -> Result<Array1<f64>, scirs2_integrate::error::IntegrateError> {
//!     // diffusion step (placeholder)
//!     Ok(u.clone())
//! };
//! let (ts, _us) = strang_split(&u0, (0.0, 1.0), 0.1, solve_a, solve_b)?;
//! # Ok(())
//! # }
//! ```

use scirs2_core::ndarray::{Array1, Array2};

use crate::error::IntegrateError;

// ─────────────────────────────────────────────────────────────────────────────
// Output type alias
// ─────────────────────────────────────────────────────────────────────────────

/// Trajectory output: `(time_points, state_matrix)`.
///
/// `time_points[k]` is the k-th output time and `state_matrix.row(k)` is the
/// state at that time.  The first row is the initial condition at `t_span.0`.
type Trajectory = (Vec<f64>, Array2<f64>);

// ─────────────────────────────────────────────────────────────────────────────
// Lie-Trotter splitting (first order)
// ─────────────────────────────────────────────────────────────────────────────

/// Lie-Trotter (sequential) splitting: first-order accurate in time.
///
/// For each time step advances with the composition `exp(dt·B)∘exp(dt·A)`:
/// ```text
///     u* = exp(dt·A) u_n
///     u_{n+1} = exp(dt·B) u*
/// ```
///
/// # Arguments
/// * `u0` – initial state vector
/// * `t_span` – `(t_start, t_end)`
/// * `dt` – time step (must be positive)
/// * `solve_a` – closure advancing the A sub-problem by `dt`
/// * `solve_b` – closure advancing the B sub-problem by `dt`
///
/// # Returns
/// `(time_points, states)` where `states.row(k)` is the solution at
/// `time_points[k]`.
///
/// # Errors
/// Propagates any error from `solve_a` or `solve_b`.
pub fn lie_trotter_split<SA, SB>(
    u0: &Array1<f64>,
    t_span: (f64, f64),
    dt: f64,
    solve_a: SA,
    solve_b: SB,
) -> Result<Trajectory, IntegrateError>
where
    SA: Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
    SB: Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
{
    validate_inputs(u0, t_span, dt)?;

    let (t0, t_end) = t_span;
    let n_steps = ((t_end - t0) / dt).ceil() as usize;
    let n_dofs = u0.len();

    let mut times = Vec::with_capacity(n_steps + 1);
    let mut states = Vec::with_capacity((n_steps + 1) * n_dofs);

    times.push(t0);
    states.extend(u0.iter().copied());

    let mut u = u0.clone();
    let mut t = t0;

    for step in 0..n_steps {
        let dt_actual = if t + dt > t_end { t_end - t } else { dt };
        if dt_actual <= 0.0 {
            break;
        }

        // A half, then B half
        let u_star = solve_a(&u, dt_actual).map_err(|e| {
            IntegrateError::ComputationError(format!("lie_trotter solve_a step {step}: {e}"))
        })?;
        u = solve_b(&u_star, dt_actual).map_err(|e| {
            IntegrateError::ComputationError(format!("lie_trotter solve_b step {step}: {e}"))
        })?;

        t += dt_actual;
        times.push(t);
        states.extend(u.iter().copied());
    }

    let n_out = times.len();
    let mat = Array2::from_shape_vec((n_out, n_dofs), states).map_err(|e| {
        IntegrateError::ComputationError(format!("Failed to build trajectory matrix: {e}"))
    })?;

    Ok((times, mat))
}

// ─────────────────────────────────────────────────────────────────────────────
// Strang splitting (second order)
// ─────────────────────────────────────────────────────────────────────────────

/// Strang splitting: second-order symmetric splitting.
///
/// For each time step advances with
/// ```text
///     u* = exp(dt/2·A) u_n
///     u** = exp(dt·B) u*
///     u_{n+1} = exp(dt/2·A) u**
/// ```
///
/// # Arguments
/// * `u0` – initial state vector
/// * `t_span` – `(t_start, t_end)`
/// * `dt` – time step
/// * `solve_a` – closure advancing the A sub-problem by `dt`
/// * `solve_b` – closure advancing the B sub-problem by `dt`
///
/// # Returns
/// `(time_points, states)` where `states.row(k)` is the solution at
/// `time_points[k]`.
///
/// # Errors
/// Propagates any error from `solve_a` or `solve_b`.
pub fn strang_split<SA, SB>(
    u0: &Array1<f64>,
    t_span: (f64, f64),
    dt: f64,
    solve_a: SA,
    solve_b: SB,
) -> Result<Trajectory, IntegrateError>
where
    SA: Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
    SB: Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
{
    validate_inputs(u0, t_span, dt)?;

    let (t0, t_end) = t_span;
    let n_steps = ((t_end - t0) / dt).ceil() as usize;
    let n_dofs = u0.len();

    let mut times = Vec::with_capacity(n_steps + 1);
    let mut states = Vec::with_capacity((n_steps + 1) * n_dofs);

    times.push(t0);
    states.extend(u0.iter().copied());

    let mut u = u0.clone();
    let mut t = t0;

    for step in 0..n_steps {
        let dt_actual = if t + dt > t_end { t_end - t } else { dt };
        if dt_actual <= 0.0 {
            break;
        }

        let half_dt = dt_actual / 2.0;

        // A(dt/2)
        let u1 = solve_a(&u, half_dt).map_err(|e| {
            IntegrateError::ComputationError(format!("strang solve_a (half) step {step}: {e}"))
        })?;

        // B(dt)
        let u2 = solve_b(&u1, dt_actual).map_err(|e| {
            IntegrateError::ComputationError(format!("strang solve_b step {step}: {e}"))
        })?;

        // A(dt/2)
        u = solve_a(&u2, half_dt).map_err(|e| {
            IntegrateError::ComputationError(format!("strang solve_a (half2) step {step}: {e}"))
        })?;

        t += dt_actual;
        times.push(t);
        states.extend(u.iter().copied());
    }

    let n_out = times.len();
    let mat = Array2::from_shape_vec((n_out, n_dofs), states).map_err(|e| {
        IntegrateError::ComputationError(format!("Failed to build trajectory matrix: {e}"))
    })?;

    Ok((times, mat))
}

// ─────────────────────────────────────────────────────────────────────────────
// Yoshida 4th-order splitting
// ─────────────────────────────────────────────────────────────────────────────

/// Yoshida 4th-order symmetric splitting.
///
/// Applies the Yoshida composition to achieve 4th-order accuracy:
/// ```text
///     w1 = 1 / (2 - 2^(1/3))
///     w0 = -2^(1/3) / (2 - 2^(1/3))
///     coefficients: [w1, w0, w1] for B and [w1/2, (w1+w0)/2, (w1+w0)/2, w1/2] for A
/// ```
///
/// # Arguments
/// Same as [`strang_split`].
///
/// # Errors
/// Propagates any error from `solve_a` or `solve_b`.
pub fn yoshida_split<SA, SB>(
    u0: &Array1<f64>,
    t_span: (f64, f64),
    dt: f64,
    solve_a: SA,
    solve_b: SB,
) -> Result<Trajectory, IntegrateError>
where
    SA: Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
    SB: Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
{
    validate_inputs(u0, t_span, dt)?;

    // Yoshida coefficients
    let cbrt2: f64 = 2.0f64.powf(1.0 / 3.0);
    let w1 = 1.0 / (2.0 - cbrt2);
    let w0 = -cbrt2 / (2.0 - cbrt2);

    // Sequence: A(c1) B(d1) A(c2) B(d2) A(c3) B(d3) A(c4)
    // c1=c4=w1/2, c2=c3=(w1+w0)/2, d1=d3=w1, d2=w0
    let c = [w1 / 2.0, (w1 + w0) / 2.0, (w1 + w0) / 2.0, w1 / 2.0];
    let d = [w1, w0, w1];

    let (t0, t_end) = t_span;
    let n_steps = ((t_end - t0) / dt).ceil() as usize;
    let n_dofs = u0.len();

    let mut times = Vec::with_capacity(n_steps + 1);
    let mut states = Vec::with_capacity((n_steps + 1) * n_dofs);

    times.push(t0);
    states.extend(u0.iter().copied());

    let mut u = u0.clone();
    let mut t = t0;

    for step in 0..n_steps {
        let dt_actual = if t + dt > t_end { t_end - t } else { dt };
        if dt_actual <= 0.0 {
            break;
        }

        // 7-stage Yoshida: A c[0], B d[0], A c[1], B d[1], A c[2], B d[2], A c[3]
        u = solve_a(&u, c[0] * dt_actual).map_err(|e| {
            IntegrateError::ComputationError(format!("yoshida A[0] step {step}: {e}"))
        })?;
        u = solve_b(&u, d[0] * dt_actual).map_err(|e| {
            IntegrateError::ComputationError(format!("yoshida B[0] step {step}: {e}"))
        })?;
        u = solve_a(&u, c[1] * dt_actual).map_err(|e| {
            IntegrateError::ComputationError(format!("yoshida A[1] step {step}: {e}"))
        })?;
        u = solve_b(&u, d[1] * dt_actual).map_err(|e| {
            IntegrateError::ComputationError(format!("yoshida B[1] step {step}: {e}"))
        })?;
        u = solve_a(&u, c[2] * dt_actual).map_err(|e| {
            IntegrateError::ComputationError(format!("yoshida A[2] step {step}: {e}"))
        })?;
        u = solve_b(&u, d[2] * dt_actual).map_err(|e| {
            IntegrateError::ComputationError(format!("yoshida B[2] step {step}: {e}"))
        })?;
        u = solve_a(&u, c[3] * dt_actual).map_err(|e| {
            IntegrateError::ComputationError(format!("yoshida A[3] step {step}: {e}"))
        })?;

        t += dt_actual;
        times.push(t);
        states.extend(u.iter().copied());
    }

    let n_out = times.len();
    let mat = Array2::from_shape_vec((n_out, n_dofs), states).map_err(|e| {
        IntegrateError::ComputationError(format!("Failed to build trajectory matrix: {e}"))
    })?;

    Ok((times, mat))
}

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive Strang splitting (step-size control via Richardson extrapolation)
// ─────────────────────────────────────────────────────────────────────────────

/// Adaptive Strang splitting with step-size control.
///
/// Uses a local error estimate via Richardson extrapolation (one coarse step vs
/// two fine half-steps) to accept/reject steps and adapt `dt`.
///
/// # Arguments
/// * `u0` – initial state
/// * `t_span` – `(t0, t_end)`
/// * `dt0` – initial step size
/// * `tol` – local error tolerance (absolute)
/// * `solve_a`, `solve_b` – sub-problem solvers
///
/// # Returns
/// `(time_points, states)`.  The number of rows equals the number of accepted
/// steps plus 1 (for the initial condition).
pub fn strang_split_adaptive<SA, SB>(
    u0: &Array1<f64>,
    t_span: (f64, f64),
    dt0: f64,
    tol: f64,
    solve_a: SA,
    solve_b: SB,
) -> Result<Trajectory, IntegrateError>
where
    SA: Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
    SB: Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
{
    validate_inputs(u0, t_span, dt0)?;
    if tol <= 0.0 {
        return Err(IntegrateError::InvalidInput(
            "tol must be positive".to_string(),
        ));
    }

    let (t0, t_end) = t_span;
    let n_dofs = u0.len();
    let mut times = vec![t0];
    let mut states: Vec<f64> = u0.iter().copied().collect();

    let mut u = u0.clone();
    let mut t = t0;
    let mut dt = dt0;
    let dt_min = dt0 * 1e-8;
    let dt_max = (t_end - t0) * 0.5;

    let max_steps = 100_000usize;
    for _ in 0..max_steps {
        if t >= t_end {
            break;
        }
        let dt_use = dt.min(t_end - t);
        if dt_use < dt_min {
            return Err(IntegrateError::ComputationError(
                "Adaptive Strang: step size too small".to_string(),
            ));
        }

        // One full Strang step with dt_use
        let u_coarse = strang_step(&u, dt_use, &solve_a, &solve_b)?;

        // Two Strang steps with dt_use/2
        let u_half = strang_step(&u, dt_use / 2.0, &solve_a, &solve_b)?;
        let u_fine = strang_step(&u_half, dt_use / 2.0, &solve_a, &solve_b)?;

        // Error estimate (Strang is 2nd order, Richardson: err ~ (u_fine - u_coarse)/3)
        let err: f64 = u_fine
            .iter()
            .zip(u_coarse.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        if err <= tol || dt_use <= dt_min {
            // Accept step (use Richardson-corrected solution)
            u = u_fine
                .iter()
                .zip(u_coarse.iter())
                .map(|(a, b)| a + (a - b) / 3.0)
                .collect::<Vec<f64>>()
                .into();
            t += dt_use;
            times.push(t);
            states.extend(u.iter().copied());

            // Increase step
            let factor = if err > 0.0 { 0.9 * (tol / err).sqrt() } else { 2.0 };
            dt = (dt * factor).min(dt_max);
        } else {
            // Reject step: decrease dt
            let factor = 0.9 * (tol / err).sqrt();
            dt = (dt * factor).max(dt_min);
        }
    }

    let n_out = times.len();
    let mat = Array2::from_shape_vec((n_out, n_dofs), states).map_err(|e| {
        IntegrateError::ComputationError(format!("Failed to build trajectory matrix: {e}"))
    })?;

    Ok((times, mat))
}

/// Perform a single Strang step (internal helper).
fn strang_step<SA, SB>(
    u: &Array1<f64>,
    dt: f64,
    solve_a: &SA,
    solve_b: &SB,
) -> Result<Array1<f64>, IntegrateError>
where
    SA: Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
    SB: Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
{
    let half = dt / 2.0;
    let u1 = solve_a(u, half)?;
    let u2 = solve_b(&u1, dt)?;
    solve_a(&u2, half)
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation helper
// ─────────────────────────────────────────────────────────────────────────────

fn validate_inputs(
    u0: &Array1<f64>,
    t_span: (f64, f64),
    dt: f64,
) -> Result<(), IntegrateError> {
    if u0.is_empty() {
        return Err(IntegrateError::InvalidInput(
            "u0 must not be empty".to_string(),
        ));
    }
    let (t0, t_end) = t_span;
    if t_end <= t0 {
        return Err(IntegrateError::InvalidInput(
            "t_span: t_end must be > t_start".to_string(),
        ));
    }
    if dt <= 0.0 {
        return Err(IntegrateError::InvalidInput(
            "dt must be positive".to_string(),
        ));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;
    use std::f64::consts::PI;

    /// Exact test: scalar linear ODE  du/dt = A(u) + B(u) = -λu
    /// Split A: du/dt = -λ₁u, B: du/dt = -λ₂u, with λ = λ₁ + λ₂.
    /// Exact: u(t) = u0 * exp(-λt)
    fn make_decay_solvers(
        lam_a: f64,
        lam_b: f64,
    ) -> (
        impl Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
        impl Fn(&Array1<f64>, f64) -> Result<Array1<f64>, IntegrateError>,
    ) {
        let solve_a = move |u: &Array1<f64>, dt: f64| -> Result<Array1<f64>, IntegrateError> {
            Ok(u.mapv(|v| v * (-lam_a * dt).exp()))
        };
        let solve_b = move |u: &Array1<f64>, dt: f64| -> Result<Array1<f64>, IntegrateError> {
            Ok(u.mapv(|v| v * (-lam_b * dt).exp()))
        };
        (solve_a, solve_b)
    }

    #[test]
    fn test_lie_trotter_convergence() {
        let u0 = Array1::from_vec(vec![1.0]);
        let lam = 2.0;
        let t_end = 1.0;
        let (solve_a, solve_b) = make_decay_solvers(1.0, 1.0);
        let (ts, states) =
            lie_trotter_split(&u0, (0.0, t_end), 0.01, solve_a, solve_b).expect("LT failed");

        let exact = (-lam * t_end).exp();
        let computed = states[[ts.len() - 1, 0]];
        let err = (computed - exact).abs();
        assert!(err < 1e-3, "Lie-Trotter error {err:.3e} too large");
    }

    #[test]
    fn test_strang_second_order() {
        // Strang should be 2nd order: err ∝ dt²
        let u0 = Array1::from_vec(vec![1.0]);
        let t_end = 1.0;
        let exact = (-2.0f64 * t_end).exp();

        let mut prev_err = f64::INFINITY;
        let mut order_sum = 0.0;
        let mut count = 0;

        for &dt in &[0.1, 0.05, 0.025] {
            let (solve_a, solve_b) = make_decay_solvers(1.0, 1.0);
            let (ts, states) =
                strang_split(&u0, (0.0, t_end), dt, solve_a, solve_b).expect("Strang failed");
            let computed = states[[ts.len() - 1, 0]];
            let err = (computed - exact).abs();

            if prev_err < f64::INFINITY && err > 1e-15 {
                let order = (prev_err / err).log2();
                order_sum += order;
                count += 1;
            }
            prev_err = err;
        }
        if count > 0 {
            let avg_order = order_sum / count as f64;
            assert!(
                avg_order > 1.8,
                "Strang order {avg_order:.2} < 2 expected"
            );
        }
    }

    #[test]
    fn test_yoshida_fourth_order() {
        let u0 = Array1::from_vec(vec![1.0]);
        let t_end = 0.5;
        let exact = (-2.0f64 * t_end).exp();

        let mut prev_err = f64::INFINITY;
        let mut order_sum = 0.0;
        let mut count = 0;

        for &dt in &[0.1, 0.05] {
            let (solve_a, solve_b) = make_decay_solvers(1.0, 1.0);
            let (ts, states) =
                yoshida_split(&u0, (0.0, t_end), dt, solve_a, solve_b).expect("Yoshida failed");
            let computed = states[[ts.len() - 1, 0]];
            let err = (computed - exact).abs();

            if prev_err < f64::INFINITY && err > 1e-15 {
                let order = (prev_err / err).log2();
                order_sum += order;
                count += 1;
            }
            prev_err = err;
        }
        if count > 0 {
            let avg_order = order_sum / count as f64;
            assert!(
                avg_order > 3.5,
                "Yoshida order {avg_order:.2} < 4 expected"
            );
        }
    }

    #[test]
    fn test_strang_multidimensional() {
        // 2D system: du₁/dt = -u₁, du₂/dt = -2u₂
        // A: affects u₁ only, B: affects u₂ only
        let u0 = Array1::from_vec(vec![1.0, 1.0]);
        let t_end = 1.0;

        let solve_a = |u: &Array1<f64>, dt: f64| -> Result<Array1<f64>, IntegrateError> {
            let mut v = u.clone();
            v[0] *= (-dt).exp();
            Ok(v)
        };
        let solve_b = |u: &Array1<f64>, dt: f64| -> Result<Array1<f64>, IntegrateError> {
            let mut v = u.clone();
            v[1] *= (-2.0 * dt).exp();
            Ok(v)
        };

        let (ts, states) =
            strang_split(&u0, (0.0, t_end), 0.05, solve_a, solve_b).expect("Strang 2D failed");

        let last = ts.len() - 1;
        let u1_exact = (-1.0f64 * t_end).exp();
        let u2_exact = (-2.0f64 * t_end).exp();
        let err1 = (states[[last, 0]] - u1_exact).abs();
        let err2 = (states[[last, 1]] - u2_exact).abs();

        assert!(err1 < 1e-4, "u1 error {err1:.3e}");
        assert!(err2 < 1e-4, "u2 error {err2:.3e}");
    }

    #[test]
    fn test_adaptive_strang() {
        let u0 = Array1::from_vec(vec![1.0]);
        let lam = 3.0;
        let t_end = 1.0;
        let (solve_a, solve_b) = make_decay_solvers(1.5, 1.5);
        let (ts, states) =
            strang_split_adaptive(&u0, (0.0, t_end), 0.1, 1e-8, solve_a, solve_b)
                .expect("Adaptive Strang failed");

        let exact = (-lam * t_end).exp();
        let computed = states[[ts.len() - 1, 0]];
        let err = (computed - exact).abs();
        assert!(err < 1e-5, "Adaptive Strang error {err:.3e}");
    }

    #[test]
    fn test_invalid_inputs() {
        let u0 = Array1::<f64>::zeros(3);
        // t_end < t_start
        let (sa, sb) = make_decay_solvers(1.0, 1.0);
        assert!(strang_split(&u0, (1.0, 0.0), 0.1, sa, sb).is_err());
        // dt <= 0
        let (sa, sb) = make_decay_solvers(1.0, 1.0);
        assert!(strang_split(&u0, (0.0, 1.0), -0.1, sa, sb).is_err());
        // empty u0
        let empty = Array1::<f64>::zeros(0);
        let (sa, sb) = make_decay_solvers(1.0, 1.0);
        assert!(strang_split(&empty, (0.0, 1.0), 0.1, sa, sb).is_err());
    }

    /// Test with a simple advection-diffusion split to verify output shape
    #[test]
    fn test_trajectory_shape() {
        let n = 8;
        let u0 = Array1::linspace(0.0, 1.0, n);
        let solve_a = |u: &Array1<f64>, _dt: f64| -> Result<Array1<f64>, IntegrateError> {
            Ok(u.clone())
        };
        let solve_b = |u: &Array1<f64>, _dt: f64| -> Result<Array1<f64>, IntegrateError> {
            Ok(u.clone())
        };

        let dt = 0.1;
        let t_end = 1.0;
        let (ts, states) =
            strang_split(&u0, (0.0, t_end), dt, solve_a, solve_b).expect("shape test failed");

        assert_eq!(states.ncols(), n, "Wrong number of DOFs");
        assert_eq!(states.nrows(), ts.len(), "Rows/time mismatch");
        // Initial state preserved
        for j in 0..n {
            assert!((states[[0, j]] - u0[j]).abs() < 1e-15);
        }
    }
}
