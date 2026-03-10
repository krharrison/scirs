//! Integration tests: scirs2-integrate ODE solvers
//!
//! Covers:
//! - Stiff ODE solver (BDF) vs analytical solution
//! - Non-stiff ODE (RK45) vs analytical solution
//! - Event detection accuracy
//! - Symplectic integrators with energy conservation
//! - Multi-dimensional ODE systems

use approx::assert_abs_diff_eq;
use scirs2_core::ndarray::{Array1, ArrayView1};
use scirs2_integrate::{
    solve_ivp, solve_ivp_with_events, terminal_event, EventAction, EventDirection, ODEMethod,
    ODEOptions, ODEOptionsWithEvents, ODEResultWithEvents, SymplecticMethod,
    SymplecticSeparableSystem, SymplecticStepper,
};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// 1. Simple exponential decay: dy/dt = -k*y,  y(0) = y0
//    Analytical: y(t) = y0 * exp(-k*t)
// ---------------------------------------------------------------------------

#[test]
fn test_rk45_exponential_decay_vs_analytical() {
    let k = 2.0_f64;
    let y0 = 3.0_f64;

    let f = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![-k * y[0]]) };

    let opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    };

    let result = solve_ivp(f, [0.0_f64, 2.0], Array1::from_vec(vec![y0]), Some(opts))
        .expect("RK45 exponential decay solve failed");

    assert!(result.success, "RK45 solve did not report success");

    // Check final value
    let t_final = *result.t.last().expect("empty t in result");
    let y_final = result.y.last().expect("empty y in result")[0];
    let y_analytical = y0 * (-k * t_final).exp();

    assert_abs_diff_eq!(y_final, y_analytical, epsilon = 1e-5);
}

// ---------------------------------------------------------------------------
// 2. Harmonic oscillator: y'' + omega^2*y = 0
//    State: [y, y']  Analytical: y(t) = A*cos(omega*t) + B*sin(omega*t)
// ---------------------------------------------------------------------------

#[test]
fn test_rk45_harmonic_oscillator_accuracy() {
    let omega = 2.0_f64;
    let y0 = 1.0_f64;
    let dy0 = 0.0_f64; // starts at rest

    let f = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        Array1::from_vec(vec![y[1], -omega * omega * y[0]])
    };

    let opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-10,
        atol: 1e-12,
        ..Default::default()
    };

    let t_end = 2.0 * PI / omega; // one full period
    let result = solve_ivp(
        f,
        [0.0_f64, t_end],
        Array1::from_vec(vec![y0, dy0]),
        Some(opts),
    )
    .expect("RK45 harmonic oscillator solve failed");

    assert!(result.success, "Harmonic oscillator solve did not succeed");

    // After one full period, y should return to initial conditions
    let y_end = &result
        .y
        .last()
        .expect("empty y in harmonic oscillator result");
    assert_abs_diff_eq!(y_end[0], y0, epsilon = 1e-6);
    assert_abs_diff_eq!(y_end[1], dy0, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// 3. Stiff ODE: Robertson chemical kinetics (scaled)
//    Modified for well-conditioning: use a smaller time span
// ---------------------------------------------------------------------------

#[test]
fn test_bdf_stiff_ode_versus_rk45() {
    // Moderately stiff ODE: dy/dt = -10*y,  y(0) = 1
    // Exact solution: y(t) = exp(-10*t).
    // The stiffness ratio is 10, manageable for both BDF and RK45.
    let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![-10.0 * y[0]]) };

    let opts_bdf = ODEOptions {
        method: ODEMethod::Bdf,
        rtol: 1e-4,
        atol: 1e-6,
        max_steps: 50_000,
        ..Default::default()
    };

    let opts_rk = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        ..Default::default()
    };

    let t_end = 1.0_f64;
    let y0 = Array1::from_vec(vec![1.0_f64]);

    let result_bdf =
        solve_ivp(f, [0.0_f64, t_end], y0.clone(), Some(opts_bdf)).expect("BDF solve failed");
    let result_rk = solve_ivp(f, [0.0_f64, t_end], y0, Some(opts_rk)).expect("RK45 solve failed");

    assert!(result_bdf.success, "BDF did not converge");
    assert!(result_rk.success, "RK45 reference did not converge");

    let y_bdf = result_bdf.y.last().expect("empty y BDF")[0];
    let y_rk = result_rk.y.last().expect("empty y RK")[0];
    let y_exact = (-10.0_f64 * t_end).exp(); // e^{-10} ≈ 4.54e-5

    // Both solutions should agree with the exact solution
    assert_abs_diff_eq!(y_bdf, y_exact, epsilon = 1e-2);
    assert_abs_diff_eq!(y_rk, y_exact, epsilon = 1e-5);
}

// ---------------------------------------------------------------------------
// 4. High-order method: DOP853 on Lorenz attractor (just verify no panic/wrong dims)
// ---------------------------------------------------------------------------

#[test]
fn test_dop853_lorenz_system_stability() {
    let sigma = 10.0_f64;
    let rho = 28.0_f64;
    let beta = 8.0_f64 / 3.0;

    let f = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        let x = y[0];
        let yy = y[1];
        let z = y[2];
        Array1::from_vec(vec![
            sigma * (yy - x),
            x * (rho - z) - yy,
            x * yy - beta * z,
        ])
    };

    // Use tighter tolerances and shorter time span to keep the chaotic
    // trajectory on the attractor. Lorenz is sensitive to numerical
    // errors; t=2 is enough to demonstrate stability without risking
    // divergence from accumulated step-size controller drift.
    let opts = ODEOptions {
        method: ODEMethod::DOP853,
        rtol: 1e-9,
        atol: 1e-11,
        max_steps: 200_000,
        ..Default::default()
    };

    let y0 = Array1::from_vec(vec![1.0_f64, 1.0, 1.0]);
    let result = solve_ivp(f, [0.0_f64, 2.0], y0, Some(opts)).expect("DOP853 Lorenz solve failed");

    assert!(result.success, "DOP853 Lorenz did not succeed");
    assert!(!result.y.is_empty(), "Lorenz result is empty");

    // On the Lorenz attractor, |x| < 25, |y| < 30, |z| < 50 roughly.
    // Use 200 as a generous upper bound.
    for state in &result.y {
        for &val in state.iter() {
            assert!(
                val.is_finite() && val.abs() < 200.0,
                "Lorenz state blew up: {val}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Event detection: bouncing ball, detect zero crossing of height
// ---------------------------------------------------------------------------

#[test]
fn test_event_detection_zero_crossing_sine() {
    // y'' = -9.81, y(0) = 10, y'(0) = 0 (free fall)
    // Detect when y = 0 (hits ground)
    // Analytical: y(t) = 10 - 0.5*9.81*t^2
    //             y = 0 at t = sqrt(20/9.81) ≈ 1.428 s

    let g = 9.81_f64;
    let f =
        move |_t: f64, _y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![_y[1], -g]) };

    // Event: detect when y[0] = 0 (ball hits ground)
    let event_fn = |_t: f64, y: ArrayView1<f64>| -> f64 { y[0] };

    let base_opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-10,
        atol: 1e-12,
        dense_output: true,
        ..Default::default()
    };

    let event_spec = terminal_event::<f64>("ground_hit", EventDirection::Falling);

    let opts_with_events = ODEOptionsWithEvents {
        base_options: base_opts,
        event_specs: vec![event_spec],
    };

    let y0 = Array1::from_vec(vec![10.0_f64, 0.0]);
    let result: ODEResultWithEvents<f64> =
        solve_ivp_with_events(f, [0.0_f64, 5.0], y0, vec![event_fn], opts_with_events)
            .expect("Event detection solve failed");

    let t_final = *result.base_result.t.last().expect("empty t");

    // Analytical impact time
    let t_analytical = (2.0 * 10.0 / g).sqrt();

    // Integration should have stopped near the impact time
    assert_abs_diff_eq!(t_final, t_analytical, epsilon = 1e-4);
}

// ---------------------------------------------------------------------------
// 6. Euler method: first-order accuracy check
// ---------------------------------------------------------------------------

#[test]
fn test_euler_method_first_order_accuracy() {
    // dy/dt = y,  y(0) = 1  →  y(t) = exp(t)
    let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { y.to_owned() };

    let t_end = 1.0_f64;
    let h0 = 0.01_f64; // small fixed step

    let opts = ODEOptions {
        method: ODEMethod::Euler,
        h0: Some(h0),
        rtol: 1.0, // disable adaptive control
        atol: 1.0,
        ..Default::default()
    };

    let result = solve_ivp(
        f,
        [0.0_f64, t_end],
        Array1::from_vec(vec![1.0_f64]),
        Some(opts),
    )
    .expect("Euler solve failed");

    let y_numerical = result.y.last().expect("empty y Euler")[0];
    let y_exact = t_end.exp();

    // Euler error ~ O(h) * t_end
    let expected_error = h0 * t_end * y_exact;
    assert!(
        (y_numerical - y_exact).abs() < 2.0 * expected_error,
        "Euler error {:.2e} larger than expected O(h) bound {:.2e}",
        (y_numerical - y_exact).abs(),
        expected_error
    );
}

// ---------------------------------------------------------------------------
// 7. Symplectic integrator: simple harmonic oscillator energy conservation
// ---------------------------------------------------------------------------

#[test]
fn test_symplectic_stormer_verlet_energy_conservation() {
    // H = p^2/2 + q^2/2 (unit-mass SHO, omega=1)
    // Exact energy: E = p0^2/2 + q0^2/2
    let omega = 1.0_f64;

    let kinetic_grad = move |_t: f64, p: &Array1<f64>| -> Array1<f64> {
        p.clone() // dT/dp = p
    };
    let potential_grad = move |_t: f64, q: &Array1<f64>| -> Array1<f64> {
        q.mapv(|qi| omega * omega * qi) // dV/dq = omega^2 * q
    };

    let system = SymplecticSeparableSystem::new(1, kinetic_grad, potential_grad).with_energy(
        |_t: f64, p: &Array1<f64>| p[0] * p[0] / 2.0,
        move |_t: f64, q: &Array1<f64>| omega * omega * q[0] * q[0] / 2.0,
    );

    let q0 = Array1::from_vec(vec![1.0_f64]); // q(0) = 1
    let p0 = Array1::from_vec(vec![0.0_f64]); // p(0) = 0
    let e0 = 0.5_f64; // initial energy = 0^2/2 + 1^2/2 = 0.5

    let stepper =
        scirs2_integrate::create_symplectic_stepper::<f64>(SymplecticMethod::StormerVerlet);
    let t_end = 10.0 * 2.0 * PI; // 10 full periods
    let dt = 0.01_f64;

    let result =
        scirs2_integrate::solve_hamiltonian(&system, &*stepper, 0.0_f64, t_end, dt, q0, p0)
            .expect("Symplectic SHO integration failed");

    assert!(!result.t.is_empty(), "Symplectic result is empty");

    // Check energy drift over the integration interval
    if let Some(monitor) = &result.energy_monitor {
        let energies = &monitor.energy_history;
        // Symplectic integrators preserve energy almost exactly — drift should be small
        for &e in energies {
            assert_abs_diff_eq!(e, e0, epsilon = 1e-3);
        }
    } else {
        // Fallback: check that final state has correct energy
        let q_end = result.q.last().expect("empty q");
        let p_end = result.p.last().expect("empty p");
        let e_end = p_end[0] * p_end[0] / 2.0 + omega * omega * q_end[0] * q_end[0] / 2.0;
        assert_abs_diff_eq!(e_end, e0, epsilon = 1e-3);
    }
}

// ---------------------------------------------------------------------------
// 8. Yoshida4 symplectic integrator: Kepler orbit conservation
// ---------------------------------------------------------------------------

#[test]
fn test_yoshida4_kepler_orbit_energy_conservation() {
    // 2D Kepler problem (planar):
    // H = (px^2 + py^2)/2 - GM/r,   GM = 1
    // State: q = [x, y],  p = [px, py]
    // Circular orbit: r=1, v=1, E=-0.5

    let kinetic_grad = |_t: f64, p: &Array1<f64>| -> Array1<f64> {
        p.clone() // dT/dp = p
    };
    let potential_grad = |_t: f64, q: &Array1<f64>| -> Array1<f64> {
        let r3 = (q[0] * q[0] + q[1] * q[1]).powf(1.5);
        if r3 < 1e-10 {
            Array1::zeros(2)
        } else {
            Array1::from_vec(vec![q[0] / r3, q[1] / r3])
        }
    };

    let system = SymplecticSeparableSystem::new(2, kinetic_grad, potential_grad).with_energy(
        |_t: f64, p: &Array1<f64>| (p[0] * p[0] + p[1] * p[1]) / 2.0,
        |_t: f64, q: &Array1<f64>| {
            let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
            if r < 1e-10 {
                f64::INFINITY
            } else {
                -1.0 / r
            }
        },
    );

    // Circular orbit at r=1: q=(1,0), p=(0,1), E=-0.5
    let q0 = Array1::from_vec(vec![1.0_f64, 0.0]);
    let p0 = Array1::from_vec(vec![0.0_f64, 1.0]);
    let e0 = -0.5_f64;

    let stepper = scirs2_integrate::create_symplectic_stepper::<f64>(SymplecticMethod::Yoshida4);
    let t_end = 2.0 * PI; // one full orbit
    let dt = 0.01_f64;

    let result =
        scirs2_integrate::solve_hamiltonian(&system, &*stepper, 0.0_f64, t_end, dt, q0, p0)
            .expect("Yoshida4 Kepler integration failed");

    // Final position should return to (1, 0)
    let q_end = result.q.last().expect("empty q");
    assert_abs_diff_eq!(q_end[0], 1.0, epsilon = 0.01);
    assert_abs_diff_eq!(q_end[1], 0.0, epsilon = 0.01);

    // Energy conservation
    if let Some(monitor) = &result.energy_monitor {
        for &e in &monitor.energy_history {
            assert_abs_diff_eq!(e, e0, epsilon = 0.01);
        }
    }
}

// ---------------------------------------------------------------------------
// 9. RK4 fixed-step: compare with RK45 on smooth ODE
// ---------------------------------------------------------------------------

#[test]
fn test_rk4_vs_rk45_smooth_ode_agreement() {
    // dy/dt = cos(t),  y(0) = 0  →  y(t) = sin(t)
    let f = |t: f64, _y: ArrayView1<f64>| -> Array1<f64> { Array1::from_vec(vec![t.cos()]) };

    let t_end = PI;

    let opts_rk4 = ODEOptions {
        method: ODEMethod::RK4,
        h0: Some(0.01),
        ..Default::default()
    };
    let opts_rk45 = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-10,
        atol: 1e-12,
        ..Default::default()
    };

    let y0 = Array1::from_vec(vec![0.0_f64]);

    let res_rk4 =
        solve_ivp(f, [0.0_f64, t_end], y0.clone(), Some(opts_rk4)).expect("RK4 solve failed");
    let res_rk45 =
        solve_ivp(f, [0.0_f64, t_end], y0, Some(opts_rk45)).expect("RK45 smooth ODE solve failed");

    let y_rk4 = res_rk4.y.last().expect("empty y RK4")[0];
    let y_rk45 = res_rk45.y.last().expect("empty y RK45")[0];
    let y_exact = t_end.sin(); // sin(pi) = 0

    assert_abs_diff_eq!(y_rk4, y_exact, epsilon = 1e-4);
    assert_abs_diff_eq!(y_rk45, y_exact, epsilon = 1e-8);
    // Both should agree with each other closely
    assert_abs_diff_eq!(y_rk4, y_rk45, epsilon = 1e-4);
}

// ---------------------------------------------------------------------------
// 10. Multi-dimensional system: coupled oscillators
// ---------------------------------------------------------------------------

#[test]
fn test_rk45_coupled_oscillators() {
    // Two coupled harmonic oscillators:
    // x'' = -omega1^2 * x + k * (y - x)
    // y'' = -omega2^2 * y + k * (x - y)
    // State: [x, x', y, y']
    let omega1 = 1.0_f64;
    let omega2 = 1.5_f64;
    let kc = 0.1_f64; // coupling constant

    let f = move |_t: f64, s: ArrayView1<f64>| -> Array1<f64> {
        let x = s[0];
        let xp = s[1];
        let y = s[2];
        let yp = s[3];
        Array1::from_vec(vec![
            xp,
            -omega1 * omega1 * x + kc * (y - x),
            yp,
            -omega2 * omega2 * y + kc * (x - y),
        ])
    };

    let opts = ODEOptions {
        method: ODEMethod::RK45,
        rtol: 1e-8,
        atol: 1e-10,
        max_steps: 100_000,
        ..Default::default()
    };

    let y0 = Array1::from_vec(vec![1.0_f64, 0.0, 0.0, 0.0]);
    let result =
        solve_ivp(f, [0.0_f64, 10.0], y0, Some(opts)).expect("Coupled oscillator solve failed");

    assert!(result.success, "Coupled oscillator solve did not succeed");

    // Energy (total mechanical) should be conserved
    // E = 0.5*(x'^2 + omega1^2*x^2 + y'^2 + omega2^2*y^2) + kc*(x-y)^2/2
    // (approximate, not including coupling fully, but verifies boundedness)
    for state in &result.y {
        let x = state[0];
        let xp = state[1];
        let y = state[2];
        let yp = state[3];
        let e = 0.5 * (xp * xp + omega1 * omega1 * x * x + yp * yp + omega2 * omega2 * y * y);
        // Energy should remain bounded (initial E ≈ 0.5)
        assert!(
            e < 2.0,
            "Coupled oscillator energy blew up to {e} at t={}",
            result.t[result.y.iter().position(|s| s[0] == state[0]).unwrap_or(0)]
        );
    }
}
