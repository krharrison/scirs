//! Tutorial: Numerical Integration with SciRS2
//!
//! This tutorial covers numerical quadrature, ODE solving,
//! and basic PDE solving using the scirs2-integrate crate.
//!
//! Run with: cargo run -p scirs2-integrate --example tutorial_integration

use scirs2_core::ndarray::{array, Array1, ArrayView1};
use scirs2_integrate::error::IntegrateResult;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 Integration Tutorial ===\n");

    section_quadrature()?;
    section_ode_solving()?;
    section_advanced_quadrature()?;

    println!("\n=== Tutorial Complete ===");
    Ok(())
}

/// Section 1: Numerical quadrature (definite integrals)
fn section_quadrature() -> IntegrateResult<()> {
    println!("--- 1. Numerical Quadrature ---\n");

    // Integrate sin(x) from 0 to pi
    // Exact answer: -cos(pi) - (-cos(0)) = 1 + 1 = 2
    let result = scirs2_integrate::quad(|x: f64| x.sin(), 0.0, std::f64::consts::PI, None)?;
    println!("integral(sin(x), 0, pi):");
    println!("  Result: {:.10}", result.value);
    println!("  Error:  {:.2e}", result.abs_error);
    println!("  Exact:  2.0");
    println!("  Abs diff: {:.2e}\n", (result.value - 2.0).abs());

    // Integrate x^2 from 0 to 1
    // Exact: 1/3
    let result2 = scirs2_integrate::quad(|x: f64| x * x, 0.0, 1.0, None)?;
    println!("integral(x^2, 0, 1):");
    println!("  Result: {:.10}", result2.value);
    println!("  Exact:  {:.10}\n", 1.0 / 3.0);

    // Simpson's rule (simpler, fixed-order method)
    let result_simp = scirs2_integrate::simpson(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 100)?;
    println!("Simpson's rule for sin(x):");
    println!("  Result (n=100): {:.10}\n", result_simp);

    // Trapezoidal rule
    let result_trap = scirs2_integrate::trapezoid(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 100);
    println!("Trapezoidal rule for sin(x):");
    println!("  Result (n=100): {:.10}\n", result_trap);

    Ok(())
}

/// Section 2: Solving ordinary differential equations (ODE)
fn section_ode_solving() -> IntegrateResult<()> {
    println!("--- 2. ODE Solving ---\n");

    // Example 1: Simple exponential decay
    // dy/dt = -y, y(0) = 1
    // Exact solution: y(t) = exp(-t)
    let f = |_t: f64, y: ArrayView1<f64>| -> Array1<f64> { array![-y[0]] };

    let result = scirs2_integrate::ode::solve_ivp(
        f,
        [0.0, 5.0],  // time span
        array![1.0], // initial condition
        Some(scirs2_integrate::ode::ODEOptions {
            method: scirs2_integrate::ode::ODEMethod::RK45,
            rtol: 1e-8,
            atol: 1e-10,
            ..Default::default()
        }),
    )?;

    println!("dy/dt = -y, y(0) = 1  (exact: y = exp(-t))");
    println!("  Method: RK45 (Runge-Kutta 4th/5th order)");
    println!("  Number of time steps: {}", result.t.len());

    // Print a few solution points
    let check_times = [0.0, 1.0, 2.0, 3.0, 5.0];
    for &tc in &check_times {
        // Find closest time point
        if let Some(idx) = result.t.iter().position(|&t| (t - tc).abs() < 0.1) {
            let exact = (-tc).exp();
            let computed = result.y[idx][0];
            println!(
                "  t={:.1}: computed={:.8}, exact={:.8}, error={:.2e}",
                result.t[idx],
                computed,
                exact,
                (computed - exact).abs()
            );
        }
    }
    println!();

    // Example 2: Lotka-Volterra (predator-prey) equations
    // dx/dt = alpha*x - beta*x*y
    // dy/dt = delta*x*y - gamma*y
    let alpha = 1.5;
    let beta_param = 1.0;
    let delta = 1.0;
    let gamma_param = 3.0;

    let lotka_volterra = move |_t: f64, y: ArrayView1<f64>| -> Array1<f64> {
        let prey = y[0];
        let pred = y[1];
        array![
            alpha * prey - beta_param * prey * pred,
            delta * prey * pred - gamma_param * pred
        ]
    };

    let lv_result = scirs2_integrate::ode::solve_ivp(
        lotka_volterra,
        [0.0, 15.0],
        array![10.0, 5.0], // Initial: 10 prey, 5 predators
        Some(scirs2_integrate::ode::ODEOptions {
            method: scirs2_integrate::ode::ODEMethod::RK45,
            rtol: 1e-6,
            atol: 1e-8,
            ..Default::default()
        }),
    )?;

    println!("Lotka-Volterra (predator-prey) equations:");
    println!("  Initial: prey=10.0, predators=5.0");
    println!("  Time steps: {}", lv_result.t.len());
    if let Some(last_y) = lv_result.y.last() {
        println!(
            "  Final state: prey={:.4}, predators={:.4}",
            last_y[0], last_y[1]
        );
    }
    // Show a few intermediate points
    let n_points = lv_result.t.len();
    let step = n_points / 5;
    for i in (0..n_points).step_by(step.max(1)) {
        println!(
            "  t={:.2}: prey={:.4}, predators={:.4}",
            lv_result.t[i], lv_result.y[i][0], lv_result.y[i][1]
        );
    }
    println!();

    Ok(())
}

/// Section 3: Advanced quadrature (Romberg, tanh-sinh)
fn section_advanced_quadrature() -> IntegrateResult<()> {
    println!("--- 3. Advanced Quadrature Methods ---\n");

    // Romberg integration: uses Richardson extrapolation on trapezoidal rule
    let result = scirs2_integrate::romberg::romberg(
        |x: f64| (-x * x).exp(), // Gaussian bell curve
        0.0,
        1.0,
        None,
    )?;
    println!("Romberg: integral(exp(-x^2), 0, 1):");
    println!("  Result: {:.10}", result.value);
    println!("  Error:  {:.2e}\n", result.abs_error);

    // Tanh-sinh (double exponential) quadrature
    // Excellent for integrands with endpoint singularities
    let result_ts = scirs2_integrate::tanhsinh(
        |x: f64| 1.0 / (1.0 + x * x), // arctan integrand
        0.0,
        1.0,
        None,
    )?;
    println!("Tanh-sinh: integral(1/(1+x^2), 0, 1):");
    println!("  Result: {:.10}", result_ts.integral);
    println!("  Exact:  {:.10} (pi/4)", std::f64::consts::PI / 4.0);
    println!(
        "  Error:  {:.2e}\n",
        (result_ts.integral - std::f64::consts::PI / 4.0).abs()
    );

    // Gauss-Legendre quadrature (high-order polynomial exactness)
    let result_gl = scirs2_integrate::gaussian::gauss_legendre(
        |x: f64| x.sin(),
        0.0,
        std::f64::consts::PI,
        16, // Number of quadrature points
    )?;
    println!("Gauss-Legendre (16 points): integral(sin(x), 0, pi):");
    println!("  Result: {:.10}", result_gl);
    println!("  Exact:  2.0");
    println!("  Error:  {:.2e}\n", (result_gl - 2.0).abs());

    Ok(())
}
