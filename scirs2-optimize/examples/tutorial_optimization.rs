//! Tutorial: Optimization with SciRS2
//!
//! This tutorial covers unconstrained minimization, scalar optimization,
//! root finding, and constrained optimization.
//!
//! Run with: cargo run -p scirs2-optimize --example tutorial_optimization

use scirs2_core::ndarray::{array, Array1, Array2, ArrayView1};
use scirs2_optimize::error::OptimizeResult as OptResult;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 Optimization Tutorial ===\n");

    section_unconstrained()?;
    section_scalar()?;
    section_root_finding()?;

    println!("\n=== Tutorial Complete ===");
    Ok(())
}

/// Section 1: Unconstrained minimization
fn section_unconstrained() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 1. Unconstrained Minimization ---\n");

    // Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
    // Minimum at (1, 1) with f(1,1) = 0
    let rosenbrock = |x: &ArrayView1<f64>| -> f64 {
        let a = 1.0 - x[0];
        let b = x[1] - x[0] * x[0];
        a * a + 100.0 * b * b
    };

    // Minimize using Nelder-Mead (derivative-free simplex method)
    let x0 = [0.0, 0.0]; // Starting point
    let result = scirs2_optimize::unconstrained::minimize(
        rosenbrock,
        &x0,
        scirs2_optimize::unconstrained::Method::NelderMead,
        None,
    )?;

    println!("Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2");
    println!("Method: Nelder-Mead");
    println!(
        "  Solution:   x = {:.6}, y = {:.6}",
        result.x[0], result.x[1]
    );
    println!("  f(x,y):     {:.6}", result.fun);
    println!("  Iterations: {}", result.nit);
    println!("  Success:    {}\n", result.success);

    // Minimize using BFGS (quasi-Newton method, uses gradient)
    let result_bfgs = scirs2_optimize::unconstrained::minimize(
        rosenbrock,
        &x0,
        scirs2_optimize::unconstrained::Method::BFGS,
        None,
    )?;

    println!("Method: BFGS");
    println!(
        "  Solution:   x = {:.6}, y = {:.6}",
        result_bfgs.x[0], result_bfgs.x[1]
    );
    println!("  f(x,y):     {:.10}", result_bfgs.fun);
    println!("  Iterations: {}", result_bfgs.nit);
    println!("  Success:    {}\n", result_bfgs.success);

    // Minimize using L-BFGS (limited-memory BFGS, good for large problems)
    let result_lbfgs = scirs2_optimize::unconstrained::minimize(
        rosenbrock,
        &x0,
        scirs2_optimize::unconstrained::Method::LBFGS,
        None,
    )?;

    println!("Method: L-BFGS");
    println!(
        "  Solution:   x = {:.6}, y = {:.6}",
        result_lbfgs.x[0], result_lbfgs.x[1]
    );
    println!("  f(x,y):     {:.10}", result_lbfgs.fun);
    println!("  Iterations: {}\n", result_lbfgs.nit);

    Ok(())
}

/// Section 2: Scalar optimization (single-variable minimization)
fn section_scalar() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 2. Scalar Minimization ---\n");

    // Minimize f(x) = (x - 3)^2 + 1
    // Minimum at x = 3 with f(3) = 1
    let f = |x: f64| -> f64 { (x - 3.0) * (x - 3.0) + 1.0 };

    // Brent's method (combines parabolic interpolation with golden section)
    let result = scirs2_optimize::minimize_scalar(
        f,
        None, // No bounds
        scirs2_optimize::scalar::Method::Brent,
        None, // Default options
    )?;

    println!("f(x) = (x - 3)^2 + 1");
    println!(
        "  Brent's method: x = {:.6}, f(x) = {:.6}",
        result.x, result.fun
    );
    println!();

    // Golden section search with bounds
    let result_golden = scirs2_optimize::minimize_scalar(
        f,
        Some((0.0, 10.0)),
        scirs2_optimize::scalar::Method::Golden,
        None,
    )?;

    println!(
        "  Golden section: x = {:.6}, f(x) = {:.6}",
        result_golden.x, result_golden.fun
    );
    println!();

    // Bounded method
    let result_bounded = scirs2_optimize::minimize_scalar(
        f,
        Some((-5.0, 5.0)),
        scirs2_optimize::scalar::Method::Bounded,
        None,
    )?;

    println!(
        "  Bounded [-5, 5]: x = {:.6}, f(x) = {:.6}\n",
        result_bounded.x, result_bounded.fun
    );

    Ok(())
}

/// Section 3: Root finding (solving f(x) = 0)
fn section_root_finding() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 3. Root Finding ---\n");

    // Find the root of a system of nonlinear equations:
    //   x^2 + y^2 = 1   (unit circle)
    //   x - y = 0        (y = x line)
    // Solutions: (sqrt(0.5), sqrt(0.5)) and (-sqrt(0.5), -sqrt(0.5))
    fn system(x: &[f64]) -> Array1<f64> {
        array![
            x[0] * x[0] + x[1] * x[1] - 1.0, // Circle equation
            x[0] - x[1]                      // Line equation
        ]
    }

    let x0 = array![2.0, 2.0]; // Starting guess

    let result = scirs2_optimize::root(
        system,
        &x0,
        scirs2_optimize::roots::Method::Hybr,
        None::<fn(&[f64]) -> Array2<f64>>,
        None,
    )?;

    let expected = (0.5_f64).sqrt();
    println!("System: x^2 + y^2 = 1, x = y");
    println!("  Root: x = {:.6}, y = {:.6}", result.x[0], result.x[1]);
    println!("  Expected: ({:.6}, {:.6})", expected, expected);
    println!("  Success: {}", result.success);

    // Verify the solution
    let residual = system(result.x.as_slice().unwrap_or(&[]));
    println!("  Residual: {:?}\n", residual.to_vec());

    // Another example: finding where exp(x) = 2x + 1
    // Rearranged: exp(x) - 2x - 1 = 0
    fn transcendental(x: &[f64]) -> Array1<f64> {
        array![x[0].exp() - 2.0 * x[0] - 1.0]
    }

    let x0_trans = array![1.0];
    let result2 = scirs2_optimize::root(
        transcendental,
        &x0_trans,
        scirs2_optimize::roots::Method::Hybr,
        None::<fn(&[f64]) -> Array2<f64>>,
        None,
    )?;

    println!("Equation: exp(x) = 2x + 1");
    println!("  Root: x = {:.6}", result2.x[0]);
    println!(
        "  Verification: exp({:.4}) = {:.4}, 2*{:.4}+1 = {:.4}",
        result2.x[0],
        result2.x[0].exp(),
        result2.x[0],
        2.0 * result2.x[0] + 1.0
    );
    println!("  Success: {}\n", result2.success);

    Ok(())
}
