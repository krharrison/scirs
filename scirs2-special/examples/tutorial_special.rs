//! Tutorial: Special Functions with SciRS2
//!
//! This tutorial covers gamma functions, Bessel functions,
//! error functions, and elliptic integrals.
//!
//! Run with: cargo run -p scirs2-special --example tutorial_special

use scirs2_special::{
    beta, betaln, digamma, ellipe, ellipk, erf, erfc, erfinv, gamma, gammaln, i0, j0, j1, jn, k0,
    y0, y1,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 Special Functions Tutorial ===\n");

    section_gamma()?;
    section_bessel()?;
    section_error_functions()?;
    section_elliptic()?;

    println!("\n=== Tutorial Complete ===");
    Ok(())
}

/// Section 1: Gamma function and relatives
fn section_gamma() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 1. Gamma Function Family ---\n");

    // Gamma function: generalization of factorial
    // gamma(n) = (n-1)! for positive integers
    println!("Gamma function (generalized factorial):");
    println!("  gamma(1) = {:.6}  (0! = 1)", gamma(1.0_f64));
    println!("  gamma(2) = {:.6}  (1! = 1)", gamma(2.0_f64));
    println!("  gamma(5) = {:.6}  (4! = 24)", gamma(5.0_f64));
    println!(
        "  gamma(0.5) = {:.6}  (sqrt(pi) = {:.6})",
        gamma(0.5_f64),
        std::f64::consts::PI.sqrt()
    );
    println!();

    // Log-gamma: more numerically stable for large arguments
    println!("Log-gamma (for large values):");
    println!("  gammaln(100) = {:.6}", gammaln(100.0_f64));
    println!("  gammaln(1000) = {:.6}", gammaln(1000.0_f64));
    println!();

    // Digamma function: psi(x) = d/dx ln(gamma(x))
    println!("Digamma function psi(x):");
    println!(
        "  digamma(1) = {:.6}  (-euler_gamma = {:.6})",
        digamma(1.0_f64),
        -0.5772156649
    );
    println!("  digamma(2) = {:.6}  (1 - euler_gamma)", digamma(2.0_f64));
    println!();

    // Beta function: B(a,b) = gamma(a)*gamma(b) / gamma(a+b)
    println!("Beta function B(a,b):");
    println!(
        "  beta(2, 3) = {:.6}  (1/12 = {:.6})",
        beta(2.0_f64, 3.0_f64),
        1.0 / 12.0
    );
    println!(
        "  beta(0.5, 0.5) = {:.6}  (pi = {:.6})",
        beta(0.5_f64, 0.5_f64),
        std::f64::consts::PI
    );
    println!("  betaln(100, 200) = {:.6}", betaln(100.0_f64, 200.0_f64));
    println!();

    Ok(())
}

/// Section 2: Bessel functions
fn section_bessel() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 2. Bessel Functions ---\n");

    // Bessel functions of the first kind: J_n(x)
    // Solutions to Bessel's differential equation that are finite at the origin
    println!("Bessel functions of the first kind J_n(x):");
    let x_vals = [0.0, 1.0, 2.0, 5.0, 10.0];
    println!("  x       J0(x)       J1(x)       J2(x)");
    println!("  ------- ----------- ----------- -----------");
    for &x in &x_vals {
        println!(
            "  {:<7.1} {:<11.6} {:<11.6} {:<11.6}",
            x,
            j0(x),
            j1(x),
            jn(2, x)
        );
    }
    println!();

    // Bessel functions of the second kind: Y_n(x)
    // Second linearly independent solution (singular at origin)
    println!("Bessel functions of the second kind Y_n(x):");
    let x_vals_y = [0.5, 1.0, 2.0, 5.0, 10.0];
    println!("  x       Y0(x)       Y1(x)");
    println!("  ------- ----------- -----------");
    for &x in &x_vals_y {
        println!("  {:<7.1} {:<11.6} {:<11.6}", x, y0(x), y1(x));
    }
    println!();

    // Modified Bessel functions: I_n(x) and K_n(x)
    // Solutions to the modified Bessel equation (exponential behavior)
    println!("Modified Bessel functions:");
    println!("  I0(1.0) = {:.6}", i0(1.0_f64));
    println!("  K0(1.0) = {:.6}", k0(1.0_f64));
    println!();

    Ok(())
}

/// Section 3: Error functions
fn section_error_functions() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 3. Error Functions ---\n");

    // Error function: erf(x) = (2/sqrt(pi)) * integral(exp(-t^2), 0, x)
    // Used extensively in probability and statistics
    println!("Error function erf(x):");
    let x_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
    for &x in &x_vals {
        println!("  erf({:.1}) = {:.8}", x, erf(x));
    }
    println!();

    // Complementary error function: erfc(x) = 1 - erf(x)
    // More accurate for large x where erf(x) is close to 1
    println!("Complementary error function erfc(x):");
    println!("  erfc(0)   = {:.8}", erfc(0.0_f64));
    println!("  erfc(1)   = {:.8}", erfc(1.0_f64));
    println!("  erfc(3)   = {:.2e}  (very small)", erfc(3.0_f64));
    println!("  erfc(5)   = {:.2e}  (even smaller)", erfc(5.0_f64));
    println!();

    // Inverse error function
    // erfinv(erf(x)) = x
    println!("Inverse error function:");
    println!("  erfinv(0.0)  = {:.6}", erfinv(0.0_f64));
    println!("  erfinv(0.5)  = {:.6}", erfinv(0.5_f64));
    println!("  erfinv(0.95) = {:.6}", erfinv(0.95_f64));

    // Connection to normal distribution:
    // For standard normal: CDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
    let x = 1.96_f64;
    let normal_cdf = 0.5 * (1.0 + erf(x / 2.0_f64.sqrt()));
    println!("\n  Normal CDF via erf:");
    println!(
        "  CDF(1.96) = 0.5 * (1 + erf(1.96/sqrt(2))) = {:.6}",
        normal_cdf
    );
    println!("  (This is the 97.5th percentile)\n");

    Ok(())
}

/// Section 4: Elliptic integrals
fn section_elliptic() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 4. Elliptic Integrals ---\n");

    // Complete elliptic integral of the first kind: K(m)
    // K(m) = integral(1/sqrt(1 - m*sin^2(t)), 0, pi/2)
    // Used in: pendulum period, electromagnetic calculations, conformal mapping
    println!("Complete elliptic integral K(m):");
    let m_vals = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99];
    for &m in &m_vals {
        println!("  K({:.2}) = {:.8}", m, ellipk(m));
    }
    println!("  K(0) = pi/2 = {:.8}", std::f64::consts::PI / 2.0);
    println!("  (K(m) -> infinity as m -> 1)\n");

    // Complete elliptic integral of the second kind: E(m)
    // E(m) = integral(sqrt(1 - m*sin^2(t)), 0, pi/2)
    // Used in: ellipse perimeter, geodesics on ellipsoids
    println!("Complete elliptic integral E(m):");
    for &m in &m_vals {
        println!("  E({:.2}) = {:.8}", m, ellipe(m));
    }
    println!("  E(0) = pi/2 = {:.8}", std::f64::consts::PI / 2.0);
    println!("  E(1) = 1.0\n");

    // Application: pendulum period
    // T = 4*sqrt(L/g) * K(sin^2(theta_max/2))
    // For small angles, K ~ pi/2, so T ~ 2*pi*sqrt(L/g)
    let l = 1.0_f64; // length in meters
    let g = 9.81_f64; // gravity
    let theta_max = 0.1_f64; // small angle (radians)
    let m_pend = (theta_max / 2.0).sin().powi(2);
    let period = 4.0 * (l / g).sqrt() * ellipk(m_pend);
    let period_approx = 2.0 * std::f64::consts::PI * (l / g).sqrt();
    println!("Pendulum period (L=1m, theta_max=0.1 rad):");
    println!("  Exact:       T = {:.6} s", period);
    println!("  Small-angle: T = {:.6} s", period_approx);
    println!("  Difference:  {:.6} s\n", (period - period_approx).abs());

    Ok(())
}
