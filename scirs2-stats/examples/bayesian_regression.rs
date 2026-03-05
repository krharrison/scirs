//! Bayesian Linear Regression Example — SciRS2 Stats
//!
//! Demonstrates Bayesian linear regression with conjugate (exact) inference:
//!   1. Generate synthetic data  y = 3x + 1.5 + noise
//!   2. Fit with `bayesian_linear_regression_exact`
//!   3. Print posterior mean, std-dev per coefficient, and predictive table
//!
//! Run with: cargo run -p scirs2-stats --example bayesian_regression

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_stats::bayesian::enhanced_bayesian_linear_regression_exact as bayesian_linear_regression_exact;

fn main() {
    println!("=== SciRS2 Bayesian Linear Regression (Conjugate Exact) ===\n");

    // ------------------------------------------------------------------ //
    //  1. Generate synthetic data: y = 1.5 + 3.0 * x + Normal(0, 0.5)   //
    // ------------------------------------------------------------------ //
    const N: usize = 40;
    const TRUE_INTERCEPT: f64 = 1.5;
    const TRUE_SLOPE: f64 = 3.0;
    const NOISE_STD: f64 = 0.5;

    // Deterministic pseudo-noise using a simple LCG so the example is
    // reproducible without pulling in rand as a dependency.
    let mut lcg_state: u64 = 0xDEAD_BEEF_CAFE_1234;
    let mut next_normal = || -> f64 {
        // Box-Muller (two uniform → one normal)
        lcg_state = lcg_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u1 = (lcg_state >> 33) as f64 / u32::MAX as f64 + 1e-15;
        lcg_state = lcg_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u2 = (lcg_state >> 33) as f64 / u32::MAX as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // Design matrix X = [1, x]  (intercept column + predictor column)
    let xs: Vec<f64> = (0..N)
        .map(|i| (i as f64 / (N - 1) as f64) * 4.0 - 2.0)
        .collect();
    let mut x_data = Array2::<f64>::zeros((N, 2));
    let mut y_data = Array1::<f64>::zeros(N);
    for i in 0..N {
        x_data[[i, 0]] = 1.0; // intercept term
        x_data[[i, 1]] = xs[i]; // predictor
        y_data[i] = TRUE_INTERCEPT + TRUE_SLOPE * xs[i] + next_normal() * NOISE_STD;
    }

    println!("Data overview:");
    println!("  Samples          : {N}");
    println!("  True intercept   : {TRUE_INTERCEPT}");
    println!("  True slope       : {TRUE_SLOPE}");
    println!("  Noise std-dev    : {NOISE_STD}");
    println!("  x range          : [{:.2}, {:.2}]\n", xs[0], xs[N - 1]);

    // ------------------------------------------------------------------ //
    //  2. Fit Bayesian regression with uninformative (default) prior      //
    // ------------------------------------------------------------------ //
    let result = bayesian_linear_regression_exact(x_data.clone(), y_data.clone(), None)
        .expect("Bayesian regression failed");

    // ------------------------------------------------------------------ //
    //  3. Print posterior summary                                         //
    // ------------------------------------------------------------------ //
    let beta_mean = &result.beta_mean;
    let beta_cov = &result.beta_covariance;

    println!("--- Posterior Coefficients ---");
    println!(
        "{:<16} {:>12} {:>12} {:>12}",
        "Parameter", "True", "Post. Mean", "Post. Std"
    );
    println!("{}", "-".repeat(56));
    let true_vals = [TRUE_INTERCEPT, TRUE_SLOPE];
    let labels = ["intercept (β₀)", "slope (β₁)"];
    for (j, (&lbl, &tv)) in labels.iter().zip(true_vals.iter()).enumerate() {
        let mean_j = beta_mean[j];
        let std_j = beta_cov[[j, j]].sqrt();
        println!("{:<16} {:>12.4} {:>12.4} {:>12.4}", lbl, tv, mean_j, std_j);
    }
    println!();

    // Noise precision → noise std estimate
    let noise_prec = result.noise_precision_mean;
    let noise_std_est = if noise_prec > 0.0 {
        1.0 / noise_prec.sqrt()
    } else {
        f64::NAN
    };
    println!("Noise precision (posterior mean) : {:.4}", noise_prec);
    println!(
        "Noise std (estimated)            : {:.4}  (true: {:.4})",
        noise_std_est, NOISE_STD
    );
    println!(
        "Log marginal likelihood          : {:.4}\n",
        result.log_marginal_likelihood
    );

    // ------------------------------------------------------------------ //
    //  4. Predictive table at selected x values                          //
    // ------------------------------------------------------------------ //
    let x_new_vals = [-2.0_f64, -1.0, 0.0, 1.0, 2.0];
    // Build new design matrix for predictions
    let mut x_pred = Array2::<f64>::zeros((x_new_vals.len(), 2));
    for (i, &xv) in x_new_vals.iter().enumerate() {
        x_pred[[i, 0]] = 1.0;
        x_pred[[i, 1]] = xv;
    }
    // Posterior predictive mean = X_new @ beta_mean
    println!("--- Predictive Table ---");
    println!(
        "{:>8}  {:>10}  {:>10}  {:>12}",
        "x", "True y", "Pred Mean", "95% Half-Width"
    );
    println!("{}", "-".repeat(48));
    for (i, &xv) in x_new_vals.iter().enumerate() {
        let true_y = TRUE_INTERCEPT + TRUE_SLOPE * xv;
        let mut pred_mean = 0.0_f64;
        for j in 0..2usize {
            pred_mean += x_pred[[i, j]] * beta_mean[j];
        }
        // Predictive variance: x^T Σ_post x + 1/noise_prec
        let mut pred_var = if noise_prec > 0.0 {
            1.0 / noise_prec
        } else {
            0.0
        };
        for j in 0..2usize {
            for k in 0..2usize {
                pred_var += x_pred[[i, j]] * beta_cov[[j, k]] * x_pred[[i, k]];
            }
        }
        let half_width = 1.96 * pred_var.sqrt();
        println!(
            "{:>8.2}  {:>10.4}  {:>10.4}  {:>12.4}",
            xv, true_y, pred_mean, half_width
        );
    }

    println!("\nDone.");
}
