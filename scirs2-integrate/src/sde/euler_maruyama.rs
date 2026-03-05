//! Euler-Maruyama (EM) scheme for SDEs
//!
//! The Euler-Maruyama method is the simplest explicit numerical method for SDEs.
//! It extends the Euler method for ODEs by adding a stochastic increment term.
//!
//! ## Scheme
//!
//! Given the SDE:
//!
//! ```text
//! dx = f(t, x) dt + g(t, x) dW
//! ```
//!
//! The Euler-Maruyama update is:
//!
//! ```text
//! x_{n+1} = x_n + f(t_n, x_n) * dt + g(t_n, x_n) * ΔW_n
//! ```
//!
//! where `ΔW_n ~ N(0, dt * I_m)` is an m-dimensional Gaussian increment.
//!
//! ## Convergence
//!
//! - **Strong order**: 0.5 (pathwise approximation quality)
//! - **Weak order**: 1.0 (distributional approximation quality)
//!
//! The weak Euler-Maruyama uses two-point distributed random variables
//! (Rademacher/√dt increments) instead of Gaussian increments to achieve the
//! same weak order 1.0 at lower computational cost per step.

use crate::error::{IntegrateError, IntegrateResult};
use crate::sde::{compute_n_steps, SdeOptions, SdeProblem, SdeSolution};
use scirs2_core::ndarray::Array1;
use scirs2_core::random::prelude::{Normal, Rng, StdRng};
use scirs2_core::Distribution;

/// Euler-Maruyama method for SDEs.
///
/// Solves the SDE:
/// ```text
/// dx = f(t, x) dt + g(t, x) dW,   x(t0) = x0
/// ```
///
/// using the explicit update:
/// ```text
/// x_{n+1} = x_n + f(t_n, x_n) * dt + g(t_n, x_n) * ΔW_n
/// ```
/// where `ΔW_n ~ N(0, dt)` component-wise.
///
/// **Strong convergence order**: 0.5
/// **Weak convergence order**: 1.0
///
/// # Arguments
///
/// * `prob` - The SDE problem definition
/// * `dt` - Step size (must be positive)
/// * `rng` - Mutable reference to a seeded random number generator
///
/// # Returns
///
/// A `SdeSolution` containing the time points and state trajectory.
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::{SdeProblem, SdeSolution};
/// use scirs2_integrate::sde::euler_maruyama::euler_maruyama;
/// use scirs2_core::ndarray::{array, Array1, Array2};
/// use scirs2_core::random::prelude::*;
///
/// // Ornstein-Uhlenbeck: dX = θ(μ - X) dt + σ dW
/// let (theta, mu_ou, sigma) = (2.0_f64, 1.0_f64, 0.3_f64);
/// let x0 = array![0.0_f64];
/// let prob = SdeProblem::new(
///     x0, [0.0, 2.0], 1,
///     move |_t, x| array![theta * (mu_ou - x[0])],
///     move |_t, _x| { let mut g = Array2::zeros((1, 1)); g[[0, 0]] = sigma; g },
/// );
/// let mut rng = seeded_rng(42);
/// let sol = euler_maruyama(&prob, 0.01, &mut rng).unwrap();
/// assert!(!sol.t.is_empty());
/// assert!((sol.t[0] - 0.0).abs() < 1e-12);
/// ```
pub fn euler_maruyama<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    euler_maruyama_with_options(prob, dt, rng, &SdeOptions::default())
}

/// Euler-Maruyama method with additional solver options.
///
/// # Arguments
///
/// * `prob` - The SDE problem definition
/// * `dt` - Step size
/// * `rng` - Random number generator
/// * `opts` - Solver options
pub fn euler_maruyama_with_options<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
    opts: &SdeOptions,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    prob.validate()?;
    let t0 = prob.t_span[0];
    let t1 = prob.t_span[1];
    let n_steps = compute_n_steps(t0, t1, dt, opts.max_steps)?;
    let n_state = prob.dim();
    let m = prob.n_brownian;

    let capacity = if opts.save_all_steps { n_steps + 1 } else { 2 };
    let mut sol = SdeSolution::with_capacity(capacity);
    sol.push(t0, prob.x0.clone());

    let normal = Normal::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::ComputationError(format!("Normal distribution error: {}", e)))?;

    let mut x = prob.x0.clone();
    let mut t = t0;

    for step in 0..n_steps {
        // Adjust last step size to hit t1 exactly
        let dt_actual = if step == n_steps - 1 {
            t1 - t
        } else {
            dt.min(t1 - t)
        };
        if dt_actual <= 0.0 {
            break;
        }
        let sqrt_dt = dt_actual.sqrt();

        // Generate m-dimensional Brownian increment ΔW ~ N(0, dt_actual * I_m)
        let dw: Array1<f64> = Array1::from_shape_fn(m, |_| {
            normal.sample(rng) * sqrt_dt
        });

        // Evaluate drift f(t, x) and diffusion g(t, x)
        let drift = (prob.f_drift)(t, &x);
        let diff_matrix = (prob.g_diffusion)(t, &x);

        // Validate dimensions
        if drift.len() != n_state {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Drift output dimension {} != state dimension {}",
                drift.len(),
                n_state
            )));
        }
        if diff_matrix.nrows() != n_state || diff_matrix.ncols() != m {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Diffusion matrix shape ({},{}) != expected ({},{})",
                diff_matrix.nrows(),
                diff_matrix.ncols(),
                n_state,
                m
            )));
        }

        // x_{n+1} = x_n + f * dt + g * ΔW
        let stochastic_increment = diff_matrix.dot(&dw);
        x = x + drift * dt_actual + stochastic_increment;
        t += dt_actual;

        if opts.save_all_steps {
            sol.push(t, x.clone());
        }
    }

    if !opts.save_all_steps {
        sol.push(t, x);
    }

    Ok(sol)
}

/// Weak Euler-Maruyama method using two-point distributed random variables.
///
/// This variant replaces Gaussian increments with Rademacher-scaled increments:
/// ```text
/// ΔW_n = ±√dt  (each component, equally likely)
/// ```
///
/// This achieves the same **weak convergence order 1.0** as the standard
/// Euler-Maruyama method, but may be cheaper per step since it avoids
/// computing normal random variates.
///
/// **Note**: This method does NOT achieve strong order 0.5; it is only suitable
/// when the distributional (weak) properties of the solution are needed.
///
/// # Arguments
///
/// * `prob` - The SDE problem definition
/// * `dt` - Step size
/// * `rng` - Mutable reference to a seeded RNG
///
/// # Returns
///
/// A `SdeSolution` containing the time points and state trajectory.
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::{SdeProblem, SdeSolution};
/// use scirs2_integrate::sde::euler_maruyama::weak_euler_maruyama;
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_core::random::prelude::*;
///
/// let x0 = array![1.0_f64];
/// let prob = SdeProblem::new(
///     x0, [0.0, 1.0], 1,
///     |_t, x| array![0.1 * x[0]],
///     |_t, x| { let mut g = Array2::zeros((1, 1)); g[[0, 0]] = 0.2 * x[0]; g },
/// );
/// let mut rng = seeded_rng(99);
/// let sol = weak_euler_maruyama(&prob, 0.01, &mut rng).unwrap();
/// assert!(sol.len() > 1);
/// ```
pub fn weak_euler_maruyama<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    weak_euler_maruyama_with_options(prob, dt, rng, &SdeOptions::default())
}

/// Weak Euler-Maruyama with solver options.
pub fn weak_euler_maruyama_with_options<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
    opts: &SdeOptions,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    prob.validate()?;
    let t0 = prob.t_span[0];
    let t1 = prob.t_span[1];
    let n_steps = compute_n_steps(t0, t1, dt, opts.max_steps)?;
    let n_state = prob.dim();
    let m = prob.n_brownian;

    let capacity = if opts.save_all_steps { n_steps + 1 } else { 2 };
    let mut sol = SdeSolution::with_capacity(capacity);
    sol.push(t0, prob.x0.clone());

    let mut x = prob.x0.clone();
    let mut t = t0;

    for step in 0..n_steps {
        let dt_actual = if step == n_steps - 1 {
            t1 - t
        } else {
            dt.min(t1 - t)
        };
        if dt_actual <= 0.0 {
            break;
        }
        let sqrt_dt = dt_actual.sqrt();

        // Two-point Rademacher increments: ±√dt each component
        let dw: Array1<f64> = Array1::from_shape_fn(m, |_| {
            if rng.random::<bool>() { sqrt_dt } else { -sqrt_dt }
        });

        let drift = (prob.f_drift)(t, &x);
        let diff_matrix = (prob.g_diffusion)(t, &x);

        if drift.len() != n_state {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Drift output dimension {} != state dimension {}",
                drift.len(),
                n_state
            )));
        }
        if diff_matrix.nrows() != n_state || diff_matrix.ncols() != m {
            return Err(IntegrateError::DimensionMismatch(format!(
                "Diffusion matrix shape ({},{}) != expected ({},{})",
                diff_matrix.nrows(),
                diff_matrix.ncols(),
                n_state,
                m
            )));
        }

        let stochastic_increment = diff_matrix.dot(&dw);
        x = x + drift * dt_actual + stochastic_increment;
        t += dt_actual;

        if opts.save_all_steps {
            sol.push(t, x.clone());
        }
    }

    if !opts.save_all_steps {
        sol.push(t, x);
    }

    Ok(sol)
}

// Type alias for 2D arrays used in diffusion functions
use scirs2_core::ndarray::Array2;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sde::SdeProblem;
    use scirs2_core::ndarray::{array, Array2};
    use scirs2_core::random::prelude::{seeded_rng, SeedableRng};

    fn make_gbm_prob(mu: f64, sigma: f64, x0: f64) -> SdeProblem<
        impl Fn(f64, &Array1<f64>) -> Array1<f64>,
        impl Fn(f64, &Array1<f64>) -> Array2<f64>,
    > {
        SdeProblem::new(
            array![x0],
            [0.0, 1.0],
            1,
            move |_t, x| array![mu * x[0]],
            move |_t, x| {
                let mut g = Array2::zeros((1, 1));
                g[[0, 0]] = sigma * x[0];
                g
            },
        )
    }

    /// GBM analytic mean: E[S(T)] = S0 * exp(mu * T)
    #[test]
    fn test_em_gbm_weak_convergence() {
        let mu = 0.1_f64;
        let sigma = 0.2_f64;
        let x0 = 1.0_f64;
        let t1 = 1.0_f64;
        let dt = 0.001;
        let n_paths = 500;
        let analytic_mean = x0 * (mu * t1).exp();

        let mut sum = 0.0;
        for seed in 0..n_paths_u64(n_paths) {
            let prob = make_gbm_prob(mu, sigma, x0);
            let mut rng = seeded_rng(seed);
            let sol = euler_maruyama(&prob, dt, &mut rng).expect("euler_maruyama should succeed");
            sum += sol.x_final().expect("solution has state")[0];
        }
        let sample_mean = sum / n_paths as f64;
        let rel_error = (sample_mean - analytic_mean).abs() / analytic_mean;
        assert!(
            rel_error < 0.05,
            "GBM mean {:.4} vs analytic {:.4}, rel error {:.4}",
            sample_mean,
            analytic_mean,
            rel_error
        );
    }

    #[test]
    fn test_em_solution_length() {
        let prob = make_gbm_prob(0.05, 0.2, 1.0);
        let mut rng = seeded_rng(0);
        let sol = euler_maruyama(&prob, 0.1, &mut rng).expect("euler_maruyama should succeed");
        // t in [0, 1] with dt=0.1 → 11 points (including t=0)
        assert_eq!(sol.len(), 11);
        assert!((sol.t[0] - 0.0).abs() < 1e-12);
        assert!((sol.t_final().expect("solution has time steps") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weak_em_solution_length() {
        let prob = make_gbm_prob(0.05, 0.2, 1.0);
        let mut rng = seeded_rng(1);
        let sol = weak_euler_maruyama(&prob, 0.1, &mut rng).expect("weak_euler_maruyama should succeed");
        assert_eq!(sol.len(), 11);
    }

    #[test]
    fn test_em_invalid_dt() {
        let prob = make_gbm_prob(0.05, 0.2, 1.0);
        let mut rng = seeded_rng(0);
        assert!(euler_maruyama(&prob, -0.1, &mut rng).is_err());
    }

    #[test]
    fn test_em_multivariate() {
        // 2D Brownian motion: dX = dt + dW, dY = dt + dW (independent)
        let x0 = array![0.0_f64, 0.0_f64];
        let prob = SdeProblem::new(
            x0,
            [0.0, 1.0],
            2,
            |_t, _x| array![1.0_f64, 1.0_f64],
            |_t, _x| {
                let mut g = Array2::zeros((2, 2));
                g[[0, 0]] = 1.0;
                g[[1, 1]] = 1.0;
                g
            },
        );
        let mut rng = seeded_rng(42);
        let sol = euler_maruyama(&prob, 0.01, &mut rng).expect("euler_maruyama should succeed");
        assert_eq!(sol.x[0].len(), 2);
    }

    #[test]
    fn test_em_save_only_last() {
        let prob = make_gbm_prob(0.05, 0.2, 1.0);
        let mut rng = seeded_rng(0);
        let opts = SdeOptions {
            save_all_steps: false,
            ..Default::default()
        };
        let sol = euler_maruyama_with_options(&prob, 0.01, &mut rng, &opts).expect("euler_maruyama_with_options should succeed");
        // Should have initial + final = 2 entries
        assert_eq!(sol.len(), 2);
    }

    fn n_paths_u64(n: usize) -> u64 {
        n as u64
    }
}
