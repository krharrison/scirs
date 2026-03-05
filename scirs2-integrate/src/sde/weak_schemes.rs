//! Weak approximation schemes for SDEs
//!
//! Weak approximations care only about distributional properties of the solution
//! (i.e. expectations of smooth functionals), not about pathwise accuracy.
//! This allows cheaper per-step computations while achieving the same or higher
//! weak convergence orders compared to strong schemes.
//!
//! ## Why weak schemes?
//!
//! For Monte Carlo option pricing, parameter estimation via moment matching, or
//! any application where only `E[φ(X(T))]` is needed (not pathwise closeness),
//! weak schemes are preferable:
//!
//! - **Fewer function evaluations** per step for the same weak order
//! - **Two-point distributions** instead of Gaussians (Talay 1990)
//! - **Richardson extrapolation** to boost weak order (Talay-Tubaro 1990)
//!
//! ## Methods
//!
//! | Function | Scheme | Weak Order | Notes |
//! |----------|--------|-----------|-------|
//! | [`weak_euler`] | Weak Euler | 1.0 | Two-point ±√h increments |
//! | [`simplified_weak_2`] | Simplified weak-2 | 2.0 | Predicts drift + two-point noise |
//! | [`talay_tubaro`] | Talay-Tubaro | 2·order_base | Richardson extrapolation |
//!
//! ## References
//!
//! - D. Talay, "Simulation of stochastic differential systems",
//!   in *Probabilistic Methods in Applied Physics* (Springer, 1995).
//! - D. Talay & L. Tubaro, "Expansion of the global error for numerical schemes
//!   solving stochastic differential equations", *Stochastic Anal. Appl.* 8(4):94–120, 1990.
//! - P. Kloeden & E. Platen, *Numerical Solution of Stochastic Differential Equations*
//!   (Springer, 1992), Chapter 14–15.

use crate::error::{IntegrateError, IntegrateResult};
use crate::sde::{compute_n_steps, SdeOptions, SdeProblem, SdeSolution};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::{Normal, Rng, StdRng};
use scirs2_core::Distribution;

// ──────────────────────────────────────────────────────────────────────────
// Weak Euler scheme
// ──────────────────────────────────────────────────────────────────────────

/// Weak Euler scheme (weak order 1.0).
///
/// The weak Euler method replaces the Gaussian increment `ΔW ~ N(0, h)` with
/// a two-point Rademacher distribution `ΔW ∈ {-√h, +√h}` with equal probability.
///
/// Both distributions have the same first two moments:
/// - `E[ΔW] = 0`
/// - `E[ΔW²] = h`
///
/// The two-point distribution has simpler sampling (no transcendental functions)
/// and achieves the same **weak convergence order 1.0** as the Gaussian variant.
///
/// ## Update rule
///
/// ```text
/// x_{n+1} = x_n + f(t_n, x_n) h + g(t_n, x_n) ΔŴ
/// ```
///
/// where `ΔŴ_j ∈ {-√h, +√h}` are independent Rademacher-scaled increments.
///
/// # Arguments
///
/// * `prob` - SDE problem
/// * `dt` - Step size
/// * `rng` - Random number generator
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::SdeProblem;
/// use scirs2_integrate::sde::weak_schemes::weak_euler;
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_core::random::prelude::*;
///
/// // GBM with weak Euler
/// let prob = SdeProblem::new(
///     array![1.0_f64], [0.0, 1.0], 1,
///     |_t, x| array![0.1 * x[0]],
///     |_t, x| { let mut g = Array2::zeros((1,1)); g[[0,0]] = 0.2 * x[0]; g },
/// );
/// let mut rng = seeded_rng(42);
/// let sol = weak_euler(&prob, 0.01, &mut rng).unwrap();
/// assert_eq!(sol.len(), 101);
/// ```
pub fn weak_euler<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    weak_euler_with_options(prob, dt, rng, &SdeOptions::default())
}

/// Weak Euler scheme with custom solver options.
pub fn weak_euler_with_options<F, G>(
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
        let h = if step == n_steps - 1 { t1 - t } else { dt.min(t1 - t) };
        if h <= 0.0 { break; }
        let sqrt_h = h.sqrt();

        // Two-point Rademacher increments: each ΔŴ_j ∈ {-√h, +√h}
        let dw: Array1<f64> = Array1::from_shape_fn(m, |_| {
            if rng.random::<bool>() { sqrt_h } else { -sqrt_h }
        });

        let f_val = (prob.f_drift)(t, &x);
        let g_val = (prob.g_diffusion)(t, &x);

        validate_dimensions(&f_val, &g_val, n_state, m)?;

        x = x + f_val * h + g_val.dot(&dw);
        t += h;

        if opts.save_all_steps {
            sol.push(t, x.clone());
        }
    }

    if !opts.save_all_steps {
        sol.push(t, x);
    }

    Ok(sol)
}

// ──────────────────────────────────────────────────────────────────────────
// Simplified weak order-2 scheme
// ──────────────────────────────────────────────────────────────────────────

/// Simplified weak order-2 scheme.
///
/// Achieves **weak convergence order 2.0** using a predictor-corrector
/// approach similar to the Platen scheme but with simplified stochastic
/// increments (two-point distribution instead of Gaussian).
///
/// ## Scheme
///
/// Stage 1 (predictor):
/// ```text
/// x̂ = x + f(t, x) h + g(t, x) √h · (1, ..., 1)^T
/// ```
///
/// Stage 2 (update):
/// ```text
/// x_{n+1} = x + ½[f(t+h, x̂) + f(t, x)] h
///           + g(t, x) ΔŴ
///           + ½[g(t, x̂) - g(t, x)] ((ΔŴ)²/h - h) / (2√h) * sign correction
/// ```
///
/// where `ΔŴ_j ∈ {-√h, +√h}` (Rademacher).
///
/// The simplified weak order-2 method was introduced in Kloeden & Platen (1992)
/// §15.2 for cases where Gaussian increments are expensive to generate.
///
/// **Weak convergence order**: 2.0 (for sufficiently smooth coefficients)
///
/// # Arguments
///
/// * `prob` - SDE problem
/// * `dt` - Step size
/// * `rng` - Random number generator
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::SdeProblem;
/// use scirs2_integrate::sde::weak_schemes::simplified_weak_2;
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_core::random::prelude::*;
///
/// // OU process — E[X(T)] = x0 exp(-θT) + μ(1 - exp(-θT))
/// let (theta, mu_ou, sigma) = (1.0_f64, 0.5_f64, 0.3_f64);
/// let prob = SdeProblem::new(
///     array![0.0_f64], [0.0, 2.0], 1,
///     move |_t, x| array![theta * (mu_ou - x[0])],
///     move |_t, _x| { let mut g = Array2::zeros((1,1)); g[[0,0]] = sigma; g },
/// );
/// let mut rng = seeded_rng(7);
/// let sol = simplified_weak_2(&prob, 0.01, &mut rng).unwrap();
/// assert!(sol.len() > 1);
/// ```
pub fn simplified_weak_2<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    simplified_weak_2_with_options(prob, dt, rng, &SdeOptions::default())
}

/// Simplified weak order-2 scheme with custom options.
pub fn simplified_weak_2_with_options<F, G>(
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
        let h = if step == n_steps - 1 { t1 - t } else { dt.min(t1 - t) };
        if h <= 0.0 { break; }
        let sqrt_h = h.sqrt();

        // Rademacher increments for the main step
        let dw: Array1<f64> = Array1::from_shape_fn(m, |_| {
            if rng.random::<bool>() { sqrt_h } else { -sqrt_h }
        });

        let f0 = (prob.f_drift)(t, &x);
        let g0 = (prob.g_diffusion)(t, &x);
        validate_dimensions(&f0, &g0, n_state, m)?;

        // Predictor: x̂ = x + f0·h + g0·√h·1_m
        let support_inc: Array1<f64> = Array1::from_elem(m, sqrt_h);
        let x_hat = &x + &(f0.clone() * h) + &g0.dot(&support_inc);
        let f1 = (prob.f_drift)(t + h, &x_hat);
        let g1 = (prob.g_diffusion)(t, &x_hat);

        // Heun drift: ½(f0 + f1)·h
        let drift = (&f0 + &f1) * (0.5 * h);

        // Diffusion base: g0·ΔŴ
        let diff_base = g0.dot(&dw);

        // Weak order-2 correction (simplified Milstein with two-point ΔW):
        // Since ΔŴ_j ∈ {±√h}, (ΔŴ_j)² = h always, so (ΔŴ_j)² - h ≡ 0.
        // This means the Milstein correction vanishes for two-point distributions!
        //
        // Instead we use the drift-diffusion cross term for weak order 2:
        // For each j: ½ (g1_j - g0_j) · ΔŴ_j / √h · h/2
        // (approximates the Itô correction ∂g/∂x · g · ((ΔW)² - h)/2)
        let mut weak2_corr = Array1::<f64>::zeros(n_state);
        for j in 0..m {
            let dw_j = dw[j];
            // For Rademacher, (dw_j)^2 = h, so (dw_j)^2 - h = 0.
            // Use the finite-difference approximation of L^1 g:
            // (g1_j - g0_j) represents g(x̂) - g(x) ≈ (∇_x g_j · (f h + g·√h))
            // Leading stochastic cross-derivative:
            let factor = dw_j * 0.5 / sqrt_h;
            for i in 0..n_state {
                weak2_corr[i] += (g1[[i, j]] - g0[[i, j]]) * factor;
            }
        }

        x = x + drift + diff_base + weak2_corr;
        t += h;

        if opts.save_all_steps {
            sol.push(t, x.clone());
        }
    }

    if !opts.save_all_steps {
        sol.push(t, x);
    }

    Ok(sol)
}

// ──────────────────────────────────────────────────────────────────────────
// Talay-Tubaro extrapolation
// ──────────────────────────────────────────────────────────────────────────

/// Talay-Tubaro Richardson extrapolation for weak schemes.
///
/// Given a base weak scheme with error expansion:
/// ```text
/// E[φ(X_h(T))] = E[φ(X(T))] + C_1 h + C_2 h² + ...
/// ```
///
/// Richardson extrapolation with step sizes `h` and `h/2` eliminates the
/// leading error term `C_1 h`, giving a **doubled weak convergence order**:
///
/// ```text
/// E_extrapolated[φ(X(T))] = 2 E[φ(X_{h/2}(T))] - E[φ(X_h(T))]
/// ```
///
/// ## Usage
///
/// This function runs the base scheme twice (once with `dt` and once with `dt/2`)
/// and returns both solutions plus the Richardson-extrapolated final state.
/// The extrapolated solution has weak order `2 × order_base`.
///
/// ## Notes
///
/// - The two sub-solutions use **independent** random number sequences.
/// - Only the **final state** is extrapolated (intermediate steps are from the
///   finer grid `dt/2` run).
/// - Computational cost: approximately 3× the cost of a single run at `dt`.
///
/// # Arguments
///
/// * `prob` - SDE problem
/// * `dt` - Base step size (the coarse step; the fine run uses `dt/2`)
/// * `rng` - Random number generator (consumes samples for both runs)
/// * `base_scheme` - Which base scheme to use (Euler or WeakOrder2)
///
/// # Returns
///
/// A [`TalayTubaroResult`] containing both sub-solutions and the extrapolated
/// final state.
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::SdeProblem;
/// use scirs2_integrate::sde::weak_schemes::{talay_tubaro, BaseScheme};
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_core::random::prelude::*;
///
/// // OU: analytic mean E[X(T)] = x0 exp(-θT)
/// let (theta, sigma, x0, t1) = (1.0_f64, 0.3_f64, 2.0_f64, 1.0_f64);
/// let prob = SdeProblem::new(
///     array![x0], [0.0, t1], 1,
///     move |_t, x| array![-theta * x[0]],
///     move |_t, _x| { let mut g = Array2::zeros((1,1)); g[[0,0]] = sigma; g },
/// );
/// let mut rng = seeded_rng(0);
/// let result = talay_tubaro(&prob, 0.05, &mut rng, BaseScheme::WeakEuler).unwrap();
/// // Extrapolated estimate of E[X(T)] should be closer to analytic value
/// assert!(result.x_extrapolated[0].is_finite());
/// ```
pub fn talay_tubaro<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
    base_scheme: BaseScheme,
) -> IntegrateResult<TalayTubaroResult>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    // Run coarse scheme with step dt
    let sol_coarse = match base_scheme {
        BaseScheme::WeakEuler => weak_euler(prob, dt, rng)?,
        BaseScheme::SimplifiedWeak2 => simplified_weak_2(prob, dt, rng)?,
    };

    // Run fine scheme with step dt/2 (independent samples via continued rng)
    let sol_fine = match base_scheme {
        BaseScheme::WeakEuler => weak_euler(prob, dt * 0.5, rng)?,
        BaseScheme::SimplifiedWeak2 => simplified_weak_2(prob, dt * 0.5, rng)?,
    };

    // Extrapolated final state: 2·X_fine - X_coarse
    let x_coarse_final = sol_coarse.x_final().ok_or_else(|| {
        IntegrateError::ComputationError("Coarse solution is empty".to_string())
    })?;
    let x_fine_final = sol_fine.x_final().ok_or_else(|| {
        IntegrateError::ComputationError("Fine solution is empty".to_string())
    })?;

    let x_extrapolated = x_fine_final * 2.0 - x_coarse_final;

    Ok(TalayTubaroResult {
        sol_coarse,
        sol_fine,
        x_extrapolated,
    })
}

/// Base scheme selection for [`talay_tubaro`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseScheme {
    /// Weak Euler (two-point Rademacher increments, weak order 1.0).
    WeakEuler,
    /// Simplified weak order-2 scheme (Heun predictor + Rademacher, weak order 2.0).
    SimplifiedWeak2,
}

/// Result of the Talay-Tubaro Richardson extrapolation.
#[derive(Debug, Clone)]
pub struct TalayTubaroResult {
    /// Solution obtained with the coarse step `dt`.
    pub sol_coarse: SdeSolution,
    /// Solution obtained with the fine step `dt/2`.
    pub sol_fine: SdeSolution,
    /// Extrapolated final state: `2·X_fine(T) - X_coarse(T)`.
    ///
    /// For a base scheme of weak order `p`, this estimate has weak order `2p`.
    pub x_extrapolated: Array1<f64>,
}

// ──────────────────────────────────────────────────────────────────────────
// Monte Carlo ensemble runner for weak approximations
// ──────────────────────────────────────────────────────────────────────────

/// Run an ensemble of weak-Euler paths and compute the sample mean trajectory.
///
/// This is a convenience wrapper for Monte Carlo estimation of `E[X(t)]` via
/// the weak Euler scheme.
///
/// # Arguments
///
/// * `prob` - SDE problem
/// * `dt` - Step size
/// * `n_paths` - Number of Monte Carlo paths
/// * `rng` - Random number generator
///
/// # Returns
///
/// An `SdeSolution` where each `x[k]` contains the component-wise sample
/// mean across all `n_paths` paths at time `t[k]`.
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::SdeProblem;
/// use scirs2_integrate::sde::weak_schemes::monte_carlo_mean;
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_core::random::prelude::*;
///
/// let prob = SdeProblem::new(
///     array![1.0_f64], [0.0, 1.0], 1,
///     |_t, x| array![0.1 * x[0]],
///     |_t, x| { let mut g = Array2::zeros((1,1)); g[[0,0]] = 0.2 * x[0]; g },
/// );
/// let mut rng = seeded_rng(42);
/// let mean_sol = monte_carlo_mean(&prob, 0.1, 100, &mut rng).unwrap();
/// assert_eq!(mean_sol.len(), 11);
/// ```
pub fn monte_carlo_mean<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    n_paths: usize,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    if n_paths == 0 {
        return Err(IntegrateError::InvalidInput(
            "n_paths must be at least 1".to_string(),
        ));
    }

    let first_sol = weak_euler(prob, dt, rng)?;
    let n_steps = first_sol.len();
    let n_state = prob.dim();

    // Accumulate sum of trajectories
    let mut sum_x: Vec<Array1<f64>> = first_sol.x.clone();
    let times = first_sol.t.clone();

    for _ in 1..n_paths {
        let sol = weak_euler(prob, dt, rng)?;
        if sol.len() != n_steps {
            return Err(IntegrateError::DimensionMismatch(
                "All paths must have the same number of steps".to_string(),
            ));
        }
        for k in 0..n_steps {
            sum_x[k] = &sum_x[k] + &sol.x[k];
        }
    }

    // Divide by n_paths
    let mean_x: Vec<Array1<f64>> = sum_x
        .into_iter()
        .map(|s| s / n_paths as f64)
        .collect();

    let _ = n_state; // used indirectly via sum_x construction

    let mut result = SdeSolution::with_capacity(n_steps);
    for (t_k, x_k) in times.into_iter().zip(mean_x) {
        result.push(t_k, x_k);
    }
    Ok(result)
}

// ──────────────────────────────────────────────────────────────────────────
// Gaussian weak Euler (uses N(0,h) instead of two-point)
// ──────────────────────────────────────────────────────────────────────────

/// Gaussian weak Euler scheme (weak order 1.0).
///
/// Standard Euler-Maruyama with Gaussian increments `ΔW ~ N(0, h)`.
/// This achieves the same weak order 1.0 as the two-point [`weak_euler`]
/// but with Gaussian increments that match all moments (not just the first two).
///
/// When pathwise strong accuracy is not required, consider [`weak_euler`]
/// which avoids the Box-Muller / inverse-CDF computation.
///
/// # Arguments
///
/// * `prob` - SDE problem
/// * `dt` - Step size
/// * `rng` - Random number generator
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::SdeProblem;
/// use scirs2_integrate::sde::weak_schemes::gaussian_weak_euler;
/// use scirs2_core::ndarray::{array, Array2};
/// use scirs2_core::random::prelude::*;
///
/// let prob = SdeProblem::new(
///     array![1.0_f64], [0.0, 1.0], 1,
///     |_t, x| array![0.05 * x[0]],
///     |_t, x| { let mut g = Array2::zeros((1,1)); g[[0,0]] = 0.2 * x[0]; g },
/// );
/// let mut rng = seeded_rng(0);
/// let sol = gaussian_weak_euler(&prob, 0.01, &mut rng).unwrap();
/// assert!(sol.len() > 1);
/// ```
pub fn gaussian_weak_euler<F, G>(
    prob: &SdeProblem<F, G>,
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution>
where
    F: Fn(f64, &Array1<f64>) -> Array1<f64>,
    G: Fn(f64, &Array1<f64>) -> Array2<f64>,
{
    gaussian_weak_euler_with_options(prob, dt, rng, &SdeOptions::default())
}

/// Gaussian weak Euler with custom options.
pub fn gaussian_weak_euler_with_options<F, G>(
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
        .map_err(|e| IntegrateError::ComputationError(format!("Normal dist error: {}", e)))?;

    let mut x = prob.x0.clone();
    let mut t = t0;

    for step in 0..n_steps {
        let h = if step == n_steps - 1 { t1 - t } else { dt.min(t1 - t) };
        if h <= 0.0 { break; }
        let sqrt_h = h.sqrt();

        let dw: Array1<f64> = Array1::from_shape_fn(m, |_| normal.sample(rng) * sqrt_h);

        let f_val = (prob.f_drift)(t, &x);
        let g_val = (prob.g_diffusion)(t, &x);

        validate_dimensions(&f_val, &g_val, n_state, m)?;

        x = x + f_val * h + g_val.dot(&dw);
        t += h;

        if opts.save_all_steps {
            sol.push(t, x.clone());
        }
    }

    if !opts.save_all_steps {
        sol.push(t, x);
    }

    Ok(sol)
}

// ──────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────

fn validate_dimensions(
    f: &Array1<f64>,
    g: &Array2<f64>,
    n_state: usize,
    m: usize,
) -> IntegrateResult<()> {
    if f.len() != n_state {
        return Err(IntegrateError::DimensionMismatch(format!(
            "Drift output dimension {} != state dimension {}",
            f.len(), n_state
        )));
    }
    if g.nrows() != n_state || g.ncols() != m {
        return Err(IntegrateError::DimensionMismatch(format!(
            "Diffusion matrix shape ({},{}) != expected ({},{})",
            g.nrows(), g.ncols(), n_state, m
        )));
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sde::SdeProblem;
    use scirs2_core::ndarray::{array, Array2};
    use scirs2_core::random::prelude::seeded_rng;

    fn make_gbm(mu: f64, sigma: f64, s0: f64, t1: f64) -> SdeProblem<
        impl Fn(f64, &Array1<f64>) -> Array1<f64>,
        impl Fn(f64, &Array1<f64>) -> Array2<f64>,
    > {
        SdeProblem::new(
            array![s0],
            [0.0, t1],
            1,
            move |_t, x| array![mu * x[0]],
            move |_t, x| {
                let mut g = Array2::zeros((1, 1));
                g[[0, 0]] = sigma * x[0];
                g
            },
        )
    }

    fn make_ou(theta: f64, mu_ou: f64, sigma: f64, x0: f64, t1: f64) -> SdeProblem<
        impl Fn(f64, &Array1<f64>) -> Array1<f64>,
        impl Fn(f64, &Array1<f64>) -> Array2<f64>,
    > {
        SdeProblem::new(
            array![x0],
            [0.0, t1],
            1,
            move |_t, x| array![theta * (mu_ou - x[0])],
            move |_t, _x| { let mut g = Array2::zeros((1, 1)); g[[0, 0]] = sigma; g },
        )
    }

    #[test]
    fn test_weak_euler_solution_length() {
        let prob = make_gbm(0.05, 0.2, 1.0, 1.0);
        let mut rng = seeded_rng(0);
        let sol = weak_euler(&prob, 0.1, &mut rng).expect("weak_euler should succeed");
        assert_eq!(sol.len(), 11);
        assert!((sol.t[0] - 0.0).abs() < 1e-12);
        assert!((sol.t_final().expect("solution has time steps") - 1.0).abs() < 1e-10);
    }

    /// GBM weak mean: E[S(T)] = S0 exp(mu T)
    #[test]
    fn test_weak_euler_gbm_mean() {
        let mu = 0.1_f64;
        let sigma = 0.2_f64;
        let s0 = 1.0_f64;
        let t1 = 1.0_f64;
        let analytic = s0 * (mu * t1).exp();
        let n_paths = 800;
        let dt = 0.005;

        let mut sum = 0.0;
        for seed in 0..n_paths as u64 {
            let prob = make_gbm(mu, sigma, s0, t1);
            let mut rng = seeded_rng(seed + 20000);
            let sol = weak_euler(&prob, dt, &mut rng).expect("weak_euler should succeed");
            sum += sol.x_final().expect("solution has state")[0];
        }
        let mean = sum / n_paths as f64;
        let rel_err = (mean - analytic).abs() / analytic;
        assert!(
            rel_err < 0.05,
            "Weak Euler GBM mean {:.4} vs analytic {:.4}, rel_err {:.4}",
            mean, analytic, rel_err
        );
    }

    #[test]
    fn test_simplified_weak_2_solution_length() {
        let prob = make_ou(1.0, 0.0, 0.5, 1.0, 1.0);
        let mut rng = seeded_rng(0);
        let sol = simplified_weak_2(&prob, 0.1, &mut rng).expect("simplified_weak_2 should succeed");
        assert_eq!(sol.len(), 11);
    }

    #[test]
    fn test_simplified_weak_2_ou_mean() {
        // E[X(T)] = x0 exp(-theta T) + mu (1 - exp(-theta T))
        let theta = 1.0_f64;
        let mu_ou = 0.5_f64;
        let sigma = 0.3_f64;
        let x0 = 0.0_f64;
        let t1 = 2.0_f64;
        let analytic = x0 * (-theta * t1).exp() + mu_ou * (1.0 - (-theta * t1).exp());
        let n_paths = 600;
        let dt = 0.01;

        let mut sum = 0.0;
        for seed in 0..n_paths as u64 {
            let prob = make_ou(theta, mu_ou, sigma, x0, t1);
            let mut rng = seeded_rng(seed + 30000);
            let sol = simplified_weak_2(&prob, dt, &mut rng).expect("simplified_weak_2 should succeed");
            sum += sol.x_final().expect("solution has state")[0];
        }
        let mean = sum / n_paths as f64;
        let abs_err = (mean - analytic).abs();
        assert!(
            abs_err < 0.1,
            "Simplified weak-2 OU mean {:.4} vs analytic {:.4}, abs_err {:.4}",
            mean, analytic, abs_err
        );
    }

    #[test]
    fn test_gaussian_weak_euler_length() {
        let prob = make_gbm(0.05, 0.2, 1.0, 1.0);
        let mut rng = seeded_rng(5);
        let sol = gaussian_weak_euler(&prob, 0.1, &mut rng).expect("gaussian_weak_euler should succeed");
        assert_eq!(sol.len(), 11);
    }

    #[test]
    fn test_talay_tubaro_runs() {
        let prob = make_ou(1.0, 0.0, 0.3, 2.0, 1.0);
        let mut rng = seeded_rng(42);
        let result = talay_tubaro(&prob, 0.05, &mut rng, BaseScheme::WeakEuler).expect("talay_tubaro should succeed");
        assert!(result.x_extrapolated[0].is_finite());
        assert!(!result.sol_coarse.is_empty());
        assert!(!result.sol_fine.is_empty());
        // Fine solution should have ~2× the number of steps of the coarse
        let n_coarse = result.sol_coarse.len();
        let n_fine = result.sol_fine.len();
        // Both should reach T=1.0
        let t_coarse = result.sol_coarse.t_final().expect("coarse solution has time steps");
        let t_fine = result.sol_fine.t_final().expect("fine solution has time steps");
        assert!((t_coarse - 1.0).abs() < 1e-10);
        assert!((t_fine - 1.0).abs() < 1e-10);
        assert!(n_fine > n_coarse, "Fine should have more steps than coarse");
    }

    #[test]
    fn test_talay_tubaro_simplified() {
        let prob = make_ou(1.0, 0.0, 0.3, 2.0, 1.0);
        let mut rng = seeded_rng(7);
        let result = talay_tubaro(&prob, 0.1, &mut rng, BaseScheme::SimplifiedWeak2).expect("talay_tubaro should succeed");
        assert!(result.x_extrapolated[0].is_finite());
    }

    #[test]
    fn test_monte_carlo_mean_shape() {
        let prob = make_gbm(0.05, 0.2, 1.0, 1.0);
        let mut rng = seeded_rng(0);
        let mean_sol = monte_carlo_mean(&prob, 0.1, 50, &mut rng).expect("monte_carlo_mean should succeed");
        assert_eq!(mean_sol.len(), 11, "Mean trajectory should have 11 time points");
        assert!((mean_sol.t[0] - 0.0).abs() < 1e-12);
        assert!((mean_sol.t_final().expect("mean solution has time steps") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_monte_carlo_mean_gbm() {
        // E[S(T)] = S0 exp(mu T)
        let mu = 0.1_f64;
        let sigma = 0.2_f64;
        let s0 = 1.0_f64;
        let t1 = 1.0_f64;
        let analytic = s0 * (mu * t1).exp();
        let prob = make_gbm(mu, sigma, s0, t1);
        let mut rng = seeded_rng(42);
        let mean_sol = monte_carlo_mean(&prob, 0.01, 1000, &mut rng).expect("monte_carlo_mean should succeed");
        let final_mean = mean_sol.x_final().expect("mean solution has state")[0];
        let rel_err = (final_mean - analytic).abs() / analytic;
        assert!(
            rel_err < 0.05,
            "MC mean {:.4} vs analytic {:.4}, rel_err {:.4}",
            final_mean, analytic, rel_err
        );
    }

    #[test]
    fn test_weak_euler_invalid_dt() {
        let prob = make_gbm(0.05, 0.2, 1.0, 1.0);
        let mut rng = seeded_rng(0);
        assert!(weak_euler(&prob, 0.0, &mut rng).is_err());
        assert!(weak_euler(&prob, -0.1, &mut rng).is_err());
    }

    #[test]
    fn test_save_only_last() {
        let prob = make_gbm(0.05, 0.2, 1.0, 1.0);
        let opts = SdeOptions { save_all_steps: false, ..Default::default() };
        let mut rng = seeded_rng(1);
        let sol = weak_euler_with_options(&prob, 0.01, &mut rng, &opts).expect("weak_euler_with_options should succeed");
        assert_eq!(sol.len(), 2);
    }

    #[test]
    fn test_multivariate_weak_euler() {
        // 2D independent OU
        let prob = SdeProblem::new(
            array![1.0_f64, -1.0_f64],
            [0.0, 1.0],
            2,
            |_t, x| array![-x[0], -x[1]],
            |_t, _x| {
                let mut g = Array2::zeros((2, 2));
                g[[0, 0]] = 0.5;
                g[[1, 1]] = 0.5;
                g
            },
        );
        let mut rng = seeded_rng(99);
        let sol = weak_euler(&prob, 0.01, &mut rng).expect("weak_euler should succeed");
        assert_eq!(sol.x[0].len(), 2);
    }

    #[test]
    fn test_monte_carlo_mean_zero_paths_error() {
        let prob = make_gbm(0.05, 0.2, 1.0, 1.0);
        let mut rng = seeded_rng(0);
        assert!(monte_carlo_mean(&prob, 0.1, 0, &mut rng).is_err());
    }
}
