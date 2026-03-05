//! Standard SDE examples and reference processes
//!
//! This module provides ready-to-use implementations of well-known stochastic
//! differential equations that arise frequently in finance, physics, and
//! statistical modeling.
//!
//! ## Available Processes
//!
//! | Process | SDE | Use Case |
//! |---------|-----|----------|
//! | Geometric Brownian Motion | `dS = μS dt + σS dW` | Stock prices (Black-Scholes) |
//! | Ornstein-Uhlenbeck | `dX = θ(μ-X) dt + σ dW` | Mean-reverting noise, velocity |
//! | Cox-Ingersoll-Ross | `dr = κ(θ-r) dt + σ√r dW` | Short interest rates |
//! | Arithmetic Brownian Motion | `dX = μ dt + σ dW` | Basic diffusion |
//! | Vasicek | `dr = (a - br) dt + σ dW` | Interest rates (linear OU) |
//! | Heston Volatility | Coupled SDEs | Stochastic volatility |

use crate::error::{IntegrateError, IntegrateResult};
use crate::sde::{SdeSolution};
use crate::sde::euler_maruyama::euler_maruyama;
use crate::sde::milstein::scalar_milstein;
use crate::sde::runge_kutta_sde::platen_scheme;
use crate::sde::SdeProblem;
use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::prelude::StdRng;

/// Simulate Geometric Brownian Motion (GBM).
///
/// The GBM satisfies:
/// ```text
/// dS = μ S dt + σ S dW,   S(t0) = S0
/// ```
///
/// The analytic solution is:
/// ```text
/// S(t) = S0 exp((μ - σ²/2) t + σ W(t))
/// ```
///
/// This is the fundamental model for stock prices in the Black-Scholes framework.
///
/// # Arguments
///
/// * `mu` - Drift rate (e.g. expected return)
/// * `sigma` - Volatility coefficient (must be > 0)
/// * `s0` - Initial price/value (must be > 0)
/// * `t_span` - Simulation interval [t0, t1]
/// * `dt` - Step size
/// * `rng` - Random number generator
///
/// # Returns
///
/// An `SdeSolution` representing the simulated path.
///
/// # Errors
///
/// Returns an error if `sigma <= 0`, `s0 <= 0`, or the time span is invalid.
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::examples::geometric_brownian_motion;
/// use scirs2_core::random::prelude::*;
///
/// let mut rng = seeded_rng(42);
/// let sol = geometric_brownian_motion(0.05, 0.2, 100.0, [0.0, 1.0], 0.01, &mut rng).unwrap();
/// // GBM price is always positive
/// for xi in &sol.x {
///     assert!(xi[0] > 0.0, "GBM price must be positive");
/// }
/// ```
pub fn geometric_brownian_motion(
    mu: f64,
    sigma: f64,
    s0: f64,
    t_span: [f64; 2],
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution> {
    if sigma <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "GBM volatility sigma must be > 0, got {}",
            sigma
        )));
    }
    if s0 <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "GBM initial value s0 must be > 0, got {}",
            s0
        )));
    }

    // Use scalar Milstein for exact strong order 1.0
    // For GBM with g(x) = sigma*x, g'(x) = sigma, the Milstein correction is exact
    scalar_milstein(
        move |_t, x| mu * x,
        move |_t, x| sigma * x,
        s0,
        t_span,
        dt,
        rng,
    )
}

/// Simulate the Ornstein-Uhlenbeck (OU) process.
///
/// The OU process satisfies:
/// ```text
/// dX = θ(μ - X) dt + σ dW,   X(t0) = x0
/// ```
///
/// It is a mean-reverting Gaussian process with:
/// - **Stationary mean**: μ
/// - **Stationary variance**: σ²/(2θ)
/// - **Correlation time**: 1/θ
///
/// The analytic mean is:
/// ```text
/// E[X(t)] = x0 exp(-θ t) + μ (1 - exp(-θ t))
/// ```
///
/// Applications: velocity in Brownian motion, interest rates (Vasicek model),
/// commodity prices, neural membrane potentials.
///
/// # Arguments
///
/// * `theta` - Mean reversion speed (must be > 0)
/// * `mu` - Long-run mean
/// * `sigma` - Diffusion coefficient (must be > 0)
/// * `x0` - Initial condition
/// * `t_span` - Time interval [t0, t1]
/// * `dt` - Step size
/// * `rng` - RNG reference
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::examples::ornstein_uhlenbeck;
/// use scirs2_core::random::prelude::*;
///
/// let mut rng = seeded_rng(0);
/// let sol = ornstein_uhlenbeck(2.0, 1.0, 0.3, 0.0, [0.0, 5.0], 0.01, &mut rng).unwrap();
/// // After long time, mean should approach mu = 1.0
/// let final_x = sol.x_final().unwrap()[0];
/// assert!((final_x - 1.0).abs() < 1.5);  // statistical tolerance
/// ```
pub fn ornstein_uhlenbeck(
    theta: f64,
    mu: f64,
    sigma: f64,
    x0: f64,
    t_span: [f64; 2],
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution> {
    if theta <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "OU theta (mean reversion speed) must be > 0, got {}",
            theta
        )));
    }
    if sigma <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "OU sigma must be > 0, got {}",
            sigma
        )));
    }

    // OU has additive noise (g'(x) = 0), so EM = Milstein = same accuracy.
    // Use scalar_milstein for interface consistency (correction term is zero).
    scalar_milstein(
        move |_t, x| theta * (mu - x),
        move |_t, _x| sigma,
        x0,
        t_span,
        dt,
        rng,
    )
}

/// Simulate the Cox-Ingersoll-Ross (CIR) process.
///
/// The CIR process satisfies:
/// ```text
/// dr = κ(θ - r) dt + σ √r dW,   r(t0) = r0
/// ```
///
/// It is a mean-reverting process with non-negative values (for r0 > 0 and
/// the Feller condition 2κθ ≥ σ² satisfied).
///
/// The **Feller condition** 2κθ ≥ σ² ensures r(t) > 0 a.s. When violated,
/// r(t) can touch zero, and we apply reflection to keep it non-negative.
///
/// Applications:
/// - Short-rate models in finance (original CIR 1985)
/// - Stochastic volatility (square-root diffusion in Heston model)
/// - Population dynamics
///
/// # Arguments
///
/// * `kappa` - Mean reversion speed (must be > 0)
/// * `theta` - Long-run mean (must be ≥ 0)
/// * `sigma` - Volatility coefficient (must be > 0)
/// * `r0` - Initial value (must be ≥ 0)
/// * `t_span` - Time interval [t0, t1]
/// * `dt` - Step size
/// * `rng` - RNG reference
///
/// # Notes
///
/// For numerical stability, the diffusion uses `√max(r, 0)` to prevent
/// negative arguments to the square root, implementing a truncation scheme
/// (Deelstra & Delbaen 1998, Lord et al. 2010).
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::examples::cir_process;
/// use scirs2_core::random::prelude::*;
///
/// // CIR with Feller condition satisfied: 2κθ = 2 > σ² = 0.09
/// let mut rng = seeded_rng(42);
/// let sol = cir_process(1.0, 1.0, 0.3, 1.0, [0.0, 1.0], 0.01, &mut rng).unwrap();
/// // CIR stays non-negative (with truncation scheme)
/// for xi in &sol.x {
///     assert!(xi[0] >= 0.0, "CIR should be non-negative");
/// }
/// ```
pub fn cir_process(
    kappa: f64,
    theta: f64,
    sigma: f64,
    r0: f64,
    t_span: [f64; 2],
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution> {
    if kappa <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "CIR kappa must be > 0, got {}",
            kappa
        )));
    }
    if theta < 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "CIR theta must be >= 0, got {}",
            theta
        )));
    }
    if sigma <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "CIR sigma must be > 0, got {}",
            sigma
        )));
    }
    if r0 < 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "CIR initial value r0 must be >= 0, got {}",
            r0
        )));
    }

    // Use Platen scheme for better accuracy with sqrt diffusion
    let prob = SdeProblem::new(
        array![r0],
        t_span,
        1,
        move |_t, x| array![kappa * (theta - x[0])],
        move |_t, x| {
            let mut g = Array2::zeros((1, 1));
            // Truncated square root for non-negativity preservation
            g[[0, 0]] = sigma * x[0].max(0.0).sqrt();
            g
        },
    );

    // Apply reflection to keep r >= 0 after each step
    let sol_raw = platen_scheme(&prob, dt, rng)?;

    // Apply absorbing/reflecting boundary at 0
    let reflected: Vec<Array1<f64>> = sol_raw.x.into_iter().map(|mut xi| {
        xi[0] = xi[0].max(0.0);
        xi
    }).collect();

    Ok(SdeSolution {
        t: sol_raw.t,
        x: reflected,
    })
}

/// Simulate Arithmetic Brownian Motion (ABM).
///
/// ABM (also called Bachelier model or Brownian motion with drift) satisfies:
/// ```text
/// dX = μ dt + σ dW,   X(t0) = x0
/// ```
///
/// This is additive noise with constant drift and diffusion. The analytic
/// solution is: `X(t) = x0 + μ t + σ W(t)`
///
/// # Arguments
///
/// * `mu` - Drift coefficient
/// * `sigma` - Diffusion coefficient (must be > 0)
/// * `x0` - Initial value
/// * `t_span` - Time interval [t0, t1]
/// * `dt` - Step size
/// * `rng` - RNG reference
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::examples::arithmetic_brownian_motion;
/// use scirs2_core::random::prelude::*;
///
/// let mut rng = seeded_rng(42);
/// let sol = arithmetic_brownian_motion(0.1, 0.5, 0.0, [0.0, 1.0], 0.01, &mut rng).unwrap();
/// assert_eq!(sol.len(), 101);
/// ```
pub fn arithmetic_brownian_motion(
    mu: f64,
    sigma: f64,
    x0: f64,
    t_span: [f64; 2],
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution> {
    if sigma <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "ABM sigma must be > 0, got {}",
            sigma
        )));
    }
    // Additive noise: EM is exact (no Milstein correction needed)
    scalar_milstein(
        move |_t, _x| mu,
        move |_t, _x| sigma,
        x0,
        t_span,
        dt,
        rng,
    )
}

/// Simulate the Vasicek (linear OU) interest rate model.
///
/// Vasicek model satisfies:
/// ```text
/// dr = (a - b·r) dt + σ dW,   r(t0) = r0
/// ```
///
/// This is equivalent to the OU process with `θ = a/b`, `κ = b`.
/// Allows r to become negative (unlike CIR).
///
/// # Arguments
///
/// * `a` - Mean-reversion level times speed
/// * `b` - Mean-reversion speed (must be > 0)
/// * `sigma` - Volatility (must be > 0)
/// * `r0` - Initial rate
/// * `t_span` - Time interval [t0, t1]
/// * `dt` - Step size
/// * `rng` - RNG reference
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::examples::vasicek;
/// use scirs2_core::random::prelude::*;
///
/// let mut rng = seeded_rng(1);
/// // a=0.1, b=1.0 → long-run mean = a/b = 0.1
/// let sol = vasicek(0.1, 1.0, 0.05, 0.05, [0.0, 10.0], 0.01, &mut rng).unwrap();
/// assert!(!sol.is_empty());
/// ```
pub fn vasicek(
    a: f64,
    b: f64,
    sigma: f64,
    r0: f64,
    t_span: [f64; 2],
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution> {
    if b <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Vasicek b (mean reversion speed) must be > 0, got {}",
            b
        )));
    }
    if sigma <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Vasicek sigma must be > 0, got {}",
            sigma
        )));
    }
    scalar_milstein(
        move |_t, x| a - b * x,
        move |_t, _x| sigma,
        r0,
        t_span,
        dt,
        rng,
    )
}

/// Simulate the Heston stochastic volatility model.
///
/// The Heston model consists of two coupled SDEs:
/// ```text
/// dS = μ S dt + √v S dW₁                   (price process)
/// dv = κ(θ - v) dt + σ_v √v dW₂            (variance process)
/// ```
///
/// where `dW₁ dW₂ = ρ dt` (correlated Brownian motions, correlation ρ).
///
/// The variance `v(t)` is a CIR process, and `S(t)` is a GBM with stochastic volatility.
///
/// # Arguments
///
/// * `mu` - Asset drift
/// * `kappa` - Variance mean reversion speed
/// * `theta` - Long-run variance
/// * `sigma_v` - Volatility of volatility
/// * `rho` - Correlation between asset and variance Brownian motions (in [-1, 1])
/// * `s0` - Initial asset price (must be > 0)
/// * `v0` - Initial variance (must be ≥ 0)
/// * `t_span` - Time interval [t0, t1]
/// * `dt` - Step size
/// * `rng` - RNG reference
///
/// # Returns
///
/// An `SdeSolution` where `x[i][0] = S(t_i)` and `x[i][1] = v(t_i)`.
///
/// # Examples
///
/// ```rust
/// use scirs2_integrate::sde::examples::heston;
/// use scirs2_core::random::prelude::*;
///
/// let mut rng = seeded_rng(42);
/// // Typical Heston parameters: Feller condition 2κθ = 0.5 > σ_v² = 0.04
/// let sol = heston(0.05, 2.0, 0.04, 0.2, -0.7, 100.0, 0.04, [0.0, 1.0], 0.005, &mut rng).unwrap();
/// for xi in &sol.x {
///     assert!(xi[0] > 0.0, "Price must be positive");
///     assert!(xi[1] >= 0.0, "Variance must be non-negative");
/// }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn heston(
    mu: f64,
    kappa: f64,
    theta: f64,
    sigma_v: f64,
    rho: f64,
    s0: f64,
    v0: f64,
    t_span: [f64; 2],
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<SdeSolution> {
    if s0 <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Heston s0 must be > 0, got {}",
            s0
        )));
    }
    if v0 < 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Heston v0 must be >= 0, got {}",
            v0
        )));
    }
    if rho.abs() > 1.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Heston rho must be in [-1, 1], got {}",
            rho
        )));
    }
    if sigma_v <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Heston sigma_v must be > 0, got {}",
            sigma_v
        )));
    }
    if kappa <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Heston kappa must be > 0, got {}",
            kappa
        )));
    }

    // Cholesky factorization for correlated Brownian motions:
    // W1 = Z1, W2 = ρ Z1 + √(1-ρ²) Z2
    let rho_perp = (1.0 - rho * rho).sqrt();

    // State: x = [S, v], driven by 2 independent BMs Z1, Z2
    // The diffusion matrix maps (Z1, Z2) → (dS_noise, dv_noise):
    //   dS = μ S dt + √v S dW1 = μ S dt + √v S Z1 dZ
    //   dv = κ(θ-v) dt + σ_v √v dW2 = κ(θ-v) dt + σ_v √v (ρ Z1 + √(1-ρ²) Z2) dZ
    //
    // Drift: f([S, v]) = [μ S, κ(θ-v)]
    // Diffusion: g([S, v]) = [[√v S, 0], [σ_v √v ρ, σ_v √v √(1-ρ²)]]
    let prob = SdeProblem::new(
        array![s0, v0],
        t_span,
        2,
        move |_t, x| {
            let s = x[0];
            let v = x[1].max(0.0);
            array![mu * s, kappa * (theta - v)]
        },
        move |_t, x| {
            let s = x[0].max(0.0);
            let v = x[1].max(0.0);
            let sqrt_v = v.sqrt();
            let mut g = Array2::zeros((2, 2));
            // Column 0: effect of Z1
            g[[0, 0]] = sqrt_v * s;
            g[[1, 0]] = sigma_v * sqrt_v * rho;
            // Column 1: effect of Z2
            g[[0, 1]] = 0.0;
            g[[1, 1]] = sigma_v * sqrt_v * rho_perp;
            g
        },
    );

    let sol_raw = euler_maruyama(&prob, dt, rng)?;

    // Reflect variance at 0 (truncation scheme)
    let reflected: Vec<Array1<f64>> = sol_raw.x.into_iter().map(|mut xi| {
        xi[1] = xi[1].max(0.0);
        xi
    }).collect();

    Ok(SdeSolution {
        t: sol_raw.t,
        x: reflected,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::prelude::seeded_rng;

    #[test]
    fn test_gbm_positive() {
        let mut rng = seeded_rng(42);
        let sol = geometric_brownian_motion(0.05, 0.2, 100.0, [0.0, 1.0], 0.01, &mut rng).expect("geometric_brownian_motion should succeed");
        for xi in &sol.x {
            assert!(xi[0] > 0.0, "GBM must stay positive");
        }
    }

    #[test]
    fn test_gbm_weak_mean() {
        // E[S(T)] = S0 * exp(mu * T)
        let mu = 0.1_f64;
        let sigma = 0.2_f64;
        let s0 = 1.0_f64;
        let t1 = 1.0_f64;
        let analytic = s0 * (mu * t1).exp();
        let n_paths = 500;
        let dt = 0.005;

        let mut sum = 0.0;
        for seed in 0..n_paths as u64 {
            let mut rng = seeded_rng(seed + 100);
            let sol = geometric_brownian_motion(mu, sigma, s0, [0.0, t1], dt, &mut rng).expect("geometric_brownian_motion should succeed");
            sum += sol.x_final().expect("solution has state")[0];
        }
        let mean = sum / n_paths as f64;
        assert!(
            (mean - analytic).abs() / analytic < 0.05,
            "GBM mean {:.4} vs analytic {:.4}",
            mean,
            analytic
        );
    }

    #[test]
    fn test_ou_mean_reversion() {
        // OU: E[X(t)] = x0 exp(-θ t) + μ (1 - exp(-θ t))
        let theta = 2.0_f64;
        let mu_ou = 1.0_f64;
        let sigma = 0.3_f64;
        let x0 = 0.0_f64;
        let t1 = 3.0_f64;
        let analytic_mean = x0 * (-theta * t1).exp() + mu_ou * (1.0 - (-theta * t1).exp());
        let n_paths = 400;
        let dt = 0.01;

        let mut sum = 0.0;
        for seed in 0..n_paths as u64 {
            let mut rng = seeded_rng(seed + 200);
            let sol = ornstein_uhlenbeck(theta, mu_ou, sigma, x0, [0.0, t1], dt, &mut rng).expect("ornstein_uhlenbeck should succeed");
            sum += sol.x_final().expect("solution has state")[0];
        }
        let mean = sum / n_paths as f64;
        assert!(
            (mean - analytic_mean).abs() < 0.1,
            "OU mean {:.4} vs analytic {:.4}",
            mean,
            analytic_mean
        );
    }

    #[test]
    fn test_cir_non_negative() {
        // Feller condition: 2κθ = 2.0 > σ² = 0.09 → stays positive
        let mut rng = seeded_rng(7);
        let sol = cir_process(1.0, 1.0, 0.3, 1.0, [0.0, 5.0], 0.005, &mut rng).expect("cir_process should succeed");
        for xi in &sol.x {
            assert!(xi[0] >= 0.0, "CIR must be non-negative, got {}", xi[0]);
        }
    }

    #[test]
    fn test_cir_mean_reversion() {
        // E[r(t)] = r0 exp(-κ t) + θ (1 - exp(-κ t))
        let kappa = 1.0_f64;
        let theta = 0.5_f64;
        let sigma = 0.2_f64;
        let r0 = 0.1_f64;
        let t1 = 5.0_f64;
        let analytic_mean = r0 * (-kappa * t1).exp() + theta * (1.0 - (-kappa * t1).exp());
        let n_paths = 400;
        let dt = 0.01;

        let mut sum = 0.0;
        for seed in 0..n_paths as u64 {
            let mut rng = seeded_rng(seed + 300);
            let sol = cir_process(kappa, theta, sigma, r0, [0.0, t1], dt, &mut rng).expect("cir_process should succeed");
            sum += sol.x_final().expect("solution has state")[0];
        }
        let mean = sum / n_paths as f64;
        assert!(
            (mean - analytic_mean).abs() < 0.1,
            "CIR mean {:.4} vs analytic {:.4}",
            mean,
            analytic_mean
        );
    }

    #[test]
    fn test_abm_mean() {
        // E[X(T)] = x0 + mu * T
        let mu = 0.5_f64;
        let sigma = 1.0_f64;
        let x0 = 0.0_f64;
        let t1 = 2.0_f64;
        let analytic_mean = x0 + mu * t1;
        let n_paths = 400;
        let dt = 0.01;

        let mut sum = 0.0;
        for seed in 0..n_paths as u64 {
            let mut rng = seeded_rng(seed + 400);
            let sol = arithmetic_brownian_motion(mu, sigma, x0, [0.0, t1], dt, &mut rng).expect("arithmetic_brownian_motion should succeed");
            sum += sol.x_final().expect("solution has state")[0];
        }
        let mean = sum / n_paths as f64;
        assert!(
            (mean - analytic_mean).abs() < 0.2,
            "ABM mean {:.4} vs analytic {:.4}",
            mean,
            analytic_mean
        );
    }

    #[test]
    fn test_vasicek_basic() {
        let mut rng = seeded_rng(5);
        let sol = vasicek(0.1, 1.0, 0.05, 0.05, [0.0, 10.0], 0.01, &mut rng).expect("vasicek should succeed");
        assert!(!sol.is_empty());
        // Long-run mean = a/b = 0.1
        // Final value should be near 0.1 on average (we just test it ran)
    }

    #[test]
    fn test_heston_non_negative_variance() {
        let mut rng = seeded_rng(42);
        let sol = heston(
            0.05, 2.0, 0.04, 0.2, -0.7,
            100.0, 0.04,
            [0.0, 1.0], 0.005, &mut rng,
        ).expect("heston should succeed");
        for xi in &sol.x {
            assert!(xi[0] > 0.0, "Heston S must be positive");
            assert!(xi[1] >= 0.0, "Heston v must be non-negative");
        }
    }

    #[test]
    fn test_gbm_invalid_sigma() {
        let mut rng = seeded_rng(0);
        assert!(geometric_brownian_motion(-0.05, -0.2, 100.0, [0.0, 1.0], 0.01, &mut rng).is_err());
    }

    #[test]
    fn test_gbm_invalid_s0() {
        let mut rng = seeded_rng(0);
        assert!(geometric_brownian_motion(0.05, 0.2, -1.0, [0.0, 1.0], 0.01, &mut rng).is_err());
    }

    #[test]
    fn test_cir_invalid_params() {
        let mut rng = seeded_rng(0);
        assert!(cir_process(-1.0, 0.5, 0.3, 1.0, [0.0, 1.0], 0.01, &mut rng).is_err());
    }
}
