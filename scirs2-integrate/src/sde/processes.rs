//! Standard stochastic processes
//!
//! This module provides high-level struct-based interfaces for common stochastic processes
//! that arise in finance, physics, and statistical modeling.
//!
//! ## Processes
//!
//! | Struct | SDE | Application |
//! |--------|-----|-------------|
//! | [`BrownianMotion`] | `dW` | Base Wiener process |
//! | [`GeometricBrownianMotion`] | `dS = μS dt + σS dW` | Stock prices (Black-Scholes) |
//! | [`OrnsteinUhlenbeck`] | `dX = θ(μ-X) dt + σ dW` | Mean-reverting processes |
//! | [`CoxIngersollRoss`] | `dr = κ(θ-r) dt + σ√r dW` | Interest rates |
//! | [`HestonModel`] | Coupled SDEs | Stochastic volatility |

use crate::error::{IntegrateError, IntegrateResult};
use crate::sde::{SdeProblem, SdeSolution};
use crate::sde::euler_maruyama::euler_maruyama;
use crate::sde::milstein::scalar_milstein;
use scirs2_core::ndarray::{array, Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Distribution, Normal};

// ─────────────────────────────────────────────────────────────────────────────
// Brownian Motion
// ─────────────────────────────────────────────────────────────────────────────

/// Standard Brownian motion (Wiener process).
///
/// `W_0 = 0`, increments `ΔW ~ N(0, dt)`.
pub struct BrownianMotion;

impl BrownianMotion {
    /// Sample a discrete Brownian motion path.
    ///
    /// Returns a vector of length `n_steps + 1` with `W_0 = 0`.
    ///
    /// # Arguments
    ///
    /// * `n_steps` — number of time steps
    /// * `dt` — step size (> 0)
    /// * `rng` — random number generator
    ///
    /// # Errors
    ///
    /// Returns an error if `dt <= 0`.
    pub fn sample(n_steps: usize, dt: f64, rng: &mut StdRng) -> IntegrateResult<Vec<f64>> {
        if dt <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "dt must be > 0, got {dt}"
            )));
        }
        let normal = Normal::new(0.0, dt.sqrt())
            .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;
        let mut path = Vec::with_capacity(n_steps + 1);
        path.push(0.0);
        let mut w = 0.0;
        for _ in 0..n_steps {
            w += normal.sample(rng);
            path.push(w);
        }
        Ok(path)
    }

    /// Sample a Brownian motion at specific time points (given as an ordered slice).
    ///
    /// Uses exact increments `ΔW ~ N(0, dt_i)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `times` is empty or not strictly increasing.
    pub fn sample_at(times: &[f64], rng: &mut StdRng) -> IntegrateResult<Vec<f64>> {
        if times.is_empty() {
            return Err(IntegrateError::InvalidInput(
                "times must be non-empty".to_string(),
            ));
        }
        let mut path = Vec::with_capacity(times.len() + 1);
        path.push(0.0_f64); // W(0) = 0
        let mut prev = 0.0_f64;
        let mut w = 0.0_f64;
        for &t in times {
            let dt = t - prev;
            if dt <= 0.0 {
                return Err(IntegrateError::InvalidInput(
                    "times must be strictly increasing and start after 0".to_string(),
                ));
            }
            let normal = Normal::new(0.0, dt.sqrt())
                .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;
            w += normal.sample(rng);
            path.push(w);
            prev = t;
        }
        Ok(path)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Geometric Brownian Motion
// ─────────────────────────────────────────────────────────────────────────────

/// Geometric Brownian Motion.
///
/// ```text
/// dS = μ S dt + σ S dW,   S(0) = S_0
/// ```
///
/// Exact solution: `S(t) = S_0 exp((μ - σ²/2) t + σ W(t))`.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::sde::processes::GeometricBrownianMotion;
/// use scirs2_core::random::prelude::*;
///
/// let gbm = GeometricBrownianMotion { mu: 0.05, sigma: 0.2, s0: 100.0 };
/// let mut rng = seeded_rng(42);
/// let path = gbm.simulate((0.0, 1.0), 0.01, &mut rng).unwrap();
/// assert!(path.iter().all(|&(_, s)| s > 0.0), "GBM must stay positive");
/// ```
#[derive(Debug, Clone)]
pub struct GeometricBrownianMotion {
    /// Drift rate.
    pub mu: f64,
    /// Volatility (must be > 0).
    pub sigma: f64,
    /// Initial value (must be > 0).
    pub s0: f64,
}

impl GeometricBrownianMotion {
    /// Create a new GBM with validation.
    pub fn new(mu: f64, sigma: f64, s0: f64) -> IntegrateResult<Self> {
        if sigma <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "GBM sigma must be > 0, got {sigma}"
            )));
        }
        if s0 <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "GBM s0 must be > 0, got {s0}"
            )));
        }
        Ok(Self { mu, sigma, s0 })
    }

    /// Simulate a GBM path using the Milstein method (strong order 1.0).
    ///
    /// Returns a vector of `(t, S(t))` pairs.
    pub fn simulate(
        &self,
        t_span: (f64, f64),
        dt: f64,
        rng: &mut StdRng,
    ) -> IntegrateResult<Vec<(f64, f64)>> {
        if t_span.0 >= t_span.1 {
            return Err(IntegrateError::InvalidInput(format!(
                "t_span must satisfy t0 < t1, got {:?}", t_span
            )));
        }
        let mu = self.mu;
        let sigma = self.sigma;
        let sol = scalar_milstein(
            move |_t, x| mu * x,
            move |_t, x| sigma * x,
            self.s0,
            [t_span.0, t_span.1],
            dt,
            rng,
        )?;
        Ok(sol.t.into_iter().zip(sol.x.into_iter().map(|x| x[0])).collect())
    }

    /// Simulate using the exact analytical formula.
    ///
    /// `S(t) = S_0 * exp((μ - σ²/2) * t + σ * W(t))`
    ///
    /// This avoids discretization error entirely.
    pub fn simulate_exact(
        &self,
        t_span: (f64, f64),
        n_steps: usize,
        rng: &mut StdRng,
    ) -> IntegrateResult<Vec<(f64, f64)>> {
        if t_span.0 >= t_span.1 {
            return Err(IntegrateError::InvalidInput(format!(
                "t_span must satisfy t0 < t1, got {:?}", t_span
            )));
        }
        if n_steps == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_steps must be > 0".to_string(),
            ));
        }
        let dt = (t_span.1 - t_span.0) / n_steps as f64;
        let bm = BrownianMotion::sample(n_steps, dt, rng)?;
        let drift = self.mu - 0.5 * self.sigma * self.sigma;
        let mut path = Vec::with_capacity(n_steps + 1);
        path.push((t_span.0, self.s0));
        for i in 1..=n_steps {
            let t = t_span.0 + i as f64 * dt;
            let s = self.s0 * (drift * (t - t_span.0) + self.sigma * bm[i]).exp();
            path.push((t, s));
        }
        Ok(path)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ornstein-Uhlenbeck
// ─────────────────────────────────────────────────────────────────────────────

/// Ornstein-Uhlenbeck mean-reverting process.
///
/// ```text
/// dX = θ (μ - X) dt + σ dW,   X(0) = x_0
/// ```
///
/// The process is mean-reverting to `μ` with speed `θ`.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::sde::processes::OrnsteinUhlenbeck;
/// use scirs2_core::random::prelude::*;
///
/// let ou = OrnsteinUhlenbeck { theta: 2.0, mu: 1.0, sigma: 0.3, x0: 0.0 };
/// let mut rng = seeded_rng(42);
/// let path = ou.simulate_exact((0.0, 5.0), 500, &mut rng).unwrap();
/// assert!(!path.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct OrnsteinUhlenbeck {
    /// Mean-reversion speed (θ > 0).
    pub theta: f64,
    /// Long-run mean level.
    pub mu: f64,
    /// Diffusion coefficient (σ > 0).
    pub sigma: f64,
    /// Initial value.
    pub x0: f64,
}

impl OrnsteinUhlenbeck {
    /// Create a new OU process with validation.
    pub fn new(theta: f64, mu: f64, sigma: f64, x0: f64) -> IntegrateResult<Self> {
        if theta <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "OU theta (mean-reversion speed) must be > 0, got {theta}"
            )));
        }
        if sigma <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "OU sigma must be > 0, got {sigma}"
            )));
        }
        Ok(Self { theta, mu, sigma, x0 })
    }

    /// Simulate using Euler-Maruyama.
    pub fn simulate(
        &self,
        t_span: (f64, f64),
        dt: f64,
        rng: &mut StdRng,
    ) -> IntegrateResult<Vec<(f64, f64)>> {
        if t_span.0 >= t_span.1 {
            return Err(IntegrateError::InvalidInput(format!(
                "t_span must satisfy t0 < t1, got {:?}", t_span
            )));
        }
        let theta = self.theta;
        let mu = self.mu;
        let sigma = self.sigma;
        let x0 = array![self.x0];
        let prob = SdeProblem::new(
            x0,
            [t_span.0, t_span.1],
            1,
            move |_t, x| array![theta * (mu - x[0])],
            move |_t, _x| {
                let mut g = Array2::zeros((1, 1));
                g[[0, 0]] = sigma;
                g
            },
        );
        let sol = euler_maruyama(&prob, dt, rng)?;
        Ok(sol.t.into_iter().zip(sol.x.into_iter().map(|x| x[0])).collect())
    }

    /// Simulate using the exact conditional distribution.
    ///
    /// The exact transition is:
    /// ```text
    /// X(t+dt) | X(t) ~ N(μ + (X(t)-μ) exp(-θ dt),  σ²(1 - exp(-2θ dt)) / (2θ))
    /// ```
    pub fn simulate_exact(
        &self,
        t_span: (f64, f64),
        n_steps: usize,
        rng: &mut StdRng,
    ) -> IntegrateResult<Vec<(f64, f64)>> {
        if t_span.0 >= t_span.1 {
            return Err(IntegrateError::InvalidInput(format!(
                "t_span must satisfy t0 < t1, got {:?}", t_span
            )));
        }
        if n_steps == 0 {
            return Err(IntegrateError::InvalidInput(
                "n_steps must be > 0".to_string(),
            ));
        }
        let dt = (t_span.1 - t_span.0) / n_steps as f64;
        let exp_neg = (-self.theta * dt).exp();
        let var_incr = self.sigma * self.sigma * (1.0 - exp_neg * exp_neg) / (2.0 * self.theta);
        let std_incr = var_incr.sqrt();
        let normal = Normal::new(0.0, std_incr)
            .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;

        let mut path = Vec::with_capacity(n_steps + 1);
        path.push((t_span.0, self.x0));
        let mut x = self.x0;
        for i in 1..=n_steps {
            let t = t_span.0 + i as f64 * dt;
            let noise: f64 = normal.sample(rng);
            x = self.mu + (x - self.mu) * exp_neg + noise;
            path.push((t, x));
        }
        Ok(path)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cox-Ingersoll-Ross
// ─────────────────────────────────────────────────────────────────────────────

/// Cox-Ingersoll-Ross interest-rate process.
///
/// ```text
/// dr = κ (θ - r) dt + σ √r dW,   r(0) = r_0
/// ```
///
/// The Feller condition `2κθ ≥ σ²` ensures the process stays positive.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::sde::processes::CoxIngersollRoss;
/// use scirs2_core::random::prelude::*;
///
/// let cir = CoxIngersollRoss { kappa: 1.0, theta: 0.5, sigma: 0.2, r0: 0.3 };
/// let mut rng = seeded_rng(7);
/// let path = cir.simulate((0.0, 5.0), 0.005, &mut rng).unwrap();
/// assert!(path.iter().all(|&(_, r)| r >= 0.0), "CIR must stay non-negative");
/// ```
#[derive(Debug, Clone)]
pub struct CoxIngersollRoss {
    /// Mean-reversion speed (κ > 0).
    pub kappa: f64,
    /// Long-run mean (θ > 0).
    pub theta: f64,
    /// Volatility (σ > 0).
    pub sigma: f64,
    /// Initial rate (r₀ ≥ 0).
    pub r0: f64,
}

impl CoxIngersollRoss {
    /// Create a new CIR process with validation.
    ///
    /// Issues a warning (non-fatal) if the Feller condition is violated.
    pub fn new(kappa: f64, theta: f64, sigma: f64, r0: f64) -> IntegrateResult<Self> {
        if kappa <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "CIR kappa must be > 0, got {kappa}"
            )));
        }
        if theta <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "CIR theta must be > 0, got {theta}"
            )));
        }
        if sigma <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "CIR sigma must be > 0, got {sigma}"
            )));
        }
        if r0 < 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "CIR r0 must be >= 0, got {r0}"
            )));
        }
        Ok(Self { kappa, theta, sigma, r0 })
    }

    /// Whether the Feller condition `2κθ ≥ σ²` holds (guarantees positivity).
    pub fn feller_satisfied(&self) -> bool {
        2.0 * self.kappa * self.theta >= self.sigma * self.sigma
    }

    /// Simulate using Euler-Maruyama with reflection at 0.
    ///
    /// The truncation scheme (`r ← max(r, 0)`) keeps the process non-negative.
    pub fn simulate(
        &self,
        t_span: (f64, f64),
        dt: f64,
        rng: &mut StdRng,
    ) -> IntegrateResult<Vec<(f64, f64)>> {
        if t_span.0 >= t_span.1 {
            return Err(IntegrateError::InvalidInput(format!(
                "t_span must satisfy t0 < t1, got {:?}", t_span
            )));
        }
        if dt <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "dt must be > 0, got {dt}"
            )));
        }
        let kappa = self.kappa;
        let theta = self.theta;
        let sigma = self.sigma;
        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;
        let sqrt_dt = dt.sqrt();
        let n_steps = ((t_span.1 - t_span.0) / dt).ceil() as usize;

        let mut path = Vec::with_capacity(n_steps + 1);
        path.push((t_span.0, self.r0));
        let mut r = self.r0;
        let mut t = t_span.0;

        for _ in 0..n_steps {
            let step = (t_span.1 - t).min(dt);
            let sqrt_step = step.sqrt();
            let z: f64 = normal.sample(rng);
            r = (r + kappa * (theta - r) * step + sigma * r.max(0.0).sqrt() * sqrt_step * z)
                .max(0.0);
            t += step;
            path.push((t, r));
        }
        Ok(path)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Heston Model
// ─────────────────────────────────────────────────────────────────────────────

/// Heston stochastic volatility model.
///
/// ```text
/// dS = μ S dt + √v S dW₁
/// dv = κ (θ - v) dt + σ √v dW₂
/// corr(dW₁, dW₂) = ρ
/// ```
///
/// # Examples
///
/// ```
/// use scirs2_integrate::sde::processes::HestonModel;
/// use scirs2_core::random::prelude::*;
///
/// let heston = HestonModel {
///     kappa: 2.0, theta: 0.04, sigma: 0.2, rho: -0.7,
///     v0: 0.04, mu: 0.05, s0: 100.0,
/// };
/// let mut rng = seeded_rng(42);
/// let path = heston.simulate((0.0, 1.0), 0.005, &mut rng).unwrap();
/// assert!(path.iter().all(|&(_, s, v)| s > 0.0 && v >= 0.0));
/// ```
#[derive(Debug, Clone)]
pub struct HestonModel {
    /// Variance mean-reversion speed (κ > 0).
    pub kappa: f64,
    /// Long-run variance level (θ > 0).
    pub theta: f64,
    /// Vol-of-vol (σ > 0).
    pub sigma: f64,
    /// Correlation between price and vol Brownians (|ρ| < 1).
    pub rho: f64,
    /// Initial variance (v₀ ≥ 0).
    pub v0: f64,
    /// Price drift (μ).
    pub mu: f64,
    /// Initial price (S₀ > 0).
    pub s0: f64,
}

impl HestonModel {
    /// Create a new Heston model with validation.
    pub fn new(kappa: f64, theta: f64, sigma: f64, rho: f64, v0: f64, mu: f64, s0: f64) -> IntegrateResult<Self> {
        if kappa <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!("Heston kappa must be > 0, got {kappa}")));
        }
        if theta <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!("Heston theta must be > 0, got {theta}")));
        }
        if sigma <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!("Heston sigma must be > 0, got {sigma}")));
        }
        if rho.abs() >= 1.0 {
            return Err(IntegrateError::InvalidInput(format!("Heston rho must satisfy |ρ| < 1, got {rho}")));
        }
        if v0 < 0.0 {
            return Err(IntegrateError::InvalidInput(format!("Heston v0 must be >= 0, got {v0}")));
        }
        if s0 <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!("Heston s0 must be > 0, got {s0}")));
        }
        Ok(Self { kappa, theta, sigma, rho, v0, mu, s0 })
    }

    /// Simulate the Heston model using Euler-Maruyama with variance truncation.
    ///
    /// Returns a vector of `(t, S(t), v(t))` triples.
    pub fn simulate(
        &self,
        t_span: (f64, f64),
        dt: f64,
        rng: &mut StdRng,
    ) -> IntegrateResult<Vec<(f64, f64, f64)>> {
        if t_span.0 >= t_span.1 {
            return Err(IntegrateError::InvalidInput(format!(
                "t_span must satisfy t0 < t1, got {:?}", t_span
            )));
        }
        if dt <= 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "dt must be > 0, got {dt}"
            )));
        }
        let kappa = self.kappa;
        let theta = self.theta;
        let sigma = self.sigma;
        let rho = self.rho;
        let rho_perp = (1.0 - rho * rho).sqrt();
        let mu = self.mu;

        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;
        let n_steps = ((t_span.1 - t_span.0) / dt).ceil() as usize;

        let mut path = Vec::with_capacity(n_steps + 1);
        path.push((t_span.0, self.s0, self.v0));
        let mut s = self.s0;
        let mut v = self.v0;
        let mut t = t_span.0;

        for _ in 0..n_steps {
            let step = (t_span.1 - t).min(dt);
            let sqrt_step = step.sqrt();
            let z1: f64 = normal.sample(rng);
            let z2: f64 = normal.sample(rng);
            // Cholesky: dW1 = z1, dW2 = rho*z1 + rho_perp*z2
            let dw1 = z1 * sqrt_step;
            let dw2 = (rho * z1 + rho_perp * z2) * sqrt_step;
            let sqrt_v = v.max(0.0).sqrt();
            s = s * (1.0 + mu * step + sqrt_v * dw1).max(1e-12);
            v = (v + kappa * (theta - v) * step + sigma * sqrt_v * dw2).max(0.0);
            t += step;
            path.push((t, s, v));
        }
        Ok(path)
    }

    /// Simulate using an Euler-Maruyama SdeProblem (returns SdeSolution with 2D state [S, v]).
    pub fn simulate_as_sde(
        &self,
        t_span: (f64, f64),
        dt: f64,
        rng: &mut StdRng,
    ) -> IntegrateResult<SdeSolution> {
        if t_span.0 >= t_span.1 {
            return Err(IntegrateError::InvalidInput(format!(
                "t_span must satisfy t0 < t1, got {:?}", t_span
            )));
        }
        let kappa = self.kappa;
        let theta = self.theta;
        let sigma = self.sigma;
        let rho = self.rho;
        let rho_perp = (1.0 - rho * rho).sqrt();
        let mu = self.mu;
        let x0 = array![self.s0, self.v0];
        let prob = SdeProblem::new(
            x0,
            [t_span.0, t_span.1],
            2,
            move |_t, x| {
                let s = x[0].max(0.0);
                let v = x[1].max(0.0);
                array![mu * s, kappa * (theta - v)]
            },
            move |_t, x| {
                let s = x[0].max(0.0);
                let v = x[1].max(0.0);
                let sqrt_v = v.sqrt();
                let mut g = Array2::zeros((2, 2));
                g[[0, 0]] = sqrt_v * s;
                g[[1, 0]] = sigma * sqrt_v * rho;
                g[[0, 1]] = 0.0;
                g[[1, 1]] = sigma * sqrt_v * rho_perp;
                g
            },
        );
        let sol_raw = euler_maruyama(&prob, dt, rng)?;
        // Truncate variance at 0
        let reflected: Vec<Array1<f64>> = sol_raw
            .x
            .into_iter()
            .map(|mut xi| {
                xi[1] = xi[1].max(0.0);
                xi
            })
            .collect();
        Ok(SdeSolution {
            t: sol_raw.t,
            x: reflected,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::prelude::seeded_rng;

    #[test]
    fn test_brownian_motion_starts_at_zero() {
        let mut rng = seeded_rng(1);
        let path = BrownianMotion::sample(100, 0.01, &mut rng).expect("BrownianMotion::sample should succeed");
        assert_eq!(path.len(), 101);
        assert!((path[0]).abs() < 1e-12, "W_0 must be 0");
    }

    #[test]
    fn test_brownian_motion_invalid_dt() {
        let mut rng = seeded_rng(0);
        assert!(BrownianMotion::sample(10, 0.0, &mut rng).is_err());
        assert!(BrownianMotion::sample(10, -0.1, &mut rng).is_err());
    }

    #[test]
    fn test_brownian_motion_variance() {
        // Var[W(T)] = T
        let n_paths = 2000;
        let n_steps = 100;
        let dt = 0.01;
        let t_final = n_steps as f64 * dt; // 1.0
        let mut sum_sq = 0.0;
        for seed in 0..n_paths as u64 {
            let mut rng = seeded_rng(seed + 1000);
            let path = BrownianMotion::sample(n_steps, dt, &mut rng).expect("BrownianMotion::sample should succeed");
            let w_final = path[n_steps];
            sum_sq += w_final * w_final;
        }
        let empirical_var = sum_sq / n_paths as f64;
        assert!(
            (empirical_var - t_final).abs() / t_final < 0.1,
            "BM variance: got {empirical_var:.4}, expected {t_final:.4}"
        );
    }

    #[test]
    fn test_gbm_positive() {
        let gbm = GeometricBrownianMotion { mu: 0.05, sigma: 0.2, s0: 100.0 };
        let mut rng = seeded_rng(42);
        let path = gbm.simulate((0.0, 1.0), 0.01, &mut rng).expect("gbm.simulate should succeed");
        assert!(path.iter().all(|&(_, s)| s > 0.0), "GBM must stay positive");
    }

    #[test]
    fn test_gbm_exact_mean() {
        // E[S(T)] = S0 * exp(mu * T)
        let mu = 0.1_f64;
        let sigma = 0.2_f64;
        let s0 = 1.0_f64;
        let t1 = 1.0_f64;
        let analytic = s0 * (mu * t1).exp();
        let n_paths = 1000;
        let mut sum = 0.0;
        for seed in 0..n_paths as u64 {
            let mut rng = seeded_rng(seed + 200);
            let gbm = GeometricBrownianMotion { mu, sigma, s0 };
            let path = gbm.simulate_exact((0.0, t1), 100, &mut rng).expect("gbm.simulate_exact should succeed");
            sum += path.last().map(|&(_, s)| s).unwrap_or(0.0);
        }
        let mean = sum / n_paths as f64;
        assert!(
            (mean - analytic).abs() / analytic < 0.05,
            "GBM mean {mean:.4} vs analytic {analytic:.4}"
        );
    }

    #[test]
    fn test_gbm_invalid() {
        assert!(GeometricBrownianMotion::new(0.05, -0.2, 100.0).is_err());
        assert!(GeometricBrownianMotion::new(0.05, 0.2, -1.0).is_err());
    }

    #[test]
    fn test_ou_mean_reversion() {
        // E[X(T)] → mu as T → ∞
        let ou = OrnsteinUhlenbeck { theta: 2.0, mu: 1.0, sigma: 0.3, x0: 0.0 };
        let t1 = 5.0;
        let n_paths = 500;
        let analytic = ou.mu + (ou.x0 - ou.mu) * (-ou.theta * t1).exp();
        let mut sum = 0.0;
        for seed in 0..n_paths as u64 {
            let mut rng = seeded_rng(seed + 300);
            let path = ou.simulate_exact((0.0, t1), 500, &mut rng).expect("ou.simulate_exact should succeed");
            sum += path.last().map(|&(_, x)| x).unwrap_or(0.0);
        }
        let mean = sum / n_paths as f64;
        assert!(
            (mean - analytic).abs() < 0.1,
            "OU mean {mean:.4} vs analytic {analytic:.4}"
        );
    }

    #[test]
    fn test_ou_invalid() {
        assert!(OrnsteinUhlenbeck::new(-1.0, 0.0, 0.3, 0.0).is_err());
        assert!(OrnsteinUhlenbeck::new(1.0, 0.0, -0.3, 0.0).is_err());
    }

    #[test]
    fn test_cir_non_negative() {
        // Feller: 2*1.0*0.5 = 1.0 >= 0.04 = sigma^2
        let cir = CoxIngersollRoss { kappa: 1.0, theta: 0.5, sigma: 0.2, r0: 0.1 };
        assert!(cir.feller_satisfied());
        let mut rng = seeded_rng(7);
        let path = cir.simulate((0.0, 5.0), 0.005, &mut rng).expect("cir.simulate should succeed");
        assert!(
            path.iter().all(|&(_, r)| r >= 0.0),
            "CIR must stay non-negative"
        );
    }

    #[test]
    fn test_cir_invalid() {
        assert!(CoxIngersollRoss::new(-1.0, 0.5, 0.2, 0.1).is_err());
        assert!(CoxIngersollRoss::new(1.0, 0.5, 0.2, -0.1).is_err());
    }

    #[test]
    fn test_heston_positive_price_non_negative_var() {
        let heston = HestonModel {
            kappa: 2.0,
            theta: 0.04,
            sigma: 0.2,
            rho: -0.7,
            v0: 0.04,
            mu: 0.05,
            s0: 100.0,
        };
        let mut rng = seeded_rng(42);
        let path = heston.simulate((0.0, 1.0), 0.005, &mut rng).expect("heston.simulate should succeed");
        assert!(path.iter().all(|&(_, s, v)| s > 0.0 && v >= 0.0));
    }

    #[test]
    fn test_heston_invalid() {
        assert!(HestonModel::new(-1.0, 0.04, 0.2, -0.7, 0.04, 0.05, 100.0).is_err());
        assert!(HestonModel::new(2.0, 0.04, 0.2, 1.0, 0.04, 0.05, 100.0).is_err()); // |rho| = 1
        assert!(HestonModel::new(2.0, 0.04, 0.2, -0.7, 0.04, 0.05, -1.0).is_err());
    }

    #[test]
    fn test_heston_sde_variant() {
        let heston = HestonModel {
            kappa: 2.0, theta: 0.04, sigma: 0.2, rho: -0.5,
            v0: 0.04, mu: 0.05, s0: 50.0,
        };
        let mut rng = seeded_rng(11);
        let sol = heston.simulate_as_sde((0.0, 0.5), 0.01, &mut rng).expect("heston.simulate_as_sde should succeed");
        assert!(!sol.is_empty());
        for xi in &sol.x {
            assert!(xi[0] > 0.0, "S must be positive");
            assert!(xi[1] >= 0.0, "v must be non-negative");
        }
    }
}
