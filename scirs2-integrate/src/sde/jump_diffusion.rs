//! Jump-diffusion processes
//!
//! This module provides simulation of processes that combine a continuous diffusion
//! component with a discrete jump component driven by a Poisson process.
//!
//! ## Processes
//!
//! | Function/Struct | Model |
//! |-----------------|-------|
//! | [`merton_jump_diffusion`] | Merton (1976) log-normal jumps |
//! | [`compound_poisson_process`] | General compound Poisson process |
//! | [`kou_double_exponential`] | Kou (2002) double-exponential jumps |
//! | [`JumpDiffusionProblem`] | Generic jump-diffusion solver |

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::random::prelude::*;
use scirs2_core::random::{Normal, Uniform};
use scirs2_core::Distribution;

// ─────────────────────────────────────────────────────────────────────────────
// Merton Jump-Diffusion
// ─────────────────────────────────────────────────────────────────────────────

/// Simulate the Merton (1976) jump-diffusion model.
///
/// ```text
/// dS = (μ - λ·k̄) S dt + σ S dW + J·S dN
/// ```
///
/// where:
/// - `N` is a Poisson process with intensity `λ`
/// - Jump size `J = exp(μ_j + σ_j·Z) - 1`, `Z ~ N(0,1)`
/// - `k̄ = exp(μ_j + σ_j²/2) - 1` (mean jump)
///
/// # Arguments
///
/// * `mu` — drift rate
/// * `sigma` — diffusion volatility (σ > 0)
/// * `lambda` — Poisson jump intensity (λ ≥ 0)
/// * `mu_j` — mean of log-jump size
/// * `sigma_j` — std of log-jump size (σ_j ≥ 0)
/// * `s0` — initial price (S₀ > 0)
/// * `t_span` — simulation interval `(t0, t1)`
/// * `dt` — time step (> 0)
/// * `rng` — random number generator
///
/// # Returns
///
/// A vector of `(t, S(t))` pairs.
///
/// # Errors
///
/// Returns an error if any parameter is invalid.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::sde::jump_diffusion::merton_jump_diffusion;
/// use scirs2_core::random::prelude::*;
///
/// let mut rng = seeded_rng(42);
/// let path = merton_jump_diffusion(0.05, 0.2, 1.0, -0.1, 0.2, 100.0, (0.0, 1.0), 0.01, &mut rng).unwrap();
/// assert!(path.iter().all(|&(_, s)| s > 0.0), "Merton S must stay positive");
/// ```
pub fn merton_jump_diffusion(
    mu: f64,
    sigma: f64,
    lambda: f64,
    mu_j: f64,
    sigma_j: f64,
    s0: f64,
    t_span: (f64, f64),
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<Vec<(f64, f64)>> {
    if sigma <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "sigma must be > 0, got {sigma}"
        )));
    }
    if lambda < 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "lambda must be >= 0, got {lambda}"
        )));
    }
    if sigma_j < 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "sigma_j must be >= 0, got {sigma_j}"
        )));
    }
    if s0 <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "s0 must be > 0, got {s0}"
        )));
    }
    validate_t_span(t_span)?;
    validate_dt(dt)?;

    // Mean jump size: k_bar = E[J] = exp(mu_j + sigma_j^2/2) - 1
    let k_bar = (mu_j + 0.5 * sigma_j * sigma_j).exp() - 1.0;
    // Compensated drift
    let mu_comp = mu - lambda * k_bar;

    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;
    let uniform = Uniform::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;

    let n_steps = ((t_span.1 - t_span.0) / dt).ceil() as usize;
    let mut path = Vec::with_capacity(n_steps + 1);
    path.push((t_span.0, s0));
    let mut s = s0;
    let mut t = t_span.0;

    for _ in 0..n_steps {
        let step = (t_span.1 - t).min(dt);
        let sqrt_step = step.sqrt();
        let z: f64 = normal.sample(rng);
        let dw = z * sqrt_step;

        // Continuous component (Euler-Maruyama)
        let ds_cont = mu_comp * s * step + sigma * s * dw;

        // Jump component: number of jumps in [t, t+step] ~ Poisson(lambda*step)
        let n_jumps = poisson_sample(lambda * step, rng, &uniform)?;
        let mut jump_factor = 1.0;
        for _ in 0..n_jumps {
            let z_j: f64 = normal.sample(rng);
            let log_jump = mu_j + sigma_j * z_j;
            jump_factor *= log_jump.exp(); // multiplicative: J + 1 = exp(log-jump)
        }
        // S_{t+dt} = S_t * jump_factor + continuous component
        s = ((s + ds_cont) * jump_factor).max(1e-10);
        t += step;
        path.push((t, s));
    }
    Ok(path)
}

// ─────────────────────────────────────────────────────────────────────────────
// Compound Poisson Process
// ─────────────────────────────────────────────────────────────────────────────

/// Simulate a compound Poisson process.
///
/// At each Poisson event (with intensity `lambda`) the process jumps by a random
/// amount drawn from `jump_dist`.
///
/// Uses the thinning (Lewis-Shedler) algorithm to place Poisson events exactly:
/// inter-arrival times are exponentially distributed with rate `lambda`.
///
/// # Arguments
///
/// * `lambda` — Poisson intensity (λ > 0)
/// * `jump_dist` — callable returning a jump size given a mutable `StdRng`
/// * `t_span` — simulation interval `(t0, t1)`
/// * `rng` — random number generator
///
/// # Returns
///
/// A vector of `(t, X(t))` pairs at all jump times plus `t0` and `t1`.
///
/// # Errors
///
/// Returns an error if `lambda <= 0` or `t_span` is invalid.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::sde::jump_diffusion::compound_poisson_process;
/// use scirs2_core::random::prelude::*;
/// use scirs2_core::random::{Normal, Distribution};
///
/// let mut rng = seeded_rng(7);
/// // Jumps ~ N(0, 1)
/// let normal = Normal::new(0.0, 1.0).unwrap();
/// let path = compound_poisson_process(
///     2.0,
///     |rng| normal.sample(rng),
///     (0.0, 5.0),
///     &mut rng,
/// ).unwrap();
/// // At least start and end points
/// assert!(path.len() >= 2);
/// ```
pub fn compound_poisson_process<F>(
    lambda: f64,
    jump_dist: F,
    t_span: (f64, f64),
    rng: &mut StdRng,
) -> IntegrateResult<Vec<(f64, f64)>>
where
    F: Fn(&mut StdRng) -> f64,
{
    if lambda <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "lambda must be > 0, got {lambda}"
        )));
    }
    validate_t_span(t_span)?;

    // Inter-arrival times: Exp(lambda)
    let uniform = Uniform::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;

    let mut path = Vec::new();
    path.push((t_span.0, 0.0));
    let mut x = 0.0_f64;
    let mut t = t_span.0;

    loop {
        // Sample next inter-arrival time: -ln(U)/lambda
        let u: f64 = uniform.sample(rng);
        let inter = -u.ln() / lambda;
        t += inter;
        if t >= t_span.1 {
            break;
        }
        let jump = jump_dist(rng);
        x += jump;
        path.push((t, x));
    }
    path.push((t_span.1, x));
    Ok(path)
}

// ─────────────────────────────────────────────────────────────────────────────
// Kou Double-Exponential Jump-Diffusion
// ─────────────────────────────────────────────────────────────────────────────

/// Simulate the Kou (2002) double-exponential jump-diffusion model.
///
/// Jump sizes follow an asymmetric double-exponential distribution:
/// - Upward jumps: Exp(η₁) with probability p
/// - Downward jumps: -Exp(η₂) with probability (1-p)
///
/// # Arguments
///
/// * `mu` — drift (μ)
/// * `sigma` — volatility (σ > 0)
/// * `lambda` — jump intensity (λ ≥ 0)
/// * `p` — probability of an upward jump (0 < p < 1)
/// * `eta1` — rate of upward exponential jumps (η₁ > 1, for finite expectation)
/// * `eta2` — rate of downward exponential jumps (η₂ > 0)
/// * `s0` — initial price (S₀ > 0)
/// * `t_span` — simulation interval
/// * `dt` — time step
/// * `rng` — random number generator
///
/// # Returns
///
/// A vector of `(t, S(t))` pairs.
///
/// # Errors
///
/// Returns an error if parameter constraints are violated.
pub fn kou_double_exponential(
    mu: f64,
    sigma: f64,
    lambda: f64,
    p: f64,
    eta1: f64,
    eta2: f64,
    s0: f64,
    t_span: (f64, f64),
    dt: f64,
    rng: &mut StdRng,
) -> IntegrateResult<Vec<(f64, f64)>> {
    if sigma <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!("sigma must be > 0, got {sigma}")));
    }
    if lambda < 0.0 {
        return Err(IntegrateError::InvalidInput(format!("lambda must be >= 0, got {lambda}")));
    }
    if !(0.0 < p && p < 1.0) {
        return Err(IntegrateError::InvalidInput(format!("p must be in (0,1), got {p}")));
    }
    if eta1 <= 1.0 {
        return Err(IntegrateError::InvalidInput(format!("eta1 must be > 1, got {eta1}")));
    }
    if eta2 <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!("eta2 must be > 0, got {eta2}")));
    }
    if s0 <= 0.0 {
        return Err(IntegrateError::InvalidInput(format!("s0 must be > 0, got {s0}")));
    }
    validate_t_span(t_span)?;
    validate_dt(dt)?;

    // Mean jump E[J] = p/(eta1-1) - (1-p)/eta2  (log-price jump)
    // k_bar = E[exp(Y) - 1] where Y is the jump size in log-price
    let k_bar = p * eta1 / (eta1 - 1.0) + (1.0 - p) * eta2 / (eta2 + 1.0) - 1.0;
    let mu_comp = mu - lambda * k_bar;

    let normal = Normal::new(0.0, 1.0)
        .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;
    let uniform = Uniform::new(0.0_f64, 1.0_f64)
        .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;

    let n_steps = ((t_span.1 - t_span.0) / dt).ceil() as usize;
    let mut path = Vec::with_capacity(n_steps + 1);
    path.push((t_span.0, s0));
    let mut s = s0;
    let mut t = t_span.0;

    for _ in 0..n_steps {
        let step = (t_span.1 - t).min(dt);
        let sqrt_step = step.sqrt();
        let z: f64 = normal.sample(rng);
        let dw = z * sqrt_step;

        let ds_cont = mu_comp * s * step + sigma * s * dw;
        let n_jumps = poisson_sample(lambda * step, rng, &uniform)?;
        let mut log_jump_sum: f64 = 0.0;
        for _ in 0..n_jumps {
            let u_type: f64 = uniform.sample(rng);
            let u_mag: f64 = uniform.sample(rng);
            if u_type < p {
                // Upward: exponential with rate eta1
                log_jump_sum += -u_mag.ln() / eta1;
            } else {
                // Downward: exponential with rate eta2
                log_jump_sum -= -u_mag.ln() / eta2;
            }
        }
        let jump_mult = log_jump_sum.exp();
        s = ((s + ds_cont) * jump_mult).max(1e-10);
        t += step;
        path.push((t, s));
    }
    Ok(path)
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic Jump-Diffusion Problem
// ─────────────────────────────────────────────────────────────────────────────

/// Generic jump-diffusion problem specification.
///
/// ```text
/// dX = f(X,t) dt + g(X,t) dW + dJ
/// ```
///
/// where `dJ` is a compound Poisson process with intensity `jump_intensity`
/// and jumps sampled from `jump_sampler`.
pub struct JumpDiffusionProblem<Drift, Diffusion, JumpSampler>
where
    Drift: Fn(f64, f64) -> f64,
    Diffusion: Fn(f64, f64) -> f64,
    JumpSampler: Fn(&mut StdRng) -> f64,
{
    /// Drift coefficient f(x, t)
    pub drift: Drift,
    /// Diffusion coefficient g(x, t)
    pub diffusion: Diffusion,
    /// Poisson jump intensity (λ ≥ 0)
    pub jump_intensity: f64,
    /// Callable returning a jump size: `fn(&mut StdRng) -> f64`
    pub jump_sampler: JumpSampler,
    /// Initial value X(t0)
    pub x0: f64,
    /// Time span (t0, t1)
    pub t_span: (f64, f64),
}

impl<Drift, Diffusion, JumpSampler> JumpDiffusionProblem<Drift, Diffusion, JumpSampler>
where
    Drift: Fn(f64, f64) -> f64,
    Diffusion: Fn(f64, f64) -> f64,
    JumpSampler: Fn(&mut StdRng) -> f64,
{
    /// Create a new jump-diffusion problem.
    pub fn new(
        drift: Drift,
        diffusion: Diffusion,
        jump_intensity: f64,
        jump_sampler: JumpSampler,
        x0: f64,
        t_span: (f64, f64),
    ) -> IntegrateResult<Self> {
        if jump_intensity < 0.0 {
            return Err(IntegrateError::InvalidInput(format!(
                "jump_intensity must be >= 0, got {jump_intensity}"
            )));
        }
        validate_t_span(t_span)?;
        Ok(Self {
            drift,
            diffusion,
            jump_intensity,
            jump_sampler,
            x0,
            t_span,
        })
    }

    /// Solve using Euler-Maruyama with exact Poisson jump timing.
    ///
    /// Returns a vector of `(t, X(t))` pairs.
    pub fn solve(&self, dt: f64, rng: &mut StdRng) -> IntegrateResult<Vec<(f64, f64)>> {
        validate_dt(dt)?;
        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;
        let uniform = Uniform::new(0.0_f64, 1.0_f64)
            .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;

        let n_steps = ((self.t_span.1 - self.t_span.0) / dt).ceil() as usize;
        let mut path = Vec::with_capacity(n_steps + 1);
        path.push((self.t_span.0, self.x0));
        let mut x = self.x0;
        let mut t = self.t_span.0;

        for _ in 0..n_steps {
            let step = (self.t_span.1 - t).min(dt);
            let sqrt_step = step.sqrt();
            let z: f64 = normal.sample(rng);
            let fx = (self.drift)(x, t);
            let gx = (self.diffusion)(x, t);
            let n_jumps = poisson_sample(self.jump_intensity * step, rng, &uniform)?;
            let jump_sum: f64 = (0..n_jumps).map(|_| (self.jump_sampler)(rng)).sum();
            x = x + fx * step + gx * z * sqrt_step + jump_sum;
            t += step;
            path.push((t, x));
        }
        Ok(path)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sample from a Poisson distribution with parameter `lambda` using the
/// Knuth algorithm (efficient for small lambda; for large lambda uses normal approx).
fn poisson_sample(lambda: f64, rng: &mut StdRng, uniform: &Uniform<f64>) -> IntegrateResult<usize> {
    if lambda < 0.0 {
        return Err(IntegrateError::InvalidInput(format!(
            "Poisson lambda must be >= 0, got {lambda}"
        )));
    }
    if lambda == 0.0 {
        return Ok(0);
    }
    // For large lambda, use Normal approximation
    if lambda > 30.0 {
        let normal = Normal::new(lambda, lambda.sqrt())
            .map_err(|e| IntegrateError::InvalidInput(e.to_string()))?;
        let sample: f64 = normal.sample(rng);
        return Ok(sample.round().max(0.0) as usize);
    }
    // Knuth algorithm: P(N=k) = exp(-lambda)*lambda^k/k!
    let threshold = (-lambda).exp();
    let mut count = 0usize;
    let mut product = uniform.sample(rng);
    while product > threshold {
        product *= uniform.sample(rng);
        count += 1;
    }
    Ok(count)
}

fn validate_t_span(t_span: (f64, f64)) -> IntegrateResult<()> {
    if t_span.0 >= t_span.1 {
        Err(IntegrateError::InvalidInput(format!(
            "t_span must satisfy t0 < t1, got {:?}", t_span
        )))
    } else {
        Ok(())
    }
}

fn validate_dt(dt: f64) -> IntegrateResult<()> {
    if dt <= 0.0 {
        Err(IntegrateError::InvalidInput(format!(
            "dt must be > 0, got {dt}"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::prelude::seeded_rng;

    #[test]
    fn test_merton_positive_price() {
        let mut rng = seeded_rng(42);
        let path = merton_jump_diffusion(
            0.05, 0.2, 1.0, -0.1, 0.2, 100.0, (0.0, 1.0), 0.01, &mut rng,
        ).expect("merton_jump_diffusion should succeed");
        assert!(!path.is_empty());
        assert!(path.iter().all(|&(_, s)| s > 0.0), "Merton S must stay positive");
    }

    #[test]
    fn test_merton_no_jumps() {
        // When lambda = 0, Merton reduces to GBM
        let mut rng = seeded_rng(1);
        let path = merton_jump_diffusion(
            0.05, 0.2, 0.0, 0.0, 0.0, 100.0, (0.0, 1.0), 0.01, &mut rng,
        ).expect("merton_jump_diffusion should succeed");
        assert!(path.iter().all(|&(_, s)| s > 0.0));
    }

    #[test]
    fn test_merton_invalid_sigma() {
        let mut rng = seeded_rng(0);
        assert!(merton_jump_diffusion(0.05, -0.2, 1.0, 0.0, 0.1, 100.0, (0.0, 1.0), 0.01, &mut rng).is_err());
    }

    #[test]
    fn test_merton_invalid_s0() {
        let mut rng = seeded_rng(0);
        assert!(merton_jump_diffusion(0.05, 0.2, 1.0, 0.0, 0.1, -1.0, (0.0, 1.0), 0.01, &mut rng).is_err());
    }

    #[test]
    fn test_compound_poisson_basic() {
        let mut rng = seeded_rng(7);
        let normal = Normal::new(0.0_f64, 1.0).expect("Normal::new should succeed with valid params");
        let path = compound_poisson_process(
            2.0,
            |r| normal.sample(r),
            (0.0, 5.0),
            &mut rng,
        ).expect("compound_poisson_process should succeed");
        // At minimum: (0.0, 0.0) and (5.0, x_final)
        assert!(path.len() >= 2);
        assert!((path[0].0).abs() < 1e-12, "starts at t0");
        assert!((path[0].1).abs() < 1e-12, "starts at 0");
        assert!((path.last().expect("path is non-empty").0 - 5.0).abs() < 1e-12, "ends at t1");
    }

    #[test]
    fn test_compound_poisson_mean_jumps() {
        // E[N(T)] = lambda * T
        let lambda = 3.0;
        let t_end = 2.0;
        let n_paths = 500;
        let mut total_jumps = 0usize;
        for seed in 0..n_paths as u64 {
            let mut rng = seeded_rng(seed + 1000);
            let path = compound_poisson_process(
                lambda,
                |_r| 1.0, // unit jumps
                (0.0, t_end),
                &mut rng,
            ).expect("compound_poisson_process should succeed");
            // subtract 2 for start/end points
            total_jumps += path.len().saturating_sub(2);
        }
        let mean_jumps = total_jumps as f64 / n_paths as f64;
        let expected = lambda * t_end;
        assert!(
            (mean_jumps - expected).abs() / expected < 0.15,
            "mean jumps {mean_jumps:.3} vs expected {expected:.3}"
        );
    }

    #[test]
    fn test_compound_poisson_invalid_lambda() {
        let mut rng = seeded_rng(0);
        assert!(compound_poisson_process(0.0, |_| 1.0, (0.0, 1.0), &mut rng).is_err());
        assert!(compound_poisson_process(-1.0, |_| 1.0, (0.0, 1.0), &mut rng).is_err());
    }

    #[test]
    fn test_kou_positive_price() {
        let mut rng = seeded_rng(42);
        let path = kou_double_exponential(
            0.05, 0.2, 1.0, 0.4, 5.0, 3.0, 100.0, (0.0, 1.0), 0.01, &mut rng,
        ).expect("kou_double_exponential should succeed");
        assert!(path.iter().all(|&(_, s)| s > 0.0), "Kou S must stay positive");
    }

    #[test]
    fn test_kou_invalid_params() {
        let mut rng = seeded_rng(0);
        // eta1 must be > 1
        assert!(kou_double_exponential(0.05, 0.2, 1.0, 0.4, 0.5, 3.0, 100.0, (0.0, 1.0), 0.01, &mut rng).is_err());
        // p must be in (0,1)
        assert!(kou_double_exponential(0.05, 0.2, 1.0, 0.0, 5.0, 3.0, 100.0, (0.0, 1.0), 0.01, &mut rng).is_err());
    }

    #[test]
    fn test_jump_diffusion_problem() {
        let mut rng = seeded_rng(99);
        let normal = Normal::new(0.0_f64, 0.5).expect("Normal::new should succeed with valid params");
        let prob = JumpDiffusionProblem::new(
            |x, _t| 0.05 * x,
            |x, _t| 0.2 * x,
            1.0,
            move |r| normal.sample(r),
            100.0,
            (0.0, 1.0),
        ).expect("JumpDiffusionProblem::new should succeed");
        let path = prob.solve(0.01, &mut rng).expect("prob.solve should succeed");
        assert!(!path.is_empty());
        assert!((path[0].0 - 0.0).abs() < 1e-12);
        assert!((path[0].1 - 100.0).abs() < 1e-12);
    }

    #[test]
    fn test_poisson_sample_zero_rate() {
        let uniform = Uniform::new(0.0_f64, 1.0).expect("Uniform::new should succeed with valid range");
        let mut rng = seeded_rng(0);
        let n = poisson_sample(0.0, &mut rng, &uniform).expect("poisson_sample should succeed");
        assert_eq!(n, 0);
    }

    #[test]
    fn test_validate_t_span() {
        assert!(validate_t_span((0.0, 1.0)).is_ok());
        assert!(validate_t_span((1.0, 0.0)).is_err());
        assert!(validate_t_span((1.0, 1.0)).is_err());
    }
}
