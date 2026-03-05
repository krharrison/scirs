//! Option pricing models for financial computing
//!
//! Implements Black-Scholes analytical formulas, binomial tree methods,
//! and Monte Carlo simulation for European and American options.

use crate::error::{CoreError, CoreResult};
use std::f64::consts::{PI, SQRT_2};

/// Pure-Rust erfc approximation (Abramowitz & Stegun 7.1.26, maximum error < 1.5e-7).
#[inline]
fn erfc_approx(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_approx(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    poly * (-x * x).exp()
}

// ============================================================
// Types
// ============================================================

/// Type of option: call or put
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionType {
    /// Right to buy the underlying at strike price
    Call,
    /// Right to sell the underlying at strike price
    Put,
}

/// Exercise style: European (at maturity only) or American (any time)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExerciseType {
    /// Can only be exercised at expiration
    European,
    /// Can be exercised at any time before expiration
    American,
}

/// Collection of Black-Scholes option Greeks
///
/// Greeks measure sensitivities of option price to various factors.
#[derive(Debug, Clone)]
pub struct Greeks {
    /// Delta: sensitivity to underlying price (dV/dS)
    pub delta: f64,
    /// Gamma: second-order sensitivity to underlying (d²V/dS²)
    pub gamma: f64,
    /// Theta: sensitivity to time decay (dV/dt), per day
    pub theta: f64,
    /// Vega: sensitivity to volatility (dV/dσ), per 1% vol change
    pub vega: f64,
    /// Rho: sensitivity to interest rate (dV/dr), per 1% rate change
    pub rho: f64,
}

// ============================================================
// Normal distribution utilities
// ============================================================

/// Cumulative standard normal distribution Φ(x)
///
/// Uses the complementary error function for precision.
#[inline]
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc_approx(-x / SQRT_2)
}

/// Standard normal probability density function φ(x)
#[inline]
pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

// ============================================================
// Black-Scholes formulas
// ============================================================

/// Price a European call option using the Black-Scholes formula.
///
/// # Arguments
/// * `s` - Current underlying price (spot)
/// * `k` - Strike price
/// * `t` - Time to expiration in years
/// * `r` - Continuously compounded risk-free rate
/// * `sigma` - Volatility (annualised standard deviation of log returns)
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] when parameters are out of valid range.
pub fn black_scholes_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> CoreResult<f64> {
    validate_bs_params(s, k, t, sigma)?;
    Ok(bs_call_inner(s, k, t, r, sigma))
}

/// Price a European put option using the Black-Scholes formula.
///
/// # Arguments
/// * `s` - Current underlying price (spot)
/// * `k` - Strike price
/// * `t` - Time to expiration in years
/// * `r` - Continuously compounded risk-free rate
/// * `sigma` - Volatility (annualised standard deviation of log returns)
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] when parameters are out of valid range.
pub fn black_scholes_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> CoreResult<f64> {
    validate_bs_params(s, k, t, sigma)?;
    Ok(bs_put_inner(s, k, t, r, sigma))
}

/// Compute all five Black-Scholes Greeks for an option.
///
/// # Arguments
/// * `s` - Current underlying price (spot)
/// * `k` - Strike price
/// * `t` - Time to expiration in years
/// * `r` - Continuously compounded risk-free rate
/// * `sigma` - Volatility
/// * `option_type` - Call or Put
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] when parameters are out of valid range.
pub fn black_scholes_greeks(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    option_type: OptionType,
) -> CoreResult<Greeks> {
    validate_bs_params(s, k, t, sigma)?;

    let sqrt_t = t.sqrt();
    let d1 = bs_d1(s, k, t, r, sigma);
    let d2 = d1 - sigma * sqrt_t;
    let pdf_d1 = normal_pdf(d1);
    let disc = (-r * t).exp();

    let (delta, theta, rho) = match option_type {
        OptionType::Call => {
            let nd1 = normal_cdf(d1);
            let nd2 = normal_cdf(d2);
            let delta = nd1;
            // theta per calendar day (divide annual by 365)
            let theta = (-s * pdf_d1 * sigma / (2.0 * sqrt_t) - r * k * disc * nd2) / 365.0;
            let rho = k * t * disc * nd2 / 100.0;
            (delta, theta, rho)
        }
        OptionType::Put => {
            // Put delta = N(d1) - 1  (ranges from -1 to 0)
            // Put theta and rho use N(-d2) = 1 - N(d2)
            let nd1 = normal_cdf(d1);
            let nd2_neg = normal_cdf(-d2);
            let delta = nd1 - 1.0;
            let theta = (-s * pdf_d1 * sigma / (2.0 * sqrt_t) + r * k * disc * nd2_neg) / 365.0;
            let rho = -k * t * disc * nd2_neg / 100.0;
            (delta, theta, rho)
        }
    };

    let gamma = pdf_d1 / (s * sigma * sqrt_t);
    // vega per 1% change in vol
    let vega = s * sqrt_t * pdf_d1 / 100.0;

    Ok(Greeks {
        delta,
        gamma,
        theta,
        vega,
        rho,
    })
}

// ============================================================
// Binomial tree (Cox-Ross-Rubinstein)
// ============================================================

/// Price an option using the Cox-Ross-Rubinstein binomial tree.
///
/// Supports both European and American exercise styles.
///
/// # Arguments
/// * `s` - Current underlying price (spot)
/// * `k` - Strike price
/// * `t` - Time to expiration in years
/// * `r` - Continuously compounded risk-free rate
/// * `sigma` - Volatility
/// * `n` - Number of time steps in the tree
/// * `option_type` - Call or Put
/// * `exercise_type` - European or American
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] when parameters are invalid or the risk-neutral
/// probability lies outside [0, 1].
pub fn binomial_option(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    n: usize,
    option_type: OptionType,
    exercise_type: ExerciseType,
) -> CoreResult<f64> {
    validate_bs_params(s, k, t, sigma)?;
    if n == 0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Number of steps n must be at least 1",
        )));
    }

    let dt = t / n as f64;
    let discount = (-r * dt).exp();

    // CRR up/down factors
    let u = (sigma * dt.sqrt()).exp();
    let d = 1.0 / u;
    let grow = (r * dt).exp();
    let p = (grow - d) / (u - d);

    if !(0.0..=1.0).contains(&p) {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            format!(
                "Risk-neutral probability {p:.6} is outside [0,1]; reduce dt or check parameters"
            ),
        )));
    }

    // Allocate terminal payoff vector
    let mut values: Vec<f64> = (0..=n)
        .map(|j| {
            let s_t = s * u.powi(j as i32) * d.powi((n - j) as i32);
            payoff(s_t, k, option_type)
        })
        .collect();

    // Backward induction
    for step in (0..n).rev() {
        for j in 0..=step {
            let continuation = discount * (p * values[j + 1] + (1.0 - p) * values[j]);
            values[j] = match exercise_type {
                ExerciseType::European => continuation,
                ExerciseType::American => {
                    let s_node = s * u.powi(j as i32) * d.powi((step - j) as i32);
                    continuation.max(payoff(s_node, k, option_type))
                }
            };
        }
    }

    Ok(values[0])
}

// ============================================================
// Monte Carlo option pricing
// ============================================================

/// Price a European option via Monte Carlo simulation (Geometric Brownian Motion).
///
/// Uses the Park-Miller LCG for reproducible pseudo-random numbers via a seed.
/// Antithetic variates are applied automatically to halve variance.
///
/// # Arguments
/// * `s` - Current underlying price (spot)
/// * `k` - Strike price
/// * `t` - Time to expiration in years
/// * `r` - Risk-free rate
/// * `sigma` - Volatility
/// * `n_paths` - Number of Monte Carlo paths (must be even for antithetics)
/// * `n_steps` - Number of time steps per path
/// * `option_type` - Call or Put
/// * `seed` - RNG seed for reproducibility
///
/// # Returns
/// `(price, std_error)` - estimated option price and its standard error.
///
/// # Errors
/// Returns [`CoreError::InvalidArgument`] for invalid parameters.
pub fn monte_carlo_option(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    n_paths: usize,
    n_steps: usize,
    option_type: OptionType,
    seed: u64,
) -> CoreResult<(f64, f64)> {
    validate_bs_params(s, k, t, sigma)?;
    if n_paths < 2 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "n_paths must be at least 2",
        )));
    }
    if n_steps == 0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "n_steps must be at least 1",
        )));
    }

    let dt = t / n_steps as f64;
    let drift = (r - 0.5 * sigma * sigma) * dt;
    let vol_sqrt_dt = sigma * dt.sqrt();
    let discount = (-r * t).exp();

    // Use antithetic variates: run n_paths/2 pairs
    let half = n_paths / 2;
    let mut rng = ParkMillerRng::new(seed);
    let mut sum_payoffs = 0.0;
    let mut sum_sq = 0.0;

    for _ in 0..half {
        // Generate one path and its antithetic mirror simultaneously
        let mut s_fwd = s;
        let mut s_anti = s;

        for _ in 0..n_steps {
            let z = rng.next_normal();
            s_fwd *= (drift + vol_sqrt_dt * z).exp();
            s_anti *= (drift - vol_sqrt_dt * z).exp();
        }

        let pf = payoff(s_fwd, k, option_type);
        let pa = payoff(s_anti, k, option_type);
        let avg = 0.5 * (pf + pa);
        sum_payoffs += avg;
        sum_sq += avg * avg;
    }

    let n = half as f64;
    let mean_payoff = sum_payoffs / n;
    let variance = (sum_sq / n - mean_payoff * mean_payoff).max(0.0);
    let std_error = (variance / n).sqrt() * discount;
    let price = mean_payoff * discount;

    Ok((price, std_error))
}

// ============================================================
// Internal helpers
// ============================================================

#[inline]
fn bs_d1(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt())
}

#[inline]
fn bs_call_inner(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    let d1 = bs_d1(s, k, t, r, sigma);
    let d2 = d1 - sigma * t.sqrt();
    s * normal_cdf(d1) - k * (-r * t).exp() * normal_cdf(d2)
}

#[inline]
fn bs_put_inner(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    let d1 = bs_d1(s, k, t, r, sigma);
    let d2 = d1 - sigma * t.sqrt();
    k * (-r * t).exp() * normal_cdf(-d2) - s * normal_cdf(-d1)
}

#[inline]
fn payoff(s: f64, k: f64, option_type: OptionType) -> f64 {
    match option_type {
        OptionType::Call => (s - k).max(0.0),
        OptionType::Put => (k - s).max(0.0),
    }
}

fn validate_bs_params(s: f64, k: f64, t: f64, sigma: f64) -> CoreResult<()> {
    if s <= 0.0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Spot price s must be positive",
        )));
    }
    if k <= 0.0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Strike price k must be positive",
        )));
    }
    if t <= 0.0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Time to expiry t must be positive",
        )));
    }
    if sigma <= 0.0 {
        return Err(CoreError::InvalidArgument(crate::error::ErrorContext::new(
            "Volatility sigma must be positive",
        )));
    }
    Ok(())
}

// ============================================================
// Minimal deterministic RNG (Park-Miller LCG + Box-Muller)
// ============================================================

/// Simple Park-Miller multiplicative congruential generator
struct ParkMillerRng {
    state: u64,
    /// Cached second normal variate from Box-Muller
    spare: Option<f64>,
}

impl ParkMillerRng {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 12345678 } else { seed };
        Self { state, spare: None }
    }

    /// Next uniform [0, 1) value
    fn next_uniform(&mut self) -> f64 {
        // Lehmer/Park-Miller with multiplier 48271, modulus 2^31-1
        self.state = (self.state.wrapping_mul(48271)) % 2_147_483_647;
        self.state as f64 / 2_147_483_647.0
    }

    /// Next standard normal variate via Box-Muller transform
    fn next_normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        // Box-Muller: generate two normals from two uniforms
        let u1 = self.next_uniform().max(1e-15);
        let u2 = self.next_uniform();
        let mag = (-2.0 * u1.ln()).sqrt();
        let angle = 2.0 * PI * u2;
        self.spare = Some(mag * angle.sin());
        mag * angle.cos()
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 1e-4;

    // --- normal CDF ---
    #[test]
    fn test_normal_cdf_standard_values() {
        // A&S 7.1.26 approximation has max error < 1.5e-7; use 1e-6 tolerance
        assert!(
            (normal_cdf(0.0) - 0.5).abs() < 1e-6,
            "cdf(0) = 0.5: got {}",
            normal_cdf(0.0)
        );
        assert!((normal_cdf(1.6449) - 0.95).abs() < 1e-3);
        assert!((normal_cdf(-1.6449) - 0.05).abs() < 1e-3);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
    }

    #[test]
    fn test_normal_pdf_symmetry() {
        assert!((normal_pdf(1.0) - normal_pdf(-1.0)).abs() < 1e-15);
    }

    // --- Black-Scholes call ---
    #[test]
    fn test_bs_call_atm() {
        // Classic ATM: S=100, K=100, T=1, r=0.05, sigma=0.2
        // Known value ≈ 10.4506
        let price = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2).expect("should succeed");
        assert!(
            (price - 10.4506).abs() < 0.01,
            "BS call ATM: got {price:.4}"
        );
    }

    #[test]
    fn test_bs_call_itm() {
        let price = black_scholes_call(110.0, 100.0, 1.0, 0.05, 0.2).expect("should succeed");
        assert!(price > 10.0, "ITM call should have intrinsic value");
    }

    #[test]
    fn test_bs_call_otm() {
        // S=90, K=100, T=1, r=0.05, σ=0.2 → OTM call ≈ 5.57 (has significant time value)
        let price = black_scholes_call(90.0, 100.0, 1.0, 0.05, 0.2).expect("should succeed");
        assert!(price > 0.0, "OTM call has positive time value");
        // Far OTM: S=70, K=100 — should be cheap
        let far_otm = black_scholes_call(70.0, 100.0, 0.25, 0.05, 0.2).expect("should succeed");
        assert!(
            far_otm < 1.0,
            "Far OTM call should be very cheap: {far_otm:.4}"
        );
    }

    // --- Black-Scholes put ---
    #[test]
    fn test_bs_put_atm() {
        // Known ATM put ≈ 5.5735 (put-call parity: put = call - S + Ke^{-rT})
        let price = black_scholes_put(100.0, 100.0, 1.0, 0.05, 0.2).expect("should succeed");
        assert!((price - 5.5735).abs() < 0.01, "BS put ATM: got {price:.4}");
    }

    #[test]
    fn test_put_call_parity() {
        let (s, k, t, r, sigma) = (100.0, 105.0, 0.5, 0.03, 0.25);
        let call = black_scholes_call(s, k, t, r, sigma).expect("should succeed");
        let put = black_scholes_put(s, k, t, r, sigma).expect("should succeed");
        // Put-call parity: C - P = S - K*e^{-rT}
        let lhs = call - put;
        let rhs = s - k * (-r * t).exp();
        assert!(
            (lhs - rhs).abs() < 1e-8,
            "Put-call parity violated: {lhs:.8} vs {rhs:.8}"
        );
    }

    #[test]
    fn test_bs_invalid_params() {
        assert!(black_scholes_call(-1.0, 100.0, 1.0, 0.05, 0.2).is_err());
        assert!(black_scholes_call(100.0, 0.0, 1.0, 0.05, 0.2).is_err());
        assert!(black_scholes_call(100.0, 100.0, 0.0, 0.05, 0.2).is_err());
        assert!(black_scholes_call(100.0, 100.0, 1.0, 0.05, -0.1).is_err());
    }

    // --- Greeks ---
    #[test]
    fn test_delta_call_between_0_and_1() {
        let greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call)
            .expect("should succeed");
        assert!(
            greeks.delta > 0.0 && greeks.delta < 1.0,
            "Call delta must be in (0,1), got {}",
            greeks.delta
        );
    }

    #[test]
    fn test_delta_put_between_neg1_and_0() {
        let greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Put)
            .expect("should succeed");
        assert!(
            greeks.delta > -1.0 && greeks.delta < 0.0,
            "Put delta must be in (-1,0), got {}",
            greeks.delta
        );
    }

    #[test]
    fn test_gamma_positive() {
        let greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call)
            .expect("should succeed");
        assert!(greeks.gamma > 0.0, "Gamma must be positive");
    }

    #[test]
    fn test_vega_positive() {
        let greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call)
            .expect("should succeed");
        assert!(greeks.vega > 0.0, "Vega must be positive");
    }

    #[test]
    fn test_theta_negative_call() {
        let greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call)
            .expect("should succeed");
        assert!(
            greeks.theta < 0.0,
            "Call theta (time decay) must be negative"
        );
    }

    #[test]
    fn test_call_delta_atm_near_half() {
        // ATM call delta ≈ 0.5 + small positive adjustment
        let greeks = black_scholes_greeks(100.0, 100.0, 1.0, 0.0, 0.01, OptionType::Call)
            .expect("should succeed");
        assert!(
            (greeks.delta - 0.5).abs() < 0.1,
            "ATM delta ≈ 0.5, got {}",
            greeks.delta
        );
    }

    // --- Binomial tree ---
    #[test]
    fn test_binomial_european_converges_to_bs() {
        let (s, k, t, r, sigma) = (100.0, 100.0, 1.0, 0.05, 0.2);
        let bs = black_scholes_call(s, k, t, r, sigma).expect("should succeed");
        let bin = binomial_option(
            s,
            k,
            t,
            r,
            sigma,
            500,
            OptionType::Call,
            ExerciseType::European,
        )
        .expect("should succeed");
        assert!(
            (bin - bs).abs() < 0.05,
            "Binomial European call should converge to BS: bin={bin:.4} bs={bs:.4}"
        );
    }

    #[test]
    fn test_binomial_american_put_ge_european() {
        // American put >= European put (early exercise premium)
        let (s, k, t, r, sigma) = (100.0, 100.0, 1.0, 0.08, 0.2);
        let eu = binomial_option(
            s,
            k,
            t,
            r,
            sigma,
            200,
            OptionType::Put,
            ExerciseType::European,
        )
        .expect("should succeed");
        let am = binomial_option(
            s,
            k,
            t,
            r,
            sigma,
            200,
            OptionType::Put,
            ExerciseType::American,
        )
        .expect("should succeed");
        assert!(
            am >= eu - 1e-8,
            "American put must be >= European put: am={am:.4} eu={eu:.4}"
        );
    }

    #[test]
    fn test_binomial_invalid_params() {
        assert!(binomial_option(
            100.0,
            100.0,
            1.0,
            0.05,
            0.2,
            0,
            OptionType::Call,
            ExerciseType::European
        )
        .is_err());
        assert!(binomial_option(
            -1.0,
            100.0,
            1.0,
            0.05,
            0.2,
            10,
            OptionType::Call,
            ExerciseType::European
        )
        .is_err());
    }

    // --- Monte Carlo ---
    #[test]
    fn test_mc_call_near_bs() {
        let (s, k, t, r, sigma) = (100.0, 100.0, 1.0, 0.05, 0.2);
        let bs = black_scholes_call(s, k, t, r, sigma).expect("should succeed");
        let (mc_price, mc_se) =
            monte_carlo_option(s, k, t, r, sigma, 50_000, 50, OptionType::Call, 42)
                .expect("should succeed");
        // MC price should be within ~3 standard errors of BS
        assert!(
            (mc_price - bs).abs() < 3.0 * mc_se + 0.5,
            "MC call={mc_price:.4} BS={bs:.4} se={mc_se:.4}"
        );
    }

    #[test]
    fn test_mc_put_near_bs() {
        let (s, k, t, r, sigma) = (100.0, 110.0, 0.5, 0.04, 0.25);
        let bs = black_scholes_put(s, k, t, r, sigma).expect("should succeed");
        let (mc_price, _) = monte_carlo_option(s, k, t, r, sigma, 50_000, 50, OptionType::Put, 99)
            .expect("should succeed");
        assert!(
            (mc_price - bs).abs() < 0.5,
            "MC put={mc_price:.4} BS={bs:.4}"
        );
    }

    #[test]
    fn test_mc_invalid_params() {
        assert!(
            monte_carlo_option(100.0, 100.0, 1.0, 0.05, 0.2, 1, 10, OptionType::Call, 1).is_err()
        );
        assert!(
            monte_carlo_option(100.0, 100.0, 1.0, 0.05, 0.2, 100, 0, OptionType::Call, 1).is_err()
        );
    }

    #[test]
    fn test_mc_std_error_decreases_with_paths() {
        let (s, k, t, r, sigma) = (100.0, 100.0, 1.0, 0.05, 0.2);
        let (_, se_small) = monte_carlo_option(s, k, t, r, sigma, 1_000, 20, OptionType::Call, 1)
            .expect("should succeed");
        let (_, se_large) = monte_carlo_option(s, k, t, r, sigma, 100_000, 20, OptionType::Call, 1)
            .expect("should succeed");
        assert!(
            se_large < se_small,
            "Larger sample should give smaller std error"
        );
    }

    #[test]
    fn test_park_miller_rng_deterministic() {
        let mut r1 = ParkMillerRng::new(12345);
        let mut r2 = ParkMillerRng::new(12345);
        for _ in 0..20 {
            assert!((r1.next_normal() - r2.next_normal()).abs() < 1e-15);
        }
    }

    #[test]
    fn test_bs_call_deep_itm_approaches_intrinsic() {
        // Deep ITM call: price ≈ S - K*e^{-rT}
        let (s, k, t, r, sigma) = (200.0, 100.0, 0.01, 0.05, 0.01);
        let price = black_scholes_call(s, k, t, r, sigma).expect("should succeed");
        let intrinsic = s - k * (-r * t).exp();
        assert!(
            (price - intrinsic).abs() < 0.1,
            "Deep ITM call ≈ intrinsic: price={price:.4} intrinsic={intrinsic:.4}"
        );
    }

    #[test]
    fn test_bs_put_otm_nearly_zero() {
        // Far OTM put: very low probability of expiring ITM
        let price = black_scholes_put(200.0, 50.0, 0.1, 0.05, 0.2).expect("should succeed");
        assert!(price < 0.001, "Far OTM put should be near zero: {price:.6}");
    }

    #[test]
    fn test_greeks_put_call_delta_sum() {
        // For a European call and put with same params: delta_call - delta_put = 1 (no dividends)
        // delta_call = N(d1), delta_put = N(d1) - 1, so difference = 1 exactly
        // Tolerance 1e-5 accounts for erfc approximation errors
        let (s, k, t, r, sigma) = (100.0, 100.0, 1.0, 0.05, 0.2);
        let gc = black_scholes_greeks(s, k, t, r, sigma, OptionType::Call).expect("should succeed");
        let gp = black_scholes_greeks(s, k, t, r, sigma, OptionType::Put).expect("should succeed");
        assert!(
            (gc.delta - gp.delta - 1.0).abs() < 1e-5,
            "delta_call - delta_put = 1, got diff={}",
            gc.delta - gp.delta
        );
    }

    #[test]
    fn test_binomial_call_put_parity() {
        let (s, k, t, r, sigma) = (100.0, 100.0, 1.0, 0.05, 0.2);
        let call = binomial_option(
            s,
            k,
            t,
            r,
            sigma,
            200,
            OptionType::Call,
            ExerciseType::European,
        )
        .expect("should succeed");
        let put = binomial_option(
            s,
            k,
            t,
            r,
            sigma,
            200,
            OptionType::Put,
            ExerciseType::European,
        )
        .expect("should succeed");
        let parity_lhs = call - put;
        let parity_rhs = s - k * (-r * t).exp();
        assert!(
            (parity_lhs - parity_rhs).abs() < 0.05,
            "Binomial put-call parity: lhs={parity_lhs:.4} rhs={parity_rhs:.4}"
        );
    }
}
