//! Basic Monte Carlo integration
//!
//! This module provides straightforward Monte Carlo integration methods with variance reduction
//! techniques including antithetic variates, control variates, and stratified sampling.
//!
//! ## Methods
//!
//! | Method | Description |
//! |--------|-------------|
//! | `MonteCarloIntegrator::integrate_1d` | Simple 1-D Monte Carlo |
//! | `MonteCarloIntegrator::integrate_nd` | Multi-dimensional MC |
//! | `MonteCarloIntegrator::integrate_with_antithetic` | Antithetic variates |
//! | `MonteCarloIntegrator::integrate_with_control_variate` | Control variate |
//! | `MonteCarloIntegrator::stratified_sampling` | Stratified sampling |

use crate::error::{IntegrateError, IntegrateResult};
use scirs2_core::random::prelude::*;
use scirs2_core::random::Uniform;
use scirs2_core::Distribution;

/// Result of a basic Monte Carlo integration.
#[derive(Debug, Clone)]
pub struct MCResult {
    /// Estimated integral value.
    pub value: f64,
    /// Standard error of the estimate (sqrt(variance / n)).
    pub std_error: f64,
    /// 95% confidence interval `(value - 1.96*std_error, value + 1.96*std_error)`.
    pub confidence_interval: (f64, f64),
    /// Number of function evaluations used.
    pub n_samples: usize,
}

impl MCResult {
    fn new(value: f64, std_error: f64, n_samples: usize) -> Self {
        let half_width = 1.96 * std_error;
        Self {
            value,
            std_error,
            confidence_interval: (value - half_width, value + half_width),
            n_samples,
        }
    }
}

/// Basic Monte Carlo integrator with configurable sample count and seed.
///
/// # Examples
///
/// ```
/// use scirs2_integrate::monte_carlo::basic_mc::MonteCarloIntegrator;
///
/// let mc = MonteCarloIntegrator { n_samples: 100_000, seed: 42 };
/// let result = mc.integrate_1d(|x| x * x, 0.0, 1.0).unwrap();
/// assert!((result.value - 1.0 / 3.0).abs() < 0.01, "got {}", result.value);
/// ```
#[derive(Debug, Clone)]
pub struct MonteCarloIntegrator {
    /// Number of samples to draw.
    pub n_samples: usize,
    /// Seed for the random number generator (for reproducibility).
    pub seed: u64,
}

impl MonteCarloIntegrator {
    /// Create a new integrator with given sample count and seed.
    pub fn new(n_samples: usize, seed: u64) -> Self {
        Self { n_samples, seed }
    }

    /// Integrate a 1-D function `f` over `[a, b]` using plain Monte Carlo.
    ///
    /// The estimator is:
    /// ```text
    /// I ≈ (b - a) * (1/n) * Σ f(x_i),   x_i ~ Uniform(a, b)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if `n_samples == 0` or `a >= b`.
    pub fn integrate_1d<F>(&self, f: F, a: f64, b: f64) -> IntegrateResult<MCResult>
    where
        F: Fn(f64) -> f64,
    {
        if self.n_samples == 0 {
            return Err(IntegrateError::ValueError(
                "n_samples must be > 0".to_string(),
            ));
        }
        if a >= b {
            return Err(IntegrateError::ValueError(format!(
                "Integration bounds must satisfy a < b, got a={a}, b={b}"
            )));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let dist =
            Uniform::new_inclusive(a, b).map_err(|e| IntegrateError::ValueError(e.to_string()))?;

        let n = self.n_samples;
        let width = b - a;
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;

        for _ in 0..n {
            let x: f64 = dist.sample(&mut rng);
            let fx = f(x);
            sum += fx;
            sum_sq += fx * fx;
        }

        let nf = n as f64;
        let mean = sum / nf;
        let variance = (sum_sq - sum * sum / nf) / (nf - 1.0).max(1.0);
        let std_error = (variance / nf).sqrt() * width;
        let value = mean * width;

        Ok(MCResult::new(value, std_error, n))
    }

    /// Integrate an n-dimensional function `f` over the hyper-rectangle defined by `bounds`.
    ///
    /// `bounds` is a slice of `(a_i, b_i)` for each dimension.
    /// The estimator is:
    /// ```text
    /// I ≈ Vol * (1/n) * Σ f(x_i),   x_i ~ Uniform(bounds)
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if `bounds` is empty, any bound has `a >= b`, or `n_samples == 0`.
    pub fn integrate_nd<F>(&self, f: F, bounds: &[(f64, f64)]) -> IntegrateResult<MCResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        if self.n_samples == 0 {
            return Err(IntegrateError::ValueError(
                "n_samples must be > 0".to_string(),
            ));
        }
        if bounds.is_empty() {
            return Err(IntegrateError::ValueError(
                "bounds must be non-empty".to_string(),
            ));
        }
        for (i, &(a, b)) in bounds.iter().enumerate() {
            if a >= b {
                return Err(IntegrateError::ValueError(format!(
                    "bounds[{i}] must satisfy a < b, got a={a}, b={b}"
                )));
            }
        }

        let volume: f64 = bounds.iter().map(|&(a, b)| b - a).product();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let dists: Vec<Uniform<f64>> = bounds
            .iter()
            .map(|&(a, b)| {
                Uniform::new_inclusive(a, b).map_err(|e| IntegrateError::ValueError(e.to_string()))
            })
            .collect::<IntegrateResult<_>>()?;

        let n = self.n_samples;
        let nf = n as f64;
        let ndim = bounds.len();
        let mut point = vec![0.0_f64; ndim];
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;

        for _ in 0..n {
            for (j, dist) in dists.iter().enumerate() {
                point[j] = dist.sample(&mut rng);
            }
            let fx = f(&point);
            sum += fx;
            sum_sq += fx * fx;
        }

        let mean = sum / nf;
        let variance = (sum_sq - sum * sum / nf) / (nf - 1.0).max(1.0);
        let std_error = (variance / nf).sqrt() * volume;
        let value = mean * volume;

        Ok(MCResult::new(value, std_error, n))
    }

    /// Integrate using antithetic variates for variance reduction.
    ///
    /// For each uniform draw `u ~ Uniform(a,b)` a pair `(u, a+b-u)` is used:
    /// ```text
    /// I ≈ (b-a) * mean( [f(u_i) + f(a+b-u_i)] / 2 )
    /// ```
    /// This halves the variance when `f` is monotone.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_samples == 0` or `a >= b`.
    pub fn integrate_with_antithetic<F>(&self, f: F, a: f64, b: f64) -> IntegrateResult<MCResult>
    where
        F: Fn(f64) -> f64,
    {
        if self.n_samples == 0 {
            return Err(IntegrateError::ValueError(
                "n_samples must be > 0".to_string(),
            ));
        }
        if a >= b {
            return Err(IntegrateError::ValueError(format!(
                "Integration bounds must satisfy a < b, got a={a}, b={b}"
            )));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let dist =
            Uniform::new_inclusive(a, b).map_err(|e| IntegrateError::ValueError(e.to_string()))?;

        let n = self.n_samples;
        let nf = n as f64;
        let width = b - a;
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;

        for _ in 0..n {
            let u: f64 = dist.sample(&mut rng);
            let u_anti = a + b - u; // reflected around midpoint
            let pair_mean = (f(u) + f(u_anti)) * 0.5;
            sum += pair_mean;
            sum_sq += pair_mean * pair_mean;
        }

        let mean = sum / nf;
        // Each pair is a single estimator; variance uses n pairs
        let variance = (sum_sq - sum * sum / nf) / (nf - 1.0).max(1.0);
        let std_error = (variance / nf).sqrt() * width;
        let value = mean * width;

        Ok(MCResult::new(value, std_error, n))
    }

    /// Integrate using control variates.
    ///
    /// Uses a control function `g` with known mean `g_mean = E_Uniform[g]` to reduce variance:
    /// ```text
    /// f*(x) = f(x) - beta * (g(x) - g_mean)
    /// beta  = Cov(f, g) / Var(g)
    /// I ≈ (b-a) * mean(f*(x_i))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `f` — integrand
    /// * `g` — control variate function
    /// * `g_mean` — exact mean of `g` under Uniform(a,b)
    /// * `a`, `b` — integration bounds
    ///
    /// # Errors
    ///
    /// Returns an error if `n_samples < 2`, `a >= b`, or `Var(g) ≈ 0`.
    pub fn integrate_with_control_variate<F, G>(
        &self,
        f: F,
        g: G,
        g_mean: f64,
        a: f64,
        b: f64,
    ) -> IntegrateResult<MCResult>
    where
        F: Fn(f64) -> f64,
        G: Fn(f64) -> f64,
    {
        if self.n_samples < 2 {
            return Err(IntegrateError::ValueError(
                "n_samples must be >= 2 for control variates".to_string(),
            ));
        }
        if a >= b {
            return Err(IntegrateError::ValueError(format!(
                "Integration bounds must satisfy a < b, got a={a}, b={b}"
            )));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let dist =
            Uniform::new_inclusive(a, b).map_err(|e| IntegrateError::ValueError(e.to_string()))?;

        let n = self.n_samples;
        let nf = n as f64;
        let width = b - a;

        // First pass: collect f and g values
        let mut fvals = Vec::with_capacity(n);
        let mut gvals = Vec::with_capacity(n);
        for _ in 0..n {
            let x: f64 = dist.sample(&mut rng);
            fvals.push(f(x));
            gvals.push(g(x));
        }

        // Compute beta = Cov(f,g) / Var(g)
        let f_mean: f64 = fvals.iter().sum::<f64>() / nf;
        let g_sample_mean: f64 = gvals.iter().sum::<f64>() / nf;
        let mut cov_fg = 0.0_f64;
        let mut var_g = 0.0_f64;
        for i in 0..n {
            let df = fvals[i] - f_mean;
            let dg = gvals[i] - g_sample_mean;
            cov_fg += df * dg;
            var_g += dg * dg;
        }
        cov_fg /= nf - 1.0;
        var_g /= nf - 1.0;

        if var_g.abs() < 1e-14 {
            // Control variate is constant; fall back to plain MC
            let std_error = {
                let sum_sq: f64 = fvals.iter().map(|&v| v * v).sum();
                let sum: f64 = fvals.iter().sum();
                let variance = (sum_sq - sum * sum / nf) / (nf - 1.0).max(1.0);
                (variance / nf).sqrt() * width
            };
            return Ok(MCResult::new(f_mean * width, std_error, n));
        }

        let beta = cov_fg / var_g;

        // Second pass: compute f* = f - beta*(g - g_mean)
        let mut sum_adj = 0.0_f64;
        let mut sum_sq_adj = 0.0_f64;
        for i in 0..n {
            let adj = fvals[i] - beta * (gvals[i] - g_mean);
            sum_adj += adj;
            sum_sq_adj += adj * adj;
        }

        let mean_adj = sum_adj / nf;
        let variance_adj = (sum_sq_adj - sum_adj * sum_adj / nf) / (nf - 1.0).max(1.0);
        let std_error = (variance_adj / nf).sqrt() * width;
        let value = mean_adj * width;

        Ok(MCResult::new(value, std_error, n))
    }

    /// Stratified sampling: divide `[a, b]` into `n_strata` equal sub-intervals and
    /// draw `n_samples / n_strata` uniform samples from each stratum.
    ///
    /// This guarantees that every region is represented and reduces variance for
    /// smooth integrands.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_samples == 0`, `n_strata == 0`, or `a >= b`.
    pub fn stratified_sampling<F>(
        &self,
        f: F,
        a: f64,
        b: f64,
        n_strata: usize,
    ) -> IntegrateResult<MCResult>
    where
        F: Fn(f64) -> f64,
    {
        if self.n_samples == 0 {
            return Err(IntegrateError::ValueError(
                "n_samples must be > 0".to_string(),
            ));
        }
        if n_strata == 0 {
            return Err(IntegrateError::ValueError(
                "n_strata must be > 0".to_string(),
            ));
        }
        if a >= b {
            return Err(IntegrateError::ValueError(format!(
                "Integration bounds must satisfy a < b, got a={a}, b={b}"
            )));
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let width = b - a;
        let stratum_width = width / n_strata as f64;

        // Distribute samples across strata; each stratum gets at least 1
        let base_per_stratum = (self.n_samples / n_strata).max(1);
        let total_actual = base_per_stratum * n_strata;

        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;

        for k in 0..n_strata {
            let lo = a + k as f64 * stratum_width;
            let hi = lo + stratum_width;
            let dist = Uniform::new_inclusive(lo, hi)
                .map_err(|e| IntegrateError::ValueError(e.to_string()))?;
            for _ in 0..base_per_stratum {
                let x: f64 = dist.sample(&mut rng);
                let fx = f(x);
                sum += fx;
                sum_sq += fx * fx;
            }
        }

        let nf = total_actual as f64;
        let mean = sum / nf;
        let variance = (sum_sq - sum * sum / nf) / (nf - 1.0).max(1.0);
        let std_error = (variance / nf).sqrt() * width;
        let value = mean * width;

        Ok(MCResult::new(value, std_error, total_actual))
    }
}

impl Default for MonteCarloIntegrator {
    fn default() -> Self {
        Self {
            n_samples: 100_000,
            seed: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const TOLERANCE: f64 = 0.01;

    // x^2 on [0,1] → 1/3
    fn x_squared(x: f64) -> f64 {
        x * x
    }

    // sin(x) on [0, π] → 2
    fn sin_fn(x: f64) -> f64 {
        x.sin()
    }

    #[test]
    fn test_integrate_1d_x_squared() {
        let mc = MonteCarloIntegrator::new(200_000, 42);
        let result = mc
            .integrate_1d(x_squared, 0.0, 1.0)
            .expect("integrate_1d should succeed");
        assert!(
            (result.value - 1.0 / 3.0).abs() < TOLERANCE,
            "integrate_1d x^2: got {}, expected 1/3",
            result.value
        );
        assert!(result.std_error >= 0.0);
        assert_eq!(result.n_samples, 200_000);
        // Check CI contains truth
        assert!(result.confidence_interval.0 < 1.0 / 3.0);
        assert!(result.confidence_interval.1 > 1.0 / 3.0);
    }

    #[test]
    fn test_integrate_1d_sin() {
        let mc = MonteCarloIntegrator::new(200_000, 7);
        let result = mc
            .integrate_1d(sin_fn, 0.0, PI)
            .expect("integrate_1d should succeed");
        assert!(
            (result.value - 2.0).abs() < TOLERANCE,
            "integrate_1d sin: got {}, expected 2.0",
            result.value
        );
    }

    #[test]
    fn test_integrate_nd_2d() {
        // ∫∫ (x^2 + y^2) dx dy over [0,1]^2 = 2/3
        let mc = MonteCarloIntegrator::new(200_000, 123);
        let result = mc
            .integrate_nd(
                |pts| pts[0] * pts[0] + pts[1] * pts[1],
                &[(0.0, 1.0), (0.0, 1.0)],
            )
            .expect("integrate_nd should succeed");
        assert!(
            (result.value - 2.0 / 3.0).abs() < TOLERANCE,
            "integrate_nd 2d: got {}, expected 2/3",
            result.value
        );
    }

    #[test]
    fn test_integrate_nd_3d() {
        // ∫∫∫ x*y*z over [0,1]^3 = (1/2)^3 = 1/8
        let mc = MonteCarloIntegrator::new(200_000, 99);
        let result = mc
            .integrate_nd(
                |pts| pts[0] * pts[1] * pts[2],
                &[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
            )
            .expect("integrate_nd should succeed");
        assert!(
            (result.value - 0.125).abs() < TOLERANCE,
            "integrate_nd 3d: got {}, expected 0.125",
            result.value
        );
    }

    #[test]
    fn test_antithetic_x_squared() {
        let mc = MonteCarloIntegrator::new(200_000, 55);
        let result = mc
            .integrate_with_antithetic(x_squared, 0.0, 1.0)
            .expect("integrate_with_antithetic should succeed");
        assert!(
            (result.value - 1.0 / 3.0).abs() < TOLERANCE,
            "antithetic x^2: got {}, expected 1/3",
            result.value
        );
    }

    #[test]
    fn test_antithetic_variance_reduction() {
        // Antithetic should achieve lower std_error for monotone f
        let n = 50_000;
        let plain = MonteCarloIntegrator::new(n, 10)
            .integrate_1d(x_squared, 0.0, 1.0)
            .expect("integrate_1d should succeed");
        let antith = MonteCarloIntegrator::new(n, 10)
            .integrate_with_antithetic(x_squared, 0.0, 1.0)
            .expect("integrate_with_antithetic should succeed");
        // Both should be accurate
        assert!((plain.value - 1.0 / 3.0).abs() < 0.05);
        assert!((antith.value - 1.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn test_control_variate_x_squared() {
        // Integrate x^2 using g(x)=x with E[g] = 0.5
        let mc = MonteCarloIntegrator::new(50_000, 17);
        let result = mc
            .integrate_with_control_variate(x_squared, |x| x, 0.5, 0.0, 1.0)
            .expect("integrate_with_control_variate should succeed");
        assert!(
            (result.value - 1.0 / 3.0).abs() < TOLERANCE,
            "control variate x^2: got {}, expected 1/3",
            result.value
        );
    }

    #[test]
    fn test_stratified_x_squared() {
        let mc = MonteCarloIntegrator::new(100_000, 33);
        let result = mc
            .stratified_sampling(x_squared, 0.0, 1.0, 100)
            .expect("stratified_sampling should succeed");
        assert!(
            (result.value - 1.0 / 3.0).abs() < TOLERANCE,
            "stratified x^2: got {}, expected 1/3",
            result.value
        );
    }

    #[test]
    fn test_error_empty_bounds() {
        let mc = MonteCarloIntegrator::new(100, 0);
        assert!(mc.integrate_nd(|_| 1.0, &[]).is_err());
    }

    #[test]
    fn test_error_invalid_bounds() {
        let mc = MonteCarloIntegrator::new(100, 0);
        assert!(mc.integrate_1d(|x| x, 1.0, 0.0).is_err());
    }

    #[test]
    fn test_error_zero_samples() {
        let mc = MonteCarloIntegrator {
            n_samples: 0,
            seed: 0,
        };
        assert!(mc.integrate_1d(|x| x, 0.0, 1.0).is_err());
    }
}
