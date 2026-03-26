//! Electricity price forecasting via Ornstein-Uhlenbeck and spike models.
//!
//! ## Models
//!
//! * [`MeanReversionModel`] — continuous-time OU: dP = κ(μ−P)dt + σ dW
//! * [`SpikePriceModel`]   — OU extended with a Poisson jump process

use super::market_types::{MarketClearingPrice, ProbabilisticForecast};
use crate::error::{Result, TimeSeriesError};

// ─── LCG PRNG ─────────────────────────────────────────────────────────────────

/// Minimal LCG pseudo-random number generator (no external rand crate).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    /// Advance state and return next u64.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Uniform sample in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller (consume two uniforms, return one normal).
    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ─── MeanReversionModel (Ornstein–Uhlenbeck) ─────────────────────────────────

/// Ornstein–Uhlenbeck mean-reversion electricity price model.
///
/// Continuous-time dynamics:
///   dP = κ(μ − P) dt + σ dW
///
/// Exact discrete update (Δt = 1 step):
///   P(t+Δt) = P(t)·e^{−κΔt} + μ(1−e^{−κΔt}) + σ·√[(1−e^{−2κΔt})/(2κ)] · N(0,1)
#[derive(Debug, Clone)]
pub struct MeanReversionModel {
    /// Mean-reversion speed κ > 0.
    pub kappa: f64,
    /// Long-run mean μ.
    pub mu: f64,
    /// Volatility σ > 0.
    pub sigma: f64,
}

impl MeanReversionModel {
    /// Create a model with explicit parameters.
    pub fn new(kappa: f64, mu: f64, sigma: f64) -> Result<Self> {
        if kappa <= 0.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "kappa".to_string(),
                message: format!("must be positive, got {kappa}"),
            });
        }
        if sigma <= 0.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "sigma".to_string(),
                message: format!("must be positive, got {sigma}"),
            });
        }
        Ok(Self { kappa, mu, sigma })
    }

    /// Calibrate (κ, μ, σ) from historical price data using moment matching.
    ///
    /// Moment matching:
    ///   μ̂ = mean(prices)
    ///   σ̂_obs = std(prices)  →  σ̂ = σ̂_obs · √(2κ) (from stationary variance σ²/(2κ))
    ///   κ̂  from AR(1) coefficient: ρ = e^{−κ} → κ = −ln(ρ) where ρ = corr(P_t, P_{t−1})
    pub fn calibrate(prices: &[f64]) -> Result<Self> {
        let n = prices.len();
        if n < 4 {
            return Err(TimeSeriesError::InsufficientData {
                message: "need at least 4 price observations for calibration".to_string(),
                required: 4,
                actual: n,
            });
        }

        // Long-run mean
        let mu = prices.iter().sum::<f64>() / n as f64;

        // Lag-1 autocorrelation via Pearson on (P_{t-1}, P_t) pairs
        let n_pairs = n - 1;
        let p_prev: Vec<f64> = prices[..n_pairs].to_vec();
        let p_next: Vec<f64> = prices[1..].to_vec();
        let mean_prev = p_prev.iter().sum::<f64>() / n_pairs as f64;
        let mean_next = p_next.iter().sum::<f64>() / n_pairs as f64;

        let cov: f64 = p_prev
            .iter()
            .zip(p_next.iter())
            .map(|(a, b)| (a - mean_prev) * (b - mean_next))
            .sum::<f64>();
        let var_prev: f64 = p_prev.iter().map(|a| (a - mean_prev).powi(2)).sum::<f64>();
        let var_next: f64 = p_next.iter().map(|b| (b - mean_next).powi(2)).sum::<f64>();

        let rho = if var_prev > 0.0 && var_next > 0.0 {
            cov / (var_prev * var_next).sqrt()
        } else {
            0.99
        };
        // Clamp ρ to (0, 1) to avoid log domain issues
        let rho_clamped = rho.clamp(1e-6, 1.0 - 1e-6);
        let kappa = -rho_clamped.ln();

        // Stationary variance = σ²/(2κ)  →  σ = std · √(2κ)
        let var_obs: f64 = prices.iter().map(|p| (p - mu).powi(2)).sum::<f64>() / n as f64;
        let sigma = (var_obs * 2.0 * kappa).sqrt().max(1e-6);

        Ok(Self { kappa, mu, sigma })
    }

    /// Simulate one step of the OU process.
    ///
    /// `dt` is the time step size (typically 1.0 for hourly data).
    fn step(&self, price: f64, dt: f64, rng: &mut Lcg) -> f64 {
        let e_neg_kdt = (-self.kappa * dt).exp();
        let mean = price * e_neg_kdt + self.mu * (1.0 - e_neg_kdt);
        let variance =
            self.sigma.powi(2) * (1.0 - (-2.0 * self.kappa * dt).exp()) / (2.0 * self.kappa);
        let std = variance.sqrt();
        mean + std * rng.standard_normal()
    }

    /// Simulate `n_scenarios` price paths of length `horizon` starting from `p0`.
    pub fn simulate_paths(
        &self,
        p0: f64,
        horizon: usize,
        n_scenarios: usize,
        seed: u64,
    ) -> Vec<Vec<f64>> {
        let mut rng = Lcg::new(seed);
        let mut paths = Vec::with_capacity(n_scenarios);
        for _ in 0..n_scenarios {
            let mut path = Vec::with_capacity(horizon);
            let mut p = p0;
            for _ in 0..horizon {
                p = self.step(p, 1.0, &mut rng);
                path.push(p);
            }
            paths.push(path);
        }
        paths
    }

    /// Forecast probabilistic prices.
    pub fn forecast_prices(
        &self,
        historical: &[f64],
        horizon: usize,
        n_scenarios: usize,
        quantiles: &[f64],
    ) -> Result<ProbabilisticForecast> {
        if historical.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "historical prices must not be empty".to_string(),
            ));
        }
        if horizon == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "horizon must be at least 1".to_string(),
            ));
        }

        let p0 = *historical
            .last()
            .expect("non-empty historical guaranteed by check above");
        let paths = self.simulate_paths(p0, horizon, n_scenarios, 12345);

        // Compute mean and quantiles at each step
        let mut mean = vec![0.0; horizon];
        let mut quantile_forecasts: Vec<Vec<f64>> = vec![vec![0.0; horizon]; quantiles.len()];

        for step in 0..horizon {
            let mut step_vals: Vec<f64> = paths.iter().map(|p| p[step]).collect();
            let m = step_vals.iter().sum::<f64>() / step_vals.len() as f64;
            mean[step] = m;

            step_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let ns = step_vals.len();
            for (qi, &q) in quantiles.iter().enumerate() {
                let idx = ((ns as f64 - 1.0) * q).round() as usize;
                quantile_forecasts[qi][step] = step_vals[idx.min(ns - 1)];
            }
        }

        let scenarios = paths;
        Ok(ProbabilisticForecast {
            quantile_forecasts,
            scenarios,
            mean,
        })
    }
}

// ─── SpikePriceModel ─────────────────────────────────────────────────────────

/// OU process extended with a Poisson-intensity jump (spike) component.
///
/// A price spike occurs with probability λ·Δt per step.
/// Spike magnitude is drawn from a log-normal distribution with parameters
/// (jump_mean_log, jump_std_log) on the log scale.
/// The sign of the spike is positive for upward spikes and negative for downward.
#[derive(Debug, Clone)]
pub struct SpikePriceModel {
    /// Underlying OU model.
    pub ou: MeanReversionModel,
    /// Jump intensity (expected jumps per time step).
    pub lambda: f64,
    /// Mean of jump size on log scale.
    pub jump_mean_log: f64,
    /// Standard deviation of jump size on log scale.
    pub jump_std_log: f64,
    /// Spike detection threshold (multiples of σ).
    pub spike_threshold_sigma: f64,
}

impl SpikePriceModel {
    /// Create a spike model from an OU model and jump parameters.
    pub fn new(
        ou: MeanReversionModel,
        lambda: f64,
        jump_mean_log: f64,
        jump_std_log: f64,
        spike_threshold_sigma: f64,
    ) -> Result<Self> {
        if lambda < 0.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "lambda".to_string(),
                message: format!("jump intensity must be non-negative, got {lambda}"),
            });
        }
        Ok(Self {
            ou,
            lambda,
            jump_mean_log,
            jump_std_log,
            spike_threshold_sigma,
        })
    }

    /// Calibrate the spike model from historical prices.
    ///
    /// The OU component is calibrated first; then prices more than
    /// `spike_threshold_sigma` σ from the mean are identified as spikes.
    pub fn calibrate(prices: &[f64], spike_threshold_sigma: f64) -> Result<Self> {
        let ou = MeanReversionModel::calibrate(prices)?;
        let n = prices.len() as f64;
        let spike_count = prices
            .iter()
            .filter(|&&p| (p - ou.mu).abs() > spike_threshold_sigma * ou.sigma)
            .count();
        let lambda = spike_count as f64 / n;

        // Compute log-normal parameters from the spike magnitudes
        let spike_magnitudes: Vec<f64> = prices
            .iter()
            .filter(|&&p| (p - ou.mu).abs() > spike_threshold_sigma * ou.sigma)
            .map(|&p| (p - ou.mu).abs().ln().max(-10.0))
            .collect();

        let (jump_mean_log, jump_std_log) = if spike_magnitudes.is_empty() {
            (0.0, 1.0)
        } else {
            let m = spike_magnitudes.iter().sum::<f64>() / spike_magnitudes.len() as f64;
            let v = spike_magnitudes
                .iter()
                .map(|x| (x - m).powi(2))
                .sum::<f64>()
                / spike_magnitudes.len() as f64;
            (m, v.sqrt().max(0.1))
        };

        Ok(Self {
            ou,
            lambda,
            jump_mean_log,
            jump_std_log,
            spike_threshold_sigma,
        })
    }

    /// Check whether a price is a spike (|P - μ| > threshold * σ).
    pub fn is_spike(&self, price: f64) -> bool {
        (price - self.ou.mu).abs() > self.spike_threshold_sigma * self.ou.sigma
    }

    /// Simulate one step including possible spike.
    fn step(&self, price: f64, dt: f64, rng: &mut Lcg) -> f64 {
        // OU step
        let base = {
            let e_neg_kdt = (-self.ou.kappa * dt).exp();
            let mean = price * e_neg_kdt + self.ou.mu * (1.0 - e_neg_kdt);
            let variance = self.ou.sigma.powi(2) * (1.0 - (-2.0 * self.ou.kappa * dt).exp())
                / (2.0 * self.ou.kappa);
            mean + variance.sqrt() * rng.standard_normal()
        };

        // Poisson jump: probability = 1 - e^{-λΔt} ≈ λΔt for small λ
        let jump_prob = 1.0 - (-self.lambda * dt).exp();
        if rng.uniform() < jump_prob {
            // Log-normal jump size
            let log_size = self.jump_mean_log + self.jump_std_log * rng.standard_normal();
            let jump_size = log_size.exp();
            // Direction: positive or negative with equal probability
            let sign = if rng.uniform() < 0.5 { 1.0 } else { -1.0 };
            base + sign * jump_size
        } else {
            base
        }
    }

    /// Simulate `n_scenarios` paths with jumps.
    pub fn simulate_paths(
        &self,
        p0: f64,
        horizon: usize,
        n_scenarios: usize,
        seed: u64,
    ) -> Vec<Vec<f64>> {
        let mut rng = Lcg::new(seed);
        let mut paths = Vec::with_capacity(n_scenarios);
        for _ in 0..n_scenarios {
            let mut path = Vec::with_capacity(horizon);
            let mut p = p0;
            for _ in 0..horizon {
                p = self.step(p, 1.0, &mut rng);
                path.push(p);
            }
            paths.push(path);
        }
        paths
    }

    /// Count spikes in simulated paths (for testing/validation).
    pub fn count_spikes_in_paths(&self, paths: &[Vec<f64>]) -> usize {
        paths
            .iter()
            .flat_map(|p| p.iter())
            .filter(|&&price| self.is_spike(price))
            .count()
    }

    /// Probabilistic forecast using the spike model.
    pub fn forecast_prices(
        &self,
        historical: &[f64],
        horizon: usize,
        n_scenarios: usize,
        quantiles: &[f64],
    ) -> Result<ProbabilisticForecast> {
        if historical.is_empty() {
            return Err(TimeSeriesError::InvalidInput(
                "historical prices must not be empty".to_string(),
            ));
        }
        let p0 = *historical.last().expect("non-empty guaranteed");
        let paths = self.simulate_paths(p0, horizon, n_scenarios, 99999);

        let mut mean = vec![0.0; horizon];
        let mut quantile_forecasts: Vec<Vec<f64>> = vec![vec![0.0; horizon]; quantiles.len()];

        for step in 0..horizon {
            let mut step_vals: Vec<f64> = paths.iter().map(|p| p[step]).collect();
            mean[step] = step_vals.iter().sum::<f64>() / step_vals.len() as f64;
            step_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let ns = step_vals.len();
            for (qi, &q) in quantiles.iter().enumerate() {
                let idx = ((ns as f64 - 1.0) * q).round() as usize;
                quantile_forecasts[qi][step] = step_vals[idx.min(ns - 1)];
            }
        }

        Ok(ProbabilisticForecast {
            quantile_forecasts,
            scenarios: paths,
            mean,
        })
    }

    /// Derive a [`MarketClearingPrice`] estimate for a single horizon step.
    pub fn market_clearing_price(
        &self,
        historical: &[f64],
        step: usize,
        n_scenarios: usize,
        confidence: f64,
    ) -> Result<MarketClearingPrice> {
        let quantiles = vec![
            (1.0 - confidence) / 2.0,
            0.5,
            1.0 - (1.0 - confidence) / 2.0,
        ];
        let forecast = self.forecast_prices(historical, step + 1, n_scenarios, &quantiles)?;
        let price = forecast.mean[step];
        let lo = forecast.quantile_forecasts[0][step];
        let hi = forecast.quantile_forecasts[2][step];
        Ok(MarketClearingPrice {
            price,
            confidence_interval: (lo, hi),
        })
    }
}

// ─── standalone forecast function ─────────────────────────────────────────────

/// Forecast electricity prices using a calibrated OU model.
///
/// Convenience wrapper that calibrates from `data` and simulates `n_scenarios`
/// OU paths to produce a [`ProbabilisticForecast`].
pub fn forecast_prices(
    data: &[f64],
    horizon: usize,
    n_scenarios: usize,
    quantiles: &[f64],
) -> Result<ProbabilisticForecast> {
    let model = MeanReversionModel::calibrate(data)?;
    model.forecast_prices(data, horizon, n_scenarios, quantiles)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_prices() -> Vec<f64> {
        // Synthetic price series following OU around 50.0
        let mut rng = Lcg::new(777);
        let mut prices = Vec::with_capacity(200);
        let mut p = 50.0_f64;
        for _ in 0..200 {
            p = p * 0.9 + 50.0 * 0.1 + rng.standard_normal() * 3.0;
            prices.push(p);
        }
        prices
    }

    #[test]
    fn test_ou_calibration() {
        let prices = sample_prices();
        let model = MeanReversionModel::calibrate(&prices).expect("calibration should succeed");
        assert!(model.kappa.is_finite(), "κ should be finite");
        assert!(model.mu.is_finite(), "μ should be finite");
        assert!(model.sigma.is_finite(), "σ should be finite");
        assert!(model.kappa > 0.0, "κ should be positive");
        assert!(model.sigma > 0.0, "σ should be positive");
    }

    #[test]
    fn test_ou_simulation_paths() {
        let model = MeanReversionModel::new(0.2, 50.0, 5.0).expect("valid params");
        let paths = model.simulate_paths(50.0, 24, 100, 42);
        assert_eq!(paths.len(), 100, "should have 100 scenarios");
        for path in &paths {
            assert_eq!(path.len(), 24, "each path should have horizon=24 steps");
            for &p in path {
                assert!(p.is_finite(), "all prices should be finite");
            }
        }
    }

    #[test]
    fn test_ou_mean_reversion() {
        // With many scenarios and large horizon, mean of paths should be near μ
        let mu = 50.0;
        let model = MeanReversionModel::new(0.5, mu, 2.0).expect("valid params");
        let paths = model.simulate_paths(100.0, 100, 1000, 99); // start far from μ
        let last_step_mean: f64 = paths.iter().map(|p| p[99]).sum::<f64>() / 1000.0;
        assert!(
            (last_step_mean - mu).abs() < 10.0,
            "long-run mean {last_step_mean} should be near μ={mu}"
        );
    }

    #[test]
    fn test_price_forecast_quantiles() {
        let model = MeanReversionModel::new(0.3, 50.0, 5.0).expect("valid params");
        let data = vec![50.0_f64; 10];
        let quantiles = [0.1, 0.5, 0.9];
        let forecast = model
            .forecast_prices(&data, 12, 200, &quantiles)
            .expect("forecast ok");
        for step in 0..12 {
            let q10 = forecast.quantile_forecasts[0][step];
            let q50 = forecast.quantile_forecasts[1][step];
            let q90 = forecast.quantile_forecasts[2][step];
            assert!(
                q10 <= q50 + 1e-6 && q50 <= q90 + 1e-6,
                "quantile ordering violated at step {step}: {q10} <= {q50} <= {q90}"
            );
        }
    }

    #[test]
    fn test_spike_model_detects_spikes() {
        let prices = sample_prices();
        let model = SpikePriceModel::calibrate(&prices, 2.5).expect("calibration ok");
        // Inject a known spike: price = μ + 10σ
        let extreme = model.ou.mu + 10.0 * model.ou.sigma;
        assert!(
            model.is_spike(extreme),
            "extreme price should be flagged as spike"
        );
        // Near-mean price should NOT be a spike
        let normal = model.ou.mu + 0.5 * model.ou.sigma;
        assert!(
            !model.is_spike(normal),
            "normal price should not be flagged"
        );
    }

    #[test]
    fn test_probabilistic_forecast_scenarios() {
        let model = MeanReversionModel::new(0.3, 50.0, 5.0).expect("valid params");
        let data = vec![50.0_f64; 10];
        let quantiles = [0.5];
        let forecast = model
            .forecast_prices(&data, 6, 50, &quantiles)
            .expect("forecast ok");
        assert_eq!(forecast.scenarios.len(), 50, "should have 50 scenarios");
        for scenario in &forecast.scenarios {
            assert_eq!(
                scenario.len(),
                6,
                "each scenario should have horizon=6 steps"
            );
        }
    }
}
