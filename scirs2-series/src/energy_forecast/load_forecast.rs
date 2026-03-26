//! Probabilistic electricity load forecasting.
//!
//! The [`LoadForecaster`] performs classical time series decomposition
//! (trend + seasonal + residual) followed by AR(p) modelling of the residual
//! via Levinson–Durbin Yule–Walker equations and bootstrap quantile construction.

use super::market_types::ProbabilisticForecast;
use crate::error::{Result, TimeSeriesError};

// ─── LCG PRNG ─────────────────────────────────────────────────────────────────

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ─── Levinson–Durbin solver ────────────────────────────────────────────────────

/// Solve a symmetric positive-definite Toeplitz system T·x = b.
///
/// `r` is the first row/column of the Toeplitz matrix: `T[i,j] = r[|i-j|]`.
/// `r` must have length ≥ p+1; `b` has length p.
///
/// Uses a Cholesky-free O(p²) Levinson recursion based on the classic
/// Durbin–Levinson algorithm as presented in Golub & Van Loan §4.7.
pub fn levinson_durbin(r: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    let p = b.len();
    if r.len() < p + 1 {
        return Err(TimeSeriesError::InvalidInput(
            "Levinson–Durbin: r must have length >= p+1".to_string(),
        ));
    }
    if p == 0 {
        return Ok(Vec::new());
    }
    if r[0].abs() < 1e-15 {
        return Err(TimeSeriesError::NumericalInstability(
            "Levinson–Durbin: r[0] is effectively zero".to_string(),
        ));
    }

    // --- Durbin's algorithm (Yule-Walker companion) -------------------------
    // First solve the auxiliary problem T·y = -[r[1], r[2], …, r[p]]
    // which is the standard Yule-Walker solve.  Then use the solution
    // to build the full solve for arbitrary rhs b.
    //
    // We use the "textbook" Levinson recursion that maintains:
    //   a[1..m] : reflection coefficients (AR params of order m)
    //   alpha_m  : leading reflection coefficient
    //   P_m      : prediction error power

    // Step 1: solve T·a = b directly by building the Toeplitz inverse
    // implicitly via the LDL-like recursion.

    // We apply the Trench algorithm for symmetric Toeplitz:
    // T has first row [r0, r1, r2, …, r_{p-1}].
    // Maintain the vector of AR coefficients a[0..k] and prediction error var.

    let r0 = r[0];
    let mut a: Vec<f64> = Vec::with_capacity(p); // AR coefficients (1-indexed convention, stored 0-indexed)
    let mut pred_err = r0; // prediction error power P

    // 1-step initialisation
    let alpha0 = -r[1] / r0;
    a.push(alpha0);
    pred_err = r0 * (1.0 - alpha0 * alpha0);

    for m in 2..=p {
        // Compute reflection coefficient:
        // k_m = -(r[m] + Σ_{j=1}^{m-1} a[j-1] * r[m-j]) / P_{m-1}
        let mut lam = r[m];
        for j in 1..m {
            lam += a[j - 1] * r[m - j];
        }
        let km = -lam / pred_err;

        // Update AR coefficients:
        // a_new[j] = a[j] + km * a[m-1-j]  for j = 0..m-2
        // a_new[m-1] = km
        let a_old = a.clone();
        for j in 0..m - 1 {
            a[j] = a_old[j] + km * a_old[m - 2 - j];
        }
        a.push(km);

        // Update prediction error
        pred_err *= 1.0 - km * km;
        if pred_err.abs() < 1e-15 {
            break;
        }
    }

    // Step 2: use the reflection-coefficient representation to solve T·x = b.
    // We build the full inverse T^{-1} b column by column using the
    // staircase / Gohberg-Semencul representation.
    //
    // Simpler approach: direct Levinson for general rhs (Trench variant).
    //
    // Maintain x[0..k+1] = solution to T_{k+1}·x = b[0..k+1]
    // At each step:
    //   delta_{k+1} = b[k+1] - r[k+1..1] · x[0..k+1]   (residual)
    //   correction via the forward predictor g (which we also maintain)

    // g[0..k] satisfies T_{k+1} · [g; 0] = e_{k+1} * P_k  (backward predictor)
    // We maintain g separately.

    let mut g: Vec<f64> = Vec::with_capacity(p); // backward predictor coefficients
    g.push(1.0);

    let mut x_sol = vec![0.0_f64; p];
    x_sol[0] = b[0] / r0;

    let mut pg = r0; // prediction error for g
    g[0] = -r[1] / r0;
    // Actually rebuild from our AR coefficients:
    // a[0..p-1] are the AR(p) coefficients.
    // For the Levinson solve of general b, we use the following recursion
    // (equivalent to the partition inverse approach):

    // Restart with a cleaner, verified O(p²) algorithm:
    // T[i,j] = r[|i-j|], 0-indexed.  b is the rhs.
    // Use the "Levinson–Trench" forward recursion.
    let mut x2 = vec![0.0_f64; p];
    let mut g2 = vec![0.0_f64; p]; // forward predictor (reversed a)

    x2[0] = b[0] / r0;
    g2[0] = -r[1] / r0;
    let mut alpha = -r[1] / r0;
    let mut beta = r0;
    beta = beta * (1.0 - alpha * alpha); // P_1

    let _ = g;
    let _ = x_sol;
    let _ = pg;

    for k in 1..p {
        // Compute δ = b[k] - r[k..1] · x2[0..k]
        let mut delta = b[k];
        for j in 0..k {
            delta -= r[k - j] * x2[j];
        }

        // Compute μ = δ / β
        let mu = delta / beta;

        // Update x2: x2_new[j] = x2[j] + mu * g2_rev[j]  for j=0..k-1, x2[k] = mu
        let x_old: Vec<f64> = x2[..k].to_vec();
        let g_old: Vec<f64> = g2[..k].to_vec();
        for j in 0..k {
            x2[j] = x_old[j] + mu * g_old[k - 1 - j];
        }
        x2[k] = mu;

        if k == p - 1 {
            break;
        }

        // Compute new reflection coefficient for g
        let mut s_g = r[k + 1];
        for j in 0..k {
            s_g += g2[j] * r[k - j];
        }
        let new_alpha = -s_g / beta;

        // Update g2
        let g_snap: Vec<f64> = g2[..k].to_vec();
        for j in 0..k {
            g2[j] = g_snap[j] + new_alpha * g_snap[k - 1 - j];
        }
        g2[k] = new_alpha;
        alpha = new_alpha;
        beta *= 1.0 - alpha * alpha;
        if beta.abs() < 1e-15 {
            break;
        }
    }

    Ok(x2)
}

// ─── Seasonal decomposition ────────────────────────────────────────────────────

/// Classical additive seasonal decomposition into trend, seasonal, and residual components.
///
/// * **Trend**: centered moving average of window `period`.
/// * **Seasonal**: group-by-position mean minus grand mean, replicated over the series.
/// * **Residual**: `loads - trend - seasonal`.
///
/// Returns `(trend, seasonal, residual)` — each of the same length as `loads`.
/// NaN-initialised trend endpoints are filled by linear extrapolation.
pub fn seasonal_decompose(loads: &[f64], period: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let n = loads.len();
    if n < 2 * period {
        return Err(TimeSeriesError::InsufficientData {
            message: "seasonal_decompose requires at least 2*period observations".to_string(),
            required: 2 * period,
            actual: n,
        });
    }
    if period == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "period".to_string(),
            message: "must be at least 1".to_string(),
        });
    }

    // ── 1. Trend: centered moving average ─────────────────────────────────────
    let half = period / 2;
    let mut trend = vec![f64::NAN; n];
    for i in half..n.saturating_sub(half) {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(n);
        let window_len = end - start;
        let mean: f64 = loads[start..end].iter().sum::<f64>() / window_len as f64;
        trend[i] = mean;
    }

    // Linear extrapolation for the endpoints where CMA is undefined
    // Left side
    let left_valid = trend.iter().position(|v| !v.is_nan()).unwrap_or(0);
    if left_valid > 0 {
        let slope = if left_valid + 1 < n && !trend[left_valid + 1].is_nan() {
            trend[left_valid + 1] - trend[left_valid]
        } else {
            0.0
        };
        for i in 0..left_valid {
            trend[i] = trend[left_valid] - slope * (left_valid - i) as f64;
        }
    }
    // Right side
    let right_valid = trend.iter().rposition(|v| !v.is_nan()).unwrap_or(n - 1);
    if right_valid + 1 < n {
        let slope = if right_valid > 0 && !trend[right_valid - 1].is_nan() {
            trend[right_valid] - trend[right_valid - 1]
        } else {
            0.0
        };
        for i in (right_valid + 1)..n {
            trend[i] = trend[right_valid] + slope * (i - right_valid) as f64;
        }
    }

    // ── 2. Seasonal: group-by-position mean minus grand mean ──────────────────
    let mut group_sums = vec![0.0; period];
    let mut group_counts = vec![0usize; period];
    for (i, &load) in loads.iter().enumerate() {
        let pos = i % period;
        let t = if trend[i].is_nan() { 0.0 } else { trend[i] };
        group_sums[pos] += load - t;
        group_counts[pos] += 1;
    }
    let mut seasonal_pattern: Vec<f64> = group_sums
        .iter()
        .zip(group_counts.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    // Normalise so seasonal effects sum to zero
    let seasonal_mean: f64 = seasonal_pattern.iter().sum::<f64>() / period as f64;
    for v in &mut seasonal_pattern {
        *v -= seasonal_mean;
    }

    // Expand to full series
    let seasonal: Vec<f64> = (0..n).map(|i| seasonal_pattern[i % period]).collect();

    // ── 3. Residual ───────────────────────────────────────────────────────────
    let residual: Vec<f64> = loads
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((&l, &t), &s)| l - t - s)
        .collect();

    Ok((trend, seasonal, residual))
}

// ─── AR(p) model for residuals ─────────────────────────────────────────────────

/// Fit an AR(p) model to the residual series using Yule–Walker / Levinson–Durbin.
///
/// Returns the AR coefficients φ = [φ_1, …, φ_p].
pub fn fit_ar(residuals: &[f64], p: usize) -> Result<Vec<f64>> {
    let n = residuals.len();
    if n <= p {
        return Err(TimeSeriesError::InsufficientData {
            message: "fit_ar: need more observations than AR order".to_string(),
            required: p + 1,
            actual: n,
        });
    }
    if p == 0 {
        return Ok(Vec::new());
    }

    let mean = residuals.iter().sum::<f64>() / n as f64;

    // Compute sample autocovariances r[0], r[1], …, r[p]
    let mut r = vec![0.0; p + 1];
    for lag in 0..=p {
        let sum: f64 = residuals[lag..]
            .iter()
            .zip(residuals[..n - lag].iter())
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum();
        r[lag] = sum / n as f64;
    }

    // Yule–Walker: r[0..p] * phi = r[1..p+1]
    let rhs: Vec<f64> = r[1..=p].to_vec();
    levinson_durbin(&r, &rhs)
}

/// One-step AR(p) prediction for a residual series.
///
/// Predicts the next `horizon` values using the fitted coefficients.
pub fn auto_regression_residual(residuals: &[f64], p: usize) -> Result<Vec<f64>> {
    fit_ar(residuals, p)
}

/// Forecast `horizon` steps of AR(p) given the fitted coefficients and history.
fn ar_forecast(phi: &[f64], history: &[f64], horizon: usize) -> Vec<f64> {
    let p = phi.len();
    let mut buf: Vec<f64> = history.iter().copied().collect();
    let mut forecasts = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        let n_buf = buf.len();
        let pred: f64 = phi
            .iter()
            .enumerate()
            .map(|(k, &phi_k)| {
                if n_buf > k {
                    phi_k * buf[n_buf - 1 - k]
                } else {
                    0.0
                }
            })
            .sum();
        buf.push(pred);
        forecasts.push(pred);
    }
    forecasts
}

// ─── LoadForecaster ────────────────────────────────────────────────────────────

/// Probabilistic electricity load forecaster.
///
/// Workflow:
/// 1. Decompose historical loads into trend + seasonal + residual.
/// 2. Fit AR(p) to the residual.
/// 3. Forecast residual `horizon` steps ahead using the AR model.
/// 4. Bootstrap CIs by resampling historical residuals.
/// 5. Add the seasonal pattern and extrapolated trend to form the final forecast.
pub struct LoadForecaster {
    /// Seasonal period in observations (e.g. 24 for hourly with daily seasonality).
    pub period: usize,
    /// AR order for the residual model.
    pub ar_order: usize,
    /// Number of bootstrap samples for confidence intervals.
    pub n_bootstrap: usize,
}

impl LoadForecaster {
    /// Create a new load forecaster.
    pub fn new(period: usize, ar_order: usize, n_bootstrap: usize) -> Self {
        Self {
            period,
            ar_order,
            n_bootstrap,
        }
    }

    /// Probabilistic load forecast.
    ///
    /// Returns a [`ProbabilisticForecast`] where each "scenario" is one bootstrap replicate.
    pub fn forecast_load(
        &self,
        historical: &[f64],
        horizon: usize,
        quantiles: &[f64],
    ) -> Result<ProbabilisticForecast> {
        let n = historical.len();
        if n < 2 * self.period + self.ar_order {
            return Err(TimeSeriesError::InsufficientData {
                message: "forecast_load: insufficient historical data".to_string(),
                required: 2 * self.period + self.ar_order,
                actual: n,
            });
        }

        // 1. Decompose
        let (trend, seasonal, residuals) = seasonal_decompose(historical, self.period)?;

        // 2. Fit AR(p) to residuals
        let phi = if self.ar_order > 0 {
            fit_ar(&residuals, self.ar_order)?
        } else {
            Vec::new()
        };

        // 3. Extrapolate trend (linear from last two trend values)
        let trend_last = trend[n - 1];
        let trend_slope = if n >= 2 {
            trend[n - 1] - trend[n - 2]
        } else {
            0.0
        };
        let trend_forecast: Vec<f64> = (1..=horizon)
            .map(|h| trend_last + trend_slope * h as f64)
            .collect();

        // 4. Extrapolate seasonal pattern
        let seasonal_forecast: Vec<f64> = (0..horizon)
            .map(|h| {
                let pattern_idx = (n + h) % self.period;
                // Derive from the periodic expansion already computed
                // use seasonal pattern from end of historical series
                seasonal[(n - self.period + pattern_idx % self.period) % seasonal.len()]
            })
            .collect();

        // 5. Point forecast of residual using AR
        let resid_point = ar_forecast(&phi, &residuals, horizon);

        // 6. Bootstrap residuals for CI
        let resid_std = {
            let mean_r = residuals.iter().sum::<f64>() / residuals.len() as f64;
            let var_r = residuals.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>()
                / residuals.len() as f64;
            var_r.sqrt().max(1e-6)
        };

        let mut rng = Lcg::new(54321);
        let n_residuals = residuals.len();
        let mut scenarios: Vec<Vec<f64>> = Vec::with_capacity(self.n_bootstrap);

        for _ in 0..self.n_bootstrap {
            let mut scenario = Vec::with_capacity(horizon);
            // Bootstrap residuals by sampling with replacement
            let mut boot_resid = resid_point.clone();
            for h in 0..horizon {
                let noise_idx = (rng.next_u64() as usize) % n_residuals;
                let noise = residuals[noise_idx] * 0.5 + rng.standard_normal() * resid_std * 0.5;
                let load = trend_forecast[h] + seasonal_forecast[h] + boot_resid[h] + noise;
                boot_resid[h] += noise * 0.1;
                scenario.push(load);
            }
            scenarios.push(scenario);
        }

        // 7. Compute mean and quantiles
        let mut mean = vec![0.0; horizon];
        let mut quantile_forecasts: Vec<Vec<f64>> = vec![vec![0.0; horizon]; quantiles.len()];

        for step in 0..horizon {
            let mut step_vals: Vec<f64> = scenarios.iter().map(|s| s[step]).collect();
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
            scenarios,
            mean,
        })
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seasonal_loads(n: usize, period: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                100.0
                    + 10.0
                        * (2.0 * std::f64::consts::PI * (i % period) as f64 / period as f64).sin()
                    + 0.5 * i as f64 // slight trend
            })
            .collect()
    }

    #[test]
    fn test_seasonal_decompose_reconstruction() {
        let loads = make_seasonal_loads(96, 24);
        let (trend, seasonal, residual) = seasonal_decompose(&loads, 24).expect("decompose ok");
        assert_eq!(trend.len(), loads.len());
        assert_eq!(seasonal.len(), loads.len());
        assert_eq!(residual.len(), loads.len());

        // trend + seasonal + residual should approximately reconstruct loads
        let max_err = loads
            .iter()
            .zip(trend.iter())
            .zip(seasonal.iter())
            .zip(residual.iter())
            .map(|(((l, t), s), r)| (l - t - s - r).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_err < 1e-8,
            "reconstruction error {max_err} should be near zero"
        );
    }

    #[test]
    fn test_levinson_durbin_toeplitz() {
        // Test: solve [4 2; 2 4] x = [6; 6]  → x = [1; 1]
        let r = vec![4.0, 2.0, 0.0]; // Toeplitz row: r[0]=4, r[1]=2
        let b = vec![6.0, 6.0];
        let x = levinson_durbin(&r, &b).expect("solve ok");
        assert_eq!(x.len(), 2);
        assert!((x[0] - 1.0).abs() < 1e-6, "x[0] should be ~1, got {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-6, "x[1] should be ~1, got {}", x[1]);
    }

    #[test]
    fn test_ar_fit() {
        // Residuals from an AR(1) with phi=0.5
        let mut r = Vec::with_capacity(200);
        let mut x = 0.0_f64;
        let mut rng = Lcg::new(11);
        for _ in 0..200 {
            x = 0.5 * x + rng.standard_normal() * 0.5;
            r.push(x);
        }
        let phi = fit_ar(&r, 1).expect("fit AR(1)");
        assert_eq!(phi.len(), 1);
        // The estimated coefficient should be in (0, 1) for a stationary AR(1)
        assert!(
            phi[0] > 0.0 && phi[0] < 1.0,
            "AR(1) coefficient should be in (0,1), got {}",
            phi[0]
        );
    }

    #[test]
    fn test_load_forecast_horizon() {
        let loads = make_seasonal_loads(120, 24);
        let forecaster = LoadForecaster::new(24, 2, 50);
        let quantiles = [0.1, 0.5, 0.9];
        let result = forecaster
            .forecast_load(&loads, 24, &quantiles)
            .expect("forecast ok");
        assert_eq!(
            result.mean.len(),
            24,
            "mean forecast should have length=horizon"
        );
        for qf in &result.quantile_forecasts {
            assert_eq!(
                qf.len(),
                24,
                "each quantile forecast should have length=horizon"
            );
        }
    }
}
