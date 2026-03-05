//! Structural Time Series Models (Basic Structural Model, Harvey 1989).
//!
//! The unobserved components model:
//!   y_t = μ_t + γ_t + ε_t,  ε_t ~ N(0, σ²_ε)
//!
//! Level (local linear trend):
//!   μ_t = μ_{t-1} + β_{t-1} + η_t,  η_t ~ N(0, σ²_η)
//!   β_t = β_{t-1} + ζ_t,              ζ_t ~ N(0, σ²_ζ)
//!
//! Seasonal (dummy-variable form, period s):
//!   Σ_{j=0}^{s-1} γ_{t-j} = ω_t,  ω_t ~ N(0, σ²_ω)
//!
//! The model is cast in state-space form and estimated via the Kalman filter
//! log-likelihood, maximised using a bounded Nelder-Mead optimiser.

use super::linear_gaussian::LinearGaussianSSM;
use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Component structs
// ---------------------------------------------------------------------------

/// Trend component of a structural time series model.
#[derive(Debug, Clone)]
pub struct TrendComponent {
    /// Current level μ
    pub level: f64,
    /// Current slope β (0 for local-level model)
    pub slope: f64,
    /// Level disturbance variance σ²_η
    pub level_var: f64,
    /// Slope disturbance variance σ²_ζ (0 means deterministic slope)
    pub slope_var: f64,
}

impl TrendComponent {
    /// Create a trend with given level/slope variances.
    pub fn new(level_var: f64, slope_var: f64) -> Self {
        Self {
            level: 0.0,
            slope: 0.0,
            level_var: level_var.max(1e-10),
            slope_var: slope_var.max(0.0),
        }
    }

    /// Local level (random walk): slope = 0, σ²_ζ = 0.
    pub fn local_level(level_var: f64) -> Self {
        Self::new(level_var, 0.0)
    }

    /// State dimension: 1 for local-level, 2 for local-linear-trend.
    pub fn state_dim(&self) -> usize {
        if self.slope_var > 0.0 {
            2
        } else {
            1
        }
    }
}

/// Seasonal component in dummy-variable form.
#[derive(Debug, Clone)]
pub struct SeasonalComponent {
    /// Seasonal period s (must be >= 2)
    pub period: usize,
    /// Seasonal state values γ_1, ..., γ_{s-1}
    pub values: Vec<f64>,
    /// Seasonal disturbance variance σ²_ω
    pub var: f64,
}

impl SeasonalComponent {
    /// Create a seasonal component with given period and variance.
    pub fn new(period: usize, var: f64) -> Result<Self> {
        if period < 2 {
            return Err(TimeSeriesError::InvalidInput(
                "Seasonal period must be >= 2".to_string(),
            ));
        }
        Ok(Self {
            period,
            values: vec![0.0; period - 1],
            var: var.max(1e-10),
        })
    }

    /// State dimension = period - 1.
    pub fn state_dim(&self) -> usize {
        self.period - 1
    }
}

// ---------------------------------------------------------------------------
// StructuralModel
// ---------------------------------------------------------------------------

/// Basic Structural Model (BSM) combining trend, seasonality, and irregular.
#[derive(Debug, Clone)]
pub struct StructuralModel {
    /// Trend component (level + optional slope)
    pub trend: TrendComponent,
    /// Optional seasonal component
    pub seasonal: Option<SeasonalComponent>,
    /// Irregular (observation noise) variance σ²_ε
    pub irregular_var: f64,
}

impl StructuralModel {
    /// Create a new structural model.
    ///
    /// If `period` is `Some(s)`, a seasonal component is included.
    pub fn new(period: Option<usize>) -> Result<Self> {
        let seasonal = match period {
            Some(s) => Some(SeasonalComponent::new(s, 0.1)?),
            None => None,
        };
        Ok(Self {
            trend: TrendComponent::new(0.1, 0.01),
            seasonal,
            irregular_var: 0.5,
        })
    }

    /// Build a local-level model (no slope, no seasonality).
    pub fn local_level(level_var: f64, obs_var: f64) -> Self {
        Self {
            trend: TrendComponent::local_level(level_var),
            seasonal: None,
            irregular_var: obs_var.max(1e-10),
        }
    }

    /// Build a local linear trend model (with slope, no seasonality).
    pub fn local_linear_trend(level_var: f64, slope_var: f64, obs_var: f64) -> Self {
        Self {
            trend: TrendComponent::new(level_var, slope_var),
            seasonal: None,
            irregular_var: obs_var.max(1e-10),
        }
    }

    /// Total state dimension.
    pub fn state_dim(&self) -> usize {
        let trend_d = self.trend.state_dim();
        let seas_d = self.seasonal.as_ref().map_or(0, |s| s.state_dim());
        trend_d + seas_d
    }

    /// Convert to a `LinearGaussianSSM` in state-space form.
    ///
    /// State vector layout:
    ///   [μ, (β), γ₁, γ₂, ..., γ_{s-1}]
    pub fn to_ssm(&self) -> LinearGaussianSSM {
        let n = self.state_dim();
        let trend_d = self.trend.state_dim();
        let seas_d = self.seasonal.as_ref().map_or(0, |s| s.state_dim());

        // Transition matrix F
        let mut f = vec![vec![0.0f64; n]; n];

        // Trend block
        if self.trend.state_dim() == 1 {
            // Local level: μ_t = μ_{t-1}
            f[0][0] = 1.0;
        } else {
            // Local linear trend: [μ_t; β_t] = [[1,1],[0,1]] [μ_{t-1}; β_{t-1}]
            f[0][0] = 1.0;
            f[0][1] = 1.0;
            f[1][1] = 1.0;
        }

        // Seasonal block: dummy-variable form
        if let Some(seas) = &self.seasonal {
            let s = seas.state_dim(); // period - 1
            let off = trend_d;
            // First row: [-1, -1, ..., -1]
            for j in 0..s {
                f[off][off + j] = -1.0;
            }
            // Remaining: shift register [I_{s-1} | 0]
            for i in 1..s {
                f[off + i][off + i - 1] = 1.0;
            }
        }

        // Observation matrix H: [1, 0, (1, 0, ...)]
        let mut h = vec![vec![0.0f64; n]];
        h[0][0] = 1.0; // level
        if seas_d > 0 {
            h[0][trend_d] = 1.0; // first seasonal state
        }

        // Process noise covariance Q
        let mut q = vec![vec![0.0f64; n]; n];
        q[0][0] = self.trend.level_var;
        if self.trend.state_dim() == 2 {
            q[1][1] = self.trend.slope_var;
        }
        if let Some(seas) = &self.seasonal {
            q[trend_d][trend_d] = seas.var;
        }

        // Measurement noise covariance R
        let r = vec![vec![self.irregular_var]];

        // Initial state: diffuse
        let mu0 = vec![0.0f64; n];
        let mut p0 = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            p0[i][i] = 1e6;
        }

        LinearGaussianSSM {
            dim_state: n,
            dim_obs: 1,
            f_mat: f,
            h_mat: h,
            q_mat: q,
            r_mat: r,
            mu0,
            p0,
        }
    }

    /// Compute Kalman filter log-likelihood for the given parameter vector.
    ///
    /// `params` = [log(σ²_η), log(σ²_ζ)?, log(σ²_ω)?, log(σ²_ε)]
    fn log_likelihood_from_params(&self, params: &[f64], data: &[f64]) -> f64 {
        let mut model = self.clone();
        model.apply_params(params);
        let ssm = model.to_ssm();
        let obs: Vec<Vec<f64>> = data.iter().map(|&y| vec![y]).collect();
        ssm.filter(&obs)
            .map_or(f64::NEG_INFINITY, |k| k.log_likelihood)
    }

    /// Apply a parameter vector to update variances.
    fn apply_params(&mut self, params: &[f64]) {
        let mut idx = 0;
        // level variance
        self.trend.level_var = params[idx].exp().max(1e-10);
        idx += 1;
        // slope variance (if present)
        if self.trend.state_dim() == 2 {
            self.trend.slope_var = params[idx].exp().max(1e-10);
            idx += 1;
        }
        // seasonal variance (if present)
        if let Some(seas) = &mut self.seasonal {
            seas.var = params[idx].exp().max(1e-10);
            idx += 1;
        }
        // obs variance
        if idx < params.len() {
            self.irregular_var = params[idx].exp().max(1e-10);
        }
    }

    /// Extract initial parameter vector (log-scale) for optimisation.
    fn initial_params(&self) -> Vec<f64> {
        let mut p = Vec::new();
        p.push(self.trend.level_var.max(1e-10).ln());
        if self.trend.state_dim() == 2 {
            p.push(self.trend.slope_var.max(1e-10).ln());
        }
        if let Some(seas) = &self.seasonal {
            p.push(seas.var.max(1e-10).ln());
        }
        p.push(self.irregular_var.max(1e-10).ln());
        p
    }

    /// Fit the model by maximising the Kalman filter log-likelihood.
    ///
    /// Uses a simple coordinate-ascent / line-search optimiser on log-scale
    /// variance parameters (no external optimiser dependency).
    /// Returns the maximised log-likelihood.
    pub fn fit(&mut self, data: &[f64]) -> Result<f64> {
        let n = data.len();
        if n < 3 {
            return Err(TimeSeriesError::InsufficientData {
                message: "StructuralModel::fit requires at least 3 observations".to_string(),
                required: 3,
                actual: n,
            });
        }

        let mut params = self.initial_params();
        let np = params.len();
        let max_outer = 100;
        let tol = 1e-6;

        let mut best_ll = self.log_likelihood_from_params(&params, data);

        // Nelder-Mead style coordinate ascent on log-variance parameters
        for _outer in 0..max_outer {
            let prev_ll = best_ll;
            for pi in 0..np {
                // Golden section search along this coordinate
                let (best_v, best_local) = golden_section_search_1d(
                    |v| {
                        let mut p2 = params.clone();
                        p2[pi] = v;
                        self.log_likelihood_from_params(&p2, data)
                    },
                    params[pi] - 6.0,
                    params[pi] + 6.0,
                    30,
                );
                if best_local > best_ll {
                    params[pi] = best_v;
                    best_ll = best_local;
                }
            }
            if (best_ll - prev_ll).abs() < tol {
                break;
            }
        }

        // Apply final parameters
        self.apply_params(&params);
        Ok(best_ll)
    }

    /// Decompose the series into trend, seasonal, and irregular components.
    ///
    /// Returns `(trend, seasonal, irregular)`, each of length T.
    pub fn decompose(&self, data: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let n = data.len();
        if n == 0 {
            return Ok((vec![], vec![], vec![]));
        }

        let ssm = self.to_ssm();
        let obs: Vec<Vec<f64>> = data.iter().map(|&y| vec![y]).collect();
        let (sm_means, _sm_covs) = ssm.smooth(&obs)?;

        let trend_d = self.trend.state_dim();
        let seas_d = self.seasonal.as_ref().map_or(0, |s| s.state_dim());

        let mut trend_vec = Vec::with_capacity(n);
        let mut seas_vec = Vec::with_capacity(n);
        let mut irreg_vec = Vec::with_capacity(n);

        for t in 0..n {
            let level = sm_means[t][0];
            let seas_val = if seas_d > 0 {
                sm_means[t][trend_d]
            } else {
                0.0
            };
            let fitted = level + seas_val;
            let irregular = data[t] - fitted;

            trend_vec.push(level);
            seas_vec.push(seas_val);
            irreg_vec.push(irregular);
        }

        Ok((trend_vec, seas_vec, irreg_vec))
    }
}

// ---------------------------------------------------------------------------
// Golden section search (1D maximisation)
// ---------------------------------------------------------------------------

/// Find the x in [a, b] maximising f(x) using the golden section search.
fn golden_section_search_1d<F>(f: F, a: f64, b: f64, n_iter: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // 0.618...
    let mut lo = a;
    let mut hi = b;
    let mut x1 = hi - phi * (hi - lo);
    let mut x2 = lo + phi * (hi - lo);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    for _ in 0..n_iter {
        if f1 < f2 {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + phi * (hi - lo);
            f2 = f(x2);
        } else {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - phi * (hi - lo);
            f1 = f(x1);
        }
    }

    let best_x = (lo + hi) / 2.0;
    let best_f = f(best_x);
    (best_x, best_f)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn trend_data(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| 1.0 + 0.05 * i as f64 + 0.1 * (i as f64 * 0.7).sin())
            .collect()
    }

    fn seasonal_data(n: usize, period: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let trend = 1.0 + 0.02 * i as f64;
                let seas = (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                trend + seas + 0.05 * (i as f64 * 1.3).cos()
            })
            .collect()
    }

    #[test]
    fn test_local_level_to_ssm() {
        let m = StructuralModel::local_level(0.1, 0.5);
        assert_eq!(m.state_dim(), 1);
        let ssm = m.to_ssm();
        assert_eq!(ssm.dim_state, 1);
        assert_eq!(ssm.f_mat[0][0], 1.0);
        assert!((ssm.q_mat[0][0] - 0.1).abs() < 1e-10);
        assert!((ssm.r_mat[0][0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_local_linear_trend_to_ssm() {
        let m = StructuralModel::local_linear_trend(0.1, 0.01, 0.5);
        assert_eq!(m.state_dim(), 2);
        let ssm = m.to_ssm();
        // F = [[1,1],[0,1]]
        assert_eq!(ssm.f_mat[0][0], 1.0);
        assert_eq!(ssm.f_mat[0][1], 1.0);
        assert_eq!(ssm.f_mat[1][0], 0.0);
        assert_eq!(ssm.f_mat[1][1], 1.0);
    }

    #[test]
    fn test_seasonal_to_ssm() {
        let m = StructuralModel::new(Some(4)).expect("ok");
        let ssm = m.to_ssm();
        // state dim = 1 (local level, slope_var=0.01 > 0 so dim=2) + 3 seasonal
        // Actually from new(): slope_var=0.01 > 0, so trend_d=2, seas_d=3, total=5
        assert_eq!(ssm.dim_state, m.state_dim());
    }

    #[test]
    fn test_decompose_local_level() {
        let data = trend_data(40);
        let m = StructuralModel::local_level(0.2, 0.1);
        let (trend, seas, irreg) = m.decompose(&data).expect("decompose ok");
        assert_eq!(trend.len(), 40);
        assert_eq!(seas.len(), 40);
        assert_eq!(irreg.len(), 40);
        // Seasonal should be zero (no seasonal component)
        for &s in &seas {
            assert_eq!(s, 0.0);
        }
        // Reconstruction check: trend + seasonal + irregular ≈ data
        for i in 0..40 {
            let recon = trend[i] + seas[i] + irreg[i];
            assert!(
                (recon - data[i]).abs() < 1e-6,
                "Reconstruction failed at {i}"
            );
        }
    }

    #[test]
    fn test_decompose_seasonal() {
        let data = seasonal_data(48, 4);
        let m = StructuralModel::new(Some(4)).expect("ok");
        let (trend, _seas, irreg) = m.decompose(&data).expect("decompose ok");
        assert_eq!(trend.len(), 48);
        assert_eq!(irreg.len(), 48);
    }

    #[test]
    fn test_fit_level_extraction() {
        // Constant level plus noise: Kalman filter should track the level
        let data: Vec<f64> = (0..30)
            .map(|i| 5.0 + 0.1 * ((i as f64) * 1.23).sin())
            .collect();
        let mut m = StructuralModel::local_level(0.05, 0.2);
        let ll = m.fit(&data).expect("fit ok");
        assert!(ll.is_finite());

        let (trend, _seas, _irreg) = m.decompose(&data).expect("decompose ok");
        // After fitting, filtered level should stay close to 5.0
        let level_mean: f64 = trend[10..30].iter().sum::<f64>() / 20.0;
        assert!(
            (level_mean - 5.0).abs() < 1.0,
            "Level mean {level_mean} far from 5.0"
        );
    }

    #[test]
    fn test_seasonal_component_creation() {
        let s = SeasonalComponent::new(12, 0.05).expect("ok");
        assert_eq!(s.period, 12);
        assert_eq!(s.state_dim(), 11);
        assert_eq!(s.values.len(), 11);
    }

    #[test]
    fn test_new_with_period() {
        let m = StructuralModel::new(Some(7)).expect("ok");
        assert!(m.seasonal.is_some());
        let seas = m.seasonal.as_ref().expect("some");
        assert_eq!(seas.period, 7);
        assert_eq!(seas.state_dim(), 6);
    }

    #[test]
    fn test_new_without_period() {
        let m = StructuralModel::new(None).expect("ok");
        assert!(m.seasonal.is_none());
    }
}
