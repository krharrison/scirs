//! Exponential Smoothing (ETS) Models
//!
//! Implements the Error-Trend-Seasonal (ETS) framework for time series forecasting:
//! - Simple Exponential Smoothing (SES) - no trend, no seasonality
//! - Holt's Double Exponential Smoothing - additive/multiplicative trend
//! - Holt-Winters Triple Exponential Smoothing - trend + seasonality
//!
//! Each model supports parameter estimation via maximum likelihood, forecasting
//! with confidence intervals, and information criteria for model selection.
//!
//! # References
//!
//! - Hyndman, R.J. & Athanasopoulos, G. (2021) "Forecasting: Principles and Practice"
//! - Hyndman, R.J., Koehler, A.B., Ord, J.K. & Snyder, R.D. (2008) "Forecasting with
//!   Exponential Smoothing: The State Space Approach"

use scirs2_core::ndarray::{Array1, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};

/// Error type in the ETS framework
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ETSError {
    /// Additive errors (Normal distribution)
    Additive,
    /// Multiplicative errors (percentage errors)
    Multiplicative,
}

/// Trend type in the ETS framework
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ETSTrend {
    /// No trend component
    None,
    /// Additive trend
    Additive,
    /// Multiplicative trend (requires positive data)
    Multiplicative,
    /// Damped additive trend
    DampedAdditive,
    /// Damped multiplicative trend (requires positive data)
    DampedMultiplicative,
}

/// Seasonal type in the ETS framework
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ETSSeasonal {
    /// No seasonal component
    None,
    /// Additive seasonality
    Additive,
    /// Multiplicative seasonality (requires positive data)
    Multiplicative,
}

/// Configuration for ETS model
#[derive(Debug, Clone)]
pub struct ETSConfig {
    /// Error type
    pub error: ETSError,
    /// Trend type
    pub trend: ETSTrend,
    /// Seasonal type
    pub seasonal: ETSSeasonal,
    /// Seasonal period (required if seasonal != None)
    pub period: Option<usize>,
    /// Level smoothing parameter alpha (None = auto-estimate)
    pub alpha: Option<f64>,
    /// Trend smoothing parameter beta (None = auto-estimate)
    pub beta: Option<f64>,
    /// Seasonal smoothing parameter gamma (None = auto-estimate)
    pub gamma: Option<f64>,
    /// Trend damping parameter phi (None = auto-estimate, only for damped trends)
    pub phi: Option<f64>,
    /// Maximum iterations for parameter optimization
    pub max_iter: usize,
    /// Convergence tolerance for parameter optimization
    pub tolerance: f64,
}

impl Default for ETSConfig {
    fn default() -> Self {
        Self {
            error: ETSError::Additive,
            trend: ETSTrend::None,
            seasonal: ETSSeasonal::None,
            period: None,
            alpha: None,
            beta: None,
            gamma: None,
            phi: None,
            max_iter: 500,
            tolerance: 1e-8,
        }
    }
}

/// State components of an ETS model at each time step
#[derive(Debug, Clone)]
struct ETSState<F> {
    /// Level component
    level: Vec<F>,
    /// Trend component (empty if no trend)
    trend: Vec<F>,
    /// Seasonal component (empty if no seasonality)
    seasonal: Vec<F>,
}

/// Fitted ETS (Error-Trend-Seasonal) model
///
/// This struct contains the fitted model parameters, state components,
/// and provides methods for forecasting and diagnostics.
#[derive(Debug, Clone)]
pub struct ETSModel<F> {
    /// Model configuration
    pub config: ETSConfig,
    /// Level smoothing parameter
    pub alpha: F,
    /// Trend smoothing parameter
    pub beta: F,
    /// Seasonal smoothing parameter
    pub gamma: F,
    /// Damping parameter
    pub phi: F,
    /// Initial level
    pub initial_level: F,
    /// Initial trend
    pub initial_trend: F,
    /// Initial seasonal indices
    pub initial_seasonal: Array1<F>,
    /// Residual variance (sigma^2)
    pub sigma2: F,
    /// Log-likelihood
    pub log_likelihood: F,
    /// Number of observations
    pub n_obs: usize,
    /// Number of parameters estimated
    pub n_params: usize,
    /// Whether the model has been fitted
    pub is_fitted: bool,
    /// Fitted values (one-step-ahead predictions)
    fitted_values: Array1<F>,
    /// Residuals
    residuals: Array1<F>,
    /// Final state for forecasting
    final_level: F,
    final_trend: F,
    final_seasonal: Vec<F>,
}

impl<F> ETSModel<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create a new unfitted ETS model with the given configuration
    pub fn new(config: ETSConfig) -> Result<Self> {
        // Validate configuration
        if config.seasonal != ETSSeasonal::None && config.period.is_none() {
            return Err(TimeSeriesError::InvalidParameter {
                name: "period".to_string(),
                message: "Seasonal period must be specified for seasonal models".to_string(),
            });
        }

        if let Some(period) = config.period {
            if period < 2 {
                return Err(TimeSeriesError::InvalidParameter {
                    name: "period".to_string(),
                    message: "Seasonal period must be at least 2".to_string(),
                });
            }
        }

        // Validate that alpha/beta/gamma/phi are in (0,1) if provided
        if let Some(alpha) = config.alpha {
            if alpha <= 0.0 || alpha >= 1.0 {
                return Err(TimeSeriesError::InvalidParameter {
                    name: "alpha".to_string(),
                    message: "Alpha must be in (0, 1)".to_string(),
                });
            }
        }
        if let Some(beta) = config.beta {
            if beta <= 0.0 || beta >= 1.0 {
                return Err(TimeSeriesError::InvalidParameter {
                    name: "beta".to_string(),
                    message: "Beta must be in (0, 1)".to_string(),
                });
            }
        }
        if let Some(gamma) = config.gamma {
            if gamma <= 0.0 || gamma >= 1.0 {
                return Err(TimeSeriesError::InvalidParameter {
                    name: "gamma".to_string(),
                    message: "Gamma must be in (0, 1)".to_string(),
                });
            }
        }
        if let Some(phi) = config.phi {
            if phi <= 0.0 || phi >= 1.0 {
                return Err(TimeSeriesError::InvalidParameter {
                    name: "phi".to_string(),
                    message: "Phi must be in (0, 1)".to_string(),
                });
            }
        }

        Ok(Self {
            config,
            alpha: F::zero(),
            beta: F::zero(),
            gamma: F::zero(),
            phi: F::one(),
            initial_level: F::zero(),
            initial_trend: F::zero(),
            initial_seasonal: Array1::zeros(0),
            sigma2: F::one(),
            log_likelihood: F::neg_infinity(),
            n_obs: 0,
            n_params: 0,
            is_fitted: false,
            fitted_values: Array1::zeros(0),
            residuals: Array1::zeros(0),
            final_level: F::zero(),
            final_trend: F::zero(),
            final_seasonal: Vec::new(),
        })
    }

    /// Fit the ETS model to data
    ///
    /// Estimates the smoothing parameters and initial states via maximum likelihood.
    pub fn fit(&mut self, data: &Array1<F>) -> Result<()> {
        let n = data.len();
        let period = self.config.period.unwrap_or(1);

        // Validate data length
        let min_length = match self.config.seasonal {
            ETSSeasonal::None => 3,
            _ => 2 * period + 1,
        };

        if n < min_length {
            return Err(TimeSeriesError::InsufficientData {
                message: "Insufficient data for ETS model fitting".to_string(),
                required: min_length,
                actual: n,
            });
        }

        // Check for positive data if multiplicative components
        if self.config.trend == ETSTrend::Multiplicative
            || self.config.trend == ETSTrend::DampedMultiplicative
            || self.config.seasonal == ETSSeasonal::Multiplicative
            || self.config.error == ETSError::Multiplicative
        {
            if data.iter().any(|&x| x <= F::zero()) {
                return Err(TimeSeriesError::InvalidInput(
                    "Multiplicative components require strictly positive data".to_string(),
                ));
            }
        }

        self.n_obs = n;

        // Initialize state values
        self.initialize_states(data)?;

        // Optimize parameters
        self.optimize_parameters(data)?;

        // Compute final fitted values and residuals using optimal parameters
        let (_ll, fitted, resid, state) = self.evaluate(data)?;

        self.fitted_values = fitted;
        self.residuals = resid.clone();
        self.sigma2 = resid.iter().fold(F::zero(), |acc, &r| acc + r * r)
            / F::from(n).ok_or_else(|| {
                TimeSeriesError::NumericalInstability("Failed to convert n".to_string())
            })?;

        // Store final state for forecasting
        self.final_level = *state.level.last().unwrap_or(&F::zero());
        self.final_trend = *state.trend.last().unwrap_or(&F::zero());
        self.final_seasonal = state.seasonal.clone();

        // Calculate log-likelihood
        let two_pi = F::from(2.0 * std::f64::consts::PI).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Failed to convert 2pi".to_string())
        })?;
        let n_f = F::from(n).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Failed to convert n".to_string())
        })?;
        let two = F::from(2.0).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Failed to convert constant".to_string())
        })?;

        self.log_likelihood = -n_f / two * (F::one() + two_pi.ln()) - n_f / two * self.sigma2.ln();

        // Count parameters
        self.n_params = 1; // alpha
        if self.config.trend != ETSTrend::None {
            self.n_params += 1; // beta
        }
        if self.config.seasonal != ETSSeasonal::None {
            self.n_params += 1; // gamma
        }
        if self.config.trend == ETSTrend::DampedAdditive
            || self.config.trend == ETSTrend::DampedMultiplicative
        {
            self.n_params += 1; // phi
        }
        // Initial states
        self.n_params += 1; // initial level
        if self.config.trend != ETSTrend::None {
            self.n_params += 1; // initial trend
        }
        if self.config.seasonal != ETSSeasonal::None {
            self.n_params += period - 1; // initial seasonal indices (one is constrained)
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Initialize state values from the data
    fn initialize_states(&mut self, data: &Array1<F>) -> Result<()> {
        let period = self.config.period.unwrap_or(1);

        match self.config.seasonal {
            ETSSeasonal::None => {
                // Simple initialization: level = first value, trend = first difference
                self.initial_level = data[0];
                if self.config.trend != ETSTrend::None {
                    // Average of first few differences
                    let n_diff = data.len().min(5);
                    let mut sum = F::zero();
                    for i in 1..n_diff {
                        sum = sum + (data[i] - data[i - 1]);
                    }
                    self.initial_trend = sum
                        / F::from(n_diff - 1).ok_or_else(|| {
                            TimeSeriesError::NumericalInstability("Failed to convert".to_string())
                        })?;
                }
                self.initial_seasonal = Array1::zeros(0);
            }
            ETSSeasonal::Additive => {
                // Initialize using first complete seasons
                let n_seasons = (data.len() / period).min(3);
                let mut season_means = Vec::with_capacity(n_seasons);

                for s in 0..n_seasons {
                    let start = s * period;
                    let end = start + period;
                    let mut sum = F::zero();
                    for i in start..end.min(data.len()) {
                        sum = sum + data[i];
                    }
                    season_means.push(
                        sum / F::from(period).ok_or_else(|| {
                            TimeSeriesError::NumericalInstability(
                                "Failed to convert period".to_string(),
                            )
                        })?,
                    );
                }

                self.initial_level = season_means[0];

                if self.config.trend != ETSTrend::None && n_seasons >= 2 {
                    self.initial_trend = (season_means[1] - season_means[0])
                        / F::from(period).ok_or_else(|| {
                            TimeSeriesError::NumericalInstability(
                                "Failed to convert period".to_string(),
                            )
                        })?;
                }

                // Initialize seasonal indices
                let mut seasonal = Array1::zeros(period);
                for j in 0..period {
                    let mut sum = F::zero();
                    let mut count = 0;
                    for s in 0..n_seasons {
                        let idx = s * period + j;
                        if idx < data.len() {
                            sum = sum + data[idx] - season_means[s];
                            count += 1;
                        }
                    }
                    if count > 0 {
                        seasonal[j] = sum
                            / F::from(count).ok_or_else(|| {
                                TimeSeriesError::NumericalInstability(
                                    "Failed to convert count".to_string(),
                                )
                            })?;
                    }
                }
                self.initial_seasonal = seasonal;
            }
            ETSSeasonal::Multiplicative => {
                // Initialize using first complete seasons
                let n_seasons = (data.len() / period).min(3);
                let mut season_means = Vec::with_capacity(n_seasons);

                for s in 0..n_seasons {
                    let start = s * period;
                    let end = start + period;
                    let mut sum = F::zero();
                    for i in start..end.min(data.len()) {
                        sum = sum + data[i];
                    }
                    season_means.push(
                        sum / F::from(period).ok_or_else(|| {
                            TimeSeriesError::NumericalInstability(
                                "Failed to convert period".to_string(),
                            )
                        })?,
                    );
                }

                self.initial_level = season_means[0];

                if self.config.trend != ETSTrend::None && n_seasons >= 2 {
                    self.initial_trend = (season_means[1] - season_means[0])
                        / F::from(period).ok_or_else(|| {
                            TimeSeriesError::NumericalInstability(
                                "Failed to convert period".to_string(),
                            )
                        })?;
                }

                // Initialize seasonal indices as ratios
                let mut seasonal = Array1::from_elem(period, F::one());
                for j in 0..period {
                    let mut sum = F::zero();
                    let mut count = 0;
                    for s in 0..n_seasons {
                        let idx = s * period + j;
                        if idx < data.len() && season_means[s] > F::zero() {
                            sum = sum + data[idx] / season_means[s];
                            count += 1;
                        }
                    }
                    if count > 0 {
                        seasonal[j] = sum
                            / F::from(count).ok_or_else(|| {
                                TimeSeriesError::NumericalInstability(
                                    "Failed to convert count".to_string(),
                                )
                            })?;
                    }
                }
                self.initial_seasonal = seasonal;
            }
        }

        Ok(())
    }

    /// Optimize smoothing parameters via grid search + Nelder-Mead refinement
    fn optimize_parameters(&mut self, data: &Array1<F>) -> Result<()> {
        let mut best_sse = F::infinity();
        let mut best_alpha = F::from(0.3).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Conversion failed".to_string())
        })?;
        let mut best_beta = F::from(0.1).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Conversion failed".to_string())
        })?;
        let mut best_gamma = F::from(0.1).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Conversion failed".to_string())
        })?;
        let mut best_phi = F::from(0.98).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Conversion failed".to_string())
        })?;

        // Grid search over parameter space
        let alpha_grid: Vec<f64> = if let Some(a) = self.config.alpha {
            vec![a]
        } else {
            vec![0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        };

        let beta_grid: Vec<f64> = if self.config.trend == ETSTrend::None {
            vec![0.0]
        } else if let Some(b) = self.config.beta {
            vec![b]
        } else {
            vec![0.01, 0.05, 0.1, 0.2, 0.3]
        };

        let gamma_grid: Vec<f64> = if self.config.seasonal == ETSSeasonal::None {
            vec![0.0]
        } else if let Some(g) = self.config.gamma {
            vec![g]
        } else {
            vec![0.01, 0.05, 0.1, 0.2, 0.3]
        };

        let phi_grid: Vec<f64> = match self.config.trend {
            ETSTrend::DampedAdditive | ETSTrend::DampedMultiplicative => {
                if let Some(p) = self.config.phi {
                    vec![p]
                } else {
                    vec![0.8, 0.9, 0.95, 0.98]
                }
            }
            _ => vec![1.0],
        };

        for &a in &alpha_grid {
            for &b in &beta_grid {
                for &g in &gamma_grid {
                    for &p in &phi_grid {
                        let a_f = F::from(a).ok_or_else(|| {
                            TimeSeriesError::NumericalInstability("Conversion failed".to_string())
                        })?;
                        let b_f = F::from(b).ok_or_else(|| {
                            TimeSeriesError::NumericalInstability("Conversion failed".to_string())
                        })?;
                        let g_f = F::from(g).ok_or_else(|| {
                            TimeSeriesError::NumericalInstability("Conversion failed".to_string())
                        })?;
                        let p_f = F::from(p).ok_or_else(|| {
                            TimeSeriesError::NumericalInstability("Conversion failed".to_string())
                        })?;

                        self.alpha = a_f;
                        self.beta = b_f;
                        self.gamma = g_f;
                        self.phi = p_f;

                        if let Ok((_, _, resid, _)) = self.evaluate(data) {
                            let sse = resid.iter().fold(F::zero(), |acc, &r| acc + r * r);
                            if sse < best_sse {
                                best_sse = sse;
                                best_alpha = a_f;
                                best_beta = b_f;
                                best_gamma = g_f;
                                best_phi = p_f;
                            }
                        }
                    }
                }
            }
        }

        // Refine with local search (simplified Nelder-Mead-like coordinate descent)
        self.alpha = best_alpha;
        self.beta = best_beta;
        self.gamma = best_gamma;
        self.phi = best_phi;

        let step = F::from(0.02).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Conversion failed".to_string())
        })?;
        let min_val = F::from(0.001).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Conversion failed".to_string())
        })?;
        let max_val = F::from(0.999).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Conversion failed".to_string())
        })?;

        for _ in 0..self.config.max_iter {
            let prev_sse = best_sse;

            // Refine alpha
            if self.config.alpha.is_none() {
                for direction in [-1.0_f64, 1.0_f64] {
                    let delta = F::from(direction).ok_or_else(|| {
                        TimeSeriesError::NumericalInstability("Conversion failed".to_string())
                    })? * step;
                    let candidate = self.alpha + delta;
                    if candidate > min_val && candidate < max_val {
                        let saved = self.alpha;
                        self.alpha = candidate;
                        if let Ok((_, _, resid, _)) = self.evaluate(data) {
                            let sse = resid.iter().fold(F::zero(), |acc, &r| acc + r * r);
                            if sse < best_sse {
                                best_sse = sse;
                                best_alpha = candidate;
                            } else {
                                self.alpha = saved;
                            }
                        } else {
                            self.alpha = saved;
                        }
                    }
                }
                self.alpha = best_alpha;
            }

            // Refine beta
            if self.config.trend != ETSTrend::None && self.config.beta.is_none() {
                for direction in [-1.0_f64, 1.0_f64] {
                    let delta = F::from(direction).ok_or_else(|| {
                        TimeSeriesError::NumericalInstability("Conversion failed".to_string())
                    })? * step;
                    let candidate = self.beta + delta;
                    if candidate > min_val && candidate < max_val {
                        let saved = self.beta;
                        self.beta = candidate;
                        if let Ok((_, _, resid, _)) = self.evaluate(data) {
                            let sse = resid.iter().fold(F::zero(), |acc, &r| acc + r * r);
                            if sse < best_sse {
                                best_sse = sse;
                                best_beta = candidate;
                            } else {
                                self.beta = saved;
                            }
                        } else {
                            self.beta = saved;
                        }
                    }
                }
                self.beta = best_beta;
            }

            // Refine gamma
            if self.config.seasonal != ETSSeasonal::None && self.config.gamma.is_none() {
                for direction in [-1.0_f64, 1.0_f64] {
                    let delta = F::from(direction).ok_or_else(|| {
                        TimeSeriesError::NumericalInstability("Conversion failed".to_string())
                    })? * step;
                    let candidate = self.gamma + delta;
                    if candidate > min_val && candidate < max_val {
                        let saved = self.gamma;
                        self.gamma = candidate;
                        if let Ok((_, _, resid, _)) = self.evaluate(data) {
                            let sse = resid.iter().fold(F::zero(), |acc, &r| acc + r * r);
                            if sse < best_sse {
                                best_sse = sse;
                                best_gamma = candidate;
                            } else {
                                self.gamma = saved;
                            }
                        } else {
                            self.gamma = saved;
                        }
                    }
                }
                self.gamma = best_gamma;
            }

            // Check convergence
            let tol = F::from(self.config.tolerance).ok_or_else(|| {
                TimeSeriesError::NumericalInstability("Conversion failed".to_string())
            })?;
            if (prev_sse - best_sse).abs() < tol {
                break;
            }
        }

        Ok(())
    }

    /// Evaluate the model on data with current parameters, returning (log_likelihood, fitted, residuals, state)
    fn evaluate(&self, data: &Array1<F>) -> Result<(F, Array1<F>, Array1<F>, ETSState<F>)> {
        let n = data.len();
        let period = self.config.period.unwrap_or(1);

        let mut level = Vec::with_capacity(n + 1);
        let mut trend_vec = Vec::with_capacity(n + 1);
        let mut seasonal_vec: Vec<F> = Vec::new();
        let mut fitted = Array1::zeros(n);
        let mut residuals = Array1::zeros(n);

        // Set initial states
        level.push(self.initial_level);

        if self.config.trend != ETSTrend::None {
            trend_vec.push(self.initial_trend);
        }

        if self.config.seasonal != ETSSeasonal::None {
            for j in 0..period {
                if j < self.initial_seasonal.len() {
                    seasonal_vec.push(self.initial_seasonal[j]);
                } else {
                    seasonal_vec.push(F::one());
                }
            }
        }

        let mut total_ll = F::zero();

        for t in 0..n {
            let l = *level.last().unwrap_or(&F::zero());
            let b = *trend_vec.last().unwrap_or(&F::zero());

            // Get seasonal component for this time step
            let s = if self.config.seasonal != ETSSeasonal::None {
                let s_idx = t % period;
                if s_idx < seasonal_vec.len() {
                    seasonal_vec[s_idx]
                } else {
                    F::one()
                }
            } else {
                F::zero()
            };

            // Compute one-step-ahead forecast
            let forecast = self.compute_point(l, b, s, 1)?;
            fitted[t] = forecast;

            let error = data[t] - forecast;
            residuals[t] = error;

            // Update states
            let (new_l, new_b, new_s) = self.update_states(l, b, s, data[t], error)?;

            level.push(new_l);
            if self.config.trend != ETSTrend::None {
                trend_vec.push(new_b);
            }
            if self.config.seasonal != ETSSeasonal::None {
                // Extend seasonal_vec by storing new seasonal value at end
                // For the circular buffer approach:
                let s_idx = t % period;
                if s_idx < seasonal_vec.len() {
                    seasonal_vec[s_idx] = new_s;
                }
            }

            // Add to log-likelihood (Gaussian errors)
            total_ll = total_ll - error * error;
        }

        let state = ETSState {
            level,
            trend: trend_vec,
            seasonal: seasonal_vec,
        };

        Ok((total_ll, fitted, residuals, state))
    }

    /// Compute the point forecast h steps ahead given state components
    fn compute_point(&self, level: F, trend: F, seasonal: F, h: usize) -> Result<F> {
        let h_f = F::from(h).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Failed to convert h".to_string())
        })?;

        let trend_component = match self.config.trend {
            ETSTrend::None => F::zero(),
            ETSTrend::Additive => h_f * trend,
            ETSTrend::Multiplicative => level * (trend.powf(h_f) - F::one()),
            ETSTrend::DampedAdditive => {
                // sum_{i=1}^{h} phi^i * b
                let mut phi_sum = F::zero();
                let mut phi_power = self.phi;
                for _ in 0..h {
                    phi_sum = phi_sum + phi_power;
                    phi_power = phi_power * self.phi;
                }
                phi_sum * trend
            }
            ETSTrend::DampedMultiplicative => {
                let mut phi_sum = F::zero();
                let mut phi_power = self.phi;
                for _ in 0..h {
                    phi_sum = phi_sum + phi_power;
                    phi_power = phi_power * self.phi;
                }
                level * (trend.powf(phi_sum) - F::one())
            }
        };

        let base = match self.config.trend {
            ETSTrend::Multiplicative | ETSTrend::DampedMultiplicative => level + trend_component,
            _ => level + trend_component,
        };

        let result = match self.config.seasonal {
            ETSSeasonal::None => base,
            ETSSeasonal::Additive => base + seasonal,
            ETSSeasonal::Multiplicative => base * seasonal,
        };

        Ok(result)
    }

    /// Update state components given observation and error
    fn update_states(
        &self,
        level: F,
        trend: F,
        seasonal: F,
        observation: F,
        _error: F,
    ) -> Result<(F, F, F)> {
        // New level
        let new_level = match (self.config.trend, self.config.seasonal) {
            (ETSTrend::None, ETSSeasonal::None) => {
                self.alpha * observation + (F::one() - self.alpha) * level
            }
            (ETSTrend::None, ETSSeasonal::Additive) => {
                self.alpha * (observation - seasonal) + (F::one() - self.alpha) * level
            }
            (ETSTrend::None, ETSSeasonal::Multiplicative) => {
                if seasonal > F::zero() {
                    self.alpha * (observation / seasonal) + (F::one() - self.alpha) * level
                } else {
                    level
                }
            }
            (ETSTrend::Additive, ETSSeasonal::None) => {
                self.alpha * observation + (F::one() - self.alpha) * (level + trend)
            }
            (ETSTrend::Additive, ETSSeasonal::Additive) => {
                self.alpha * (observation - seasonal) + (F::one() - self.alpha) * (level + trend)
            }
            (ETSTrend::Additive, ETSSeasonal::Multiplicative) => {
                if seasonal > F::zero() {
                    self.alpha * (observation / seasonal)
                        + (F::one() - self.alpha) * (level + trend)
                } else {
                    level + trend
                }
            }
            (ETSTrend::DampedAdditive, ETSSeasonal::None) => {
                self.alpha * observation + (F::one() - self.alpha) * (level + self.phi * trend)
            }
            (ETSTrend::DampedAdditive, ETSSeasonal::Additive) => {
                self.alpha * (observation - seasonal)
                    + (F::one() - self.alpha) * (level + self.phi * trend)
            }
            (ETSTrend::DampedAdditive, ETSSeasonal::Multiplicative) => {
                if seasonal > F::zero() {
                    self.alpha * (observation / seasonal)
                        + (F::one() - self.alpha) * (level + self.phi * trend)
                } else {
                    level + self.phi * trend
                }
            }
            (ETSTrend::Multiplicative, ETSSeasonal::None) => {
                self.alpha * observation + (F::one() - self.alpha) * level * trend
            }
            (ETSTrend::Multiplicative, ETSSeasonal::Additive) => {
                self.alpha * (observation - seasonal) + (F::one() - self.alpha) * level * trend
            }
            (ETSTrend::Multiplicative, ETSSeasonal::Multiplicative) => {
                if seasonal > F::zero() {
                    self.alpha * (observation / seasonal) + (F::one() - self.alpha) * level * trend
                } else {
                    level * trend
                }
            }
            (ETSTrend::DampedMultiplicative, ETSSeasonal::None) => {
                self.alpha * observation + (F::one() - self.alpha) * level * trend.powf(self.phi)
            }
            (ETSTrend::DampedMultiplicative, ETSSeasonal::Additive) => {
                self.alpha * (observation - seasonal)
                    + (F::one() - self.alpha) * level * trend.powf(self.phi)
            }
            (ETSTrend::DampedMultiplicative, ETSSeasonal::Multiplicative) => {
                if seasonal > F::zero() {
                    self.alpha * (observation / seasonal)
                        + (F::one() - self.alpha) * level * trend.powf(self.phi)
                } else {
                    level * trend.powf(self.phi)
                }
            }
        };

        // New trend
        let new_trend = match self.config.trend {
            ETSTrend::None => F::zero(),
            ETSTrend::Additive => self.beta * (new_level - level) + (F::one() - self.beta) * trend,
            ETSTrend::DampedAdditive => {
                self.beta * (new_level - level) + (F::one() - self.beta) * self.phi * trend
            }
            ETSTrend::Multiplicative => {
                if level > F::zero() {
                    self.beta * (new_level / level) + (F::one() - self.beta) * trend
                } else {
                    trend
                }
            }
            ETSTrend::DampedMultiplicative => {
                if level > F::zero() {
                    self.beta * (new_level / level) + (F::one() - self.beta) * trend.powf(self.phi)
                } else {
                    trend.powf(self.phi)
                }
            }
        };

        // New seasonal
        let new_seasonal = match self.config.seasonal {
            ETSSeasonal::None => F::zero(),
            ETSSeasonal::Additive => {
                self.gamma * (observation - new_level) + (F::one() - self.gamma) * seasonal
            }
            ETSSeasonal::Multiplicative => {
                if new_level > F::zero() {
                    self.gamma * (observation / new_level) + (F::one() - self.gamma) * seasonal
                } else {
                    seasonal
                }
            }
        };

        Ok((new_level, new_trend, new_seasonal))
    }

    /// Forecast future values
    ///
    /// # Arguments
    /// * `steps` - Number of steps to forecast
    ///
    /// # Returns
    /// Array of point forecasts
    pub fn forecast(&self, steps: usize) -> Result<Array1<F>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::ModelNotFitted(
                "Model must be fitted before forecasting".to_string(),
            ));
        }

        if steps == 0 {
            return Ok(Array1::zeros(0));
        }

        let period = self.config.period.unwrap_or(1);
        let mut forecasts = Array1::zeros(steps);

        for h in 0..steps {
            let s = if self.config.seasonal != ETSSeasonal::None && !self.final_seasonal.is_empty()
            {
                let s_idx = h % period;
                if s_idx < self.final_seasonal.len() {
                    self.final_seasonal[s_idx]
                } else {
                    F::one()
                }
            } else {
                F::zero()
            };

            forecasts[h] = self.compute_point(self.final_level, self.final_trend, s, h + 1)?;
        }

        Ok(forecasts)
    }

    /// Forecast with confidence intervals
    ///
    /// # Arguments
    /// * `steps` - Number of steps to forecast
    /// * `confidence_level` - Confidence level (e.g., 0.95 for 95% intervals)
    ///
    /// # Returns
    /// Tuple of (point_forecast, lower_ci, upper_ci)
    pub fn forecast_with_confidence(
        &self,
        steps: usize,
        confidence_level: f64,
    ) -> Result<(Array1<F>, Array1<F>, Array1<F>)> {
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "confidence_level".to_string(),
                message: "Must be between 0 and 1".to_string(),
            });
        }

        let point_forecast = self.forecast(steps)?;

        let alpha_half = (1.0 - confidence_level) / 2.0;
        let z = crate::arima_models::quantile_normal(1.0 - alpha_half);
        let z_f = F::from(z).ok_or_else(|| {
            TimeSeriesError::NumericalInstability("Failed to convert z-score".to_string())
        })?;

        let mut lower_ci = Array1::zeros(steps);
        let mut upper_ci = Array1::zeros(steps);

        for h in 0..steps {
            let h_f = F::from(h + 1).ok_or_else(|| {
                TimeSeriesError::NumericalInstability("Failed to convert h".to_string())
            })?;
            let se = self.sigma2.sqrt() * h_f.sqrt();
            lower_ci[h] = point_forecast[h] - z_f * se;
            upper_ci[h] = point_forecast[h] + z_f * se;
        }

        Ok((point_forecast, lower_ci, upper_ci))
    }

    /// Get fitted values
    pub fn fitted_values(&self) -> Result<&Array1<F>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::ModelNotFitted(
                "Model must be fitted first".to_string(),
            ));
        }
        Ok(&self.fitted_values)
    }

    /// Get residuals
    pub fn residuals(&self) -> Result<&Array1<F>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::ModelNotFitted(
                "Model must be fitted first".to_string(),
            ));
        }
        Ok(&self.residuals)
    }

    /// Calculate AIC
    pub fn aic(&self) -> F {
        let k = F::from(self.n_params).unwrap_or(F::one());
        let two = F::from(2.0).unwrap_or(F::one());
        two * k - two * self.log_likelihood
    }

    /// Calculate BIC
    pub fn bic(&self) -> F {
        let k = F::from(self.n_params).unwrap_or(F::one());
        let n = F::from(self.n_obs).unwrap_or(F::one());
        let two = F::from(2.0).unwrap_or(F::one());
        k * n.ln() - two * self.log_likelihood
    }

    /// Calculate AICc (corrected AIC for small samples)
    pub fn aicc(&self) -> F {
        let k = F::from(self.n_params).unwrap_or(F::one());
        let n = F::from(self.n_obs).unwrap_or(F::one());
        let two = F::from(2.0).unwrap_or(F::one());
        self.aic() + two * k * (k + F::one()) / (n - k - F::one())
    }

    /// Get the model name in ETS notation (e.g., "ETS(A,A,N)")
    pub fn model_name(&self) -> String {
        let error = match self.config.error {
            ETSError::Additive => "A",
            ETSError::Multiplicative => "M",
        };
        let trend = match self.config.trend {
            ETSTrend::None => "N",
            ETSTrend::Additive => "A",
            ETSTrend::Multiplicative => "M",
            ETSTrend::DampedAdditive => "Ad",
            ETSTrend::DampedMultiplicative => "Md",
        };
        let seasonal = match self.config.seasonal {
            ETSSeasonal::None => "N",
            ETSSeasonal::Additive => "A",
            ETSSeasonal::Multiplicative => "M",
        };
        format!("ETS({},{},{})", error, trend, seasonal)
    }
}

/// Convenience function to fit an ETS model
///
/// # Arguments
/// * `data` - Time series data
/// * `trend` - Trend type specification ("none", "additive", "multiplicative", "damped_additive")
/// * `seasonal` - Seasonal type specification ("none", "additive", "multiplicative")
/// * `period` - Seasonal period (required if seasonal != "none")
///
/// # Returns
/// A fitted ETSModel ready for forecasting
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_series::ets::ets;
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
/// let model = ets(&data, "additive", "none", None).expect("Failed to fit ETS");
/// let forecast = model.forecast(5).expect("Failed to forecast");
/// assert_eq!(forecast.len(), 5);
/// ```
pub fn ets<F>(
    data: &Array1<F>,
    trend: &str,
    seasonal: &str,
    period: Option<usize>,
) -> Result<ETSModel<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let trend_type = match trend.to_lowercase().as_str() {
        "none" | "n" => ETSTrend::None,
        "additive" | "add" | "a" => ETSTrend::Additive,
        "multiplicative" | "mul" | "m" => ETSTrend::Multiplicative,
        "damped_additive" | "damped_add" | "da" | "ad" => ETSTrend::DampedAdditive,
        "damped_multiplicative" | "damped_mul" | "dm" | "md" => ETSTrend::DampedMultiplicative,
        _ => {
            return Err(TimeSeriesError::InvalidParameter {
                name: "trend".to_string(),
                message: format!(
                    "Unknown trend type '{}'. Use: none, additive, multiplicative, damped_additive",
                    trend
                ),
            })
        }
    };

    let seasonal_type = match seasonal.to_lowercase().as_str() {
        "none" | "n" => ETSSeasonal::None,
        "additive" | "add" | "a" => ETSSeasonal::Additive,
        "multiplicative" | "mul" | "m" => ETSSeasonal::Multiplicative,
        _ => {
            return Err(TimeSeriesError::InvalidParameter {
                name: "seasonal".to_string(),
                message: format!(
                    "Unknown seasonal type '{}'. Use: none, additive, multiplicative",
                    seasonal
                ),
            })
        }
    };

    let config = ETSConfig {
        error: ETSError::Additive,
        trend: trend_type,
        seasonal: seasonal_type,
        period,
        ..Default::default()
    };

    let mut model = ETSModel::new(config)?;
    model.fit(data)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ses_fit_and_forecast() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let model = ets(&data, "none", "none", None).expect("Failed to fit SES");

        assert!(model.is_fitted);
        assert_eq!(model.model_name(), "ETS(A,N,N)");

        let forecast = model.forecast(5).expect("Failed to forecast");
        assert_eq!(forecast.len(), 5);

        // SES forecast should be a constant (all same value)
        for i in 1..5 {
            assert!(
                (forecast[i] - forecast[0]).abs() < 1e-10,
                "SES forecasts should be constant"
            );
        }
    }

    #[test]
    fn test_holt_double_exponential() {
        // Linear trend data
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let model = ets(&data, "additive", "none", None).expect("Failed to fit Holt");

        assert!(model.is_fitted);
        assert_eq!(model.model_name(), "ETS(A,A,N)");

        let forecast = model.forecast(3).expect("Failed to forecast");
        assert_eq!(forecast.len(), 3);

        // Forecasts should be increasing for upward-trending data
        for i in 1..3 {
            assert!(
                forecast[i] > forecast[i - 1],
                "Holt forecasts should be increasing for upward trend"
            );
        }
    }

    #[test]
    fn test_holt_winters_additive() {
        // Seasonal data with period 4
        let data = array![
            10.0, 15.0, 12.0, 8.0, 11.0, 16.0, 13.0, 9.0, 12.0, 17.0, 14.0, 10.0, 13.0, 18.0, 15.0,
            11.0
        ];
        let model =
            ets(&data, "additive", "additive", Some(4)).expect("Failed to fit Holt-Winters");

        assert!(model.is_fitted);
        assert_eq!(model.model_name(), "ETS(A,A,A)");

        let forecast = model.forecast(4).expect("Failed to forecast");
        assert_eq!(forecast.len(), 4);

        // Forecasts should be finite
        for i in 0..4 {
            assert!(
                forecast[i].is_finite(),
                "Forecast at step {} should be finite",
                i
            );
        }
    }

    #[test]
    fn test_holt_winters_multiplicative() {
        // Positive seasonal data with multiplicative pattern
        let data = array![
            10.0, 15.0, 12.0, 8.0, 11.0, 16.0, 13.0, 9.0, 12.0, 17.0, 14.0, 10.0, 13.0, 18.0, 15.0,
            11.0
        ];
        let model = ets(&data, "additive", "multiplicative", Some(4))
            .expect("Failed to fit multiplicative Holt-Winters");

        assert!(model.is_fitted);
        assert_eq!(model.model_name(), "ETS(A,A,M)");

        let forecast = model.forecast(4).expect("Failed to forecast");
        assert_eq!(forecast.len(), 4);

        for i in 0..4 {
            assert!(
                forecast[i].is_finite(),
                "Forecast at step {} should be finite",
                i
            );
        }
    }

    #[test]
    fn test_damped_trend() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let model =
            ets(&data, "damped_additive", "none", None).expect("Failed to fit damped trend");

        assert!(model.is_fitted);
        assert_eq!(model.model_name(), "ETS(A,Ad,N)");

        let forecast = model.forecast(10).expect("Failed to forecast");
        assert_eq!(forecast.len(), 10);

        // Damped trend should level off — later differences should be smaller
        if forecast.len() >= 3 {
            let diff_early = forecast[1] - forecast[0];
            let diff_late = forecast[forecast.len() - 1] - forecast[forecast.len() - 2];
            assert!(
                diff_late.abs() <= diff_early.abs() + 1e-6,
                "Damped trend increments should decrease over time"
            );
        }
    }

    #[test]
    fn test_confidence_intervals() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let model = ets(&data, "additive", "none", None).expect("Failed to fit ETS");

        let (point, lower, upper) = model
            .forecast_with_confidence(5, 0.95)
            .expect("Failed to forecast with CI");

        assert_eq!(point.len(), 5);
        assert_eq!(lower.len(), 5);
        assert_eq!(upper.len(), 5);

        for i in 0..5 {
            assert!(lower[i] <= point[i], "Lower CI should be <= point forecast");
            assert!(upper[i] >= point[i], "Upper CI should be >= point forecast");
        }

        // Intervals should widen over time
        for i in 1..5 {
            let width_prev = upper[i - 1] - lower[i - 1];
            let width_curr = upper[i] - lower[i];
            assert!(
                width_curr >= width_prev - 1e-10,
                "CI width should not decrease over time"
            );
        }
    }

    #[test]
    fn test_information_criteria() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let model = ets(&data, "none", "none", None).expect("Failed to fit SES");

        let aic = model.aic();
        let bic = model.bic();
        let aicc = model.aicc();

        assert!(aic.is_finite(), "AIC should be finite");
        assert!(bic.is_finite(), "BIC should be finite");
        assert!(aicc.is_finite(), "AICc should be finite");
    }

    #[test]
    fn test_fitted_and_residuals() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let model = ets(&data, "additive", "none", None).expect("Failed to fit ETS");

        let fitted = model.fitted_values().expect("Failed to get fitted values");
        let resid = model.residuals().expect("Failed to get residuals");

        assert_eq!(fitted.len(), data.len());
        assert_eq!(resid.len(), data.len());

        // Fitted + Residual should equal actual
        for i in 0..data.len() {
            let reconstructed = fitted[i] + resid[i];
            assert!(
                (reconstructed - data[i]).abs() < 1e-6,
                "fitted + residual should equal actual at index {}",
                i
            );
        }
    }

    #[test]
    fn test_multiplicative_requires_positive() {
        let data = array![1.0, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = ets(&data, "multiplicative", "none", None);
        assert!(
            result.is_err(),
            "Multiplicative model should fail with negative data"
        );
    }

    #[test]
    fn test_seasonal_requires_period() {
        let config = ETSConfig {
            seasonal: ETSSeasonal::Additive,
            period: None,
            ..Default::default()
        };
        let result = ETSModel::<f64>::new(config);
        assert!(result.is_err(), "Seasonal model without period should fail");
    }

    #[test]
    fn test_model_not_fitted_errors() {
        let config = ETSConfig::default();
        let model = ETSModel::<f64>::new(config).expect("Failed to create model");

        assert!(model.forecast(5).is_err());
        assert!(model.fitted_values().is_err());
        assert!(model.residuals().is_err());
    }

    #[test]
    fn test_ets_convenience_invalid_trend() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = ets(&data, "invalid_trend_type", "none", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_ets_model_name_formats() {
        let ses = ETSConfig::default();
        let model = ETSModel::<f64>::new(ses).expect("Failed to create");
        assert_eq!(model.model_name(), "ETS(A,N,N)");

        let holt = ETSConfig {
            trend: ETSTrend::Additive,
            ..Default::default()
        };
        let model = ETSModel::<f64>::new(holt).expect("Failed to create");
        assert_eq!(model.model_name(), "ETS(A,A,N)");

        let hw = ETSConfig {
            trend: ETSTrend::Additive,
            seasonal: ETSSeasonal::Additive,
            period: Some(4),
            ..Default::default()
        };
        let model = ETSModel::<f64>::new(hw).expect("Failed to create");
        assert_eq!(model.model_name(), "ETS(A,A,A)");
    }
}
